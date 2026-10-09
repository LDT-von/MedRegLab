"""
Registration Engine - 配准主引擎
整合 ConvexAdam + DINOv2 + IMPACT 的统一配准流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Union, List, Tuple
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

from .convex_optimizer import ConvexAdam
from .loss_builder import LossBuilder


class RegistrationEngine:
    """
    统一配准引擎

    支持三种模式：
    1. ConvexAdam: 纯凸优化配准
    2. DINO-Reg: DINOv2 特征 + ConvexAdam
    3. IMPACT: 语义感知配准
    """

    def __init__(
        self,
        mode: str = "dino_reg",
        device: Optional[str] = None,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Args:
            mode: 配准模式
                - "convex": 纯 ConvexAdam
                - "dino_reg": DINOv2 + ConvexAdam
                - "impact": IMPACT 语义损失 + ConvexAdam
            device: 计算设备
            verbose: 是否打印进度
            **kwargs: 其他参数传递给各组件
        """
        self.mode = mode
        self.verbose = verbose
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.kwargs = kwargs

        # 初始化组件
        self._init_components()

    def _init_components(self):
        """初始化各组件"""
        # 优化器
        convex_kwargs = self.kwargs.get("convex", {})
        self.optimizer = ConvexAdam(**convex_kwargs).to(self.device)

        # 损失函数构建器
        loss_kwargs = self.kwargs.get("loss", {})
        self.loss_builder = LossBuilder(**loss_kwargs)

        # DINOv2 特征提取器
        if self.mode in ["dino_reg", "impact"]:
            self._init_dino_extractor()

    def _init_dino_extractor(self):
        """初始化 DINOv2 特征提取器"""
        dino_kwargs = self.kwargs.get("dino", {})
        model_name = dino_kwargs.get("model_name", "dinov2_vitl14")

        # 延迟导入以避免不必要的依赖
        try:
            import timm

            self.has_timm = True
        except ImportError:
            self.has_timm = False
            print(
                "Warning: timm not installed. DINOv2 features will use identity mapping."
            )

    def register(
        self,
        moving: Union[str, Path, torch.Tensor, np.ndarray, sitk.Image],
        fixed: Union[str, Path, torch.Tensor, np.ndarray, sitk.Image],
        output_dir: Optional[Union[str, Path]] = None,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        执行单例配准

        Args:
            moving: 浮动图像
            fixed: 参考图像
            output_dir: 输出目录
            save_results: 是否保存结果

        Returns:
            dict: 配准结果
        """
        # 加载图像
        moving_img, moving_tensor = self._load_image(moving)
        fixed_img, fixed_tensor = self._load_image(fixed)

        # 预处理
        moving_tensor = self._preprocess(moving_tensor)
        fixed_tensor = self._preprocess(fixed_tensor)

        # 特征提取（如果使用 DINO）
        if self.mode in ["dino_reg", "impact"] and self.has_timm:
            moving_features = self._extract_features(moving_tensor)
            fixed_features = self._extract_features(fixed_tensor)
        else:
            moving_features = moving_tensor
            fixed_features = fixed_tensor

        # 执行配准
        if self.verbose:
            print(f"[MedRegLab] Running {self.mode} registration...")

        result = self.optimizer(
            moving_features, fixed_features, mask=self._get_mask(fixed_tensor)
        )

        # 后处理
        result = self._postprocess(result, moving_img, fixed_img)

        # 保存结果
        if save_results and output_dir:
            self._save_results(result, output_dir)

        return result

    def register_batch(
        self,
        pairs: List[Tuple[Union[str, Path], Union[str, Path]]],
        output_dir: Union[str, Path],
        save_pairs: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        批量配准

        Args:
            pairs: [(moving, fixed), ...] 图像对列表
            output_dir: 输出目录
            save_pairs: 是否保存每对的结果

        Returns:
            list: 每个配准的结果
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i, (moving, fixed) in enumerate(
            tqdm(pairs, desc="Registering", disable=not self.verbose)
        ):
            pair_output = output_dir / f"pair_{i:04d}"
            pair_output.mkdir(exist_ok=True)

            result = self.register(
                moving=moving,
                fixed=fixed,
                output_dir=pair_output,
                save_results=save_pairs,
            )
            result["pair_id"] = i
            results.append(result)

        return results

    def _load_image(
        self, image: Union[str, Path, torch.Tensor, np.ndarray, sitk.Image]
    ) -> Tuple[sitk.Image, torch.Tensor]:
        """加载图像"""
        if isinstance(image, (str, Path)):
            img = sitk.ReadImage(str(image))
        elif isinstance(image, sitk.Image):
            img = image
        elif isinstance(image, np.ndarray):
            img = sitk.GetImageFromArray(image)
        elif isinstance(image, torch.Tensor):
            img = sitk.GetImageFromArray(image.cpu().numpy())
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # 转换为张量
        arr = sitk.GetArrayFromImage(img)
        tensor = torch.from_numpy(arr).float().unsqueeze(0)

        return img, tensor

    def _preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        """预处理图像张量"""
        # 归一化
        tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-8)

        # 调整维度顺序 (D, H, W) -> (1, 1, D, H, W)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 4:
            tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def _extract_features(self, tensor: torch.Tensor) -> torch.Tensor:
        """提取 DINOv2 特征"""
        if not hasattr(self, "dino_model") and self.has_timm:
            model_name = self.kwargs.get("dino", {}).get(
                "model_name", "dinov2_vitl14"
            )
            self.dino_model = torch.hub.load("facebookresearch/dinov2", model_name)
            self.dino_model = self.dino_model.to(self.device)
            self.dino_model.eval()

        if hasattr(self, "dino_model"):
            with torch.no_grad():
                features = self.dino_model(tensor)
            return features
        return tensor

    def _get_mask(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """获取感兴趣区域掩码"""
        # 简单实现：使用图像非零区域作为掩码
        mask = (tensor > tensor.min()).float()
        return mask

    def _postprocess(
        self, result: Dict[str, torch.Tensor], moving_img: sitk.Image, fixed_img: sitk.Image
    ) -> Dict[str, Any]:
        """后处理结果"""
        # 确保变形场维度正确
        phi = result["phi"]
        warped = result["warped"]

        return {
            "phi": phi,
            "phi_inv": result.get("phi_inv"),
            "warped": warped,
            "loss": result.get("loss"),
            "loss_history": result.get("loss_history", []),
            "moving_image": moving_img,
            "fixed_image": fixed_img,
        }

    def _save_results(self, result: Dict[str, Any], output_dir: Union[str, Path]):
        """保存结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存变形后的图像
        warped = result["warped"]
        if warped.dim() >= 4:
            warped = warped.squeeze(0).squeeze(0)
        warped_arr = warped.cpu().numpy()
        warped_img = sitk.GetImageFromArray(warped_arr)
        warped_img.CopyInformation(result["moving_image"])
        sitk.WriteImage(warped_img, str(output_dir / "warped.nii.gz"))

        # 保存变形场
        phi = result["phi"]
        if phi.dim() == 5:
            phi = phi.squeeze(0)
        phi_arr = phi.cpu().numpy()
        phi_img = sitk.GetImageFromArray(phi_arr)
        phi_img.CopyInformation(result["moving_image"])
        sitk.WriteImage(phi_img, str(output_dir / "deformation_field.nii.gz"))


def create_engine(mode: str = "dino_reg", **kwargs) -> RegistrationEngine:
    """创建配准引擎的便捷函数"""
    return RegistrationEngine(mode=mode, **kwargs)
