"""
Visualization - 配准可视化工具
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple, List
from pathlib import Path
import SimpleITK as sitk


class RegistrationVisualizer:
    """
    配准结果可视化工具

    支持：
    - 2D/3D 图像切片显示
    - 变形场可视化
    - 变形网格可视化
    - 切片对比图
    - Jacobian 热力图
    """

    def __init__(self, figsize: Tuple[int, int] = (15, 5), dpi: int = 100):
        """
        Args:
            figsize: 默认图像大小
            dpi: 分辨率
        """
        self.figsize = figsize
        self.dpi = dpi

    def plot_comparison(
        self,
        moving: Union[torch.Tensor, np.ndarray],
        fixed: Union[torch.Tensor, np.ndarray],
        warped: Optional[Union[torch.Tensor, np.ndarray]] = None,
        slice_idx: Optional[int] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        绘制移动、固定和变形后图像的对比

        Args:
            moving: 移动图像
            fixed: 固定图像
            warped: 变形后图像
            slice_idx: 切片索引
            save_path: 保存路径
            show: 是否显示

        Returns:
            matplotlib.Figure
        """
        if isinstance(moving, torch.Tensor):
            moving = moving.squeeze().cpu().numpy()
        if isinstance(fixed, torch.Tensor):
            fixed = fixed.squeeze().cpu().numpy()
        if warped is not None and isinstance(warped, torch.Tensor):
            warped = warped.squeeze().cpu().numpy()

        n_images = 3 if warped is not None else 2
        fig, axes = plt.subplots(1, n_images, figsize=(self.figsize[0] * n_images / 3, self.figsize[1]))

        titles = ["Moving", "Fixed"]
        images = [moving, fixed]

        if warped is not None:
            titles.append("Warped")
            images.append(warped)

        for ax, img, title in zip(axes, images, titles):
            if img.ndim == 3:
                if slice_idx is None:
                    slice_idx = img.shape[0] // 2
                img = img[slice_idx]

            im = ax.imshow(img, cmap="gray")
            ax.set_title(title)
            ax.axis("off")
            plt.colorbar(im, ax=ax)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_deformation_field(
        self,
        phi: Union[torch.Tensor, np.ndarray],
        subsample: int = 10,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        绘制变形场向量图

        Args:
            phi: 变形场
            subsample: 采样间隔
            save_path: 保存路径
            show: 是否显示

        Returns:
            matplotlib.Figure
        """
        if isinstance(phi, torch.Tensor):
            phi = phi.squeeze().cpu().numpy()

        if phi.ndim == 3:  # 2D [2, H, W]
            phi_x = phi[0]
            phi_y = phi[1]

            # 下采样
            if subsample > 1:
                phi_x = phi_x[::subsample, ::subsample]
                phi_y = phi_y[::subsample, ::subsample]

            H, W = phi_x.shape
            y, x = np.mgrid[0:H, 0:W]

            fig, ax = plt.subplots(figsize=(8, 8))
            q = ax.quiver(x, y, phi_x, phi_y, np.sqrt(phi_x**2 + phi_y**2),
                         cmap="jet", scale=5, width=0.005)
            ax.set_aspect("equal")
            ax.set_title("Deformation Field")
            plt.colorbar(q, ax=ax)

        else:
            raise ValueError("3D deformation field visualization not supported yet")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_jacobian(
        self,
        jacobian: Union[torch.Tensor, np.ndarray],
        slice_idx: Optional[int] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        绘制 Jacobian 行列式热力图

        Args:
            jacobian: Jacobian 行列式
            slice_idx: 切片索引
            save_path: 保存路径
            show: 是否显示

        Returns:
            matplotlib.Figure
        """
        if isinstance(jacobian, torch.Tensor):
            jacobian = jacobian.squeeze().cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 8))

        if jacobian.ndim == 3:
            if slice_idx is None:
                slice_idx = jacobian.shape[0] // 2
            jacobian = jacobian[slice_idx]

        # 使用发散色图，0 为中心
        vmax = max(abs(jacobian.max()), abs(jacobian.min()))
        im = ax.imshow(jacobian, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Jacobian Determinant (slice {slice_idx})")
        ax.axis("off")
        plt.colorbar(im, ax=ax, label="det(J)")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_loss_curve(
        self,
        loss_history: List[float],
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        绘制损失曲线

        Args:
            loss_history: 损失历史
            save_path: 保存路径
            show: 是否显示

        Returns:
            matplotlib.Figure
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(loss_history)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Registration Loss Curve")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

        return fig


def save_comparison_grid(
    moving: np.ndarray,
    fixed: np.ndarray,
    warped: np.ndarray,
    output_path: str,
    slice_idx: Optional[int] = None,
) -> None:
    """
    保存对比网格图像

    Args:
        moving: 移动图像
        fixed: 固定图像
        warped: 变形后图像
        output_path: 输出路径
        slice_idx: 切片索引
    """
    if slice_idx is not None:
        moving = moving[slice_idx]
        fixed = fixed[slice_idx]
        warped = warped[slice_idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, img, title in zip(axes, [moving, fixed, warped], ["Moving", "Fixed", "Warped"]):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
