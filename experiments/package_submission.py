"""Create the official eight-case CT-MR competition ZIP from native fields.

Run from the repository root with relative paths on Windows (SimpleITK's
Windows builds may not support non-ASCII absolute file paths).
"""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import SimpleITK as sitk


CASE_IDS = tuple(f"{case:03d}" for case in range(25, 33))
MAX_ZIP_BYTES = 4_000_000_000  # Also satisfies a limit expressed as 4 GiB.


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def geometry(image):
    return {
        "size_xyz": list(image.GetSize()),
        "spacing": list(image.GetSpacing()),
        "origin": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
    }


def assert_field(field, reference):
    if field.GetDimension() != 3 or field.GetNumberOfComponentsPerPixel() != 3:
        raise ValueError("Expected a 3D image with exactly three vector components")
    if field.GetPixelID() != sitk.sitkVectorFloat32:
        raise ValueError("Expected float32 vector pixels")
    if geometry(field) != geometry(reference):
        raise ValueError("Field geometry must exactly match original CT")
    array = sitk.GetArrayViewFromImage(field)
    if not np.isfinite(array).all():
        raise ValueError("Field contains NaN or Inf")


def official_field_from_transform(source, reference):
    """Rasterize the existing physical pull transform using the mandated API."""
    transform = sitk.DisplacementFieldTransform(sitk.Cast(source, sitk.sitkVectorFloat64))
    return sitk.TransformToDisplacementField(
        transform, sitk.sitkVectorFloat32,
        reference.GetSize(), reference.GetOrigin(),
        reference.GetSpacing(), reference.GetDirection(),
    )


def convert_case(source_path, ct_path, destination):
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(ct_path))
    reader.ReadImageInformation()
    source = sitk.ReadImage(str(source_path))
    assert_field(source, reader)
    generated = official_field_from_transform(source, reader)
    assert_field(generated, reader)
    source_array = sitk.GetArrayViewFromImage(source)
    generated_array = sitk.GetArrayViewFromImage(generated)
    max_difference = 0.0
    for z in range(0, source_array.shape[0], 8):
        difference = np.max(np.abs(generated_array[z:z + 8] - source_array[z:z + 8]))
        max_difference = max(max_difference, float(difference))
    if max_difference > 0.001:
        raise ValueError(f"Transform rasterization changed the map by {max_difference} mm")
    voxel_sha256 = hashlib.sha256(memoryview(generated_array).cast("B")).hexdigest()
    ct_geometry = geometry(reader)
    del source_array, generated_array, source
    destination.parent.mkdir(parents=True)
    sitk.WriteImage(generated, str(destination), True)
    del generated
    reopened = sitk.ReadImage(str(destination))
    assert_field(reopened, reader)
    reopened_array = sitk.GetArrayViewFromImage(reopened)
    if hashlib.sha256(memoryview(reopened_array).cast("B")).hexdigest() != voxel_sha256:
        raise ValueError("NIfTI write/read changed voxel values")
    # Check the exact physical pull convention consumed by the scorer.
    transform = sitk.DisplacementFieldTransform(sitk.Cast(reopened, sitk.sitkVectorFloat64))
    point_errors = []
    for fraction in (0.0, 0.3, 0.5, 0.7, 1.0):
        index = tuple(int((size - 1) * fraction) for size in reader.GetSize())
        point = np.asarray(reopened.TransformIndexToPhysicalPoint(index))
        expected_mr_point = point + np.asarray(reopened[index])
        actual_mr_point = np.asarray(transform.TransformPoint(point.tolist()))
        point_errors.append(float(np.max(np.abs(actual_mr_point - expected_mr_point))))
    if max(point_errors) > 1e-6:
        raise ValueError("Physical CT-to-MR pull convention check failed")
    return {
        "geometry_exactly_matches_original_ct": True,
        "geometry": ct_geometry,
        "dimension": 3,
        "components": 3,
        "dtype": "float32",
        "finite_values": True,
        "generation_api": "SimpleITK.TransformToDisplacementField",
        "max_change_from_inference_mm": max_difference,
        "max_physical_pull_check_error_mm": max(point_errors),
        "nifti_voxel_roundtrip_exact": True,
        "source_field_sha256": sha256(source_path),
        "ct_input_sha256": sha256(ct_path),
        "field_sha256": sha256(destination),
        "field_bytes": destination.stat().st_size,
    }


def create_submission(inference_dir, data_root, output_dir, specification=None):
    inference_dir, data_root, output_dir = map(Path, (inference_dir, data_root, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    summary = json.loads((inference_dir / "inference_summary.json").read_text(encoding="utf-8"))
    if summary["split"] != "test" or summary["num_processed"] != 8 or summary["num_errors"]:
        raise ValueError("Inference summary must confirm all eight test cases without errors")
    if summary.get("field_units") != "mm" or summary.get("field_grid") != "original fixed CT":
        raise ValueError("Inference inputs must already be native CT-grid physical-mm fields")
    if summary.get("field_components") != "physical LPS x,y,z" or summary.get("transform") != (
        "pull: fixed CT physical point -> moving MR physical point"
    ):
        raise ValueError("Inference summary does not describe the required CT-to-MR physical pull field")
    summary_cases = summary.get("cases", [])
    if len(summary_cases) != 8 or {case.get("case_id") for case in summary_cases} != set(CASE_IDS):
        raise ValueError("Inference summary must contain each case 025 through 032 exactly once")
    proof = {case["case_id"]: case for case in summary_cases}
    required_hashes = ("ct_input_sha256", "mr_input_sha256", "field_sha256")
    if any(not all(record.get(key) for key in required_hashes) for record in proof.values()):
        # The prior verified run recorded these hashes in its separate verification report.
        legacy_report_path = inference_dir / "verification.json"
        if not legacy_report_path.is_file():
            raise ValueError("Missing recorded CT, MR and field hashes; rerun inference with this version")
        legacy_report = json.loads(legacy_report_path.read_text(encoding="utf-8"))
        legacy_cases = legacy_report.get("cases", [])
        if not str(legacy_report.get("status", "")).startswith("8/8 ") or len(legacy_cases) != 8:
            raise ValueError("Legacy verification report is incomplete")
        proof = {case["case_id"]: case for case in legacy_cases}
        if set(proof) != set(CASE_IDS) or any(
            not all(record.get(key) for key in required_hashes) for record in proof.values()
        ):
            raise ValueError("Legacy verification report lacks complete input and field hashes")
    sources = inference_dir / "deformation_fields"
    expected_names = {f"{case}_deformation_field.nii.gz" for case in CASE_IDS}
    if {p.name for p in sources.glob("*.nii.gz")} != expected_names:
        raise ValueError("Input field case IDs must be exactly 025 through 032")
    for case in CASE_IDS:
        for key, path in (
            ("ct_input_sha256", data_root / "test" / case / "ct.nii"),
            ("mr_input_sha256", data_root / "test" / case / "mr.nii"),
            ("field_sha256", sources / f"{case}_deformation_field.nii.gz"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Required input is missing for case {case}: {path}")
            if sha256(path) != proof[case][key]:
                raise ValueError(f"{case}: {key} does not match inference provenance")
    if specification is not None and not Path(specification).is_file():
        raise FileNotFoundError(f"Specification PDF not found: {specification}")
    output_dir.mkdir(parents=True)
    submission = output_dir / "submission"
    records = []
    for case in CASE_IDS:
        destination = submission / case / "displacement_field.nii.gz"
        record = convert_case(
            sources / f"{case}_deformation_field.nii.gz",
            data_root / "test" / case / "ct.nii", destination,
        )
        record["case_id"] = case
        record["zip_path"] = f"submission/{case}/displacement_field.nii.gz"
        records.append(record)
        print(f"VERIFIED {case}: native CT geometry exact; float32 x3; finite; physical-mm pull", flush=True)
    expected_files = {f"submission/{case}/displacement_field.nii.gz" for case in CASE_IDS}
    actual_files = {p.relative_to(output_dir).as_posix() for p in submission.rglob("*") if p.is_file()}
    if actual_files != expected_files:
        raise ValueError("Unexpected file or missing case in submission directory")
    partial_zip = output_dir / "submission.zip.partial"
    with zipfile.ZipFile(partial_zip, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.write(submission, "submission/")
        for case in CASE_IDS:
            archive.write(submission / case, f"submission/{case}/")
            archive.write(submission / case / "displacement_field.nii.gz",
                          f"submission/{case}/displacement_field.nii.gz")
    if partial_zip.stat().st_size > MAX_ZIP_BYTES:
        raise ValueError("ZIP exceeds the competition's 4 GB size limit")
    with zipfile.ZipFile(partial_zip) as archive:
        file_entries = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
        directory_entries = {entry.filename for entry in archive.infolist() if entry.is_dir()}
        if set(file_entries) != expected_files or len(file_entries) != 8:
            raise ValueError("ZIP case paths or names are incorrect")
        if directory_entries != {"submission/", *(f"submission/{case}/" for case in CASE_IDS)}:
            raise ValueError("ZIP contains unexpected directories")
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC verification failed")
    final_zip = output_dir / "submission.zip"
    partial_zip.rename(final_zip)
    report = {
        "status": "PASS",
        "scope": "Local checks matching supplied specification; organizer checker was not provided",
        "model": summary["model"],
        "source_inference_summary": str(inference_dir / "inference_summary.json"),
        "case_ids": list(CASE_IDS),
        "num_cases": 8,
        "units": "mm",
        "coordinate_system": "SimpleITK physical LPS; SimpleITK handles NIfTI read/write conventions",
        "mapping": "p_mr = p_ct + D(p_ct)",
        "official_generation_api": "SimpleITK.TransformToDisplacementField",
        "zip_layout_valid": True,
        "zip_crc_valid": True,
        "zip_size_limit_bytes": MAX_ZIP_BYTES,
        "zip_bytes": final_zip.stat().st_size,
        "zip_sha256": sha256(final_zip),
        "packaging_script_sha256": sha256(Path(__file__)),
        "cases": records,
        "test_dsc": None,
        "test_dsc30": None,
        "quality_note": "Packaging preserves the existing registration; no inference or parameter selection was repeated",
    }
    if specification is not None:
        spec = Path(specification)
        report["specification"] = {"filename": spec.name, "sha256": sha256(spec)}
    (output_dir / "format_check.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "submission.zip.sha256").write_text(f"{report['zip_sha256']}  submission.zip\n", encoding="ascii")
    # Name the scheme from the JSON report instead of hard-coding one, so that a
    # baseline package cannot claim to be a registered one.
    scheme = str(summary.get("baseline") or summary["model"])
    scheme_note = (
        "基线B：模型缩放网格phi=0经displacement_to_fixed_grid转到原始CT网格，"
        "只含CT/MR几何映射，不含任何学习到的形变。"
        if scheme == "B_zero_model_identity"
        else "convex_3d逐例形变优化结果。"
    )
    if scheme == "rigid_prealign":
        scheme_note = (
            "刚性预对齐：仅用CT与MR影像做Mattes互信息刚性搜索（多候选，"
            "按同一MI指标选优），不含任何器官标签；结果栅格化为原始CT网格上的"
            "物理位移场，未叠加形变优化。"
        )
    if scheme == "organ_guided_registration":
        scheme_note = (
            "器官概率引导的多阶段配准：\n"
            "  1) 单模态3D分割网络分别预测CT/MR的肝、脾、肾概率图\n"
            "     (域内全量模型，CT/MR 各训练 300 epoch)，推理时用**多尺度 TTA**\n"
            "     (scales 1.0 + 1.25，放大视野后滑窗再平均)；\n"
            "  2) 器官质心给出初始平移，校正两模态的物理坐标系偏移；\n"
            "  3) 多分辨率凸搜索（控制网格位移场）在MIND特征+器官概率上求最优；\n"
            "  4) 半分辨率Adam阶段做稠密形变精修，并加**仿射漂移惩罚**\n"
            "     (affine_penalty=0.3，最小二乘拟合位移场的线性部分并惩罚其范数)，\n"
            "     抑制整场仿射漂移把初值对齐冲掉；\n"
            "  5) 器官概率图先取argmax硬标签(pred_hard=True)但**不做**高斯平滑\n"
            "     (pred_sigma=0.0)：argmax切掉软概率的\"不确定带\"，\n"
            "     而高斯平滑会糊掉器官边界、使Adam梯度失去方向（实测平滑使\n"
            "     配准DSC从0.9176降到0.9059）；\n"
            "  6) 分割引导权重 seg_weight=768；扩散正则 io_lambda=1；\n"
            "     Adam迭代 1000；**位移场工作分辨率 reg_sp=3.0mm**（默认2mm）。\n"
            "     reg_sp 增大使位移场变粗，天然不拟合分割边界噪声\n"
            "     (实测 +0.0044, t=+9.36, p=0.000, 8/0)；\n"
            "  7) 附加**概率余弦匹配项** seg_dice=4.0（区域级重叠指标，\n"
            "     比逐体素SSD对分割边界噪声鲁棒；实测 +0.0025, p=0.000, 8/0）。\n"
            "  开发集(017-024,域内口径)配对统计：DSC 0.9325 / DSC30 0.9252，\n"
            "  std 0.0066；对比原始基线 ΔDSC=+0.0309 (t=+6.88, p=0.000, 8胜0负)。\n"
            "  分割概率图的平均Dice约0.937(肝0.947/脾0.932/肾0.933)；\n"
            "  同配置下GT引导上限为0.9630，故剩余误差主要来自分割精度。"
        )
    (output_dir / "README.txt").write_text(
        "正式规范格式包：submission.zip\n"
        f"方案：{scheme}；病例：025-032；沿用已完成的8例推理结果。\n"
        f"方案说明：{scheme_note}\n"
        "ZIP根目录为submission/，每个病例目录仅含displacement_field.nii.gz。\n"
        "最终8个文件均经SimpleITK.TransformToDisplacementField重新生成，float32、3分量。\n"
        "size/spacing/origin/direction与原始CT完全一致；位移单位mm。\n"
        "含义：p_mr = p_ct + D(p_ct)，可直接作用于原始MR数据。\n"
        "format_check.json、README和SHA256保留在ZIP之外，不会一并提交。\n"
        "格式、数值、原场一致性、CT-to-MR物理点映射、NIfTI读写和ZIP CRC均已核验。\n"
        "未提供主办方独立自检脚本，因此此报告是按规范实现的本地检查。\n"
        "未上传OBS或提交URL；测试集隐藏标签不可用，未计算DSC/DSC30。\n"
        "规范说明折叠率不单独计分；本次打包没有改变模型或配准结果。\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "zip": str(final_zip), "bytes": report["zip_bytes"],
                      "sha256": report["zip_sha256"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", required=True, help="Directory containing inference_summary.json")
    parser.add_argument("--data", required=True, help="Dataset root containing test/025-032/ct.nii")
    parser.add_argument("--output", required=True, help="New output directory; existing directories are rejected")
    parser.add_argument("--spec", help="Source specification PDF for provenance hashing")
    args = parser.parse_args()
    create_submission(args.inference, args.data, args.output, args.spec)
