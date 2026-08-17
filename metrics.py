# #
# # Copyright (C) 2023, Inria
# # GRAPHDECO research group, https://team.inria.fr/graphdeco
# # All rights reserved.
# #
# # This software is free for non-commercial, research and evaluation use 
# # under the terms of the LICENSE.md file.
# #
# # For inquiries contact  george.drettakis@inria.fr
# #

# from pathlib import Path
# import os
# from PIL import Image
# import torch
# import torchvision.transforms.functional as tf
# from utils.loss_utils import ssim
# from lpipsPyTorch import lpips
# import json
# from tqdm import tqdm
# from utils.image_utils import psnr
# from argparse import ArgumentParser

# def readImages(renders_dir, gt_dir):
#     renders = []
#     gts = []
#     image_names = []
#     for fname in os.listdir(renders_dir):
#         render = Image.open(renders_dir / fname)
#         gt = Image.open(gt_dir / fname)
#         renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
#         gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
#         image_names.append(fname)
#     return renders, gts, image_names

# def evaluate(model_paths):

#     full_dict = {}
#     per_view_dict = {}
#     full_dict_polytopeonly = {}
#     per_view_dict_polytopeonly = {}
#     print("")

#     for scene_dir in model_paths:
#         try:
#             print("Scene:", scene_dir)
#             full_dict[scene_dir] = {}
#             per_view_dict[scene_dir] = {}
#             full_dict_polytopeonly[scene_dir] = {}
#             per_view_dict_polytopeonly[scene_dir] = {}

#             test_dir = Path(scene_dir) / "test"

#             for method in os.listdir(test_dir):
#                 print("Method:", method)

#                 full_dict[scene_dir][method] = {}
#                 per_view_dict[scene_dir][method] = {}
#                 full_dict_polytopeonly[scene_dir][method] = {}
#                 per_view_dict_polytopeonly[scene_dir][method] = {}

#                 method_dir = test_dir / method
#                 gt_dir = method_dir/ "gt"
#                 renders_dir = method_dir / "renders"
#                 renders, gts, image_names = readImages(renders_dir, gt_dir)

#                 ssims = []
#                 psnrs = []
#                 lpipss = []

#                 for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
#                     ssims.append(ssim(renders[idx], gts[idx]))
#                     psnrs.append(psnr(renders[idx], gts[idx]))
#                     lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

#                 print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
#                 print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
#                 print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
#                 print("")

#                 full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
#                                                         "PSNR": torch.tensor(psnrs).mean().item(),
#                                                         "LPIPS": torch.tensor(lpipss).mean().item()})
#                 per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
#                                                             "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
#                                                             "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

#             with open(scene_dir + "/results.json", 'w') as fp:
#                 json.dump(full_dict[scene_dir], fp, indent=True)
#             with open(scene_dir + "/per_view.json", 'w') as fp:
#                 json.dump(per_view_dict[scene_dir], fp, indent=True)
#         except:
#             print("Unable to compute metrics for model", scene_dir)

# if __name__ == "__main__":
#     device = torch.device("cuda:0")
#     torch.cuda.set_device(device)

#     # Set up command line argument parser
#     parser = ArgumentParser(description="Training script parameters")
#     parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
#     args = parser.parse_args()
#     evaluate(args.model_paths)





from pathlib import Path
import os
import json
from argparse import ArgumentParser

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from tqdm import tqdm

from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []

    for fname in sorted(os.listdir(renders_dir)):
        render_path = renders_dir / fname
        gt_path = gt_dir / fname

        if not gt_path.exists():
            print(f"[Warning] GT not found for {fname}, skip.")
            continue

        render = Image.open(render_path)
        gt = Image.open(gt_path)

        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)

    return renders, gts, image_names


def resize_depth_to_shape(depth, target_h, target_w):
    """
    Resize depth spatially only. Depth values are not scaled.
    """
    depth = np.asarray(depth, dtype=np.float32)

    if depth.shape == (target_h, target_w):
        return depth

    depth_t = torch.from_numpy(depth)[None, None].float()

    depth_t = F.interpolate(
        depth_t,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )

    return depth_t[0, 0].cpu().numpy().astype(np.float32)


def find_da3_depth_file(da3_depth_dir, pgsr_depth_file):
    """
    Try to match DA3 depth according to PGSR depth filename.
    Example:
        PGSR: xxx.npy
        DA3 : xxx.npy
    """
    da3_depth_dir = Path(da3_depth_dir)
    pgsr_depth_file = Path(pgsr_depth_file)

    candidates = [
        da3_depth_dir / pgsr_depth_file.name,
        da3_depth_dir / f"{pgsr_depth_file.stem}.npy",
    ]

    for c in candidates:
        if c.exists():
            return c

    return None


def valid_depth_mask(pred, ref, eps=1e-6):
    return (
        np.isfinite(pred)
        & np.isfinite(ref)
        & (pred > eps)
        & (ref > eps)
    )


def compute_scene_scale(pred_list, ref_list, mask_list, eps=1e-12):
    """
    Solve one global scale per scene/method:

        s = argmin || s * pred - ref ||^2

    pred: PGSR depth
    ref : DA3 pseudo-reference depth
    """
    numerator = 0.0
    denominator = 0.0

    for pred, ref, mask in zip(pred_list, ref_list, mask_list):
        p = pred[mask].astype(np.float64)
        r = ref[mask].astype(np.float64)

        numerator += np.sum(p * r)
        denominator += np.sum(p * p)

    scale = numerator / (denominator + eps)

    return float(scale)


def compute_depth_metrics(pred, ref, mask, eps=1e-6):
    """
    Compute standard depth metrics.
    pred and ref should already be aligned if scale alignment is needed.
    """
    pred = pred[mask].astype(np.float64)
    ref = ref[mask].astype(np.float64)

    pred = np.maximum(pred, eps)
    ref = np.maximum(ref, eps)

    abs_rel = np.mean(np.abs(pred - ref) / ref)
    rmse = np.sqrt(np.mean((pred - ref) ** 2))

    log_diff = np.log(pred) - np.log(ref)
    silog = np.sqrt(
        max(np.mean(log_diff ** 2) - np.mean(log_diff) ** 2, 0.0)
    )

    ratio = np.maximum(pred / ref, ref / pred)
    delta1 = np.mean(ratio < 1.25)

    return {
        "Depth_AbsRel": float(abs_rel),
        "Depth_RMSE": float(rmse),
        "Depth_SILog": float(silog),
        "Depth_delta1": float(delta1),
    }


def evaluate_depth_against_da3(method_dir, da3_depth_dir, da3_depth_mask_dir=None):
    """
    Compare:
        method_dir/renders_depth_raw/*.npy
    against:
        da3_depth_dir/*.npy

    Return:
        full_depth_metrics, per_view_depth_metrics
    """
    pgsr_depth_dir = method_dir / "renders_depth_raw"
    da3_depth_dir = Path(da3_depth_dir)

    if not pgsr_depth_dir.exists():
        print(f"[Warning] PGSR raw depth dir not found: {pgsr_depth_dir}")
        return None, None

    if not da3_depth_dir.exists():
        print(f"[Warning] DA3 depth dir not found: {da3_depth_dir}")
        return None, None
    if da3_depth_mask_dir is not None:
        da3_depth_mask_dir = Path(da3_depth_mask_dir)

    pred_list = []
    ref_list = []
    mask_list = []
    names = []

    pgsr_files = sorted(pgsr_depth_dir.glob("*.npy"))

    for pgsr_file in pgsr_files:
        da3_file = find_da3_depth_file(da3_depth_dir, pgsr_file)

        if da3_file is None:
            print(f"[Warning] DA3 depth not found for {pgsr_file.name}, skip.")
            continue

        pred = np.load(pgsr_file).astype(np.float32)
        ref = np.load(da3_file).astype(np.float32)

        if pred.ndim != 2:
            pred = np.squeeze(pred)
        if ref.ndim != 2:
            ref = np.squeeze(ref)

        if pred.ndim != 2 or ref.ndim != 2:
            print(f"[Warning] Invalid depth shape: {pgsr_file.name}, pred={pred.shape}, ref={ref.shape}")
            continue

        target_h, target_w = pred.shape
        ref = resize_depth_to_shape(ref, target_h, target_w)

        mask = valid_depth_mask(pred, ref)

        if da3_depth_mask_dir is not None:
            mask_file = find_matched_file(da3_depth_mask_dir, pgsr_file)
            if mask_file is not None:
                fg_mask = np.load(mask_file)
                fg_mask = np.squeeze(fg_mask)
                fg_mask = resize_mask_to_shape(fg_mask, target_h, target_w)
                mask = mask & fg_mask

        if mask.sum() == 0:
            print(f"[Warning] No valid depth pixels for {pgsr_file.name}, skip.")
            continue

        pred_list.append(pred)
        ref_list.append(ref)
        mask_list.append(mask)
        names.append(pgsr_file.name)

    if len(pred_list) == 0:
        print("[Warning] No matched PGSR/DA3 depth pairs.")
        return None, None

    # scale = compute_scene_scale(pred_list, ref_list, mask_list)
    scale = 1.0
    per_view_metrics = {
        "Depth_AbsRel": {},
        "Depth_RMSE": {},
        "Depth_SILog": {},
        "Depth_delta1": {},
    }

    all_absrel = []
    all_rmse = []
    all_silog = []
    all_delta1 = []

    for name, pred, ref, mask in zip(names, pred_list, ref_list, mask_list):
        pred_aligned = scale * pred

        metrics = compute_depth_metrics(pred_aligned, ref, mask)

        per_view_metrics["Depth_AbsRel"][name] = metrics["Depth_AbsRel"]
        per_view_metrics["Depth_RMSE"][name] = metrics["Depth_RMSE"]
        per_view_metrics["Depth_SILog"][name] = metrics["Depth_SILog"]
        per_view_metrics["Depth_delta1"][name] = metrics["Depth_delta1"]

        all_absrel.append(metrics["Depth_AbsRel"])
        all_rmse.append(metrics["Depth_RMSE"])
        all_silog.append(metrics["Depth_SILog"])
        all_delta1.append(metrics["Depth_delta1"])

    full_metrics = {
        "Depth_AbsRel": float(np.mean(all_absrel)),
        "Depth_RMSE": float(np.mean(all_rmse)),
        "Depth_SILog": float(np.mean(all_silog)),
        "Depth_delta1": float(np.mean(all_delta1)),
        "Depth_Scale": float(scale),
        "Depth_NumViews": int(len(names)),
    }

    return full_metrics, per_view_metrics


def evaluate(model_paths):
    full_dict = {}
    per_view_dict = {}

    print("")

    for scene_dir in model_paths:
        try:
            scene_dir = Path(scene_dir)

            print("Scene:", scene_dir)

            full_dict[str(scene_dir)] = {}
            per_view_dict[str(scene_dir)] = {}

            test_dir = scene_dir / "test"

            # Your structure:
            # /home/lidar/dzz_3DGS/validation/llff_8/${s}/${var}
            # DA3:
            # /home/lidar/dzz_3DGS/validation/llff_8/${s}/depth_da3/depth_npy
            llff_scene_dir = scene_dir.parent
            da3_depth_dir = llff_scene_dir / "depth_da3" / "depth_npy"

            da3_normal_dir = llff_scene_dir / "depth_da3" / "normal_npy"
            da3_normal_mask_dir = llff_scene_dir / "depth_da3" / "normal_mask"

            for method in sorted(os.listdir(test_dir)):
                print("Method:", method)

                full_dict[str(scene_dir)][method] = {}
                per_view_dict[str(scene_dir)][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"

                # =====================================================
                # RGB metrics: original logic
                # =====================================================
                if renders_dir.exists() and gt_dir.exists():
                    renders, gts, image_names = readImages(renders_dir, gt_dir)

                    ssims = []
                    psnrs = []
                    lpipss = []

                    for idx in tqdm(range(len(renders)), desc="RGB metric evaluation progress"):
                        ssims.append(ssim(renders[idx], gts[idx]))
                        psnrs.append(psnr(renders[idx], gts[idx]))
                        lpipss.append(lpips(renders[idx], gts[idx], net_type="vgg"))

                    if len(ssims) > 0:
                        mean_ssim = torch.tensor(ssims).mean().item()
                        mean_psnr = torch.tensor(psnrs).mean().item()
                        mean_lpips = torch.tensor(lpipss).mean().item()

                        print("  SSIM : {:>12.7f}".format(mean_ssim))
                        print("  PSNR : {:>12.7f}".format(mean_psnr))
                        print("  LPIPS: {:>12.7f}".format(mean_lpips))

                        full_dict[str(scene_dir)][method].update({
                            "SSIM": mean_ssim,
                            "PSNR": mean_psnr,
                            "LPIPS": mean_lpips,
                        })

                        per_view_dict[str(scene_dir)][method].update({
                            "SSIM": {
                                name: value
                                for value, name in zip(torch.tensor(ssims).tolist(), image_names)
                            },
                            "PSNR": {
                                name: value
                                for value, name in zip(torch.tensor(psnrs).tolist(), image_names)
                            },
                            "LPIPS": {
                                name: value
                                for value, name in zip(torch.tensor(lpipss).tolist(), image_names)
                            },
                        })
                else:
                    print(f"[Warning] RGB dirs not found: {renders_dir}, {gt_dir}")

                # =====================================================
                # Depth metrics: PGSR renders_depth_raw vs DA3 depth_npy
                # =====================================================
                depth_full_metrics, depth_per_view_metrics = evaluate_depth_against_da3(
                    method_dir=method_dir,
                    da3_depth_dir=da3_depth_dir,
                )

                if depth_full_metrics is not None:
                    print("  Depth AbsRel : {:>12.7f}".format(depth_full_metrics["Depth_AbsRel"]))
                    print("  Depth RMSE   : {:>12.7f}".format(depth_full_metrics["Depth_RMSE"]))
                    print("  Depth SILog  : {:>12.7f}".format(depth_full_metrics["Depth_SILog"]))
                    print("  Depth delta1 : {:>12.7f}".format(depth_full_metrics["Depth_delta1"]))
                    print("  Depth Scale  : {:>12.7f}".format(depth_full_metrics["Depth_Scale"]))
                    print("")

                    full_dict[str(scene_dir)][method].update(depth_full_metrics)
                    per_view_dict[str(scene_dir)][method].update(depth_per_view_metrics)
                else:
                    print("[Warning] Depth metrics skipped.")
                    print("")

                # =====================================================
                # Normal metrics:
                # PGSR renders_normal_raw vs DA3 normal_npy
                # Orientation-invariant: n and -n are treated as same
                # =====================================================
                normal_full_metrics, normal_per_view_metrics = evaluate_normal_against_da3(
                    method_dir=method_dir,
                    da3_normal_dir=da3_normal_dir,
                    da3_normal_mask_dir=da3_normal_mask_dir,
                )

                if normal_full_metrics is not None:
                    print("  Normal Mean   : {:>12.7f}".format(normal_full_metrics["Normal_Mean"]))
                    print("  Normal Median : {:>12.7f}".format(normal_full_metrics["Normal_Median"]))
                    print("  Normal 22.5   : {:>12.7f}".format(normal_full_metrics["Normal_22.5"]))
                    print("  Normal Views  : {:>12d}".format(normal_full_metrics["Normal_NumViews"]))
                    print("")

                    full_dict[str(scene_dir)][method].update(normal_full_metrics)
                    per_view_dict[str(scene_dir)][method].update(normal_per_view_metrics)
                else:
                    print("[Warning] Normal metrics skipped.")
                    print("")

            with open(scene_dir / "results.json", "w") as fp:
                json.dump(full_dict[str(scene_dir)], fp, indent=True)

            with open(scene_dir / "per_view.json", "w") as fp:
                json.dump(per_view_dict[str(scene_dir)], fp, indent=True)

        except Exception as e:
            print("Unable to compute metrics for model", scene_dir)
            print("Reason:", repr(e))

def load_normal_npy(path):
    """
    Load normal npy and convert to shape (H, W, 3).
    Supports:
        (H, W, 3)
        (3, H, W)
    """
    normal = np.load(path).astype(np.float32)
    normal = np.squeeze(normal)

    if normal.ndim != 3:
        raise RuntimeError(f"Invalid normal shape: {path}, shape={normal.shape}")

    if normal.shape[-1] == 3:
        return normal

    if normal.shape[0] == 3:
        return np.transpose(normal, (1, 2, 0))

    raise RuntimeError(f"Invalid normal shape: {path}, shape={normal.shape}")


def resize_normal_to_shape(normal, target_h, target_w):
    """
    Resize normal to target shape and re-normalize.
    normal: (H, W, 3)
    """
    normal = np.asarray(normal, dtype=np.float32)

    if normal.shape[:2] == (target_h, target_w):
        out = normal
    else:
        normal_t = torch.from_numpy(normal).permute(2, 0, 1)[None].float()

        normal_t = F.interpolate(
            normal_t,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )

        out = normal_t[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)

    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    out = out / (norm + 1e-8)

    return out.astype(np.float32)


def resize_mask_to_shape(mask, target_h, target_w):
    """
    Resize bool mask by nearest neighbor.
    """
    mask = np.asarray(mask).astype(np.float32)

    if mask.shape == (target_h, target_w):
        return mask > 0.5

    mask_t = torch.from_numpy(mask)[None, None].float()

    mask_t = F.interpolate(
        mask_t,
        size=(target_h, target_w),
        mode="nearest",
    )

    return mask_t[0, 0].cpu().numpy() > 0.5


def find_matched_file(ref_dir, src_file):
    """
    Match by filename stem.
    Example:
        src: xxx.npy
        ref: xxx.npy
    """
    ref_dir = Path(ref_dir)
    src_file = Path(src_file)

    candidates = [
        ref_dir / src_file.name,
        ref_dir / f"{src_file.stem}.npy",
    ]

    for c in candidates:
        if c.exists():
            return c

    return None


def normal_valid_mask(pred_normal, ref_normal, extra_mask=None, eps=1e-6):
    pred_norm = np.linalg.norm(pred_normal, axis=-1)
    ref_norm = np.linalg.norm(ref_normal, axis=-1)

    mask = (
        np.isfinite(pred_normal).all(axis=-1)
        & np.isfinite(ref_normal).all(axis=-1)
        & (pred_norm > eps)
        & (ref_norm > eps)
    )

    if extra_mask is not None:
        mask = mask & extra_mask.astype(bool)

    return mask


def normal_angles_orientation_invariant(pred_normal, ref_normal, mask):
    """
    Orientation-invariant normal error.
    n and -n are treated as identical.
    """
    pred = pred_normal[mask].astype(np.float64)
    ref = ref_normal[mask].astype(np.float64)

    pred = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8)
    ref = ref / (np.linalg.norm(ref, axis=-1, keepdims=True) + 1e-8)

    dot = np.sum(pred * ref, axis=-1)

    # 关键：方向相反也认为一致
    dot = np.abs(dot)

    dot = np.clip(dot, 0.0, 1.0)

    angles = np.degrees(np.arccos(dot))

    return angles


def normal_metrics_from_angles(angles):
    return {
        "Normal_Mean": float(np.mean(angles)),
        "Normal_Median": float(np.median(angles)),
        "Normal_22.5": float(np.mean(angles < 22.5)),
    }


def evaluate_normal_against_da3(method_dir, da3_normal_dir, da3_normal_mask_dir=None):
    """
    Compare:
        PGSR/3DGS:
            method_dir/renders_normal_raw/*.npy

        DA3 pseudo normal:
            da3_normal_dir/*.npy

        DA3 normal mask:
            da3_normal_mask_dir/*.npy

    Metrics:
        Normal_Mean
        Normal_Median
        Normal_22.5

    Orientation-invariant:
        n and -n are treated as identical.
    """
    method_dir = Path(method_dir)
    da3_normal_dir = Path(da3_normal_dir)

    pgsr_normal_dir = method_dir / "renders_normal_raw"

    if not pgsr_normal_dir.exists():
        print(f"[Warning] PGSR raw normal dir not found: {pgsr_normal_dir}")
        return None, None

    if not da3_normal_dir.exists():
        print(f"[Warning] DA3 normal npy dir not found: {da3_normal_dir}")
        return None, None

    if da3_normal_mask_dir is not None:
        da3_normal_mask_dir = Path(da3_normal_mask_dir)
        if not da3_normal_mask_dir.exists():
            print(f"[Warning] DA3 normal mask dir not found: {da3_normal_mask_dir}")
            da3_normal_mask_dir = None

    per_view_metrics = {
        "Normal_Mean": {},
        "Normal_Median": {},
        "Normal_22.5": {},
    }

    all_angles = []
    matched_names = []

    pgsr_files = sorted(pgsr_normal_dir.glob("*.npy"))

    for pgsr_file in pgsr_files:
        da3_file = find_matched_file(da3_normal_dir, pgsr_file)

        if da3_file is None:
            print(f"[Warning] DA3 normal not found for {pgsr_file.name}, skip.")
            continue

        pred_normal = load_normal_npy(pgsr_file)
        ref_normal = load_normal_npy(da3_file)

        target_h, target_w = pred_normal.shape[:2]

        ref_normal = resize_normal_to_shape(ref_normal, target_h, target_w)

        extra_mask = None
        if da3_normal_mask_dir is not None:
            da3_mask_file = find_matched_file(da3_normal_mask_dir, pgsr_file)
            if da3_mask_file is not None:
                extra_mask = np.load(da3_mask_file)
                extra_mask = np.squeeze(extra_mask)
                extra_mask = resize_mask_to_shape(extra_mask, target_h, target_w)

        mask = normal_valid_mask(
            pred_normal=pred_normal,
            ref_normal=ref_normal,
            extra_mask=extra_mask,
        )

        if mask.sum() == 0:
            print(f"[Warning] No valid normal pixels for {pgsr_file.name}, skip.")
            continue

        angles = normal_angles_orientation_invariant(
            pred_normal=pred_normal,
            ref_normal=ref_normal,
            mask=mask,
        )

        metrics = normal_metrics_from_angles(angles)

        per_view_metrics["Normal_Mean"][pgsr_file.name] = metrics["Normal_Mean"]
        per_view_metrics["Normal_Median"][pgsr_file.name] = metrics["Normal_Median"]
        per_view_metrics["Normal_22.5"][pgsr_file.name] = metrics["Normal_22.5"]

        all_angles.append(angles)
        matched_names.append(pgsr_file.name)

    if len(all_angles) == 0:
        print("[Warning] No matched PGSR/DA3 normal pairs.")
        return None, None

    all_angles = np.concatenate(all_angles, axis=0)

    full_metrics = normal_metrics_from_angles(all_angles)
    full_metrics["Normal_NumViews"] = int(len(matched_names))

    return full_metrics, per_view_metrics

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="Evaluation script parameters")
    parser.add_argument("--model_paths", "-m", required=True, nargs="+", type=str, default=[])

    args = parser.parse_args()

    evaluate(args.model_paths)