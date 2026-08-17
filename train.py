#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
import glob
import math
import os
import torch
import random
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, lncc, get_img_grad_weight
from utils.graphics_utils import patch_offsets, patch_warp
from gaussian_renderer import render, network_gui, render_normal
import sys, time
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import cv2
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, erode
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.app_model import AppModel
from scene.cameras import Camera
# from utils.da3_utils import estimate_depth_da3
# from torchmetrics.functional.regression import pearson_corrcoef
# from depth_anything_3.api import DepthAnything3
from utils.loss_utils import compute_weighted_pca_normals
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import torch.nn.functional as F
from utils.loss_utils import NormalConsistencyLoss
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(22)

def gen_virtul_cam(cam, trans_noise=1.0, deg_noise=15.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam.R.transpose()
    Rt[:3, 3] = cam.T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)

    translation_perturbation = np.random.uniform(-trans_noise, trans_noise, 3)
    rotation_perturbation = np.random.uniform(-deg_noise, deg_noise, 3)
    rx, ry, rz = np.deg2rad(rotation_perturbation)
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
    R_perturbation = Rz @ Ry @ Rx

    C2W[:3, :3] = C2W[:3, :3] @ R_perturbation
    C2W[:3, 3] = C2W[:3, 3] + translation_perturbation
    Rt = np.linalg.inv(C2W)
    virtul_cam = Camera(100000, Rt[:3, :3].transpose(), Rt[:3, 3], cam.FoVx, cam.FoVy,
                        cam.image_width, cam.image_height,
                        cam.image_path, cam.image_name, 100000,
                        trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                        preload_img=False, data_device = "cuda")
    return virtul_cam

def load_name_map(map_path):
    name_map = {}
    with open(map_path, 'r', encoding='utf-8') as f:
        header = f.readline()  # 跳过表头
        for line in f:
            original_name, mvs_name, image_id = line.rstrip('\n').split('\t')
            name_map[original_name] = {
                'mvs_name': mvs_name,
                'image_id': int(image_id),
            }
    return name_map


def build_mvs_depth_regularization(
    prob_volume,           # (D, H, W)
    depth_values,          # (D, H, W)
    render_plane_depth,    # (B, H, W)
    temperature=1.0,
    conf_power=1.0,
    support_power=1.0
):
    D, Hp, Wp = prob_volume.shape
    assert depth_values.shape == (D, Hp, Wp), (
        f"depth_values should match prob_volume shape, "
        f"got {depth_values.shape} vs {(D, Hp, Wp)}"
    )
    assert render_plane_depth.shape[0] == 1, (
        f"render_plane_depth should have shape (1,H,W), got {render_plane_depth.shape}"
    )

    Hr, Wr = render_plane_depth.shape[1], render_plane_depth.shape[2]

    # 根据 render_plane_depth 的原始尺寸，反推出 padding 后应有尺寸
    expected_Hp = int(math.ceil(Hr / 64.0) * 64)
    expected_Wp = int(math.ceil(Wr / 64.0) * 64)

    assert (Hp, Wp) == (expected_Hp, expected_Wp), (
        f"prob_volume/depth_values spatial size {(Hp, Wp)} does not match "
        f"the expected padded size {(expected_Hp, expected_Wp)} computed from "
        f"render_plane_depth spatial size {(Hr, Wr)}"
    )

    # padding 的上下左右像素数
    pad_h = expected_Hp - Hr
    pad_w = expected_Wp - Wr

    pad_top = pad_h // 2 + (pad_h % 2)
    pad_bottom = pad_h // 2

    pad_left = pad_w // 2 + (pad_w % 2)
    pad_right = pad_w // 2

    # 将 prob_volume 和 depth_values 裁剪回 render_plane_depth 的原始尺寸
    h_start = pad_top
    h_end = Hp - pad_bottom if pad_bottom > 0 else Hp

    w_start = pad_left
    w_end = Wp - pad_right if pad_right > 0 else Wp

    prob_volume = prob_volume[:, h_start:h_end, w_start:w_end]      # (D, Hr, Wr)
    depth_values = depth_values[:, h_start:h_end, w_start:w_end]    # (D, Hr, Wr)

    assert prob_volume.shape == (D, Hr, Wr), (
        f"Cropped prob_volume shape mismatch: got {prob_volume.shape}, expected {(D, Hr, Wr)}"
    )
    assert depth_values.shape == (D, Hr, Wr), (
        f"Cropped depth_values shape mismatch: got {depth_values.shape}, expected {(D, Hr, Wr)}"
    )

    # 1) 多视支撑强度，直接求和
    support = prob_volume.sum(dim=0)  # (H, W)

    # 2) 深度维 softmax，保证可微
    prob_soft = F.softmax(prob_volume / temperature, dim=0)  # (D, H, W)

    # 3) softmax 后的最大值作为置信度
    confidence = prob_soft.max(dim=0).values  # (H, W)

    # 4) soft regression 得到 MVS 深度
    depth_mvs = (prob_soft * depth_values).sum(dim=0)  # (H, W)

    # 5) 构造最终权重
    weight_support = support.clamp_min(0.0).pow(support_power)
    weight_conf = confidence.clamp(0.0, 1.0).pow(conf_power)

    weight = weight_support * weight_conf  # (H, W)

    # 6) 加权 Charbonnier 深度一致性损失
    diff = render_plane_depth.squeeze(0) - depth_mvs
    loss_depth_reg = (torch.sqrt(diff * diff + 1e-6) * weight).sum()

    return loss_depth_reg


def _save_heatmap(arr, save_path, title=None, cmap="viridis"):
    """
    arr: torch.Tensor or np.ndarray, shape (H, W)
    """
    if torch.is_tensor(arr):
        arr = arr.detach().cpu().float().numpy()

    arr = np.asarray(arr)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(arr, cmap=cmap)
    plt.colorbar(im)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# def load_all_fused_probs(source_path: str, suffix: str = "fused_prob"):

#     fused_dict = {}
#     pattern = os.path.join(source_path, f"*{suffix}.pt")
    
#     for pt_path in sorted(glob.glob(pattern)):          # sorted 保证顺序稳定
#         filename = os.path.basename(pt_path)
#         # 提取前缀数字（例如 "23_fused_prob.pt" → 23）
#         try:
#             info_name = int(filename.split('_')[0])
#         except (ValueError, IndexError):
#             print(f"[Warning] 无法解析文件名: {filename}，跳过")
#             continue
            
#         pt_file = torch.load(pt_path, map_location='cpu')   # cpu 加载更省显存
#         fused_dict[info_name] = {
#             "weight":    pt_file["weight"],      # (H, W)
#             "depth_mvs": pt_file["depth_mvs"]    # (H, W)
#         }
    
#     print(f"[Preload] 已加载 {len(fused_dict)} 个 fused_prob 文件")
#     return fused_dict


# def load_npz(source_path: str, suffix: str = "fused_prob"):

#     fused_dict = {}
#     path = "/home/lidar/dzz_3DGS/data/flower/3_views/test/exports/mini_npz/results.npz"
#     data = np.load(path)
#     fused_dict["depth"] = data["depth"]
#     fused_dict["conf"] = data["conf"] / 100

#     return fused_dict


@torch.no_grad()
def truncate_gradient_pure_torch(grads: torch.Tensor, normals: torch.Tensor, xi_min: float = 1e-3):
    gdn = torch.sum(grads * normals, dim=1, keepdim=True)
    g_perp = gdn * normals
    g_tang = grads - g_perp
    g_perp_norm = torch.abs(gdn)
    scale = torch.minimum(xi_min / (g_perp_norm + 1e-8), torch.tensor(1.0, device=grads.device))
    scale = torch.where(g_perp_norm > 1e-8, scale, torch.tensor(1.0, device=grads.device))
    
    grads.copy_(g_tang + scale * g_perp)   # ← 真正 in-place


def align_depth_to_original(
    depth_small: torch.Tensor,   # (742, 994)
    conf_small: torch.Tensor,    # (742, 994) 可选
    original_size: tuple = (747, 996),
    mode: str = 'bilinear'
):
    """
    像素级对齐 + 返回 tensor（兼容 fused_dict）
    """
    if isinstance(depth_small, np.ndarray):
        depth_small = torch.from_numpy(depth_small).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    else:
        depth_small = depth_small.unsqueeze(0).unsqueeze(0)
    
    if isinstance(conf_small, np.ndarray):
        conf_small = torch.from_numpy(conf_small).unsqueeze(0).unsqueeze(0)
    elif conf_small is not None:
        conf_small = conf_small.unsqueeze(0).unsqueeze(0)
    
    # 双线性上采样（align_corners=False 为 CV 标准）
    depth_aligned = F.interpolate(
        depth_small, size=original_size, mode=mode, align_corners=False
    ).squeeze(0).squeeze(0)  # [H, W]
    
    conf_aligned = None
    if conf_small is not None:
        conf_aligned = F.interpolate(
            conf_small, size=original_size, mode=mode, align_corners=False
        ).squeeze(0).squeeze(0)
    
    return  depth_aligned, conf_aligned


# def edge_aware_smoothness_loss(
#     depth: torch.Tensor,          # [H,W] / [1,H,W] / [B,1,H,W]
#     image: torch.Tensor,          # [H,W], [1,H,W], [3,H,W], [B,C,H,W]
#     alpha: float = 10.0,
#     eps: float = 1e-7
# ) -> torch.Tensor:

#     # ---- shape normalize ----
#     if depth.dim() == 2:
#         depth = depth.unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
#     elif depth.dim() == 3:
#         depth = depth.unsqueeze(1)                # [B,1,H,W]

#     if image.dim() == 2:
#         image = image.unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
#     elif image.dim() == 3:
#         image = image.unsqueeze(0)                # [1,C,H,W]

#     depth = depth.float()
#     image = image.float()

#     # 可选：若图像范围是 [0,255]，归一化到 [0,1]
#     if image.max() > 1.0:
#         image = image / 255.0

#     # ---- normalized depth ----
#     mean_depth = depth.mean(dim=[1,2,3], keepdim=True) + eps
#     depth_norm = depth / mean_depth

#     # ---- forward gradient ----
#     depth_gx = depth_norm[:, :, :, 1:] - depth_norm[:, :, :, :-1]
#     depth_gy = depth_norm[:, :, 1:, :] - depth_norm[:, :, :-1, :]

#     image_gx = image[:, :, :, 1:] - image[:, :, :, :-1]   # [B,C,H,W-1]
#     image_gy = image[:, :, 1:, :] - image[:, :, :-1, :]   # [B,C,H-1,W]

#     # 对多通道图像，在通道维聚合
#     image_gx_mag = torch.mean(torch.abs(image_gx), dim=1, keepdim=True)  # [B,1,H,W-1]
#     image_gy_mag = torch.mean(torch.abs(image_gy), dim=1, keepdim=True)  # [B,1,H-1,W]

#     weight_x = torch.exp(-alpha * image_gx_mag)
#     weight_y = torch.exp(-alpha * image_gy_mag)

#     loss_x = torch.abs(depth_gx) * weight_x
#     loss_y = torch.abs(depth_gy) * weight_y

#     return (loss_x.mean() + loss_y.mean())/2.0



def edge_aware_smoothness_loss(
    depth: torch.Tensor,          # [H,W] / [B,1,H,W]
    image: torch.Tensor,          # [H,W] / [C,H,W] / [B,C,H,W]
    gamma: float = 1e-2,
    eps: float = 1e-8,
    normalize_depth: bool = True
) -> torch.Tensor:
    """
    Edge-aware smoothness:
        L = mean( |dD/dx| / max(|dI/dx|, gamma) ) +
            mean( |dD/dy| / max(|dI/dy|, gamma) )

    image can be grayscale or RGB.
    Recommended: image should be normalized to [0,1].
    """

    # ---- shape normalize ----
    if depth.dim() == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
    elif depth.dim() == 3:
        depth = depth.unsqueeze(1)                # [B,1,H,W]

    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
    elif image.dim() == 3:
        image = image.unsqueeze(0)                # [1,C,H,W]

    depth = depth.float()
    image = image.float()

    # normalize image to [0,1] if needed
    if image.max() > 1.0:
        image = image / 255.0

    # optional mean normalization for depth
    if normalize_depth:
        mean_depth = depth.mean(dim=[1,2,3], keepdim=True) + eps
        depth = depth / mean_depth

    # ---- forward differences ----
    depth_gx = depth[:, :, :, 1:] - depth[:, :, :, :-1]   # [B,1,H,W-1]
    depth_gy = depth[:, :, 1:, :] - depth[:, :, :-1, :]   # [B,1,H-1,W]

    image_gx = image[:, :, :, 1:] - image[:, :, :, :-1]   # [B,C,H,W-1]
    image_gy = image[:, :, 1:, :] - image[:, :, :-1, :]   # [B,C,H-1,W]

    # aggregate RGB gradient to single-channel magnitude
    image_gx_mag = torch.mean(torch.abs(image_gx), dim=1, keepdim=True)  # [B,1,H,W-1]
    image_gy_mag = torch.mean(torch.abs(image_gy), dim=1, keepdim=True)  # [B,1,H-1,W]

    denom_x = torch.clamp(image_gx_mag, min=gamma)
    denom_y = torch.clamp(image_gy_mag, min=gamma)

    loss_x = torch.abs(depth_gx) / denom_x
    loss_y = torch.abs(depth_gy) / denom_y

    loss = loss_x.mean() + loss_y.mean()
    return loss

def get_smallest_axis(gau, return_idx=False):
    """
    改进版：保留“取最短轴”的逻辑，但通过 detach 彻底切断 scale 的梯度。
    这样 scale 不会被 normal loss 驱动，仅 rotation 受影响。
    """
    rotation_matrices = gau.get_rotation_matrix()          # [N, 3, 3]
    
    # ← 关键修改：detach 切断 scale 的梯度
    _, smallest_axis_idx = gau.get_scaling.min(dim=-1)     # [N]
    smallest_axis_idx = smallest_axis_idx.detach()          # ← 梯度截断
    
    # 恢复原来形状用于 gather
    smallest_axis_idx = smallest_axis_idx[..., None, None].expand(-1, 3, -1)
    
    smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
    
    if return_idx:
        return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
    
    return smallest_axis.squeeze(dim=2)



def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # backup main code
    cmd = f'cp ./train.py {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./arguments {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./gaussian_renderer {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./scene {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./utils {dataset.model_path}/'
    os.system(cmd)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    app_model = AppModel()
    app_model.train()
    app_model.cuda()
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        app_model.load_weights(scene.model_path)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack, pseudo_stack = None, None
    ema_loss_for_log = 0.0
    ema_single_view_for_log = 0.0
    ema_multi_view_geo_for_log = 0.0
    ema_multi_view_pho_for_log = 0.0
    ema_depth_for_log = 0.0
    ema_depth_for_pseudo = 0.0
    normal_loss, geo_loss, ncc_loss, depth_loss, pseudo_loss= None, None, None, None, None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    debug_path = os.path.join(scene.model_path, "debug")
    os.makedirs(debug_path, exist_ok=True)

    normal_consis = NormalConsistencyLoss(dataset.interval, k_search=opt.k_search, opacity_thr=0.1)
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        gt_image, gt_image_gray = viewpoint_cam.get_image()
        if iteration > 1000 and opt.exposure_compensation:
            gaussians.use_app = True

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, app_model=app_model,
                            return_plane=iteration>opt.single_view_weight_from_iter, return_depth_normal=iteration>opt.single_view_weight_from_iter)
        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        ssim_loss = (1.0 - ssim(image, gt_image))
        if 'app_image' in render_pkg and ssim_loss < 0.5:
            app_image = render_pkg['app_image']
            Ll1 = l1_loss(app_image, gt_image)
        else:
            Ll1 = l1_loss(image, gt_image)
        image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        loss = image_loss.clone()

        if iteration > opt.start_sample_pseudo:
            pseudo_loss = normal_consis.compute(
                gaussians=gaussians,
                iteration=iteration,
                cameras_extent=scene.cameras_extent
            )
            loss += 1.0 * pseudo_loss

        if iteration > opt.start_single_view:
            weight = opt.single_view_weight
            normal = render_pkg["rendered_normal"]
            depth_normal = render_pkg["depth_normal"]

            image_weight = (1.0 - get_img_grad_weight(gt_image))
            image_weight = (image_weight).clamp(0,1).detach() ** 2
            if not opt.wo_image_weight:
                # image_weight = erode(image_weight[None,None]).squeeze()
                normal_loss = weight * (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()
            else:
                normal_loss = weight * (((depth_normal - normal)).abs().sum(0)).mean()
            loss += (normal_loss)
        

        # scale loss normal direction
        if visibility_filter.sum() > 0 and iteration > opt.densify_from_iter:
            scale = gaussians.get_scaling[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[...,0]
            loss += opt.scale_loss_weight * min_scale_loss.mean()



        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * image_loss.item() + 0.6 * ema_loss_for_log
            ema_single_view_for_log = 0.4 * normal_loss.item() if normal_loss is not None else 0.0 + 0.6 * ema_single_view_for_log
            if pipe.use_H and pipe.use_MVS:
                ema_multi_view_geo_for_log = 0.4 * geo_loss.item() if geo_loss is not None else 0.0 + 0.6 * ema_multi_view_geo_for_log
                ema_multi_view_pho_for_log = 0.4 * ncc_loss.item() if ncc_loss is not None else 0.0 + 0.6 * ema_multi_view_pho_for_log
                ema_depth_for_log = 0.4 * depth_loss.item() if depth_loss is not None else 0.0 + 0.6 * ema_depth_for_log
                ema_depth_for_pseudo = 0.4 * pseudo_loss.item() if pseudo_loss is not None else 0.0 + 0.6 * ema_depth_for_pseudo
                if iteration % 10 == 0:
                    loss_dict = {
                        "Loss": f"{ema_loss_for_log:.{4}f}",
                        "Single": f"{ema_single_view_for_log:.{4}f}",
                        "Geo": f"{ema_multi_view_geo_for_log:.{4}f}",
                        "Pho": f"{ema_multi_view_pho_for_log:.{4}f}",
                        "Depth": f"{ema_depth_for_log:.{4}f}",
                        "Pseudo": f"{ema_depth_for_pseudo:.{4}f}",
                        "Points": f"{len(gaussians.get_xyz)}"
                    }
                    progress_bar.set_postfix(loss_dict)
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()                
            elif pipe.use_H:
                ema_multi_view_geo_for_log = 0.4 * geo_loss.item() if geo_loss is not None else 0.0 + 0.6 * ema_multi_view_geo_for_log
                ema_multi_view_pho_for_log = 0.4 * ncc_loss.item() if ncc_loss is not None else 0.0 + 0.6 * ema_multi_view_pho_for_log
                if iteration % 10 == 0:
                    loss_dict = {
                        "Loss": f"{ema_loss_for_log:.{5}f}",
                        "Single": f"{ema_single_view_for_log:.{5}f}",
                        "Geo": f"{ema_multi_view_geo_for_log:.{5}f}",
                        "Pho": f"{ema_multi_view_pho_for_log:.{5}f}",
                        "Points": f"{len(gaussians.get_xyz)}"
                    }
                    progress_bar.set_postfix(loss_dict)
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()
            elif pipe.use_MVS:
                ema_depth_for_log = 0.4 * depth_loss.item() if depth_loss is not None else 0.0 + 0.6 * ema_depth_for_log
                ema_depth_for_pseudo = 0.4 * pseudo_loss.item() if pseudo_loss is not None else 0.0 + 0.6 * ema_depth_for_pseudo
                if iteration % 10 == 0:
                    loss_dict = {
                        "Loss": f"{ema_loss_for_log:.{5}f}",
                        "Single": f"{ema_single_view_for_log:.{5}f}",
                        "Depth": f"{ema_depth_for_log:.{5}f}",
                        "Pseudo": f"{ema_depth_for_pseudo:.{4}f}",
                        "Points": f"{len(gaussians.get_xyz)}"
                    }
                    progress_bar.set_postfix(loss_dict)
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), app_model)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                    
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                mask = (render_pkg["out_observe"] > 0) & visibility_filter
                gaussians.max_radii2D[mask] = torch.max(gaussians.max_radii2D[mask], radii[mask])
                viewspace_point_tensor_abs = render_pkg["viewspace_points_abs"]
                gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_abs, visibility_filter)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # size_threshold = None
                    size_threshold = 20
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_abs_grad_threshold, 
                                                opt.opacity_cull_threshold, scene.cameras_extent, size_threshold, iteration)
            
            # multi-view observe trim
            # if opt.use_multi_view_trim and iteration % 1000 == 0 and iteration < opt.densify_until_iter:
            #     observe_the = 2
            #     observe_cnt = torch.zeros_like(gaussians.get_opacity)
            #     for view in scene.getTrainCameras():
            #         render_pkg_tmp = render(view, gaussians, pipe, bg, app_model=app_model, return_plane=False, return_depth_normal=False)
            #         out_observe = render_pkg_tmp["out_observe"]
            #         observe_cnt[out_observe > 0] += 1
            #     prune_mask = (observe_cnt < observe_the).squeeze()
            #     if prune_mask.sum() > 0:
            #         gaussians.prune_points(prune_mask)

            #reset_opacity
            if iteration < 7000:
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                app_model.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                app_model.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                app_model.save_weights(scene.model_path, iteration)

    app_model.save_weights(scene.model_path, opt.iterations)
    torch.cuda.empty_cache()

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, app_model):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    out = renderFunc(viewpoint, scene.gaussians, *renderArgs, app_model=app_model)
                    image = out["render"]
                    if 'app_image' in out:
                        image = out['app_image']
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image, _ = viewpoint.get_image()
                    gt_image = torch.clamp(gt_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test), flush=True)
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--debug_from', type=int, default=-100)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[50_00, 10000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5000,10000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    # args.test_iterations = list(range(1000, args.iterations + 1, 1000))
    # args.test_iterations.append(500)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # 先 extract，方便拿到 model_path
    model_params = lp.extract(args)
    optim_params = op.extract(args)
    pipe_params = pp.extract(args)

    # 确保 model_path 存在
    os.makedirs(model_params.model_path, exist_ok=True)

    time_json_path = os.path.join(model_params.model_path, "time.json")

    start_time = time.time()
    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    status = "success"
    error_msg = None

    try:
        training(
            model_params,
            optim_params,
            pipe_params,
            args.test_iterations,
            args.save_iterations,
            args.checkpoint_iterations,
            args.start_checkpoint,
            args.debug_from
        )

    except Exception as e:
        status = "failed"
        error_msg = str(e)
        raise

    finally:
        end_time = time.time()
        end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed_seconds = end_time - start_time

        time_record = {
            "status": status,
            "start_time": start_time_str,
            "end_time": end_time_str,
            "elapsed_seconds": elapsed_seconds,
            "elapsed_minutes": elapsed_seconds / 60.0,
            "elapsed_hours": elapsed_seconds / 3600.0,
            "elapsed_hms": time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds)),
            "model_path": model_params.model_path,
            "iterations": optim_params.iterations,
            "error": error_msg
        }

        with open(time_json_path, "w", encoding="utf-8") as f:
            json.dump(time_record, f, indent=4, ensure_ascii=False)

        print(f"\nTraining time saved to: {time_json_path}")


    # All done
    print("\nTraining complete.")
