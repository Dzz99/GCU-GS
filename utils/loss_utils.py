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

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
from pytorch3d.common.workaround import symeig3x3
from pytorch3d.ops import knn_points

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim2(img1, img2, window_size=11):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean(0)

def get_img_grad_weight(img, beta=2.0):
    _, hd, wd = img.shape 
    bottom_point = img[..., 2:hd,   1:wd-1]
    top_point    = img[..., 0:hd-2, 1:wd-1]
    right_point  = img[..., 1:hd-1, 2:wd]
    left_point   = img[..., 1:hd-1, 0:wd-2]
    grad_img_x = torch.mean(torch.abs(right_point - left_point), 0, keepdim=True)
    grad_img_y = torch.mean(torch.abs(top_point - bottom_point), 0, keepdim=True)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
    grad_img, _ = torch.max(grad_img, dim=0)
    grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())
    grad_img = torch.nn.functional.pad(grad_img[None,None], (1,1,1,1), mode='constant', value=1.0).squeeze()
    return grad_img

def lncc(ref, nea):
    # ref_gray: [batch_size, total_patch_size]
    # nea_grays: [batch_size, total_patch_size]
    bs, tps = nea.shape
    patch_size = int(np.sqrt(tps))

    ref_nea = ref * nea
    ref_nea = ref_nea.view(bs, 1, patch_size, patch_size)
    ref = ref.view(bs, 1, patch_size, patch_size)
    nea = nea.view(bs, 1, patch_size, patch_size)
    ref2 = ref.pow(2)
    nea2 = nea.pow(2)

    # sum over kernel
    filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
    padding = patch_size // 2
    ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea_sum = F.conv2d(nea, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea2_sum = F.conv2d(nea2, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_nea_sum = F.conv2d(ref_nea, filters, stride=1, padding=padding)[:, :, padding, padding]

    # average over kernel
    ref_avg = ref_sum / tps
    nea_avg = nea_sum / tps

    cross = ref_nea_sum - nea_avg * ref_sum
    ref_var = ref2_sum - ref_avg * ref_sum
    nea_var = nea2_sum - nea_avg * nea_sum

    cc = cross * cross / (ref_var * nea_var + 1e-8)
    ncc = 1 - cc
    ncc = torch.clamp(ncc, 0.0, 2.0)
    ncc = torch.mean(ncc, dim=1, keepdim=True)
    mask = (ncc < 0.9)
    return ncc, mask

def get_smallest_axis(gau, return_idx=False):
    """
    改进版：保留“取最短轴”的逻辑，但通过 detach 彻底切断 scale 的梯度。
    这样 scale 不会被 normal loss 驱动，仅 rotation 受影响。
    """
    rotation_matrices = gau.get_rotation_matrix()          # [N, 3, 3]
    
    # ← 关键修改：detach 切断 scale 的梯度
    _, smallest_axis_idx = gau.get_scaling.min(dim=-1)     # [N]
    smallest_axis_idx = smallest_axis_idx.detach()
    
    # 恢复原来形状用于 gather
    smallest_axis_idx = smallest_axis_idx[..., None, None].expand(-1, 3, -1)
    
    smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
    
    if return_idx:
        return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
    
    return smallest_axis.squeeze(dim=2)

from typing import Optional
def compute_weighted_pca_normals(
    xyz: torch.Tensor,                    # (N, 3) 
    weight: torch.Tensor,                 # (N, 1) 
    knn: torch.Tensor,                    # (N, K) 
    dist2: torch.Tensor,                  # (N, K) 
    radius: Optional[float] = None,       # 距离筛选阈值
    eps: float = 1e-8
) -> torch.Tensor:                        # (N, 3) 单位法向量
    """
    【计算机视觉点云处理】加权PCA法向估计（全GPU向量化实现）
    
    算法严谨说明（基于经典加权PCA + 局部邻域）：
    1. 对每个点 i 筛选邻域：dist(i,j) < radius（即 dist2 < radius²）
    2. 使用提供的 weight[j] 作为基权重，筛选后重新归一化（∑w = 1）
    3. 计算加权均值中心：
    \[\bar{\mathbf{p}}_i = \frac{\sum_{j \in \mathcal{N}} w_j \mathbf{p}_j}{\sum w_j}\]
    4. 计算加权协方差矩阵（3×3）：
    \[\mathbf{C}_i = \frac{1}{\sum w_j} \sum_{j \in \mathcal{N}} w_j (\mathbf{p}_j - \bar{\mathbf{p}}_i)(\mathbf{p}_j - \bar{\mathbf{p}}_i)^T\]
    5. 对 C_i 特征分解，取最小特征值对应的特征向量作为法向（torch.linalg.eigh 保证数值稳定）
    6. 归一化为单位向量
    
    优势：
    - 完全向量化
    - 支持任意radius筛选
    
    参考文献：Hoppe 1992 + 现代加权变体（Sanchez 2020, Weighted Normal Estimation 2023）
    """
    # -------------------------- 输入检查与预处理 --------------------------
    device = xyz.device
    N, K = knn.shape
    if weight.dim() == 1:
        weight = weight.unsqueeze(1)          # 统一为 (N, 1)
    assert weight.shape == (N, 1), f"weight 应为 (N,1)，当前 {weight.shape}"
    assert xyz.shape == (N, 3)
    assert knn.shape == (N, K)
    assert dist2.shape == (N, K)

    # -------------------------- Gather 邻域信息 --------------------------
    weight = weight.detach()
    neigh_xyz = xyz[knn]                      # (N, K, 3)
    neigh_w_base = weight[knn].squeeze(-1)    # (N, K) 基权重

    # -------------------------- 距离筛选掩码 --------------------------
    if radius is not None:
        mask = (dist2 < radius ** 2).to(xyz.dtype)   # (N, K) 0/1
    else:
        mask = torch.ones((N, K), device=device, dtype=xyz.dtype)

    # -------------------------- 最终权重 + 归一化 --------------------------
    dist = torch.sqrt(torch.clamp_min(dist2, 1e-12))   # (N, K)

    # 局部自适应尺度
    sigma = dist.mean(dim=1, keepdim=True).clamp_min(1e-12)

    # 高斯距离权重
    w_dist = torch.exp(-dist2 / (2 * sigma * sigma))   # (N, K)

    weights = neigh_w_base * w_dist * mask             # (N, K)
    w_sum = weights.sum(dim=1, keepdim=True).clamp_min_(eps)  # (N, 1)

    # -------------------------- 加权均值中心 --------------------------
    weighted_sum = torch.sum(weights.unsqueeze(-1) * neigh_xyz, dim=1)  # (N, 3)
    bar_p = weighted_sum / w_sum                                 # (N, 3)

    # -------------------------- 中心化差值 --------------------------
    centered = neigh_xyz - bar_p.unsqueeze(1)           # (N, K, 3)

    # -------------------------- 协方差矩阵 --------------------------
    # C = (w * centered)^T @ centered / w_sum
    weighted_centered = weights.unsqueeze(-1) * centered          # (N, K, 3)
    cov = torch.bmm(weighted_centered.transpose(1, 2), centered)   # (N, 3, 3)
    cov = cov / w_sum.unsqueeze(-1)                               # (N, 3, 3)

    # 可选：轻微正则化防止数值奇异（研究中强烈推荐）
    cov = cov + 1e-6 * torch.eye(3, device=device, dtype=cov.dtype).unsqueeze(0)

    # -------------------------- 特征分解 + 最小特征向量 --------------------------
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)   # eigenvalues 升序
    # eigenvalues, eigenvectors = symeig3x3(cov, eigenvectors=True)
    normals = eigenvectors[:, :, 0]                      # (N, 3) 最小特征向量

    # -------------------------- 归一化 --------------------------
    normals = F.normalize(normals, dim=-1, eps=1e-6)

    return eigenvalues, normals

class NormalConsistencyLoss:
    """
    Normal Consistency Pseudo Loss

    主要优化：
    1. KNN 仍然按 update_freq 更新。
    2. PCA normal 和 planarity 也放进 cache，不再每个 iteration 重算。
    3. compute() 每轮只计算当前 Gaussian 的 smallest-axis normal。
    4. large mode 中缓存 global_valid_idx，避免每次 torch.where 扫全局点。
    """

    def __init__(
        self,
        update_freq: int = 5,
        k_search: int = 16,
        opacity_thr: float = 0.1,
        large_threshold: int = 1_200_000,
    ):
        self.update_freq = update_freq
        self.k_search = k_search
        self.opacity_thr = opacity_thr
        self.large_threshold = large_threshold

        self.is_large = False
        self.last_update_iter = -1
        self.sector_num = 36

        # 原始 cache
        self.cached_sector_id = None
        self.cached_knn_list = None
        self.cached_dist2_list = None
        self.cached_valid_idx_list = None

        # 新增 cache：compute 阶段直接使用
        self.cached_valid_ini_idx_list = None     # 每个 partition 中属于当前 Gaussian 的 global index
        self.cached_normals_pca_list = None       # cached PCA normals
        self.cached_planarity_list = None         # cached planarity weights
        self.cached_global_valid_idx_list = None  # large mode 用，避免 compute 中重复 torch.where

    def _should_update(self, iteration: int) -> bool:
        return (iteration - 1) % self.update_freq == 0 or self.last_update_iter == -1

    def _reset_cache(self):
        self.cached_sector_id = None
        self.cached_knn_list = []
        self.cached_dist2_list = []
        self.cached_valid_idx_list = []

        self.cached_valid_ini_idx_list = []
        self.cached_normals_pca_list = []
        self.cached_planarity_list = []
        self.cached_global_valid_idx_list = []

    def _append_empty_cache(self):
        self.cached_knn_list.append(None)
        self.cached_dist2_list.append(None)
        self.cached_valid_idx_list.append(None)

        self.cached_valid_ini_idx_list.append(None)
        self.cached_normals_pca_list.append(None)
        self.cached_planarity_list.append(None)
        self.cached_global_valid_idx_list.append(None)

    def _compute_sector_id(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        计算扇区 ID。

        注意：
        原注释写的是 6 扇区，但 self.sector_num = 24。
        这里实际会分成 24 个扇区。
        """
        centroid = xyz.mean(dim=0)
        rel = xyz[:, :2] - centroid[:2]

        angles = torch.atan2(rel[:, 1], rel[:, 0])
        angles = (angles + 2 * torch.pi) % (2 * torch.pi)

        # 原代码里使用 .item() 会触发一次 GPU -> CPU 同步。
        # 这里只在 update_cache 时执行，频率较低，一般可以接受。
        offset = torch.rand(1, device=xyz.device).item() * (2 * torch.pi / self.sector_num)

        shifted = (angles + offset) % (2 * torch.pi)
        sector_size = 2 * torch.pi / self.sector_num

        return torch.floor(shifted / sector_size).long()

    @torch.no_grad()
    def _build_pca_cache_for_valid_points(
        self,
        x_valid: torch.Tensor,
        opacity_valid: torch.Tensor,
        knn_idx: torch.Tensor,
        dist2: torch.Tensor,
        global_valid_idx: torch.Tensor,
        num_current_gaussians: int,
        cameras_extent: float,
    ):
        """
        根据已经得到的 valid points 和 KNN，计算并缓存：
        1. 参与 loss 的当前 Gaussian index
        2. PCA normal
        3. planarity weight

        这里使用 torch.no_grad() 是合理的，因为原代码中 x/opacity 已经 detach，
        PCA normal 本身并没有对 xyz 或 opacity 反传梯度。
        """

        eigenvalues, normals_pca = compute_weighted_pca_normals(
            x_valid,
            opacity_valid.unsqueeze(-1) if opacity_valid.dim() == 1 else opacity_valid,
            knn_idx,
            dist2,
            radius=0.1 * cameras_extent,
        )

        # 只监督当前 Gaussian，不监督 cat 进来的 ini_points。
        mask_current = global_valid_idx < num_current_gaussians

        if mask_current.sum().item() == 0:
            return None, None, None

        valid_ini_idx = global_valid_idx[mask_current]
        eigenvalues = eigenvalues[mask_current]
        normals_pca = normals_pca[mask_current]

        sigma = eigenvalues[:, 0] / (eigenvalues.sum(dim=1) + 1e-8)
        planarity = (1.0 - 3.0 * sigma).clamp(0.0, 1.0)

        normals_pca = F.normalize(normals_pca, dim=-1, eps=1e-6).detach()
        planarity = planarity.detach()

        return valid_ini_idx.detach(), normals_pca, planarity

    def update_cache(self, gaussians, iteration: int, cameras_extent: float):
        """
        每隔 update_freq 更新：
        1. KNN
        2. PCA normal
        3. planarity
        4. loss 中真正使用的 current Gaussian index
        """

        if not self._should_update(iteration):
            return

        self.last_update_iter = iteration

        self._reset_cache()

        num_current_gaussians = gaussians.get_xyz.shape[0]

        # PCA target 不需要梯度，保持原逻辑中的 detach。
        xyz = torch.cat(
            [gaussians.get_xyz, gaussians.ini_points],
            dim=0,
        ).detach()

        opacity = torch.cat(
            [gaussians.get_opacity.squeeze(), gaussians.ini_opacity.squeeze()],
            dim=0,
        ).detach()

        self.is_large = xyz.shape[0] > self.large_threshold

        if self.is_large:
            self.cached_sector_id = self._compute_sector_id(xyz)

            for sid in range(self.sector_num):
                sector_mask = self.cached_sector_id == sid
                sector_global_idx = torch.nonzero(sector_mask, as_tuple=False).squeeze(1)

                if sector_global_idx.numel() < self.k_search + 1:
                    self._append_empty_cache()
                    continue

                x_sec = xyz[sector_global_idx]
                opacity_sec = opacity[sector_global_idx].squeeze()

                valid_mask_sec = opacity_sec > self.opacity_thr
                valid_idx_sec = torch.nonzero(valid_mask_sec, as_tuple=False).squeeze(1)

                if valid_idx_sec.numel() < self.k_search + 1:
                    self._append_empty_cache()
                    continue

                x_valid = x_sec[valid_idx_sec]
                opacity_valid = opacity_sec[valid_idx_sec]
                global_valid_idx = sector_global_idx[valid_idx_sec]

                knn_res = knn_points(
                    x_valid.unsqueeze(0),
                    x_valid.unsqueeze(0),
                    K=self.k_search + 1,
                    return_sorted=True,
                )

                dist2 = knn_res.dists.squeeze(0)[:, 1:].contiguous().detach()
                knn_idx = knn_res.idx.squeeze(0)[:, 1:].contiguous().to(torch.int64).detach()

                valid_ini_idx, normals_pca, planarity = self._build_pca_cache_for_valid_points(
                    x_valid=x_valid,
                    opacity_valid=opacity_valid,
                    knn_idx=knn_idx,
                    dist2=dist2,
                    global_valid_idx=global_valid_idx,
                    num_current_gaussians=num_current_gaussians,
                    cameras_extent=cameras_extent,
                )

                if valid_ini_idx is None:
                    self._append_empty_cache()
                    continue

                self.cached_knn_list.append(knn_idx)
                self.cached_dist2_list.append(dist2)
                self.cached_valid_idx_list.append(valid_idx_sec)
                self.cached_global_valid_idx_list.append(global_valid_idx.detach())

                self.cached_valid_ini_idx_list.append(valid_ini_idx)
                self.cached_normals_pca_list.append(normals_pca)
                self.cached_planarity_list.append(planarity)

        else:
            self.cached_sector_id = None

            valid_mask = opacity > self.opacity_thr
            valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)

            if valid_idx.numel() < self.k_search + 1:
                self.cached_valid_idx_list = None
                self.cached_valid_ini_idx_list = None
                self.cached_normals_pca_list = None
                self.cached_planarity_list = None
                return

            x_valid = xyz[valid_idx]
            opacity_valid = opacity[valid_idx]
            global_valid_idx = valid_idx

            knn_res = knn_points(
                x_valid.unsqueeze(0),
                x_valid.unsqueeze(0),
                K=self.k_search + 1,
                return_sorted=True,
            )

            dist2 = knn_res.dists.squeeze(0)[:, 1:].contiguous().detach()
            knn_idx = knn_res.idx.squeeze(0)[:, 1:].contiguous().to(torch.int64).detach()

            valid_ini_idx, normals_pca, planarity = self._build_pca_cache_for_valid_points(
                x_valid=x_valid,
                opacity_valid=opacity_valid,
                knn_idx=knn_idx,
                dist2=dist2,
                global_valid_idx=global_valid_idx,
                num_current_gaussians=num_current_gaussians,
                cameras_extent=cameras_extent,
            )

            if valid_ini_idx is None:
                self.cached_valid_idx_list = None
                self.cached_valid_ini_idx_list = None
                self.cached_normals_pca_list = None
                self.cached_planarity_list = None
                return

            self.cached_knn_list = [knn_idx]
            self.cached_dist2_list = [dist2]
            self.cached_valid_idx_list = [valid_idx]
            self.cached_global_valid_idx_list = [global_valid_idx.detach()]

            self.cached_valid_ini_idx_list = [valid_ini_idx]
            self.cached_normals_pca_list = [normals_pca]
            self.cached_planarity_list = [planarity]

    def _zero_loss(self, gaussians):
        return torch.zeros((), device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)

    def compute(self, gaussians, iteration: int, cameras_extent: float) -> torch.Tensor:
        """
        返回 scalar loss。

        优化后：
        - update_cache() 负责低频计算 KNN + PCA normal + planarity。
        - compute() 高频阶段只计算当前 Gaussian normal，并和缓存的 PCA normal 做 loss。
        """

        self.update_cache(gaussians, iteration, cameras_extent)

        if (
            self.cached_valid_ini_idx_list is None
            or self.cached_normals_pca_list is None
            or self.cached_planarity_list is None
        ):
            return self._zero_loss(gaussians)

        # 这里仍然是全量 get_smallest_axis。
        # 如果它仍然慢，下一步应改成 get_smallest_axis_subset(gaussians, valid_idx)。
        normal_gs_all = get_smallest_axis(gaussians)

        w_all = []
        loss_per_all = []

        num_parts = len(self.cached_valid_ini_idx_list)

        for pid in range(num_parts):
            valid_ini_idx = self.cached_valid_ini_idx_list[pid]
            normals_pca = self.cached_normals_pca_list[pid]
            w = self.cached_planarity_list[pid]

            if valid_ini_idx is None or normals_pca is None or w is None:
                continue

            if valid_ini_idx.numel() == 0:
                continue

            normal_gs_valid = normal_gs_all[valid_ini_idx]
            normal_gs_valid = F.normalize(normal_gs_valid, dim=-1, eps=1e-6)

            # normals_pca 在 cache 时已经 normalize 过。
            cos = torch.sum(normals_pca * normal_gs_valid, dim=-1).clamp(-1.0, 1.0)
            loss_per = 1.0 - cos * cos

            # 不要在这里写 if w.sum() > 1e-8。
            # 那会触发 GPU -> CPU 同步。
            w_all.append(w)
            loss_per_all.append(loss_per)

        if len(w_all) == 0:
            return self._zero_loss(gaussians)

        w_all = torch.cat(w_all, dim=0)
        loss_per_all = torch.cat(loss_per_all, dim=0)

        pseudo_loss = (w_all * loss_per_all).sum() / (w_all.sum() + 1e-8)

        return pseudo_loss

# class NormalConsistencyLoss:
#     """
#     Normal Consistency Pseudo Loss 类（满足分区计算要求）
    
#     Large (>1M points):
#         - 按极角6扇区分区
#         - 每个扇区独立 KNN + compute_weighted_pca_normals + loss_per
#         - 最后聚合6个分区的loss
    
#     Small (<=1M points):
#         - 一次性 cat ini_points，全局计算（保持你原有逻辑）
#     """

#     def __init__(self, update_freq: int = 5, k_search: int = 16, 
#                  opacity_thr: float = 0.1, large_threshold: int = 1_000_000):
        
#         self.update_freq = update_freq
#         self.k_search = k_search
#         self.opacity_thr = opacity_thr
#         self.large_threshold = large_threshold

#         # 缓存
#         self.cached_sector_id = None          # (N,) only for large mode
#         self.cached_knn_list = None           # list of knn per sector (large) or single tensor (small)
#         self.cached_dist2_list = None         # list of dist2 per sector (large) or single tensor (small)
#         self.cached_valid_idx_list = None     # list of valid_idx per sector (large) or single (small)
#         self.is_large = False
#         self.last_update_iter = -1
#         self.sector_num = 24

#     def _should_update(self, iteration: int) -> bool:
#         return (iteration - 1) % self.update_freq == 0 or self.last_update_iter == -1

#     def update_cache(self, gaussians, iteration: int):
#         """每固定轮次更新 KNN / sector 缓存"""
#         if not self._should_update(iteration):
#             return

#         self.last_update_iter = iteration
#         self.is_large = gaussians.get_xyz.shape[0] > self.large_threshold

#         if self.is_large:
#             # ====================== LARGE MODE ======================
#             xyz = torch.cat([gaussians.get_xyz, gaussians.ini_points], dim=0).detach()
#             opacity = torch.cat([gaussians.get_opacity.squeeze(), 
#                                gaussians.ini_opacity.squeeze()], dim=0).detach()
#             self.cached_sector_id = self._compute_sector_id(xyz)
#             self.cached_knn_list = []
#             self.cached_dist2_list = []
#             self.cached_valid_idx_list = []

#             for sid in range(self.sector_num):
#                 mask = (self.cached_sector_id == sid)
#                 if mask.sum() < self.k_search + 10:   # 点太少跳过该扇区
#                     self.cached_knn_list.append(None)
#                     self.cached_dist2_list.append(None)
#                     self.cached_valid_idx_list.append(None)
#                     continue

#                 x_sec = xyz[mask]
#                 opacity_sec = opacity[mask].squeeze()

#                 valid_mask = opacity_sec > self.opacity_thr
#                 valid_idx_sec = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)

#                 if valid_idx_sec.numel() < self.k_search + 1:
#                     self.cached_knn_list.append(None)
#                     self.cached_dist2_list.append(None)
#                     self.cached_valid_idx_list.append(None)
#                     continue

#                 x_valid = x_sec[valid_idx_sec]

#                 # 每个扇区独立建 KNN
#                 knn_res = knn_points(
#                     x_valid.unsqueeze(0), x_valid.unsqueeze(0),
#                     K=self.k_search + 1, return_sorted=True
#                 )

#                 dist2 = knn_res.dists.squeeze(0)[:, 1:].contiguous().detach()
#                 knn_idx = knn_res.idx.squeeze(0)[:, 1:].contiguous().to(torch.int64).detach()

#                 self.cached_knn_list.append(knn_idx)
#                 self.cached_dist2_list.append(dist2)
#                 self.cached_valid_idx_list.append(valid_idx_sec)   # 相对于本扇区的索引

#         else:
#             # ====================== SMALL MODE ======================
#             # 保持你原来的全局 + cat ini_points 逻辑
#             x = torch.cat([gaussians.get_xyz, gaussians.ini_points], dim=0).detach()
#             opacity = torch.cat([gaussians.get_opacity.squeeze(), 
#                                gaussians.ini_opacity.squeeze()], dim=0).detach()

#             valid_mask = opacity > self.opacity_thr
#             valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)

#             if valid_idx.numel() < self.k_search + 1:
#                 self.cached_valid_idx_list = None
#                 return

#             x_valid = x[valid_idx]

#             knn_res = knn_points(
#                 x_valid.unsqueeze(0), x_valid.unsqueeze(0),
#                 K=self.k_search + 1, return_sorted=True
#             )

#             self.cached_dist2_list = [knn_res.dists.squeeze(0)[:, 1:].contiguous().detach()]
#             self.cached_knn_list = [knn_res.idx.squeeze(0)[:, 1:].contiguous().to(torch.int64).detach()]
#             self.cached_valid_idx_list = [valid_idx]
#             self.cached_sector_id = None

#     def _compute_sector_id(self, xyz: torch.Tensor) -> torch.Tensor:
#         """计算6扇区ID"""
#         centroid = xyz.mean(dim=0)
#         rel = xyz[:, :2] - centroid[:2]
#         angles = torch.atan2(rel[:, 1], rel[:, 0])
#         angles = (angles + 2 * torch.pi) % (2 * torch.pi)

#         offset = torch.rand(1, device=xyz.device).item() * (torch.pi / 3)
#         shifted = (angles + offset) % (2 * torch.pi)
#         sector_size = 2 * torch.pi / self.sector_num
#         return torch.floor(shifted / sector_size).long()  # 0~5

#     def compute(self, gaussians, iteration: int, cameras_extent: float) -> torch.Tensor:
#         """返回 scalar loss（可直接 loss += pseudo_loss）"""
#         self.update_cache(gaussians, iteration)

#         if self.cached_valid_idx_list is None:
#             return torch.tensor(0.0, device="cuda")

#         if self.is_large:
#             # ====================== LARGE MODE: 按扇区分区计算 ======================
#             xyz = torch.cat([gaussians.get_xyz, gaussians.ini_points], dim=0).detach()
#             opacity = torch.cat([gaussians.get_opacity.squeeze(), 
#                                gaussians.ini_opacity.squeeze()], dim=0).detach()
#             normal_gs_all = get_smallest_axis(gaussians)
#             w_all = []
#             loss_per_all = []
#             for sid in range(self.sector_num):
#                 if self.cached_knn_list[sid] is None:
#                     continue

#                 valid_idx_sec = self.cached_valid_idx_list[sid]          # 扇区内有效点相对索引
#                 global_idx = torch.where(self.cached_sector_id == sid)[0][valid_idx_sec]  # 转全局索引

#                 x_valid = xyz[global_idx]
#                 opacity_valid = opacity[global_idx]

#                 # 使用缓存的 KNN 计算 PCA normals（带梯度）
#                 eigenvalues, normals_pca = compute_weighted_pca_normals(
#                     x_valid,
#                     opacity_valid.unsqueeze(-1) if opacity_valid.dim() == 1 else opacity_valid,
#                     self.cached_knn_list[sid],
#                     self.cached_dist2_list[sid],
#                     radius=0.1 * cameras_extent
#                 )
#                 ini_num = gaussians._xyz.shape[0]
#                 mask_ini = global_idx < ini_num
#                 ini_global_idx = global_idx[mask_ini]
#                 # 对应的 GS 法向
#                 normal_gs_valid = normal_gs_all[ini_global_idx]
#                 eigenvalues = eigenvalues[mask_ini]
#                 normals_pca = normals_pca[mask_ini]

#                 # 计算 planarity 和 loss_per
#                 sigma = eigenvalues[:, 0] / (eigenvalues.sum(dim=1) + 1e-8)
#                 planarity = (1.0 - 3.0 * sigma).clamp(0.0, 1.0)

#                 normals_pca = F.normalize(normals_pca, dim=-1, eps=1e-6)
#                 normal_gs_valid = F.normalize(normal_gs_valid, dim=-1, eps=1e-6)

#                 cos = torch.sum(normals_pca * normal_gs_valid, dim=-1).clamp(-1.0, 1.0)
#                 loss_per = 1.0 - cos * cos

#                 w = planarity.detach()

#                 if w.numel() > 0 and w.sum() > 1e-8:
#                     w_all.append(w)
#                     loss_per_all.append(loss_per)

#             if len(w_all) > 0:
#                 w_all = torch.cat(w_all, dim=0)
#                 loss_per_all = torch.cat(loss_per_all, dim=0)
#                 pseudo_loss = (w_all * loss_per_all).sum() / (w_all.sum() + 1e-8)
#             else:
#                 pseudo_loss = torch.tensor(0.0, device="cuda")

#         else:
#             # ====================== SMALL MODE: 一次性全局计算 ======================
#             valid_idx = self.cached_valid_idx_list[0]
#             x_cat = (torch.cat([gaussians.get_xyz, gaussians.ini_points], dim=0)).detach()
#             opacity_cat = (torch.cat([gaussians.get_opacity.squeeze(), 
#                                      gaussians.ini_opacity.squeeze()], dim=0)).detach()
#             x_valid = x_cat[valid_idx]
#             opacity_valid = opacity_cat[valid_idx]
#             eigenvalues, normals_pca = compute_weighted_pca_normals(
#                 x_valid,
#                 opacity_valid.unsqueeze(-1) if opacity_valid.dim() == 1 else opacity_valid,
#                 self.cached_knn_list[0].detach(),
#                 self.cached_dist2_list[0].detach(),
#                 radius=0.1 * cameras_extent
#             )

#             # 只保留原始 Gaussians 的点
#             ini_num = gaussians.get_xyz.shape[0]
#             mask_ini = valid_idx < ini_num
#             valid_ini_idx = valid_idx[mask_ini]

#             if valid_ini_idx.numel() == 0:
#                 return torch.tensor(0.0, device="cuda")

#             eigenvalues = eigenvalues[mask_ini]
#             normals_pca = normals_pca[mask_ini]
#             normal_gs = get_smallest_axis(gaussians)
#             normal_gs_valid = normal_gs[valid_ini_idx]
#             sigma = eigenvalues[:, 0] / (eigenvalues.sum(dim=1) + 1e-8)
#             planarity = (1.0 - 3.0 * sigma).clamp(0.0, 1.0)

#             normals_pca = F.normalize(normals_pca, dim=-1, eps=1e-6)
#             normal_gs_valid = F.normalize(normal_gs_valid, dim=-1, eps=1e-6)

#             cos = torch.sum(normals_pca * normal_gs_valid, dim=-1).clamp(-1.0, 1.0)
#             loss_per = 1.0 - cos * cos

#             w = planarity.detach()
#             pseudo_loss = (w * loss_per).sum() / (w.sum() + 1e-8)

#         return pseudo_loss