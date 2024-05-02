from __future__ import division
import torch
from torch import nn
from inverse_warp import miccai_inverse_warp, synthetize_newImage
import numpy as np
import cv2

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


compute_ssim_loss = SSIM().to(device)


def photo_and_geometry_loss(tgt_img, ref_img, intrinsics, tgt_depth, ref_depth, tgt_disp, ref_disp, pose, pose_inv, with_ssim, with_mask, with_auto_mask, padding_mode):

    photo_loss = 0
    geometry_loss = 0

    assert type(tgt_depth) == torch.Tensor and type(ref_depth) == torch.Tensor, "We just use the max scale of depth"
    
    ref_valid_mask, tgt_projected_depth, tgt_computed_depth = miccai_inverse_warp(tgt_img, tgt_depth, ref_depth, pose_inv, intrinsics[0], intrinsics[1], padding_mode)
    tgt_valid_mask, ref_projected_depth, ref_computed_depth = miccai_inverse_warp(ref_img, ref_depth, tgt_depth, pose, intrinsics[1], intrinsics[0], padding_mode)

    # cross-validated mask
    b, _, h, w = tgt_img.shape
    tgt_disp_lb = ref_disp[:,:,:,0].detach().int()
    ref_disp_rb = tgt_disp[:,:,:,-1].detach().int()
    tgt_disp_lb = tgt_disp_lb.unsqueeze(-1).repeat(1,1,1,tgt_img.shape[-1])
    ref_disp_rb = ref_disp_rb.unsqueeze(-1).repeat(1,1,1,tgt_img.shape[-1])

    mask_l = torch.linspace(0, w-1, w)[None, None, None, :].to(tgt_img.device).repeat(1, 1, h, 1)
    mask_r = mask_l.flip(dims=[-1])

    mask_l = mask_l - tgt_disp_lb
    mask_l[mask_l > 0] = 1
    mask_l[mask_l < 0] = 0
    tgt_valid_mask = tgt_valid_mask * mask_l

    mask_r = mask_r - ref_disp_rb
    mask_r[mask_r > 0] = 1
    mask_r[mask_r < 0] = 0
    ref_valid_mask = ref_valid_mask * mask_r

    # loss compute
    photo_loss1, geometry_loss1, left_mask = cp_loss(tgt_img, ref_img, tgt_disp, ref_projected_depth, ref_computed_depth, 
                                                        tgt_valid_mask, 0, with_ssim, with_mask, with_auto_mask)
    photo_loss2, geometry_loss2, right_mask = cp_loss(ref_img, tgt_img, ref_disp, tgt_projected_depth, tgt_computed_depth,
                                                        ref_valid_mask, 1, with_ssim, with_mask, with_auto_mask)
    geometry_loss += (geometry_loss1 + geometry_loss2)

    return photo_loss1, photo_loss2, geometry_loss, left_mask, right_mask


def cp_loss(tgt_img, ref_img, tgt_disp, projected_depth, computed_depth, mask, mode, with_ssim, with_mask, with_auto_mask):

    ref_img_warped = synthetize_newImage(ref_img, tgt_disp, mode)

    diff_img = (tgt_img - ref_img_warped).abs().clamp(0, 1)

    diff_depth = ((computed_depth - projected_depth).abs() / (computed_depth + projected_depth)).clamp(0, 1)

    if with_auto_mask: # 0
        auto_mask = (diff_img.mean(dim=1, keepdim=True) < (tgt_img - ref_img).abs().mean(dim=1, keepdim=True)).float() * mask
        mask = auto_mask
 
    if with_ssim:   # 1
        ssim_map = compute_ssim_loss(tgt_img, ref_img_warped)
        diff_img = (0.15 * diff_img + 0.85 * ssim_map)

    if with_mask:   # 1
        weight_mask = (1 - diff_depth)
        diff_img = diff_img * weight_mask
        
    reconstruction_loss = mean_on_mask(diff_img, mask)
    geometry_consistency_loss = mean_on_mask(diff_depth, mask)

    return reconstruction_loss, geometry_consistency_loss, mask


# compute mean value given a binary mask
def mean_on_mask(diff, valid_mask):

    mask = valid_mask.expand_as(diff)
    mean_value = (diff * mask).sum() / mask.sum()

    return mean_value


def compute_smooth_loss(left_disp, left_img, right_disp, right_img):
    def get_smooth_loss(disp, img):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """

        # normalize
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        disp = norm_disp

        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()

    loss_left = get_smooth_loss(left_disp, left_img)

    loss_right = get_smooth_loss(right_disp, right_img)

    return loss_left + loss_right


def cal_LK(left_img_rgb, right_img_rgb, intrinsics, scale, downsample):
    """
    left_img_rgb:左图
    right_img_rgb:右图
    intrinsics: [list] 左内参+右内参
    """

    h, w, c = left_img_rgb.shape
    left_img_rgb = cv2.resize(left_img_rgb, (int(w/scale), int(h/scale)))
    right_img_rgb = cv2.resize(right_img_rgb, (int(w/scale), int(h/scale)))
    
    img_left_grey = cv2.cvtColor(left_img_rgb, cv2.COLOR_BGR2GRAY)
    img_right_grey = cv2.cvtColor(right_img_rgb, cv2.COLOR_BGR2GRAY)
    
    h, w = img_left_grey.shape
    x_range = np.linspace(0, w, int(w/downsample), endpoint=False)
    y_range = np.linspace(0, h, int(h/downsample), endpoint=False)
    X, Y = np.meshgrid(x_range, y_range)
    X = X.flatten()
    Y = Y.flatten()
    X_norm = 2*(X / w)-1
    Y_norm = 2*(Y / h)-1
    p0 = np.vstack((X, Y)).transpose().reshape(-1, 1, 2).astype(np.float32)
    p0_norm = np.vstack((X_norm, Y_norm)).transpose().reshape(-1, 1, 2).astype(np.float32)

    pt1, st, err = cv2.calcOpticalFlowPyrLK(img_left_grey, img_right_grey, p0, None, maxLevel=3, minEigThreshold=0.001)

    p0 = p0.reshape(int(h/downsample), int(w/downsample), 1, 2)
    p0_norm = p0_norm.reshape(1, int(h/downsample), int(w/downsample), 2)
    pt1 = pt1.reshape(int(h/downsample), int(w/downsample), 1, 2)
    st = st.reshape(int(h/downsample), int(w/downsample)).astype(np.bool8)
    err = err.reshape(int(h/downsample), int(w/downsample))

    tangent = np.abs((pt1[:,:,0,1] - p0[:,:,0,1]) / (pt1[:,:,0,0] - p0[:,:,0,0] + 1e-6))
    right_pos_mask = pt1[:,:,0,0] > 0
    valid_mask = (tangent < 0.05) & st & right_pos_mask

    vis_mask = [valid_mask[:,-1:]]
    copy_pt1_u = -pt1[:,:,:,0]
    max_ = copy_pt1_u[:,-1,:]
    max_[~valid_mask[:,-1:]] = -w

    for col in range(copy_pt1_u.shape[1] - 2, -1, -1):
        cur_ = copy_pt1_u[:, col, :]

        col_mask = (cur_ > max_) & valid_mask[:, col:(col+1)]
        vis_mask.append(col_mask)

        max_[col_mask] = cur_[col_mask]
    
    vis_mask = np.stack(vis_mask[::-1]).transpose(1,0,2)

    right_cx = intrinsics[1][0,0,2]
    left_cx = intrinsics[0][0,0,2]
    left_disp_discrete = p0[:,:,:,0] - pt1[:,:,:,0] + right_cx.item() - left_cx.item()
    left_disp_discrete[~vis_mask] = 0
    left_disp_discrete = left_disp_discrete[:,:,0]

    return torch.from_numpy(p0_norm), torch.from_numpy(left_disp_discrete)


@torch.no_grad()
def compute_errors(gt, pred, args, left_masks):
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = 0, 0, 0, 0, 0, 0, 0
    batch_size, h, w = gt.size()
    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    max_depth = 300

    pred[pred > 300] = 300

    invalid_gt = 0
    for current_gt, current_pred, left_mask in zip(gt, pred, left_masks):
        valid = (current_gt > 1e-3) & (current_gt < max_depth)
        valid = valid & left_mask

        if current_gt.max() == 0 or valid.sum() == 0:
            invalid_gt = invalid_gt + 1
            continue

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid]

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.1).float().mean()
        a2 += (thresh < 1.1 ** 2).float().mean()
        a3 += (thresh < 1.1 ** 3).float().mean()
        # print((thresh < 1.1).float().mean())

        rmse_temp = (valid_gt - valid_pred) ** 2
        rmse += torch.sqrt(rmse_temp.mean())
        rmse_log_temp = (torch.log(valid_gt) - torch.log(valid_pred)) ** 2
        rmse_log += torch.sqrt(rmse_log_temp.mean())

        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)
        sq_rel += torch.mean(((valid_gt - valid_pred) ** 2) / valid_gt)

    if batch_size != invalid_gt:
        return [metric.item() / (batch_size - invalid_gt) for metric in [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]]
    else:
        return None