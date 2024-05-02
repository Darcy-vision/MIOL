import argparse
import time
import cv2
import os
import numpy as np
import torch.nn.functional as F
import torch
import torch.optim
import torch.utils.data
from tqdm import tqdm
import models
import custom_transforms
from loss_functions import compute_smooth_loss, photo_and_geometry_loss
import matplotlib as mpl
import matplotlib.cm as cm


parser = argparse.ArgumentParser(description='Inference script for MOIL on test image sequence',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--lr', default=1e-4, type=float, metavar='LR', help='online learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--resnet-layers', type=int, default=18, choices=[18, 50], help='depth network architecture.')
parser.add_argument('--max-disp', type=int, default=192, help='max disparity of network')
parser.add_argument('--with-ssim', type=int, default=1, help='with ssim or not')
parser.add_argument('--with-mask', type=int, default=1, help='with the the mask for moving objects and occlusions or not')
parser.add_argument('--with-auto-mask', type=int, default=0, help='with the the mask for stationary points')
parser.add_argument('--with-pretrain', type=int, default=1, help='with or without imagenet pretrain for resnet')
parser.add_argument('--pretrained-model', dest='pretrained_model', required=True,
                        default='sceneflow_pretrained.tar', metavar='PATH', help='path to pre-trained SSDNet model')
parser.add_argument('--calib-path', dest='calib_path', required=True,
                        default='/home/dell/Codes/MIOL/data/test_02/calib.yaml', metavar='PATH', help='path to stereo calibration file')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image. zeros will null gradients outside target image.'
                        ' border will only null gradients of the coordinate outside (x or y)')

parser.add_argument("--dataset-dir", required=True, default='/home/dell/Codes/MIOL/data/test_02/', type=str, help="Dataset directory")
parser.add_argument("--output-depth", action='store_true', help="save depth img")
parser.add_argument("--output-dir", required=True, type=str, help="Output directory")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def depth_visualizer(data):
    """
    Args:
        data (HxW): depth data
    Returns:
        vis_data (HxWx3): depth visualization (RGB)
    """

    inv_depth = 1 / (data + 1e-6)
    vmax = np.percentile(inv_depth, 95)
    normalizer = mpl.colors.Normalize(vmin=inv_depth.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    vis_data = (mapper.to_rgba(inv_depth)[:, :, :3] * 255).astype(np.uint8)
    return vis_data


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

    # LK flow
    pt1, st, err = cv2.calcOpticalFlowPyrLK(img_left_grey, img_right_grey, p0, None, maxLevel=3, minEigThreshold=0.001)

    p0 = p0.reshape(int(h/downsample), int(w/downsample), 1, 2)
    p0_norm = p0_norm.reshape(1, int(h/downsample), int(w/downsample), 2)
    pt1 = pt1.reshape(int(h/downsample), int(w/downsample), 1, 2)
    st = st.reshape(int(h/downsample), int(w/downsample)).astype(np.bool8)
    err = err.reshape(int(h/downsample), int(w/downsample))

    # calculate tangent map
    tangent = np.abs((pt1[:,:,0,1] - p0[:,:,0,1]) / (pt1[:,:,0,0] - p0[:,:,0,0] + 1e-6))
    right_pos_mask = pt1[:,:,0,0] > 0
    valid_mask = (tangent < 0.05) & st & right_pos_mask
    
    # calculate occlusion mask according to the disparity
    vis_mask = [valid_mask[:,-1:]]
    copy_pt1_u = -pt1[:,:,:,0] # record the u-axis position of the right image corresponding to the left image pixel
    max_ = copy_pt1_u[:,-1,:] # the last column
    max_[~valid_mask[:,-1:]] = -w # nvalid points are directly set to -w

    for col in range(copy_pt1_u.shape[1] - 2, -1, -1):
        cur_ = copy_pt1_u[:, col, :]

        col_mask = (cur_ > max_) & valid_mask[:, col:(col+1)]
        vis_mask.append(col_mask)

        max_[col_mask] = cur_[col_mask]
    
    vis_mask = np.stack(vis_mask[::-1]).transpose(1,0,2)

    right_cx = intrinsics[1][0,2]
    left_cx = intrinsics[0][0,2]
    left_disp_discrete = p0[:,:,:,0] - pt1[:,:,:,0] + right_cx - left_cx
    left_disp_discrete[~vis_mask] = 0
    left_disp_discrete = left_disp_discrete[:,:,0]

    return p0_norm, left_disp_discrete


def disp2depth(left_disps, right_disps, intrinsics, poses):
    
    left_depth = []
    right_depth = []

    real_left_disps = []
    real_right_disps = []

    for left_disp, right_disp, left_intrinsic, right_intrinsic, pose in zip(left_disps, right_disps, intrinsics[0], intrinsics[1], poses):
        left_fx = left_intrinsic[0, 0]
        right_fx = right_intrinsic[0, 0]

        left_cx = left_intrinsic[0, 2]
        right_cx = right_intrinsic[0, 2]

        b = -pose[0, 3]  # baseline

        # avoid nan in backward
        real_left_disp = left_disp + right_cx - left_cx
        real_right_disp = right_disp + right_cx - left_cx
        real_left_disp[(real_left_disp <= 0).detach()] = 0.01
        real_right_disp[(real_right_disp <= 0).detach()] = 0.01
        
        real_left_disps.append(real_left_disp)
        real_right_disps.append(real_right_disp)

        # Explain: depth = fx * baseline / [(u_left - u_right) - (cx_left - cx_right)]
        temp_left = (left_fx * b / real_left_disp).clamp(min=1e-3, max=300)
        temp_right = (right_fx * b / real_right_disp).clamp(min=1e-3, max=300)

        left_depth.append(temp_left)
        right_depth.append(temp_right)

    real_left_disps = torch.stack(real_left_disps, dim=0)
    real_right_disps = torch.stack(real_right_disps, dim=0)

    left_depth = torch.stack(left_depth, dim=0)
    right_depth = torch.stack(right_depth, dim=0)

    return left_depth, right_depth, real_left_disps, real_right_disps


def inference(args, left_img_input, right_img_input, debug_transform, intrinsics, pose, pose_inv, disp_net, optimizer, record_idx):
    
    # compute sparse disparity from LK flow
    # scale and downsample depend on your image size
    scale = 1
    downsample = 8
    sample_points, left_LK_disp = cal_LK(left_img_input, right_img_input, intrinsics, scale, downsample)
    sample_points = torch.from_numpy(sample_points).cuda()
    left_LK_disp = torch.from_numpy(left_LK_disp).cuda()
    
    # from numpy to tensor
    imgs, intrinsics = debug_transform([left_img_input, right_img_input], intrinsics)

    # data to GPU
    left_img = imgs[0].unsqueeze(0).cuda()
    right_img = imgs[1].unsqueeze(0).cuda()
    intrinsics = [torch.from_numpy(intrinsic).unsqueeze(0).cuda() for intrinsic in intrinsics]
    pose = torch.from_numpy(pose).unsqueeze(0).cuda()
    pose_inv = torch.from_numpy(pose_inv).unsqueeze(0).cuda()

    # switch to train mode
    disp_net.train()

    # ============ Online Learning ============ #
    tgt_disps, ref_disps = disp_net([left_img, right_img])

    w, h = left_img.shape[-2], left_img.shape[-1]
    tgt_disp_new = F.interpolate(tgt_disps[0], (w, h), mode='bilinear', align_corners=False)
    ref_disp_new = F.interpolate(ref_disps[0], (w, h), mode='bilinear', align_corners=False)

    sample_disp = F.grid_sample(tgt_disp_new, grid=sample_points, mode='nearest', align_corners=True)
    err = torch.abs(sample_disp[0,0] - left_LK_disp)
    mask_lk = left_LK_disp > 0
    err[~mask_lk] = 0
    loss_lk = err.mean()

    left_depth, right_depth, rl, rr = disp2depth(tgt_disp_new, ref_disp_new, intrinsics, pose)

    loss_new_2 = compute_smooth_loss(left_depth, left_img, right_depth, right_img)

    loss_1_left, loss_1_right, loss_new_3, lm, rm = photo_and_geometry_loss(left_img, right_img, intrinsics,
                                                            left_depth, right_depth, tgt_disp_new, ref_disp_new, pose, pose_inv,
                                                            args.with_ssim, args.with_mask, args.with_auto_mask, args.padding_mode)

    loss_new = 0.01 * loss_lk + (loss_1_left + loss_1_right) + 0.1 * loss_new_2 + 0.5 * loss_new_3

    # compute gradient and do Adam step
    optimizer.zero_grad()
    loss_new.backward()
    optimizer.step()

    if args.output_depth:

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        # save img-depth map
        vis_tgt_depth = left_depth.detach().cpu().numpy()[0,0]
        vis_pred = depth_visualizer(vis_tgt_depth).astype(np.uint8)
        vis_lm = lm[0].cpu().numpy().repeat(3,0).transpose(1,2,0).astype(np.bool8)
        vis_pred[~vis_lm]=0

        hh, ww, _ = left_img_input.shape
        sum_img = np.zeros((hh, ww*2, 3))
        sum_img[:, :ww, :] = left_img_input
        sum_img[:, ww:ww*2, :] = cv2.cvtColor(vis_pred, cv2.COLOR_RGB2BGR)
        
        png_path = os.path.join(args.output_dir, "show{:06}.png".format(record_idx))
        npy_path = os.path.join(args.output_dir, "show{:06}.npy".format(record_idx))
        cv2.imwrite(png_path, sum_img)
        np.save(npy_path, vis_tgt_depth)


def main():

    args = parser.parse_args()

    # create model
    print("=> creating model")
    disp_net = models.SSDNet(args.resnet_layers, args.with_pretrain, args.max_disp).cuda()

    # load parameters
    if args.pretrained_model is not None:
        print("=> using pre-trained weights for SSDNet")
        weights = torch.load(args.pretrained_model)
        disp_net.load_state_dict(weights['state_dict'])

    # setting solver
    print('=> setting adam solver') 
    optimizer = torch.optim.Adam(disp_net.parameters(), lr=args.lr,
                                 betas=(args.momentum, args.beta), weight_decay=args.weight_decay)

    # read camera intrinsics and extrinsics
    fs = cv2.FileStorage(args.calib_path, cv2.FileStorage_READ)

    left_intrinsics = fs.getNode('K1').mat()
    right_intrinsics = fs.getNode('K2').mat()
    intrinsics = [left_intrinsics, right_intrinsics]
    translation = fs.getNode('T').mat()
    pose = np.eye(4, dtype=np.float32)
    pose_inv = np.eye(4, dtype=np.float32)
    pose[0, 3] = translation[0][0]
    pose_inv[0, 3] = -translation[0][0]

    # Data loading code
    debug_transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(),
        custom_transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])])
    
    # imgs path
    all_files = sorted(os.listdir(args.dataset_dir))
    left_imgs = [file for file in all_files if file.endswith('.jpg')]

    for j in tqdm(range(len(left_imgs))):
        
        left = cv2.imread(os.path.join(args.dataset_dir, left_imgs[j]))
        right = cv2.imread(os.path.join(args.dataset_dir.replace('02', '03'), left_imgs[j]))

        inference(args, left, right, debug_transform, intrinsics, pose, pose_inv, disp_net, optimizer, j)


if __name__ == '__main__':

    main()