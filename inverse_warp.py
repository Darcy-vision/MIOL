import torch
import torch.nn.functional as F
import cv2
import numpy as np
import time

pixel_coords = None

def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1, h, w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def synthetize_newImage(img2_colors, disps, mode, padding_mode='zeros'):
    """
    mode = 0, 通过左图坐标网格(img1_colors) - 视差图来采样右视图(img2)，重建左视图; depth2cam为左图对应在右图的坐标
    mode = 1, 通过右图坐标网格(img1_colors) + 视差图来采样左视图(img2)，重建右视图
    """

    b, _, h, w = disps.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(disps)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(disps)  # [1, H, W]
    pixel_coords = torch.stack((j_range, i_range), dim=1) # [1, 2, H, W]
    src_pixel_coords = pixel_coords.repeat(b, 1, 1, 1).permute(0, 2, 3, 1) # [4, H, W, 2]
    
    if mode == 0: # left to right
        src_pixel_coords[:, :, :, 0] = src_pixel_coords[:, :, :, 0] - disps[:, 0, :, :] # [4, H, W, 2] - [4, 1, H, W]
    else: # right to left
        src_pixel_coords[:, :, :, 0] = src_pixel_coords[:, :, :, 0] + disps[:, 0, :, :]

    # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    X_norm = 2*(src_pixel_coords[:, :, :, 0])/(w-1) - 1
    Y_norm = 2*(src_pixel_coords[:, :, :, 1])/(h-1) - 1  # Idem [B, H*W]
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1)+(X_norm < -1)).detach()
        # make sure that no point in warped image is a combinaison of im and gray
        X_norm[X_mask] = 2
        Y_mask = ((Y_norm > 1)+(Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pro_pixel_coords = torch.stack([X_norm, Y_norm], dim=3)

    projected_img = F.grid_sample(img2_colors, pro_pixel_coords, padding_mode=padding_mode, align_corners=False)  

    return projected_img


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i, size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(
        input_name, 'x'.join(expected), list(input.size()))


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size() != depth.size():
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[:, :, :h, :w].expand(b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    X_norm = 2*(X / Z)/(w-1) - 1
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b, h, w, 2)


def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:, :1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:,
                                            1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat


def inverse_warp(img, depth, pose, intrinsics, rotation_mode='euler', padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_points: Boolean array indicating point validity
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth, intrinsics.inverse())  # [B,3,H,W]

    pose_mat = pose_vec2mat(pose, rotation_mode)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords = cam2pixel(
        cam_coords, rot, tr, padding_mode)  # [B,H,W,2]
    projected_img = F.grid_sample(
        img, src_pixel_coords, padding_mode=padding_mode)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1

    return projected_img, valid_points


def cam2pixel2(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pointcloud defined in the left camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    # pixel_coords = torch.stack([X/Z, Y/Z], dim=2)

    # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    # pixel coord x = X / Z, y = Y / Z
    # |fx  0 cx|   |Xw|   |-fx*t|    |X|    |u|
    # | 0 fy cy| * |Yw| + |  0  | =  |Y| = Z|v|
    # | 0  0  1|   |Zw|   |  0  |    |Z|    |1|
    X_norm = 2*(X / Z)/(w-1) - 1    
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1)+(X_norm < -1)).detach()
        # make sure that no point in warped image is a combinaison of im and gray
        X_norm[X_mask] = 2
        Y_mask = ((Y_norm > 1)+(Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords_norm = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]

    return pixel_coords_norm.reshape(b, h, w, 2), Z.reshape(b, 1, h, w)


def inverse_warp2(img, depth, ref_depth, pose, intrinsics, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the right image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the left image -- [B, 1, H, W]
        ref_depth: the right depth map (where to sample depth) -- [B, 1, H, W] 
        pose: transformation from left to right -- [B, 4, 4]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_mask: Float array indicating point validity
        projected_depth: sampled depth from source image  
        computed_depth: computed depth of source image using the target depth
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'B1HW')
    check_sizes(ref_depth, 'ref_depth', 'B1HW')
    check_sizes(pose, 'pose', 'B44')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth.squeeze(1), intrinsics.inverse())  # [B,3,H,W]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose[:, :3, :]  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords, computed_depth = cam2pixel2(cam_coords, rot, tr, padding_mode)  # [B,H,W,2]
    projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode, align_corners=False)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1
    valid_mask = valid_points.unsqueeze(1).float()

    projected_depth = F.grid_sample(ref_depth, src_pixel_coords, padding_mode=padding_mode, align_corners=False)

    return projected_img, valid_mask, projected_depth, computed_depth


def my_inverse_warp(img, depth, ref_depth, pose, intrinsics, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the left image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the right image -- [B, 1, H, W]
        ref_depth: the left depth map (where to sample depth) -- [B, 1, H, W] 
        pose: transformation from right to left -- [B, 4, 4]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        valid_mask: Float array indicating point validity
        projected_depth: sampled depth from source image  
        computed_depth: computed depth of source image using the target depth
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'B1HW')
    check_sizes(ref_depth, 'ref_depth', 'B1HW')
    check_sizes(pose, 'pose', 'B44')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    cam_coords = pixel2cam(depth.squeeze(1), intrinsics.inverse())  # [B,3,H,W]  右图的intrinsics

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose[:, :3, :]  # [B, 3, 4]  左图的intrinsics  pose是右到左

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords, computed_depth = cam2pixel2(cam_coords, rot, tr, padding_mode)  # [B,H,W,2]
    
    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1
    valid_mask3d = valid_points.unsqueeze(1).float()

    projected_depth = F.grid_sample(ref_depth, src_pixel_coords, padding_mode=padding_mode, align_corners=False)

    return valid_mask3d, projected_depth, computed_depth


def miccai_inverse_warp(img, depth, ref_depth, pose, intrinsics, ref_intrinsics, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the left image (where to sample pixels) -- [B, 3, H, W]
        depth: the left depth map -- [B, 1, H, W]
        ref_depth: the right depth map (where to sample depth) -- [B, 1, H, W] 
        pose: transformation from right to left -- [B, 4, 4]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        valid_mask: Float array indicating point validity
        projected_depth: sampled depth from source image  
        computed_depth: computed depth of source image using the target depth
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'B1HW')
    check_sizes(ref_depth, 'ref_depth', 'B1HW')
    check_sizes(pose, 'pose', 'B44')
    check_sizes(intrinsics, 'intrinsics', 'B33')
    check_sizes(ref_intrinsics, 'ref_intrinsics', 'B33')

    cam_coords = pixel2cam(ref_depth.squeeze(1), ref_intrinsics.inverse())  # [B,3,H,W]  右图的intrinsics

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose[:, :3, :]  # [B, 3, 4]  左图的intrinsics  pose是右到左

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords_norm, computed_depth = cam2pixel2(cam_coords, rot, tr, padding_mode)  # [B,H,W,2]
    
    # cam2pixel2 中进行了归一化，正常范围[-1, 1], 取绝对值后保证x,y轴的最大值不超过1
    valid_points = src_pixel_coords_norm.abs().max(dim=-1)[0] <= 1 
    valid_mask3d = valid_points.unsqueeze(1).float()

    projected_depth = F.grid_sample(depth, src_pixel_coords_norm, padding_mode=padding_mode, align_corners=False)

    return valid_mask3d, projected_depth, computed_depth


def mycam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, intrinsics, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pointcloud defined in the left camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]

    Z = pcoords[:, 2].clamp(min=1e-3)

    return Z.reshape(b, 1, h, w)

def get_opposite_depth(img, depth, pose, intrinsics, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the right image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the left image -- [B, 1, H, W]
        pose: transformation from left to right -- [B, 4, 4]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        valid_mask: Float array indicating point validity
        computed_depth: computed depth of source image using the target depth
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'B1HW')
    check_sizes(pose, 'pose', 'B44')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    cam_coords = pixel2cam(depth.squeeze(1), intrinsics.inverse())  # [B,3,H,W]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose[:, :3, :]  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    computed_depth = mycam2pixel(cam_coords, rot, tr, intrinsics, padding_mode)  # [B,H,W,2]

    return computed_depth
