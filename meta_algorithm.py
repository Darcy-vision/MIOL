from collections import OrderedDict
import torch
import torch.nn.functional as F
import numpy as np
from utils import tensor2array
from loss_functions import compute_smooth_loss, photo_and_geometry_loss
import time


def inner_update_MAML(fast_weights, loss, inner_lr):
    grads = torch.autograd.grad(
        loss, fast_weights.values(), create_graph=False, allow_unused=True)
    # Perform SGD
    fast_weights = OrderedDict(
        (name, param - inner_lr * grad) if grad is not None else (name, param)
        for ((name, param), grad) in zip(fast_weights.items(), grads))
    return fast_weights


def outer_update_MAML(model, meta_batch_loss):
    """ Simply backwards """
    meta_batch_loss.backward()


def MetaAlgorithm(args, disp_net, optimizer, tgt_img, ref_img, left_gt_disp, right_gt_disp,
                  left_intrinsics, right_intrinsics, pose, pose_inv, data_writer, record_idx, train=True): 

    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_weight
    task_loss, task_error = [], []

    task_num = record_idx * tgt_img.shape[0]
    task_id = 0

    # task loop
    for aa, bb, cc, dd, ee, ff, gg, hh in zip(tgt_img, ref_img, left_gt_disp, right_gt_disp,
                                        left_intrinsics, right_intrinsics, pose, pose_inv):
        # data preparation
        support_tgt_img = aa[:args.k_shot]
        query_tgt_img = aa[args.k_shot:]

        support_ref_img = bb[:args.k_shot]
        query_ref_img = bb[args.k_shot:]

        query_left_gt = cc[args.k_shot:]
        query_right_gt = dd[args.k_shot:]

        support_intrinsics = [ee[:args.k_shot], ff[:args.k_shot]]
        query_intrinsics = [ee[args.k_shot:], ff[args.k_shot:]]

        support_pose = gg[:args.k_shot]
        query_pose = gg[args.k_shot:]

        support_pose_inv = hh[:args.k_shot]
        query_pose_inv = hh[args.k_shot:]

    
        fast_weights = OrderedDict(disp_net.named_parameters())

        ### ---------- INNER TRAIN LOOP ---------- ###
        tgt_disps_train, ref_disps_train = update_disparity(disp_net, [support_tgt_img, support_ref_img], params=fast_weights)

        w, h = support_tgt_img.shape[-2], support_tgt_img.shape[-1]
        tgt_disp_train = F.interpolate(tgt_disps_train[0], (w, h), mode='bilinear', align_corners=False)
        ref_disp_train = F.interpolate(ref_disps_train[0], (w, h), mode='bilinear', align_corners=False)

        tgt_depth_train, ref_depth_train, real_tgt_disp_train, real_ref_disp_train = disp2depth(tgt_disp_train , ref_disp_train, support_intrinsics, support_pose)

        loss_2 = compute_smooth_loss(tgt_depth_train, support_tgt_img, ref_depth_train, support_ref_img)

        loss_1_left, loss_1_right, loss_3, lm, rm = photo_and_geometry_loss(support_tgt_img, support_ref_img, 
                                                                support_intrinsics, tgt_depth_train, ref_depth_train,
                                                                tgt_disp_train, ref_disp_train, support_pose, support_pose_inv, args.with_ssim,
                                                                args.with_mask, args.with_auto_mask, args.padding_mode)
    
        loss_inner = w1 * (loss_1_left + loss_1_right) + w2 * loss_2 + w3 * loss_3

        # inner update
        fast_weights = inner_update_MAML(fast_weights, loss_inner, args.inner_lr)

        
        ### ---------- OUTER TRAIN LOOP ---------- ###
        if train:
            tgt_disps_new, ref_disps_new = update_disparity(disp_net, [query_tgt_img, query_ref_img], params=fast_weights)
            
            w, h = support_tgt_img.shape[-2], support_tgt_img.shape[-1]
            tgt_disp_new = F.interpolate(tgt_disps_new[0], (w, h), mode='bilinear', align_corners=False)
            ref_disp_new = F.interpolate(ref_disps_new[0], (w, h), mode='bilinear', align_corners=False)

            tgt_depth_new, ref_depth_new, real_tgt_disp_new, real_ref_disp_new = disp2depth(tgt_disp_new, ref_disp_new, query_intrinsics, query_pose)

            loss_2 = compute_smooth_loss(tgt_depth_new, query_tgt_img, ref_depth_new, query_ref_img)

            loss_1_left, loss_1_right, loss_3, lm, rm = photo_and_geometry_loss(query_tgt_img, query_ref_img, 
                                                                    query_intrinsics, tgt_depth_new, ref_depth_new,
                                                                    tgt_disp_new, ref_disp_new, query_pose, query_pose_inv, args.with_ssim,
                                                                    args.with_mask, args.with_auto_mask, args.padding_mode)

            loss = w1 * (loss_1_left + loss_1_right) + w2 * loss_2 + w3 * loss_3
            output1 = torch.squeeze(tgt_disp_new, 1)
            
        else:
            with torch.no_grad():
                tgt_disps_new, ref_disps_new = update_disparity(disp_net, [query_tgt_img, query_ref_img], params=fast_weights)
            
                w, h = support_tgt_img.shape[-2], support_tgt_img.shape[-1]
                tgt_disp_new = F.interpolate(tgt_disps_new[0], (w, h), mode='bilinear', align_corners=False)
                ref_disp_new = F.interpolate(ref_disps_new[0], (w, h), mode='bilinear', align_corners=False)

                tgt_depth_new, ref_depth_new, real_tgt_disp_new, real_ref_disp_new = disp2depth(tgt_disp_new, ref_disp_new, query_intrinsics, query_pose)

                loss_2 = compute_smooth_loss(tgt_depth_new, query_tgt_img, ref_depth_new, query_ref_img)

                loss_1_left, loss_1_right, loss_3, lm, rm = photo_and_geometry_loss(query_tgt_img, query_ref_img, 
                                                                        query_intrinsics, tgt_depth_new, ref_depth_new,
                                                                        tgt_disp_new, ref_disp_new, query_pose, query_pose_inv, args.with_ssim,
                                                                        args.with_mask, args.with_auto_mask, args.padding_mode)
            
                loss = w1 * (loss_1_left + loss_1_right) + w2 * loss_2 + w3 * loss_3
                output1 = torch.squeeze(tgt_disp_new, 1)
        
        ### ------------LOG info------------- ###
        if record_idx % 200 == 0:

            data_writer.add_scalar('inner loss', loss_inner.item(), task_num + task_id)
            data_writer.add_scalar('left photo loss', loss_1_left.item(), task_num + task_id)
            data_writer.add_scalar('smooth loss', loss_2.item(), task_num + task_id)
            data_writer.add_scalar('geometry loss', loss_3.item(), task_num + task_id)
            
            if train:
                data_writer.add_image('Train Support Left', tensor2array(support_tgt_img[0]), task_num + task_id)
                data_writer.add_image('Train Query Left', tensor2array(query_tgt_img[0]), task_num + task_id)
                data_writer.add_image('Train Support Left Disp',
                                        tensor2array(real_tgt_disp_train[0][0], max_value=None, colormap='magma'), task_num + task_id)
                data_writer.add_image('Train Query Left Disp',
                                        tensor2array(tgt_disp_new[0][0], max_value=None, colormap='magma'), task_num + task_id)
            else:
                data_writer.add_image('Val Support Left', tensor2array(support_tgt_img[0]), task_num + task_id)
                data_writer.add_image('Val Query Left', tensor2array(query_tgt_img[0]), task_num + task_id)
                data_writer.add_image('Val Support Left Disp',
                                        tensor2array(real_tgt_disp_train[0][0], max_value=None, colormap='magma'), task_num + task_id)
                data_writer.add_image('Val Query Left Disp',
                                        tensor2array(tgt_disp_new[0][0], max_value=None, colormap='magma'), task_num + task_id)

        left_errors = abs(query_left_gt - output1).mean().item()

        if left_errors is not None:
            task_loss.append(loss)
            task_error.append(left_errors)
                
        task_id = task_id + 1

    # outer loop update
    # compute gradient and do Adam step
    optimizer.zero_grad()
    meta_batch_loss = torch.stack(task_loss).mean()
    if train:
        outer_update_MAML(disp_net, meta_batch_loss)
        optimizer.step()
    task_error = np.mean(task_error)

    return meta_batch_loss, task_error


def update_disparity(disp_net, input_images, params=None):
    """
    SSDNet compute the multi-scale delta disparity;
    Upsample and add the result to the initial depth
    Args:
        disp_net(SSDNet):   compute the multi-scale delta disparity
        input_images(list): left and right images
        init_disparities(list):   left and right init disparity 
    """
    if params is None:
        left_disp, right_disp = disp_net(input_images)
    else:
        left_disp, right_disp = disp_net.functional_forward(input_images, params)

    return left_disp, right_disp


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

        # avoid zero division
        real_left_disp = left_disp + right_cx - left_cx
        real_right_disp = right_disp + right_cx - left_cx
        real_left_disp[(real_left_disp <= 0).detach()] = 0.01
        real_right_disp[(real_right_disp <= 0).detach()] = 0.01
        
        real_left_disps.append(real_left_disp)
        real_right_disps.append(real_right_disp)

        # fx * baseline / [(u_left - u_right) - (cx_left - cx_right)]
        temp_left = (left_fx * b / real_left_disp).clamp(min=1e-3, max=500)
        temp_right = (right_fx * b / real_right_disp).clamp(min=1e-3, max=500)

        left_depth.append(temp_left)
        right_depth.append(temp_right)

    real_left_disps = torch.stack(real_left_disps, dim=0)
    real_right_disps = torch.stack(real_right_disps, dim=0)

    left_depth = torch.stack(left_depth, dim=0)
    right_depth = torch.stack(right_depth, dim=0)

    return left_depth, right_depth, real_left_disps, real_right_disps