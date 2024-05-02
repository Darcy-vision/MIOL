from __future__ import division
import shutil
import numpy as np
import torch
import cv2
from path import Path
import datetime
from collections import OrderedDict
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0, 1, low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0, max_value, resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:, i])
                         for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


COLORMAPS = {'rainbow': opencv_rainbow(),
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'bone': cm.get_cmap('bone', 10000)}


def tensor2array(tensor, max_value=None, colormap='rainbow'):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        norm_array = tensor.squeeze().numpy()/max_value
        array = COLORMAPS[colormap](norm_array).astype(np.float32)
        array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.45 + tensor.numpy()*0.225
    return array

def recons_img_error(tensor1, tensor2, tensor3, ref_disp, idx):
    tensor1 = tensor1.detach().cpu()
    tensor2 = tensor2.detach().cpu()
    tensor3 = tensor3.detach().cpu()
    ref_disp = ref_disp.detach().cpu()
    assert(tensor1.size(0) == 3 and tensor2.size(0) == 3 and tensor3.size(0) == 3)
    ori_image = (0.45 + tensor1.numpy()*0.225) * 255
    ori_image = ori_image.transpose(1, 2, 0)

    ref_image = (0.45 + tensor2.numpy()*0.225) * 255
    ref_image = ref_image.transpose(1, 2, 0)

    recons_image = (0.45 + tensor3.numpy()*0.225) * 255
    recons_image = recons_image.transpose(1, 2, 0)

    np.save('misc/' + str(idx) + 'ref_disp.npy', ref_disp)

    if idx == 0:
        cv2.imwrite('misc/' + str(idx) + 'ori_image.jpg', ori_image)
        cv2.imwrite('misc/' + str(idx) + 'ref_image.jpg', ref_image)
    cv2.imwrite('misc/' + str(idx) + 'recons_image.jpg', recons_image)
    cv2.imwrite('misc/' + str(idx) + 'diff_image.jpg', ori_image - recons_image)


def save_checkpoint(save_path, iter_num, dispnet_state, filename='final.pth.tar'):
    file_prefix = 'dispnet' + str(iter_num)
    state = dispnet_state
    
    torch.save(state, save_path/'{}_{}'.format(file_prefix, filename))

def meta_overfit_save_weights(save_path, iter_num, dispnet_state, filename='.pth.tar'):
    file_prefix = 'img' + str(iter_num)
    state = dispnet_state
    
    torch.save(state, save_path/'{}{}'.format(file_prefix, filename))