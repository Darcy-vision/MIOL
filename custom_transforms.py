from __future__ import division
import torch
import random
import numpy as np
import cv2
from PIL import Image

'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''
# Setting parameters for StereoBM algorithm
numDisparities = 32
blockSize = 11

# Creating an object of StereoBM algorithm
bm = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, intrinsics, disparities=None):
        for t in self.transforms:
            images, intrinsics = t(images, intrinsics)
        return images, intrinsics


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, intrinsics):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images, intrinsics


class GetStereoDisparity(object):
    """
    Compute the disparity of left image and right image through BM algorithm in OpenCV.
    """
    def __call__(self, images):
        # images type is RGB, need to trans to Grey
        grey_imgL = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY).astype(np.uint8)
        grey_imgR = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY).astype(np.uint8)

        grey_imgR_flipped = cv2.flip(grey_imgR, 1)
        grey_imgL_flipped = cv2.flip(grey_imgL, 1)

        l2r_disp = bm.compute(grey_imgL, grey_imgR).astype(np.float32)
        r2l_disp = bm.compute(grey_imgR_flipped, grey_imgL_flipped).astype(np.float32)

        # without divide numDisparities(no need to normalize, just use the true disparity)
        # range from (-1, numDisparities)
        l2r_disp = l2r_disp/16.0
        r2l_disp = r2l_disp/16.0
        fliped_r2l_disp = cv2.flip(r2l_disp, 1)

        # cv2.imshow('left', images[0].astype(np.uint8))
        # cv2.imshow('right', images[1].astype(np.uint8))
        # l2r_disp = l2r_disp / numDisparities
        # fliped_r2l_disp = fliped_r2l_disp / numDisparities
        # cv2.imshow('left2right_disparity', l2r_disp)
        # cv2.imshow('right2left_disparity', fliped_r2l_disp)
        # cv2.waitKey(0)

        return images, [l2r_disp, fliped_r2l_disp]


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, images, intrinsics):
        tensors = []
        for im in images:
            # put it from HWC to CHW format
            im = np.transpose(im, (2, 0, 1))
            # handle numpy array
            tensors.append(torch.from_numpy(im).float()/255)
        return tensors, intrinsics


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, images, intrinsics):
        assert intrinsics is not None
        if random.random() < 0.5:
            output_intrinsics = np.copy(intrinsics)
            output_images = [np.copy(np.fliplr(im)) for im in images]
            w = output_images[0].shape[1]
            output_intrinsics[0, 2] = w - output_intrinsics[0, 2]

            # exachange the position of flipped left and right images to keep the stereo_T unchanged
            temp_image = output_images[0]
            output_images[0] = output_images[1]
            output_images[1] = temp_image

        else:
            output_images = images
            output_intrinsics = intrinsics

        return output_images, output_intrinsics


class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, images, intrinsics):
        assert intrinsics is not None
        output_intrinsics = [np.copy(intrinsic) for intrinsic in intrinsics] 

        in_h, in_w, _ = images[0].shape
        x_scaling, y_scaling = np.random.uniform(1, 1.15, 2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

        for output_intrinsic in output_intrinsics:
            output_intrinsic[0] *= x_scaling
            output_intrinsic[1] *= y_scaling

        scaled_images = [np.array(Image.fromarray(im.astype(np.uint8)).resize((scaled_w, scaled_h))).astype(np.float32) for im in images]

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)
        cropped_images = [im[offset_y:offset_y + in_h, offset_x:offset_x + in_w] for im in scaled_images]

        for output_intrinsic in output_intrinsics:
            output_intrinsic[0, 2] -= offset_x
            output_intrinsic[1, 2] -= offset_y

        return cropped_images, output_intrinsics
