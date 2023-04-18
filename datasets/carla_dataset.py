# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import random
import torch

# from .mono_dataset import MonoDataset
from .kitti_dataset import KITTIDepthDataset

class CARLADataset(KITTIDepthDataset):
    """Handmade dataset class
    """
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(CARLADataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        # edited for CARLA(fov: 100, img: 640x480)
        self.K = np.array([[0.41954981558, 0, 0.5, 0],
                           [0, 0.55939975411, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (640, 480)

    def check_depth(self):
        return True

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "leftImg8bit",
            folder,
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            "depth",
            folder,
            f_str)
        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        # depth_gt = np.array(depth_gt).astype(np.float32) / 256
        # depth_gt = np.array(depth_gt).astype(np.float32) / 255
        depth_gt = np.array(depth_gt)
        depth_gt = np.where(depth_gt==0, 1, depth_gt)
        depth_gt = np.array(depth_gt).astype(np.float32) / 255

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
