import time
import numpy as np
from open3d import *
import cv2


fs = cv2.FileStorage('data\\test_02\\calib.yaml', cv2.FileStorage_READ)
left_intrinsics = fs.getNode('K1').mat()
fx = left_intrinsics[0, 0]
fy = left_intrinsics[1, 1]
cx = left_intrinsics[0, 2]
cy = left_intrinsics[1, 2]

h = 256
w = 448

# path for imgs and npy of MICCAI
rgb_img = cv2.imread('data\\test_02\\00600.jpg')
depth_raw = np.load('output/show000600.npy')


cv2.imwrite('image.png', rgb_img)
cv2.imwrite('depth.png', depth_raw)

# 三维有颜色点云显示
color_raw = open3d.io.read_image("image.png")
depth_raw = open3d.io.read_image("depth.png")
rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, 1, 1, False)
print(rgbd_image)

inter = open3d.camera.PinholeCameraIntrinsic()
inter.set_intrinsics(w, h, fx, fy, cx, cy) # (width, height, fx, fy, cx, cy)
# inter.set_intrinsics(1280, 1024, 1072.6687, 1073.4343, 577.49968, 524.40978)
pcd = open3d.geometry.PointCloud().create_from_rgbd_image(rgbd_image, inter)

# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
open3d.visualization.draw_geometries([pcd])