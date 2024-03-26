# Real-Time and High-Accuracy Switchable Stereo Depth Estimation Method Utilizing Self-Supervised Online Learning Mechanism for MIS

## :movie_camera: Demos

You can demo the trained model on a sequence of stereo images. To predict depth for your dataset, run
```Shell
python test_inference.py --pretrained-model sceneflow_pretrained.tar --calib-path path_to_calib_yaml --dataset-dir path_to_test_02 --output-dir output --output-depth
```
To save the depth values as `.npy` files and the depth maps as `.png` images, run with the `--output-depth` flag. 

## :candy: Visualization
Here we show some results of the proposed **MIOL** framework on [Hamlyn](https://arxiv.org/abs/1705.08260) and [SCARED](https://arxiv.org/abs/2101.01133) datasets.

- Depth map predictions
![imgs](https://user-images.githubusercontent.com/131570332/233924380-e9ff65b6-380e-46e9-a259-39d2ef8eb76e.png)

- Point Clouds

https://user-images.githubusercontent.com/131570332/233920820-c1057d58-0803-44a3-b8ca-49aa056ab538.mp4

<!-- ## :rose: Acknowledgment
Our code is based on the excellent works of [SC-SfMLearner](https://github.com/JiawangBian/SC-SfMLearner-Release) and [monodepth2](https://github.com/nianticlabs/monodepth2). -->

## License
For academic usage, the code is released under the permissive MIT license. Our intension of sharing the project is for research/personal purpose. For any commercial purpose, please contact the authors.
