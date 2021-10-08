# DVC_P
These are main codes of the paper "DVC-P: Deep Video Compression with Perceptual Optimizations" [[paper](https://arxiv.org/abs/2109.10849)], which is accepted by [VCIP 2021](http://www.vcip2021.org/).

Our work is based on [OpenDVC](https://github.com/RenYang-home/OpenDVC) (an open source Tensorflow implementation of [DVC](https://arxiv.org/abs/1812.00101)), but improves it with perceptual optimizations (i.e., a discriminator network and a mixed loss are employed to help our network trade off among distortion, perception and rate and nearest-neighbor interpolation is used to eliminate checkerboard artifacts).

Please refer to [technical report](https://arxiv.org/abs/2006.15862) for more details of OpenDVC. If you find their open source codes are helpful, please cite their work
```
@article{yang2020opendvc,
  title={Open{DVC}: An Open Source Implementation of the {DVC} Video Compression Method},
  author={Yang, Ren and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint arXiv:2006.15862},
  year={2020}
}
```
Please refer to [OpenDVC](https://github.com/RenYang-home/OpenDVC) for more training details and downloading necessary dependencies.

If our paper and open source codes are helpful for your research, please cite our paper
```
@article{DVC-P,
  title={DVC-P: Deep Video Compression with Perceptual Optimizations},
  author={Saiping Zhang and Marta Mrak and Luis Herranz and Marc Gorriz Blanch and Shuai Wan and Fuzheng Yang},
  journal={arXiv preprint arXiv:2109.10849},
  year={2021}
}
```

If you have any question or find any bug, please feel free to contact:

Saiping Zhang

Email: spzhang@stu.xidian.edu.cn
