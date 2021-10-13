# DVC_P
These is the open soure implementation of the paper "DVC-P: Deep Video Compression with Perceptual Optimizations" [[paper](https://arxiv.org/abs/2109.10849)], which is accepted by [VCIP 2021](http://www.vcip2021.org/).

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

## Basis
Since our work DVC-P is totally based on [OpenDVC](https://github.com/RenYang-home/OpenDVC), [OpenDVC](https://github.com/RenYang-home/OpenDVC) is considered as our base software. To ensure that you can successfully run our codes, we strongly suggest that you firstly try to learn how to run [OpenDVC](https://github.com/RenYang-home/OpenDVC) according to their detailed instructions. For better illustration, detailed instructions are also shown below. Note that most of them are referred to those in [OpenDVC](https://github.com/RenYang-home/OpenDVC).

## Dependency

(please also refer to the *dependecy* in [OpenDVC](https://github.com/RenYang-home/OpenDVC))

- Tensorflow 1.12

- Tensorflow-compression 1.0 ([Download link](https://github.com/tensorflow/compression/releases/tag/v1.0))

(plesae put the folder "tensorflow_compression" to the same directory as the codes after downloading.)

- BPG ([Download link](https://bellard.org/bpg/))

(Note that BPG encoder is used to compress I frames, and our DVC-P is only used to generatively compress P frames.)

- VGG19.npy ([Download link](https://github.com/TachibanaYoshino/AnimeGAN/releases/download/vgg16/19.npy/vgg19.npy))

- Pre-trained models of optical flow. Download the folder "motion_flow" ([Download link](https://drive.google.com/drive/folders/1b6W3AMpHnPddZrGte2zeQJMxZDSha_Ue?usp=sharing)).

Here we give an example of the folder structure.

<img src="https://github.com/SaipingZhang/DVC_P/blob/main/FolderStructure.png" width="500" height="600">

## Input Preperation

(please also refer to the *preperation* in [OpenDVC](https://github.com/RenYang-home/OpenDVC))

Input frames need to be in RGB format. To compress a video in YUV format, please first convert the YUV to sequential PNG images with the following command.

```
ffmpeg -pix_fmt yuv420p -s WidthxHeight -i Name.yuv -vframes Frame path_to_PNG/f%03d.png
```

Since our network requires input frames with the height and width as the multiples of 16 (followed by [OpenDVC](https://github.com/RenYang-home/OpenDVC)), please make sure you have cropped input frames to meet the requirements. The following command can be used to crop images.

```
ffmpeg -pix_fmt yuv420p -s 1920x1080 -i Name.yuv -vframes Frame -filter:v "crop=1920:1072:0:0" path_to_PNG/f%03d.png
```

A prepared sequence *BasketballPass* (containing the first 100 frames in RGB format) is uploaded in [OpenDVC](https://github.com/RenYang-home/OpenDVC) as an example. Please check it if you have any questions about preperaing inputs.

## Training your own models

(please also refer to the *training your own models* in [OpenDVC](https://github.com/RenYang-home/OpenDVC))

### Preperation

- Download the training data. We train the models on the [Vimeo90k dataset](https://github.com/anchen1011/toflow) ([Download link](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip)) (82G) (followed by [OpenDVC](https://github.com/RenYang-home/OpenDVC)). After downloading, please run the following codes to generate "folder.npy" which contains the directories of all training samples.
```
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(root)
    return result

folder = find('im1.png', 'path_to_vimeo90k/vimeo_septuplet/sequences/')
np.save('folder.npy', folder)
```

- Compress I-frames. Followed by [OpenDVC](https://github.com/RenYang-home/OpenDVC), we compress I-frames (im1.png) by BPG 444 at QP = 22, 27, 32 and 37 for the models of lambda = 2048, 1024, 512 and 256, respectively. The Vimeo90k dataset has ~90k 7-frame clips, we need to compress "im1.png" in each clip as I-frame. For example:

```
bpgenc -f 444 -m 9 im1.png -o im1_QP27.bpg -q 27
bpgdec im1_QP27.bpg -o im1_bpg444_QP27.png        
```

### Training strategies

Similarly to the [OpenDVC](https://github.com/RenYang-home/OpenDVC) in which the framework design consists of various deep models, our proposed DVC-P requires carefully designed joined training strategy. In particular, the training process consists of *700k* iterations in total. When *iterations<20k*, only optical flow network, MV encoder network and MV generator network are trained together. When *iterations* reaches to *20k*, motion compensation network begins to join the training. When *iterations* reaches to *40k*, residual encoder network and residual generator network also begin their joint training. When *iterations* reaches to *400k*, the discriminator begins to be optimized. As for loss function, we only use MSE loss when *iteration<20k*, VGG-based loss is added when *iterations* reaches to *40k*. Adversarial loss is added when *iterations* reaches to *400k*.

### Training models

Run Train.py to train your models, e.g.,

```
python Train.py --l 1024
```

## Testing your own models

(please also refer to the *encoder for video* in [OpenDVC](https://github.com/RenYang-home/OpenDVC))

```
--path, the path to PNG files;

--frame, the total frame number to compress;

--GOP, the GOP size, e.g., 10;

--mode, PSNR;

--metric, PSNR;

--l, lambda value. The pre-trained PSNR models are trained by 4 lambda values, i.e., 256, 512, 1024 and 2048, with increasing bit-rate/PSNR;

--N, filter number in CNN (Do not change);

--M, channel number of latent representations (Do not change).
```

For example:
```
python Test.py --path BasketballPass --mode PSNR  --metric PSNR --l 1024
```

The OpenDVC encoder generates the encoded bit-stream and compressed frames in two folders.

```
path = args.path + '/' # path to PNG
path_com = args.path + '_com_' + args.mode  + '_' + str(args.l) + '/' # path to compressed frames
path_bin = args.path + '_bin_' + args.mode  + '_' + str(args.l) + '/' # path to encoded bit-streams
```

## Performance

Using our open source codes, compressed frames are in higher perceptual quality and without checkerboard artifacts compared with those compressed by [OpenDVC](https://github.com/RenYang-home/OpenDVC).

![](PerformanceComparison.png)
