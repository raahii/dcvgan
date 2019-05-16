DCVGAN: Depth Conditional Video Generation
--

This repository contains official pytorch implementation of DCVGAN.

> Yuki Nakahira and Kazuhiko Kawamoto, DCVGAN: Depth Conditional Video Generation, 2019 IEEE International Conference on Image Processing, ICIP 2019.



## About

This paper proposes a new GAN architecture for video generation with depth videos and color videos. 

The proposed model explicitly uses the information of depth in a video sequence as additional information for a GAN-based video generation scheme to make the model understand scene dynamics more accurately.

The model uses pairs of color video and depth video for training, and generates a video using the two steps. 

1. Generate the depth video to model the scene dynamics based on the geometrical information. 
2. To add appropriate color to the geometrical information of the scene, the domain translation from depth to color is performed for each image. 

The architecture is below. This model has three networks in the generator: frame seed generator ($R_M$), depth image generator ($G_D$), color image generator ($G_C$). 

<p align="center">
<img src="https://user-images.githubusercontent.com/13511520/57762583-baf5f300-773a-11e9-942d-858c2d834536.png" width="50%">
</p>

In addition, the model has two discriminators: image discriminator ($D_I$) and video discriminator ($D_V$). The detailed network architecture is shown below.

## Result

#### facial expression dataset

<p align="center">
<img src="https://user-images.githubusercontent.com/13511520/54088503-f58d8900-43a1-11e9-8b27-1eca5a7d8e98.gif" width="640px">
</p>


#### hand gesture dataset

<p align="center">
<img src="https://user-images.githubusercontent.com/13511520/54088434-75672380-43a1-11e9-9f7e-c6ff1bc0b77b.gif" width="640px">
</p>


## Network Architecture

### Generators

<p align="center">
<img src="https://user-images.githubusercontent.com/13511520/57746277-743cd480-770b-11e9-8066-c3b6b64426aa.png" width="60%">
</p>

### Discriminators

<p align="center">
<img src="https://user-images.githubusercontent.com/13511520/57746276-73a43e00-770b-11e9-90ec-9dcc58ffc3b6.png" width="60%">
</p>

## Usage

### 1. Clone the repository

```shell
git clone https://github.com/raahii/dcvgan.git
cd dcvgan
```



### 2. Install dependencies

#### Requirements

- Python3.7
- PyTorch
- FFmpeg
- OpenCV
- GraphViz

#### Using docker
  
  Easy. Thanks :whale:

  ```shell
  docker build -t dcvgan
  docker run --runtime=nvidia -v $(pwd):/home/user/dcvgan -it dcvgan
  ```
  
#### Manual installation

  I recommend to use pyenv and conda to install dependencies. For instance, my environment is like following. 

  ```shell
  pyenv install miniconda3-4.3.30
  pyenv local miniconda3-4.3.30
  conda install -y ffmpeg
  pip install -r requirements.txt
  ```
  
  For detail, please refer my [Dockerfile](https://github.com/raahii/dcvgan/blob/master/Dockerfile).


### 2. Prepare the dataset

- facial expression: [MUG Facial Exprssion Database](https://mug.ee.auth.gr/fed/)
- hand gesture: [Chalearn LAP IsoGD Database](http://www.cbsr.ia.ac.cn/users/jwan/database/isogd.html)

Please follow the instructions of each official page to obtain the dataset. Preprocessing codes for facial expression dataset is not available now. I recommend to place the dataset under `data/raw/`



### 3. Training

```
python src/train.py --config <config.yml>
tensorboard --logdir <result dir>
```

For the first time, preprocessing for the dataset starts automatically. Preprocessing is to format all datasets into a common format for [VideoDataset](https://github.com/raahii/dcvgan/blob/master/src/dataset.py#L18). 

Please refer and edit `configs/XXX.yml` to chage training configurations such as training epochs, batchsize, result directory.

### 4. Sampling

```
python src/generate_samples.py <result dir> <iteration> <save dir>
```



### 5. Evaluation

I have published a framework for efficient evaluation of video generation, [video-gans-evaluation](https://github.com/raahii/video-gans-evaluation). The framework supports `Inception Score`, `FID` and `PRD` now. **Please star the repository if you like it!** :relaxed:



----



## TODOS

- [ ] upload pretrained models
- [x] dockernize

