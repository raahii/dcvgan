DCVGAN: Depth Conditional Video Generation
--

This repository contains official pytorch implementation of DCVGAN.



## About





## Result

#### facial expression dataset

![result-mug](https://user-images.githubusercontent.com/13511520/54088503-f58d8900-43a1-11e9-8b27-1eca5a7d8e98.gif)



#### hand gesture dataset

![result-isogd](https://user-images.githubusercontent.com/13511520/54088434-75672380-43a1-11e9-9f7e-c6ff1bc0b77b.gif)



## Requirements

- Python3
- PyTorch
- FFmpeg
- OpenCV



### 1. Install dependencies

```
pip install -r requirements.txt
```



### 2. Prepare the dataset

- facial expression: [MUG Facial Exprssion Database](https://mug.ee.auth.gr/fed/)
- hand gesture: [Chalearn LAP IsoGD Database](http://www.cbsr.ia.ac.cn/users/jwan/database/isogd.html)

Please follow the instructions of each official page to download the dataset. Preprocessing codes for facial expression dataset is not available now. I recommend to place the dataset under `data/raw/`



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
- [ ] dockernize

