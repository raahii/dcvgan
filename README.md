DCVGAN: Depth Conditional Video Generation
--

This repository contains official pytorch implementation and further details of DCVGAN.



### About





### Requirements

- Python3
- PyTorch
- FFmpeg
- OpenCV



#### 1. Install dependencies

```
pip install -r requirements.txt
```



#### 2. Prepare the dataset

- Facial Expression: [MUG Facial Exprssion Database](https://mug.ee.auth.gr/fed/)
- Hand Gesture: [Chalearn LAP IsoGD Database](http://www.cbsr.ia.ac.cn/users/jwan/database/isogd.html)

Please Follow the instructions of each official page and download the dataset. Preprocessing codes for MUG Facial Expression Database is not available now. I recomment place the dataset under `data/raw/`



#### 3. Training

```
python src/train.py --config <config.yml>
tensorboard --logdir <result dir>
```

For the first time, preprocessing the dataset starts automatically. Preprocessing is to format all datasets into a common format for `VideoDataset`. 

Please refer and edit `config.yml` under `configs/` to chage training configurations such as training epochs, batchsize, result directory and so on.

#### 4. Sampling

```
python src/generate_samples.py <result dir> <iteration> <save dir>
```



#### 5. Evaluation

I have published a framework for efficient evaluation of video generation. The framework supports `Inception Score`, `FID` and `PRD` now. Please refer the repository: [video-gans-evaluation](https://github.com/raahii/video-gans-evaluation)



----



### TODOS

- [ ] upload pretrained models
- [ ] dockernize

