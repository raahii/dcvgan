Depth Conditional Video Generation
--

A pytorch implemention of depth condtional video generation.



### Getting Started

```
pip install -r requirements.txt
```

### Training

```
python src/train.py --config <config.yml>
tensorboard --logdir <result dir>
```

### Sampling

```
python src/generate_samples.py --result <result dir>
```

