Depth Conditional Video Generation
--

A pytorch implemention of depth condtional video generation.

### TODO:

- [ ] implement logging to the file, dump config file
- [ ] implement evaluation script, evaluation training hook
- [ ] reproduce result

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

