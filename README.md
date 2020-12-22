# IIRNet
Direct design of IIR filters using neural networks.

## Usage

```
git clone https://github.com/csteinmetz1/IIRNet.git
pip install -r requirements.txt
python train.py --lr 1e-4 --max_order 2
tensorboard --logdir=lightning_logs/    # see logging
```

At the moment you need to use the nightly Pytorch build (1.8.0).
```
pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
```

## Notes 

Right now the largest bottleneck in training comes from the for loop in the loss functions (see [`irrnet/loss.py`](irrnet/loss.py)).
This is required since `sosfreqz()` and `freqz()` both only operate on a single SOS. 
In order to address this, we first need a version of `polyval()` that can operate on batches of polynomials. 

There are a few other things we could consider. First would be some kind of normalization applied to the input features, 
which are currently the real and imaginary parts of the target spectrum. We would like to normalize between -1 and 1. 
Interestingly, in this [paper](https://arxiv.org/abs/1911.03053), they just apply `tanh()` to the features at the input. 
I am not sure that will work well if the values in the response are much larger than 1 or smaller than -1.

The second has to do with how we input the desired response and how we evaluate. 
Currently, `freqz()` simply measures the response of the filter at linearly spaced frequencies, 
which is how scipy handles this, but, from an error perspective, it would make more sense to have logarithmically spaced frequencies. 
This should be straightforward to implement as another option. 