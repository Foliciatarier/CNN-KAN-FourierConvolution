# CNN-KAN-FourierConvolution
Fourier KAN Layer to modify Convolution Layer in CNN and test on MINST

* Modified from https://github.com/eonurk/CNN-KAN
* Decrease some parameters to reduce GPU memory requirement
* Loss tend to be divergence during training
* Compared to naive convolution, acc almost no improvement

```
Train Epoch: 0 [0/6000 (0%)]    Loss: 2.282534
Train Epoch: 0 [640/6000 (11%)] Loss: 2.042386
Train Epoch: 0 [1280/6000 (21%)]        Loss: 1.815717
Train Epoch: 0 [1920/6000 (32%)]        Loss: 1.586558
Train Epoch: 0 [2560/6000 (43%)]        Loss: 1.285961
Train Epoch: 0 [3200/6000 (53%)]        Loss: 1.132277
Train Epoch: 0 [3840/6000 (64%)]        Loss: 0.957478
Train Epoch: 0 [4480/6000 (74%)]        Loss: 0.928291
Train Epoch: 0 [5120/6000 (85%)]        Loss: 1.003554
Train Epoch: 0 [5760/6000 (96%)]        Loss: 2.496895
Train Epoch: 1 [0/6000 (0%)]    Loss: 1.872281
Train Epoch: 1 [640/6000 (11%)] Loss: 0.735947
Train Epoch: 1 [1280/6000 (21%)]        Loss: 0.627827
Train Epoch: 1 [1920/6000 (32%)]        Loss: 0.660952
Train Epoch: 1 [2560/6000 (43%)]        Loss: 0.705110
Train Epoch: 1 [3200/6000 (53%)]        Loss: 0.616599
Train Epoch: 1 [3840/6000 (64%)]        Loss: 0.534389
Train Epoch: 1 [4480/6000 (74%)]        Loss: 0.611223
Train Epoch: 1 [5120/6000 (85%)]        Loss: 0.510398
Train Epoch: 1 [5760/6000 (96%)]        Loss: 0.433316
Train Epoch: 2 [0/6000 (0%)]    Loss: 0.359988
Train Epoch: 2 [640/6000 (11%)] Loss: 0.356172
Train Epoch: 2 [1280/6000 (21%)]        Loss: 0.409894
Train Epoch: 2 [1920/6000 (32%)]        Loss: 0.356990
Train Epoch: 2 [2560/6000 (43%)]        Loss: 0.322838
Train Epoch: 2 [3200/6000 (53%)]        Loss: 0.265407
Train Epoch: 2 [3840/6000 (64%)]        Loss: 0.285016
Train Epoch: 2 [4480/6000 (74%)]        Loss: 0.325980
Train Epoch: 2 [5120/6000 (85%)]        Loss: 0.260493
Train Epoch: 2 [5760/6000 (96%)]        Loss: 0.335429
Train Epoch: 3 [0/6000 (0%)]    Loss: 0.163821
Train Epoch: 3 [640/6000 (11%)] Loss: 0.136392
Train Epoch: 3 [1280/6000 (21%)]        Loss: 0.111793
Train Epoch: 3 [1920/6000 (32%)]        Loss: 0.116403
Train Epoch: 3 [2560/6000 (43%)]        Loss: 0.096282
Train Epoch: 3 [3200/6000 (53%)]        Loss: 0.120010
Train Epoch: 3 [3840/6000 (64%)]        Loss: 0.076812
Train Epoch: 3 [4480/6000 (74%)]        Loss: 0.193832
Train Epoch: 3 [5120/6000 (85%)]        Loss: 0.105416
Train Epoch: 3 [5760/6000 (96%)]        Loss: 0.058837
Train Epoch: 4 [0/6000 (0%)]    Loss: 0.076122
Train Epoch: 4 [640/6000 (11%)] Loss: 0.045793
Train Epoch: 4 [1280/6000 (21%)]        Loss: 0.049276
Train Epoch: 4 [1920/6000 (32%)]        Loss: 0.183884
Train Epoch: 4 [2560/6000 (43%)]        Loss: 0.040525
Train Epoch: 4 [3200/6000 (53%)]        Loss: 0.310321
Train Epoch: 4 [3840/6000 (64%)]        Loss: 0.099308
Train Epoch: 4 [4480/6000 (74%)]        Loss: 0.051874
Train Epoch: 4 [5120/6000 (85%)]        Loss: 0.033581
Train Epoch: 4 [5760/6000 (96%)]        Loss: 0.044794
Test Accuracy: 96.40%
```

## Acknowledgement

[pyKAN](https://github.com/KindXiaoming/pykan)

[FourierKAN](https://github.com/GistNoesis/FourierKAN/)

[CNN-KAN](https://github.com/eonurk/CNN-KAN)
