# Neural Rejuvenation @ CVPR19

[Neural Rejuvenation: Improving Deep Network Training by
Enhancing Computational Resource Utilization](https://arxiv.org/pdf/1812.00481.pdf)  
Siyuan Qiao, Zhe Lin, Jianming Zhang, Alan Yuille  
In Conference on Computer Vision and Pattern Recognition, 2019 **Oral Presentation**

```
@inproceedings{NR,
   title = {Neural Rejuvenation: Improving Deep Network Training by
   Enhancing Computational Resource Utilization},
   author = {Qiao, Siyuan and Lin, Zhe and Zhang, Jianming and Yuille, Alan},
   booktitle = {CVPR},
   year = {2019}
}
```

Neural Rejuvenation is a training method for deep neural networks that focuses on improving the computation resource utilization.
Deep neural networks are usually over-parameterized for their tasks
in order to achieve good performances, thus are likely to
have underutilized computational resources.
As models with higher computational costs (e.g. more parameters or more computations)
usually have better performances, we study the problem of
improving the resource utilization of neural networks so that
their potentials can be further realized.
To this end, we propose a novel optimization method named Neural Rejuvenation. As its name suggests, our method detects dead neurons
and computes resource utilization in real time, rejuvenates
dead neurons by resource reallocation and reinitialization,
and trains them with new training schemes.

## Training
The code was implemented and tested with PyTorch 0.4.1.post2.
If you are using other versions, please be aware that there might be some incompatibility issues.
The code is based on [pytorch-classification](https://github.com/bearpaw/pytorch-classification) by [Wei Yang](https://github.com/bearpaw/).

### CIFAR
#### VGG19 (BN)
```bash
python cifar.py -d cifar10 -a vgg19_bn --epochs 300 --schedule 150 225 --gamma 0.1 --checkpoint checkpoints/cifar10/vgg19_bn --nr-target 0.25 --nr-sparsity 1.5e-4
```

#### ResNet-164
```bash
python cifar.py -d cifar10 -a resnet --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-164 --nr-target 0.25 --nr-sparsity 1.5e-4
```

#### DenseNet-BC
```bash
python cifar.py -d cifar10 -a densenet --depth 100 --growthRate 40 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/cifar10/densenet-bc-100-12 --nr-target 0.25 --nr-sparsity 1.0e-4
```

### ImageNet
#### VGG16 (BN)
```bash
python imagenet.py -a vgg_nr --data ~/dataset/ILSVRC2012/ --epochs 100 --schedule 30 60 90 --gamma 0.1 -c checkpoints/imagenet/vgg16 --nr-target 0.5 --train-batch 128 --test-batch 100 --nr-compress-only --gpu-id 0,1,2,3 --image-size 224 -j 20
python imagenet.py -a vgg_nr --data ~/dataset/ILSVRC2012/ --epochs 100 --schedule 30 60 90 --gamma 0.1 -c checkpoints/imagenet/vgg16 --nr-bn-target 0.5 --train-batch 128 --test-batch 100 --resume checkpoints/imagenet/vgg16/NR_vgg16_nr_0.5.pth.tar --gpu-id 0,1,2,3 --image-size 224 -j 20
```
Note that without --nr-compress-only, the program will automatically go to the second line.
Writing it as two steps makes debug easier.

## Experimental Results
### CIFAR
| Model                     | # Params          |  Dataset  | nr_sparsity | Err |
| -------------------       | ----------------- | --------- | ------- | ------- |
| VGG-19                    | 9.99M             | CIFAR-10  | 1.5e-4  | 4.19    |
| VGG-19                    | 10.04M            | CIFAR-100 | 3e-4    | 21.53   |
| ResNet-164                | 0.88M             | CIFAR-10  | 1.50e-4 | 5.13    |
| ResNet-164                | 0.92M             | CIFAR-100 | 2.50e-4 | 23.84   |
| DenseNet-100-40           | 4.12M             | CIFAR-10  | 1.00e-4 | 3.40    |
| DenseNet-100-40           | 4.31M             | CIFAR-100 | 2.00e-4 | 18.59   |

### ImageNet
| Model               | # Params      | FLOPs | Top-1 | Top-5 |
| ------------------- | ------------- | ----- | ----- | ----- |
| DenseNet-121        | 8.22M | 3.13G | 24.50 | 7.49  |
| VGG-16              | 36.4M | 23.5G | 23.11 | 6.69  |
| ResNet-18           | 11.9M | 2.16G | 28.86 | 9.93  |
| ResNet-34           | 21.9M | 3.77G | 25.77 | 8.10  |
| ResNet-50           | 26.4M | 3.90G | 22.93 | 6.47  |
| ResNet-101          | 46.6M | 6.96G | 21.22 | 5.76  |
