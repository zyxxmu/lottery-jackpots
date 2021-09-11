# Efficient Weight Pruning using Pre-trained Lottery Jackpots ([Paper Link](https://arxiv.org/abs/2104.08700)) ![](https://visitor-badge.glitch.me/badge?page_id=zyxxmu.lottery-jackpots)

## Requirements

- Python >= 3.7.4
- Pytorch >= 1.6.1
- Torchvision >= 0.4.1

## Reproduce the Experiment Results 

1. Download the pre-trained models from this [link](https://drive.google.com/drive/folders/1tUdUzHguEShhmSKGPz4JVFCO4GM5oRqH?usp=sharing) and place them in the `pre-train` folder.

2. Select a configuration file in `configs` to reproduce the experiment results reported in the paper. For example, to find a lottery jackpot with 30 epochs for pruning 95% parameters of ResNet-32 on CIFAR-10, run:

   `python cifar.py --config configs/resnet32_cifar10/90sparsity30epoch.yaml --gpus 0`

   To find a lottery jackpot with 30 epochs for pruning 90% parameters of ResNet-50 on ImageNet, run:

   `python imagenet.py --config configs/resnet50_imagenet/90sparsity30epoch.yaml --gpus 0`

   To further tune the weights of a searched lottery jackpot with 10 epochs for pruning 90% parameters of ResNet-50 on ImageNet, run:

   `python imagenet-t.py --config configs/resnet50_imagenet/90sparsity30s10t.yaml --gpus 0`

   Note that the `data_path` in the yaml file should be changed to the data. 

## Evaluate Our Pruned Modelse

We provide training logs and pruned models reported in the paper, which can be downloaded from the links in the following table:

| Model     | Dataset  | Sparsity | Epoch       | Top-1 Acc. | Link                                                         |
| --------- | -------- | -------- | ----------- | ---------- | ------------------------------------------------------------ |
| VGGNet-19 | CIFAR-10 | 90%      | 30(S)       | 93.88%     | [link](https://drive.google.com/drive/folders/1QkEwAapP3WIg8TKefhVm1vUXJJOYgRy4?usp=sharing) |
| VGGNet-19 | CIFAR-10 | 90%      | 160(S)      | 93.94%     | [link](https://drive.google.com/drive/folders/1jxmmkdNf6sk61kK0SEMW8lFvYh6gM92t?usp=sharing) |
| VGGNet-19 | CIFAR-10 | 95%      | 30(S)       | 93.49%     | [link](https://drive.google.com/drive/folders/137xF5W-dbwNeFVejUK_G0Vdq4_YVPon6?usp=sharing) |
| VGGNet-19 | CIFAR-10 | 95%      | 160(S)      | 93.74%     | [link](https://drive.google.com/drive/folders/1S6emUouAi2K6ddtmlZGIqPyrQnOf3fqY?usp=sharing) |
| ResNet-32 | CIFAR-10 | 90%      | 30(S)       | 93.70%     | [link](https://drive.google.com/drive/folders/1reizJkCgiliul3-JWeZ7lVD14Kj9N40t?usp=sharing) |
| ResNet-32 | CIFAR-10 | 90%      | 160(S)     | 94.39%     | [link](https://drive.google.com/drive/folders/1fTTs3aeiyI9fUIpfnO81xssJ-wHo0pwe?usp=sharing) |
| ResNet-32 | CIFAR-10 | 95%      | 30(S)       | 92.90%     | [link](https://drive.google.com/drive/folders/1JpABQkOAjvLkgvzbxtbm_wUIUkknawz3?usp=sharing) |
| ResNet-32 | CIFAR-10 | 95%      | 160(S)      | 93.41%     | [link](https://drive.google.com/drive/folders/1K-FLAtK44RUJlGsW1zBAuntubeyRGa03?usp=sharing) |
| ResNet-50 | ImageNet | 80%      | 30(S)       | 75.19%     | [link](https://drive.google.com/drive/folders/1s0Ar5VamRndGj06hrMzPLz5WHaqehqTf?usp=sharing) |
| ResNet-50 | ImageNet | 80%      | 30(S)+10(T) | 76.66%     | [link](https://drive.google.com/drive/folders/1egmfGG4zPmuAsTmZTeTJ_ZhGkZEt7d_n?usp=sharing) |
| ResNet-50 | ImageNet | 90%      | 30(S)       | 72.43%     | [link](https://drive.google.com/drive/folders/1Ws5qjQrSeEYeOf2UhIdEmgQO04O5LwUa?usp=sharing) |
| ResNet-50 | ImageNet | 90%      | 30(S)+10(T) | 74.62%     | [link](https://drive.google.com/drive/folders/1wTzyDKr6_PW3ty2wsiZXT-iS5EFsEpzb?usp=sharing) |

To test the pruned models, Download the pruned models and place them in the `ckpt` folder.

1. Select a configuration file in `configs` to test the pruned models. For example, to evaluate a lottery jackpot for pruning ResNet-32 on CIFAR-10, run:

   `python evaluate.py --config configs/resnet32_cifar10/evaluate.yaml --gpus 0`

    To evaluate a lottery jackpot for pruning ResNet-50 on ImageNet, run:

   `python evaluate.py --config configs/resnet50_imagenet/evaluate.yaml --gpus 0`


