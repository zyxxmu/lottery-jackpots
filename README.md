# Lottery Jackpots Exist in Pre-trained Models ([Paper Link]())

## Requirements

- Python >= 3.7.4
- Pytorch >= 1.6.1
- Torchvision >= 0.4.1

## Reproduce the Experiment Results 

1. Download the pre-trained models from this [link](https://drive.google.com/drive/folders/13et0J5S2iJK9oS-twrKXVqC-tk0AO9Gn?usp=sharing) and place them in the `pre-train` folder.

2. Select a configuration file in `configs` to reproduce the experiment results reported in the paper. For example, to find a lottery jackpot with 30 epochs for pruning 95% parameters of ResNet-32 on CIFAR-10, run:

   `python cifar.py --config configs/resnet32_cifar10/90sparsity30epoch.yaml --gpus 0`

   To find a lottery jackpot with 30 epochs for pruning 90% parameters of ResNet-50 on ImageNet, run:

   `python imagenet.py --config configs/resnet50_imagenet/90sparsity30epoch.yaml --gpus 0`

   Note that the `data_path` in the yaml file should be changed to the data 

## Evaluate Our Pruned Modelse

We provide configuration, training logs, and pruned models reported in the paper, which can be downloaded from the links in the following table:

| Model     | Dataset   | Sparsity | Epoch | Top-1 Acc. | Link                                                         |
| --------- | --------- | -------- | ----- | ---------- | ------------------------------------------------------------ |
| VGGNet-19 | CIFAR-10  | 90%      | 30    | 93.88%     | [link](https://drive.google.com/drive/folders/1TnQVdbLThBdPRV4RS8kngPy3WCz1octc?usp=sharing) |
| VGGNet-19 | CIFAR-10  | 90%      | 160   | 93.94%     | [link](https://drive.google.com/drive/folders/1M0EOWKBCksAvhzdHZlP7Wf5c5GIat8G5?usp=sharing) |
| VGGNet-19 | CIFAR-10  | 95%      | 30    | 93.49%     | [link](https://drive.google.com/drive/folders/1Xf971LlD60Nryq_YwEjnfCgopdBqMtp-?usp=sharing) |
| VGGNet-19 | CIFAR-10  | 95%      | 160   | 93.74%     | [link](https://drive.google.com/drive/folders/1T9FVhdfcioDLG-KMp51A-nUVKuSHZfR0?usp=sharing) |
| VGGNet-19 | CIFAR-100 | 90%      | 30    | 72.59%     | [link](https://drive.google.com/drive/folders/1fwm5ReS0GwtvNL7RzoagdgYF5irJs4ta?usp=sharing) |
| VGGNet-19 | CIFAR-100 | 90%      | 160   | 74.61%     | [link](https://drive.google.com/drive/folders/1LriCm8jDClY4b2F_GrlgzcAEdC8QhlJP?usp=sharing) |
| VGGNet-19 | CIFAR-100 | 95%      | 30    | 71.76%     | [link](https://drive.google.com/drive/folders/1OgXsqi-M_tFfnbuVzxBqsJ5MxwQ7Hwmr?usp=sharing) |
| VGGNet-19 | CIFAR-100 | 95%      | 160   | 73.35%     | [link](https://drive.google.com/drive/folders/1YYfqN22Oy2GRHCrzmwTPUqmRdzg9z5hu?usp=sharing) |
| ResNet-32 | CIFAR-10  | 90%      | 30    | 93.70%     | [link](https://drive.google.com/drive/folders/1rqfhbVI1B1St7BIUSTqtHdpE_ZHlKHG9?usp=sharing) |
| ResNet-32 | CIFAR-10  | 90%      | 160   | 94.39%     | [link](https://drive.google.com/drive/folders/1Co1R5XqPrhqBbEpHqmckiInQdv68_k8X?usp=sharing) |
| ResNet-32 | CIFAR-10  | 95%      | 30    | 92.90%     | [link](https://drive.google.com/drive/folders/1tSXqi9LjJA-8ksbjZj9m1uqtr9BQpW4I?usp=sharing) |
| ResNet-32 | CIFAR-10  | 95%      | 160   | 93.41%     | [link](https://drive.google.com/drive/folders/1_mzK3vPHQQLc93Q_Aya5CQt5NTGAcRDk?usp=sharing) |
| ResNet-32 | CIFAR-100 | 90%      | 30    | 72.22%     | [link](https://drive.google.com/drive/folders/1BB6TdK-5nwkxgn_F3jwfrxIQjMhKZ9aw?usp=sharing) |
| ResNet-32 | CIFAR-100 | 90%      | 160   | 73.43%     | [link](https://drive.google.com/drive/folders/1vEya5kzEUuN9vJKkiNHpvF8paojzKrBQ?usp=sharing) |
| ResNet-32 | CIFAR-100 | 95%      | 30    | 69.38%     | [link](https://drive.google.com/drive/folders/1b3rrPbMbXMQ-BHN6wlQopzhPmo3EDUau?usp=sharing) |
| ResNet-32 | CIFAR-100 | 95%      | 160   | 70.31%     | [link](https://drive.google.com/drive/folders/1957ZFcigsULPg3Rlq9qI6mqR4Z0pUhQI?usp=sharing) |
| ResNet-50 | ImageNet  | 80%      | 30    | 74.53%     | [link](https://drive.google.com/drive/folders/1RsdlMSTA0S93A1bIRpiR5JThTtgBo6F0?usp=sharing) |
| ResNet-50 | ImageNet  | 80%      | 60    | 75.26%     | [link](https://drive.google.com/drive/folders/1aUojX8nWgg1gv7LgEl7o-gN4f876TWKW?usp=sharing) |
| ResNet-50 | ImageNet  | 90%      | 30    | 72.17%     | [link](https://drive.google.com/drive/folders/164S_FlcLa2IQWlXXLbOCRsTGYUu66Xhu?usp=sharing) |
| ResNet-50 | ImageNet  | 90%      | 60    | 72.46%     | [link](https://drive.google.com/drive/folders/1FtR7VuGeWB_h-WhsmWupsJsFnhSt3kyO?usp=sharing) |

To test the pruned models, Download the pruned models and place them in the `ckpt` folder.

1. Select a configuration file in `configs` to test the pruned models. For example, to evaluate a lottery jackpot for pruning ResNet-32 on CIFAR-10, run:

   `python evaluate.py --config configs/resnet32_cifar10/evaluate.yaml --gpus 0`

    To evaluate a lottery jackpot for pruning ResNet-50 on ImageNet, run:

   `python evaluate.py --config configs/resnet50_imagenet/evaluate.yaml --gpus 0`

