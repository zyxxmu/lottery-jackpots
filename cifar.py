import torch
import torch.nn as nn
import torch.optim as optim
from utils.options import args
import utils.common as utils
import os
import time
import copy
import sys
import random
import numpy as np
import heapq
from data import cifar10, cifar100
from utils.common import *
from importlib import import_module

from utils.conv_type import *

import models
import pdb

visible_gpus_str = ','.join(str(i) for i in args.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpus_str
args.gpus = [i for i in range(len(args.gpus))]
checkpoint = utils.checkpoint(args)
now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
logger = utils.get_logger(os.path.join(args.job_dir, 'logger'+now+'.log'))
device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'

if args.label_smoothing is None:
    loss_func = nn.CrossEntropyLoss().cuda()
else:
    loss_func = LabelSmoothing(smoothing=args.label_smoothing)

# Data
print('==> Loading Data..')
if args.data_set == 'cifar10':
    loader = cifar10.Data(args)
elif args.data_set == 'cifar100':
    loader = cifar100.Data(args)  
              
def train(model, optimizer, trainLoader, args, epoch):

    model.train()
    losses = utils.AverageMeter(':.4e')
    accurary = utils.AverageMeter(':6.3f')
    print_freq = len(trainLoader.dataset) // args.train_batch_size // 10
    start_time = time.time()
    for batch, (inputs, targets) in enumerate(trainLoader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        #adjust_learning_rate(optimizer, epoch, batch, print_freq, args)
        loss = loss_func(output, targets)
        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        optimizer.step()

        prec1 = utils.accuracy(output, targets)
        accurary.update(prec1[0], inputs.size(0))

        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            logger.info(
                'Epoch[{}] ({}/{}):\t'
                'Loss {:.4f}\t'
                'Accurary {:.2f}%\t\t'
                'Time {:.2f}s'.format(
                    epoch, batch * args.train_batch_size, len(trainLoader.dataset),
                    float(losses.avg), float(accurary.avg), cost_time
                )
            )
            start_time = current_time

def validate(model, testLoader):
    global best_acc
    model.eval()

    losses = utils.AverageMeter(':.4e')
    accurary = utils.AverageMeter(':6.3f')

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            predicted = utils.accuracy(outputs, targets)
            accurary.update(predicted[0], inputs.size(0))

        current_time = time.time()
        logger.info(
            'Test Loss {:.4f}\tAccurary {:.2f}%\t\tTime {:.2f}s\n'
            .format(float(losses.avg), float(accurary.avg), (current_time - start_time))
        )
    return accurary.avg

def generate_pr_cfg(model):
    cfg_len = {
        'vgg': 17,
        'resnet32': 32,
    }

    pr_cfg = []
    if args.layerwise == 'l1':
        weights = []
        for name, module in model.named_modules():
            if hasattr(module, "set_prune_rate") and name != 'fc' and name != 'classifier':
                conv_weight = module.weight.data.detach().cpu()   
                weights.append(conv_weight.view(-1)) 
        all_weights = torch.cat(weights,0)
        preserve_num = int(all_weights.size(0) * (1 - args.prune_rate))
        preserve_weight, _ = torch.topk(torch.abs(all_weights), preserve_num)
        threshold = preserve_weight[preserve_num-1]

        #Based on the pruning threshold, the prune cfg of each layer is obtained
        for weight in weights:
            pr_cfg.append(torch.sum(torch.lt(torch.abs(weight),threshold)).item()/weight.size(0))
        pr_cfg.append(0)
    elif args.layerwise == 'uniform':
        pr_cfg = [args.prune_rate] * cfg_len[args.arch]
        pr_cfg[-1] = 0

    get_prune_rate(model, pr_cfg)

    return pr_cfg

def get_prune_rate(model, pr_cfg):
    all_params = 0
    prune_params = 0

    i = 0
    for name, module in model.named_modules():
        if hasattr(module, "set_prune_rate"):
            w = module.weight.data.detach().cpu()
            params = w.size(0) * w.size(1) * w.size(2) * w.size(3)
            all_params = all_params + params
            prune_params += int(params * pr_cfg[i])
            i += 1

    logger.info('Params Compress Rate: %.2f M/%.2f M(%.2f%%)' % ((all_params-prune_params)/1000000, all_params/1000000, 100. * prune_params / all_params))

def main():
    start_epoch = 0
    best_acc = 0.0

    model, pr_cfg = get_model(args,logger)

    optimizer = get_optimizer(args, model)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)

    if args.resume == True:
        start_epoch, best_acc = resume(args, model, optimizer)
    
    if len(args.gpus) != 1:
        model = nn.DataParallel(model, device_ids=args.gpus)

    for epoch in range(start_epoch, args.num_epochs):
        train(model, optimizer, loader.trainLoader, args, epoch)
        test_acc = validate(model, loader.testLoader)
        scheduler.step()

        is_best = best_acc < test_acc
        best_acc = max(best_acc, test_acc)

        model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()

        state = {
            'state_dict': model_state_dict,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1,
            'cfg': pr_cfg,
        }

        checkpoint.save_model(state, epoch + 1, is_best)

    logger.info('Best accurary: {:.3f}'.format(float(best_acc)))

def resume(args, model, optimizer):
    if os.path.exists(args.job_dir+'/checkpoint/model_last.pt'):
        print(f"=> Loading checkpoint ")

        checkpoint = torch.load(args.job_dir+'/checkpoint/model_last.pt')

        start_epoch = checkpoint["epoch"]

        best_acc = checkpoint["best_acc"]

        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"=> Loaded checkpoint (epoch) {checkpoint['epoch']})")

        return start_epoch, best_acc
    else:
        print(f"=> No checkpoint found at '{args.job_dir}' '/checkpoint/")


def get_model(args,logger):
    pr_cfg = []

    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]().to(device)
    ckpt = torch.load(args.pretrained_model, map_location=device)
    model.load_state_dict(ckpt['state_dict'],strict=False)
    
    #applying sparsity to the network
    pr_cfg = generate_pr_cfg(model)
    set_model_prune_rate(model, pr_cfg, logger)
    
    if args.freeze_weights:
        freeze_model_weights(model)

    model = model.to(device)

    return model, pr_cfg

def get_optimizer(args, model):
    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and ("sparseThreshold" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
                },
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )

    return optimizer

if __name__ == '__main__':
    main()