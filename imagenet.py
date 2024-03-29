import models
from importlib import import_module
import torch
import torch.nn as nn
import torch.optim as optim
from utils.options import args
import utils.common as utils
from utils.common import *

import os
import copy
import time
import math
import sys
import pdb
import numpy as np
import heapq
import random

import torch.autograd as autograd
import torch.nn.functional as F

if args.use_dali:
    from data import imagenet_dali
else:
    from data import imagenet


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

# load training data
print('==> Preparing data..')
if args.use_dali:
    def get_data_set(type='train'):
        if type == 'train':
            return imagenet_dali.get_imagenet_iter_dali('train', args.data_path, args.train_batch_size,
                                                        num_threads=4, crop=224, device_id=args.gpus[0], num_gpus=1)
        else:
            return imagenet_dali.get_imagenet_iter_dali('val', args.data_path, args.eval_batch_size,
                                                        num_threads=4, crop=224, device_id=args.gpus[0], num_gpus=1)
    train_loader = get_data_set('train')
    val_loader = get_data_set('test')
else:
    data_tmp = imagenet.Data(args)
    train_loader = data_tmp.trainLoader
    val_loader = data_tmp.testLoader

def adjust_rate(epoch):
    rate = 0.5 * (1 + math.cos(math.pi * epoch / args.num_epochs))
    #rate = 0.1
    #rate = 1 - rate
    #if epoch < 10:
    #    rate = 0.9
    #elif ech < 20:
    #    rate = 0.6
    #else:
    #    rate = 0.3
    #rate = 1
    return 1
    
def train(epoch, train_loader, model, criterion, optimizer):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    model.train()
    end = time.time()

    num_iter = len(train_loader)

    print_freq = num_iter // 10
    i = 1
    if args.arch == 'MobileNetV1':
        pop_config = np.array([0] * 28)
    else:
        pop_config = np.array([0] * 54)
    rate = adjust_rate(epoch)
    logger.info("rate {}".format(rate))
    for batch_idx, (images, targets) in enumerate(train_loader):
        i+=1
        if args.debug:
            if i > 5:
                break
            i += 1
        images = images.cuda()
        targets = targets.cuda()
        data_time.update(time.time() - end)

        #if i % 5 == 0:
        adjust_learning_rate(optimizer, epoch, batch_idx, num_iter)
        # compute output
        logits = model(images)
        loss = loss_func(logits, targets)

        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)  # accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pop_num = pop_up(model, rate)
        #import pdb; pdb.set_trace()
        pop_config += pop_num
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % print_freq == 0 and batch_idx != 0:
            logger.info(
                'Epoch[{0}]({1}/{2}): '
                'Loss {loss.avg:.4f} '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                    epoch, batch_idx, num_iter, loss=losses,
                    top1=top1, top5=top5))
    logger.info("epoch{} pop_configuration {}".format(epoch, pop_config))
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, args):

    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    num_iter = len(val_loader)

    model.eval()
    with torch.no_grad():
        end = time.time()
        i = 0
        
        for batch_idx, (images, targets) in enumerate(val_loader):
            if args.debug:
                if i > 5:
                    break
                i += 1
            images = images.cuda()
            targets = targets.cuda()

            # compute output
            logits = model(images)
            loss = criterion(logits, targets)

            # measure accuracy and record loss
            pred1, pred5 = utils.accuracy(logits, targets, topk=(1, 5))
            n = images.size(0)
            
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                    .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


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

    logger.info('Params Compress Rate: %.2f M/%.2f M(%.2f%%)' % 
                                                    ((all_params - prune_params)/1000000, all_params/1000000, 100. * prune_params / all_params))


def generate_pr_cfg(model):
    pr_cfg = []
    if args.layerwise == 'l1':
        weights = []
        for name, module in model.named_modules():
            if hasattr(module, "set_prune_rate") and name != "fc":
                conv_weight = module.weight.data.detach().cpu()
                weights.append(conv_weight.view(-1))

        all_weights = torch.cat(weights, 0)
        preserve_num = int(all_weights.size(0) * (1 - args.prune_rate))
        preserve_weight, _ = torch.topk(torch.abs(all_weights), preserve_num)
        threshold = preserve_weight[preserve_num-1]

        # Based on the pruning threshold, the prune cfg of each layer is obtained
        for weight in weights:
            pr_cfg.append(torch.sum(torch.lt(torch.abs(weight), threshold)).item()/weight.size(0))

        pr_cfg.append(0)
        #import pdb; pdb.set_trace()
    
    elif args.layerwise == 'uniform':
        pr_cfg = [args.prune_rate] * 54
        pr_cfg[-1] = 0
    #pr_cfg = [0.32408588435374147, 0.406982421875, 0.6580403645833334, 0.5928955078125, 0.50701904296875, 0.620849609375, 0.6635199652777778, 0.67535400390625, 0.54150390625, 0.5310872395833334, 0.6903076171875, 0.530120849609375, 0.6433241102430556, 0.675872802734375, 0.7753448486328125, 0.7902069091796875, 0.7621527777777778, 0.7320709228515625, 0.6675567626953125, 0.7088962131076388, 0.637054443359375, 0.6290740966796875, 0.6652289496527778, 0.688873291015625, 0.5538864135742188, 0.7781507703993056, 0.6635284423828125, 0.8420219421386719, 0.8455886840820312, 0.8425309922960069, 0.7321853637695312, 0.8086471557617188, 0.8236796061197916, 0.7345085144042969, 0.7720527648925781, 0.8275027804904513, 0.7670669555664062, 0.7563896179199219, 0.8310869004991319, 0.7610435485839844, 0.7079086303710938, 0.8244544135199653, 0.7275924682617188, 0.6203765869140625, 0.8668539259168837, 0.7195043563842773, 0.8949909210205078, 0.7720861434936523, 0.8736118740505643, 0.7472705841064453, 0.6554927825927734, 0.885939704047309, 0.7277250289916992, 0]
    return pr_cfg

def get_mask(model):
    masks = []
    for n, m in model.named_modules():
        if hasattr(m, "set_prune_rate"):
            mask = m.binary_mask()
            masks.append(mask.view(-1))
    all_masks = torch.cat(masks,0)
    return all_masks

def pop_up(model, rate):
    pop_num = []
    for n, m in model.named_modules():
        if hasattr(m, "set_prune_rate"):
            pop_num.append(m.final_pop_up(rate))
    model = model.to(device)
    if len(args.gpus) != 1:
        model = nn.DataParallel(model, device_ids=args.gpus)
    #logger.info("epoch{} iter{} pop_configuration {}".format(epoch, iter, pop_num))
    return np.array(pop_num)

def get_model(args, logger):
    pr_cfg = []
    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]().to(device)
    ckpt = torch.load(args.pretrained_model, map_location=device)
    fc_weight = ckpt['fc.weight']
    ckpt['fc.weight'] = fc_weight.view(
        fc_weight.size(0), fc_weight.size(1), 1, 1)
    model.load_state_dict(ckpt, strict=False)

    # applying sparsity to the network
    pr_cfg = generate_pr_cfg(model)
    set_model_prune_rate(model, pr_cfg, logger)

    if args.freeze_weights:
        freeze_model_weights(model)

    #model = model.to(device)
    return model, pr_cfg


def main():
    start_epoch = 0
    best_acc = 0.0
    best_acc_top1 = 0.0

    model, pr_cfg = get_model(args, logger)

    optimizer = get_optimizer(args, model)

    model = model.to(device)
    if args.resume == True:
        start_epoch, best_acc = resume(args, model, optimizer)

    if len(args.gpus) != 1:
        model = nn.DataParallel(model, device_ids=args.gpus)
    #valid_obj, test_acc_top1, test_acc = validate(
     #       val_loader, model, loss_func, args)

    for epoch in range(start_epoch, args.num_epochs):
        train_obj, train_acc_top1,  train_acc = train(
            epoch,  train_loader, model, loss_func, optimizer)
        valid_obj, test_acc_top1, test_acc = validate(
            val_loader, model, loss_func, args)
    

        is_best = best_acc < test_acc
        best_acc_top1 = max(best_acc_top1, test_acc_top1)
        best_acc = max(best_acc, test_acc)

        model_state_dict = model.module.state_dict() if len(
            args.gpus) > 1 else model.state_dict()

        state = {
            'state_dict': model_state_dict,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'cfg': pr_cfg,
        }
        checkpoint.save_model(state, epoch + 1, is_best)

    logger.info('Best accurary(top5): {:.3f} (top1): {:.3f}'.format(
        float(best_acc), float(best_acc_top1)))


def resume(args, model, optimizer):
    if os.path.exists(args.job_dir+'/checkpoint/model_last.pt'):
        print(f"=> Loading checkpoint ")

        checkpoint = torch.load(args.job_dir+'/checkpoint/model_last.pt')

        start_epoch = checkpoint["epoch"]

        best_acc = checkpoint["best_acc"]

        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"=> Loaded checkpoint (epoch) {checkpoint['epoch']}")

        return start_epoch, best_acc
    else:
        print(f"=> No checkpoint found at '{args.job_dir}' '/checkpoint/")


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    # Warmup
    if args.lr_policy == 'step':
        factor = epoch // 8
        #if epoch >= 5:
        #    factor = factor + 1
        lr = args.lr * (0.1 ** factor)
    elif args.lr_policy == 'cos':
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.num_epochs))
    elif args.lr_policy == 'exp':
        step = 1
        decay = 0.96
        lr = args.lr * (decay ** (epoch // step))
    elif args.lr_policy == 'fixed':
        lr = args.lr
    else:
        raise NotImplementedError

    if epoch < args.warmup_length:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)
    if step == 0:
        print('current learning rate:{0}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_optimizer(args, model):
    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if (
            "bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if (
            "bn" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
                },
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            args.lr,
            momentum=0.9,
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
