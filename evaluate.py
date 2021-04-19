import argparse
import torch
import torch.nn as nn
from importlib import import_module
import utils.common as utils
import models
import time
from utils.options import args

from data import cifar10, cifar100, imagenet

from utils.common import *


parser = argparse.ArgumentParser()

device = torch.device(f'cuda:{args.gpus[0]}') if torch.cuda.is_available() else 'cpu'
loss_func = nn.CrossEntropyLoss()

if args.data_set == 'cifar10':
    loader = cifar10.Data(args)
elif args.data_set == 'cifar100':
    loader = cifar100.Data(args) 
elif args.data_set == 'imagenet':
    loader = imagenet.Data(args)

# Test function
if args.data_set == 'cifar10' or args.data_set == 'cifar100':
    def test(model, testLoader):
        model.eval()
        losses = utils.AverageMeter('Loss', ':.4e')
        accuracy = utils.AverageMeter('Acc@1', ':6.2f')

        start_time = time.time()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testLoader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_func(outputs, targets)

                losses.update(loss.item(), inputs.size(0))
                pred = utils.accuracy(outputs, targets)
                accuracy.update(pred[0], inputs.size(0))

            current_time = time.time()
            print(
                f'Test Loss: {float(losses.avg):.4f}\t Acc: {float(accuracy.avg):.2f}%\t\t Time: {(current_time - start_time):.2f}s'
            )
        return accuracy.avg
elif args.data_set == 'imagenet':
    def test(model, val_loader, topk=(1,)):
        batch_time = utils.AverageMeter('Time', ':6.3f')
        losses = utils.AverageMeter('Loss', ':.4e')
        top1 = utils.AverageMeter('Acc@1', ':6.2f')
        top5 = utils.AverageMeter('Acc@5', ':6.2f')

        if args.use_dali:
            num_iter = val_loader._size // args.eval_batch_size
        else:
            num_iter = len(val_loader)

        model.eval()
        with torch.no_grad():
            end = time.time()
            i = 0
            if args.use_dali:
                for batch_idx, batch_data in enumerate(val_loader):
                    if args.debug:
                        if i > 5:
                            break
                        i += 1
                    images = batch_data[0]['data'].cuda()
                    targets = batch_data[0]['label'].squeeze().long().cuda()

                    # compute output
                    logits = model(images)
                    loss = loss_func(logits, targets)

                    # measure accuracy and record loss
                    pred1, pred5 = utils.accuracy(logits, targets, topk=(1, 5))
                    n = images.size(0)
                    losses.update(loss.item(), n)
                    top1.update(pred1[0], n)
                    top5.update(pred5[0], n)

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()
            else:
                for batch_idx, (images, targets) in enumerate(val_loader):
                    if args.debug:
                        if i > 5:
                            break
                        i += 1
                    images = images.cuda()
                    targets = targets.cuda()

                    # compute output
                    logits = model(images)
                    loss = loss_func(logits, targets)

                    # measure accuracy and record loss
                    pred1, pred5 = utils.accuracy(logits, targets, topk=(1, 5))
                    n = images.size(0)
                    losses.update(loss.item(), n)
                    top1.update(pred1[0], n)
                    top5.update(pred5[0], n)

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                        .format(top1=top1, top5=top5))

        return losses.avg, top1.avg, top5.avg


def get_prune_rate(model):
    all_params = 0
    prune_params = 0

    i = 0
    for name, module in model.named_modules():
        if hasattr(module, "set_prune_rate"):
            w = module.weight.data.detach().cpu()
            params = w.size(0) * w.size(1) * w.size(2) * w.size(3)
            all_params = all_params + params
            prune_params += int(params * module.prune_rate)
            i += 1

    print('Params Compress Rate: %.2f M/%.2f M(%.2f%%)' % 
                                                    ((all_params - prune_params)/1000000, all_params/1000000, 100. * prune_params / all_params))


model = models.__dict__[args.arch]().to(device)
ckpt = torch.load(args.pruned_model, map_location=device)

i = 0
for n, m in model.named_modules():
    if hasattr(m, "set_prune_rate"):
        m.set_prune_rate(ckpt['cfg'][i])
        i += 1    

model.load_state_dict(ckpt['state_dict'])

model = model.to(device)

get_prune_rate(model)

test(model, loader.testLoader)

