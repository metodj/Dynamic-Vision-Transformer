import os
import errno
import math
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_k = correct[:1].view(-1).float()

    return correct_k


def generate_logits(model, dataloader, C=1000):

    logits_list = []
    logits_raw_list = []
    targets_list = []

    top1 = [AverageMeter() for _ in range(3)]
    model.eval()

    for i, (x, target) in tqdm(enumerate(dataloader)):

        logits_temp = torch.zeros(3, x.size(0), C)
        logits_raw = torch.zeros(3, x.size(0), C)

        target_var = target.cuda()
        input_var = x.cuda()

        with torch.no_grad():

            less_less_token_output, less_token_output, normal_output = model(input_var)
                
            logits_temp[0] = F.softmax(less_less_token_output, 1)
            logits_temp[1] = F.softmax(less_token_output, 1)
            logits_temp[2] = F.softmax(normal_output, 1)

            logits_raw[0] = less_less_token_output
            logits_raw[1] = less_token_output
            logits_raw[2] = normal_output

            acc = accuracy(less_less_token_output, target_var, topk=(1,))
            top1[0].update(acc.sum(0).mul_(100.0 / x.size(0)).data.item(), x.size(0))
            acc = accuracy(less_token_output, target_var, topk=(1,))
            top1[1].update(acc.sum(0).mul_(100.0 / x.size(0)).data.item(), x.size(0))
            acc = accuracy(normal_output, target_var, topk=(1,))
            top1[2].update(acc.sum(0).mul_(100.0 / x.size(0)).data.item(), x.size(0))
            
        logits_list.append(logits_temp)
        logits_raw_list.append(logits_raw)
        targets_list.append(target_var)

        anytime_classification = []

        for index in range(3):
            anytime_classification.append(top1[index].ave)

    return torch.cat(logits_list, 1), torch.cat(logits_raw_list, 1), torch.cat(targets_list, 0), anytime_classification
