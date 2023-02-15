import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from utils import *

import math
import argparse
import models
import models_deit
from timm.models import create_model
import pickle
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Inference code for DVT')

parser.add_argument('--data_url', default='./data', type=str,
                    help='path to the dataset (ImageNet)')

parser.add_argument('--batch_size', default=64, type=int,
                    help='mini-batch size (default: 64)')

parser.add_argument('--model', default='DVT_T2t_vit_12', type=str,
                    help='model name')

parser.add_argument('--checkpoint_path', default='', type=str,
                    help='path to the pre-train model (default: none)')

parser.add_argument('--eval_mode', default=2, type=int,
                    help='mode 0 : read the evaluation results saved in pre-trained models\
                          mode 1 : read the confidence thresholds saved in pre-trained models and infer the model on the validation set\
                          mode 2 : determine confidence thresholds on the training set and infer the model on the validation set')

args = parser.parse_args()


def main():

    # load pretrained model
    checkpoint = torch.load(args.checkpoint_path)

    try:
        flops = checkpoint['flops']
        anytime_classification = checkpoint['anytime_classification']
        budgeted_batch_classification = checkpoint['budgeted_batch_classification']
        dynamic_threshold = checkpoint['dynamic_threshold']
    except:
        print('Error: \n'
              'Please provide essential information'
              'for customized models (as we have done '
              'in pre-trained models)!\n'
              'At least the following information should be Given: \n'
              '--flops: a list containing the Multiply-Adds corresponding to each '
              'length of the input sequence during inference')

    if args.eval_mode > 0:

        model = create_model(
        args.model,
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_connect_rate=None,  
        drop_path_rate=0.1,
        drop_block_rate=None,
        global_pool=None,
        bn_tf=False,
        bn_momentum=None,
        bn_eps=None,
        checkpoint_path='')

        # traindir = args.data_url + 'train/'
        valdir = args.data_url + 'valid/'

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # train_set = datasets.ImageFolder(traindir, transforms.Compose([
        #         transforms.RandomResizedCrop(224),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         normalize ]))
        # train_set_index = torch.randperm(len(train_set))
        # train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=32, pin_memory=False,
        #         sampler=torch.utils.data.sampler.SubsetRandomSampler(train_set_index[-50000:]))

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256,interpolation=3),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize])),
            batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=False)
        
        model = model.cuda()

        model.load_state_dict(checkpoint['model_state_dict'])

        budgeted_batch_flops_list = []
        budgeted_batch_acc_list = []

        print('generate logits on test samples...')
        test_logits, test_logits_raw, test_targets, anytime_classification = generate_logits(model, val_loader)
        print(test_logits_raw.shape, test_targets.shape)
        with open(f'output/{args.model}.p', 'wb') as f:
            pickle.dump((test_logits_raw, test_targets, anytime_classification), f)
            f.close()
        
        if args.eval_mode == 2:
            raise NotImplementedError('The training set of ImageNet still needs to be downloaded.')
            # print('generate logits on training samples...')
            # dynamic_threshold = torch.zeros([59, 3])
            # train_logits, train_targets, _ = generate_logits(model, train_loader)

        for p in range(1, 60):

            print('inference: {}/60'.format(p))

            _p = torch.FloatTensor(1).fill_(p * 1.0 / 20)
            probs = torch.exp(torch.log(_p) * torch.range(1, 3))
            probs /= probs.sum()

            if args.eval_mode == 2:
                raise NotImplementedError('The training set of ImageNet still needs to be downloaded.')
                # dynamic_threshold[p-1] = dynamic_find_threshold(train_logits, train_targets, probs)
            
            acc_step, flops_step = dynamic_evaluate(test_logits, test_targets, flops, dynamic_threshold[p-1])
            
            budgeted_batch_acc_list.append(acc_step)
            budgeted_batch_flops_list.append(flops_step)
        
        budgeted_batch_classification = [budgeted_batch_flops_list, budgeted_batch_acc_list]

    print('flops :', flops)
    print('anytime_classification :', anytime_classification)
    print('budgeted_batch_classification :', budgeted_batch_classification)


def dynamic_find_threshold(logits, targets, p):

    n_stage, n_sample, c = logits.size()
    max_preds, argmax_preds = logits.max(dim=2, keepdim=False)
    _, sorted_idx = max_preds.sort(dim=1, descending=True)

    filtered = torch.zeros(n_sample)
    T = torch.Tensor(n_stage).fill_(1e8)

    for k in range(n_stage - 1):
        acc, count = 0.0, 0
        out_n = math.floor(n_sample * p[k])
        for i in range(n_sample):
            ori_idx = sorted_idx[k][i]
            if filtered[ori_idx] == 0:
                count += 1
                if count == out_n:
                    T[k] = max_preds[k][ori_idx]
                    break
        filtered.add_(max_preds[k].ge(T[k]).type_as(filtered))

    T[n_stage - 1] = -1e8
    return T
   

def dynamic_evaluate(logits, targets, flops, T):

    n_stage, n_sample, c = logits.size()
    max_preds, argmax_preds = logits.max(dim=2, keepdim=False)
    _, sorted_idx = max_preds.sort(dim=1, descending=True)

    acc_rec, exp = torch.zeros(n_stage), torch.zeros(n_stage)
    acc, expected_flops = 0, 0
    for i in range(n_sample):
        gold_label = targets[i]
        for k in range(n_stage):
            if max_preds[k][i].item() >= T[k]:  # force the sample to exit at k
                if int(gold_label.item()) == int(argmax_preds[k][i].item()):
                    acc += 1
                    acc_rec[k] += 1
                exp[k] += 1
                break
    acc_all = 0
    for k in range(n_stage):
        _t = 1.0 * exp[k] / n_sample
        expected_flops += _t * flops[k]
        acc_all += acc_rec[k]

    return acc * 100.0 / n_sample, expected_flops.item()


if __name__ == '__main__':
    main()