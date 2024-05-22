import argparse
import torch
import os

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser(description='RS Models')


parser.add_argument('--seed', type=int, default=2023, help='seed')
parser.add_argument('--model', type=str, default='MF', help='model')
parser.add_argument('--defense', type=str, default='PamaCF', help='defense')


# dataset
parser.add_argument('--dataset', type=str, default='Gowalla', help='dataset')
parser.add_argument('--max_interaction', type=int, default=50, help='Max interactions')
parser.add_argument('--min_interaction', type=int, default=10, help='Min interactions')

parser.add_argument('--with_fakes', type=str2bool, default=False, help='Whether has injected users in dataset')
parser.add_argument('--attack_type', type=str, default='bandwagon_1', help='type of injected user')


# experiment
parser.add_argument('--use_gpu', type=str2bool, default=True, help='training device')
parser.add_argument('--device', type=str, default='gpu', help='training device')
parser.add_argument('--device_id', type=str, default='0', help='device id for gpu')
parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
parser.add_argument('--test_batch_size', type=int, default=2048, help='Batch size')
parser.add_argument('--patience', type=int, default=1000, help='patience for early stop')
parser.add_argument('--val_interval', type=int, default=100, help='Validation interval')
parser.add_argument('--record_interval', type=int, default=10000, help='Validation interval')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight for L2 loss on basic models.')
parser.add_argument('--n_epochs', type=int, default=30, help='max epoch')

# Evaluation
parser.add_argument('--top_k', type=list, default=[10, 20], help='K in evaluation')
parser.add_argument('--attack_top_k', type=list, default=[10, 20, 50], help='K in evaluation of attakcs')

# AT
parser.add_argument('--adv_reg', type=float, default=1.0, help='learning rate')
parser.add_argument('--eps', type=float, default=0.3, help='learning rate')
parser.add_argument('--user_lmb', type=float, default=2, help='learning rate')

# Common Model
parser.add_argument('--out_dim', type=int, default=64, help='Output size of Model')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

if torch.cuda.is_available() and args.use_gpu:
    print('using gpu:{} to train the model'.format(args.device_id))
    args.device_id = list(range(torch.cuda.device_count()))
    args.device = torch.device("cuda")
else:
    args.device = torch.device("cpu")
    print('using cpu to train the model')

if args.defense == "PamaCF":
    if args.model == "MF":
        if args.dataset == "Gowalla":
            args.eps = 0.9
        elif args.dataset == "Yelp2018":
            args.eps = 0.5
        elif args.dataset == "MIND":
            args.eps = 0.8
    elif args.model == "LighGCN":
        if args.dataset == "Gowalla":
            args.eps = 0.5 # or 0.6
        elif args.dataset == "Yelp2018":
            args.eps = 0.5
        elif args.dataset == "MIND":
            args.eps = 0.5 # or 0.3