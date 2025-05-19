import torch
import argparse

import numpy as np
import random
from loguru import logger
from data.data_loader import load_data
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import Queryopt

from set_config import load_args
from tqdm import tqdm
import time
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False#False
    torch.backends.cudnn.deterministic = True

def run():
    args = load_config()
    log_path = 'logs/' + args.arch + '-' + args.net + '/' + args.info
    if not os.path.exists(log_path):
        os.makedirs(log_path)


    #------------------------------------------------------------
    #------------------------------------------------------------
    logger.add(log_path +"/"+ '-{time:YYYY-MM-DD-HH}.log', rotation='500 MB', level='INFO')
    logger.success(args)
    #time:YYYY-MM-DD HH
    for i in tqdm(range(1),ncols=50,leave=False):
        time.sleep(1)

    # Load dataset

    query_dataloader, train_dataloader, retrieval_dataloader = load_data(
        args.dataset,
        args.root,
        args.num_query,
        args.num_samples,
        args.batch_size,
        args.num_workers
    )
    if args.arch == 'query_opt':
        net_arch = Queryopt
    else:
        raise ValueError('Invalid architecture name: {}'.format(args.arch))


    for code_length in args.code_length:

        args.num_queries = code_length

                
        mAP = net_arch.train(query_dataloader, train_dataloader, retrieval_dataloader, code_length, args)

        logger.info('[code_length:{}][map:{:.4f}]'.format(code_length, mAP))


def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='ADSH_PyTorch')
    parser.add_argument('--dataset',
                        help='Dataset name.')
    parser.add_argument('--root',
                        help='Path of dataset')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='Batch size.(default: 64)')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Learning rate.(default: 1e-4)')
    parser.add_argument('--wd', default=1e-5, type=float,
                        help='Weight Decay.(default: 1e-5)')
    parser.add_argument('--optim', default='Adam', type=str,
                        help='Optimizer')
    parser.add_argument('--code-length', default='12,24,32,48', type=str,#
                        help='Binary hash code length.(default: 12,24,32,48)')
    parser.add_argument('--max-iter', default=50, type=int,
                        help='Number of iterations.(default: 50)')
    parser.add_argument('--max-epoch', default=3, type=int,
                        help='Number of epochs.(default: 3)')
    parser.add_argument('--num-query', default=1000, type=int,
                        help='Number of query data points.(default: 1000)')
    parser.add_argument('--num-samples', default=2000, type=int,
                        help='Number of sampling data points.(default: 2000)')
    parser.add_argument('--num-workers', default=0, type=int,
                        help='Number of loading data threads.(default: 0)')
    parser.add_argument('--topk', default=-1, type=int,
                        help='Calculate map of top k.(default: all)')
    parser.add_argument('--gpu', default=None, type=int,
                        help='Using gpu.(default: False)')
    parser.add_argument('--gamma', default=200, type=float,
                        help='Hyper-parameter.(default: 200)')
    parser.add_argument('--info', default='Trivial',
                        help='Train info')
    parser.add_argument('--arch', default='baseline',
                        help='Net arch')
    parser.add_argument('--net', default='resnet50',
                        help='Net arch')
    parser.add_argument('--save_ckpt', default='checkpoints/',
                        help='result_save')
    parser.add_argument('--lr-step', default='30,45', type=str,
                        help='lr decrease step.(default: 30,45)')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='lr decrease step.(default: 0.1WS)')

    parser.add_argument('--pretrain', action='store_true',
                        help='Using image net pretrain')

    parser.add_argument('--momen', default=0.9, type=float,
                        help='Hyper-parameter.(default: 0.9)')
    parser.add_argument('--nesterov', action='store_true',
                        help='Using SGD nesterov')

    parser.add_argument('--num-classes', default=200, type=int,
                        help='Number of classes.(default: 200)')
    parser.add_argument('--val-freq', default=10, type=int,
                        help='Number of validate frequency.(default: 10)')

    parser.add_argument('--num-roll', default=0, type=int,
                        help='Number of roll.(default: 0)')
    # MyModel------------------------------------------------
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    # * Transformer
    parser.add_argument('--enc_layers', default=1, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=1, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=768, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")

    parser.add_argument('--pre_norm', action='store_true')

    args = parser.parse_args()

    flag = "queryopt" 
    args = load_args(args,flag)

#-------------------------------

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(f"cuda:{args.gpu}")

    # Hash code length
    args.code_length = list(map(int, args.code_length.split(',')))
    args.lr_step = list(map(int, args.lr_step.split(',')))

    return args


if __name__ == '__main__':
    seed_everything(42)
    run()
