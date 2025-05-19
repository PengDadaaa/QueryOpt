import sys
import torch
import torch.optim as optim
import os
import time
import utils.evaluate as evaluate
from torch import nn
from models.build_model import build
from loguru import logger

from data.data_loader import sample_dataloader
from utils import AverageMeter
from tqdm import tqdm

from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count, flop_count_table
class ADSH_Loss(nn.Module):
    """
    Loss function of ADSH
    Args:
        code_length(int): Hashing code length.
        gamma(float): Hyper-parameter.
    """
    def __init__(self, code_length,num_roll, gamma):
        super(ADSH_Loss, self).__init__()
        self.code_length = code_length
        self.num_roll = num_roll
        self.gamma = gamma

    def forward(self, F, B, S, omega):

#######--------------
        hash_loss =((self.code_length * S - F @ B.t()) ** 2).sum() / (F.shape[0] * B.shape[0]) / self.code_length * 12
#-----------------------


        quantization_loss = ((F - B[omega, :]) ** 2).sum() / (F.shape[0] * B.shape[0]) * self.gamma / self.code_length * 12
        loss = hash_loss + quantization_loss

        return loss, hash_loss, quantization_loss

def train(
        query_dataloader,
        train_loader,
        retrieval_dataloader,
        code_length,
        args
):
    """
    Training model.

    Args
        query_dataloader, retrieval_dataloader(torch.utils.data.dataloader.DataLoader): Data loader.
        code_length(int): Hashing code length.
        args.device(torch.args.device): GPU or CPU.
        lr(float): Learning rate.
        args.max_iter(int): Number of iterations.
        args.max_epoch(int): Number of epochs.
        num_train(int): Number of sampling training data points.
        args.batch_size(int): Batch size.
        args.root(str): Path of dataset.
        dataset(str): Dataset name.
        args.gamma(float): Hyper-parameters.
        topk(int): Topk k map.

    Returns
        mAP(float): Mean Average Precision.
    """
    print('train start')

    num_roll = 96//code_length


    if args.max_iter == 50:
        step = 40
    elif args.max_iter == 120:
        step = 90
    elif args.max_iter == 40:
        step = 30
    else:
        KeyError('no step')
    print('step:',step)
    print('max_iter:',args.max_iter)

    if args.net == "resnet50":
        model = build(args,code_length,num_roll)

    # input = torch.randn(1, 3, 224, 224)
    # flop_counter = FlopCountAnalysis(model, input)
    # flop_counter.set_op_handle("aten::multihead_attention_forward", count_multihead_attention_flops)
    # flops = flop_counter.total()
    # print(f"FLOPs: {flops}")
    # print(f'Total number of parameters:\t {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    model.to(args.device)
    

   
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momen, nesterov=args.nesterov)#True
    # elif args.optim == 'Adam':
    #     optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # elif args.optim == 'AdamW':
    #     optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_step,gamma=args.lr_gamma)
    num_retrieval = len(retrieval_dataloader.dataset)



    criterion = ADSH_Loss(args.num_queries*num_roll, num_roll,args.gamma)
    U = torch.zeros(args.num_samples, args.num_queries*num_roll).to(args.device)
    B = torch.randn(num_retrieval, args.num_queries*num_roll).to(args.device)

    retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets().to(args.device)
    cnn_losses, hash_losses, quan_losses = AverageMeter(), AverageMeter(), AverageMeter()

    start = time.time()
    best_mAP = 0
    for it in range(args.max_iter):
        iter_start = time.time()

        train_dataloader, sample_index = sample_dataloader(train_loader, args.num_samples, args.batch_size, args.root, args.dataset)

        # Create Similarity matrix
        train_targets = train_dataloader.dataset.get_onehot_targets().to(args.device)
        S = (train_targets @ retrieval_targets.t() > 0).float()
        S = torch.where(S == 1, torch.full_like(S, 1), torch.full_like(S, 0))



        # Training CNN model
        for epoch in range(args.max_epoch):
            cnn_losses.reset()
            hash_losses.reset()
            quan_losses.reset()

            pbar = tqdm(enumerate(train_dataloader),total=len(train_dataloader),ncols = 50,leave=False)
            for batch, (data, targets, index) in pbar:## cifat-10==2000
                data, targets, index = data.to(args.device), targets.to(args.device), index.to(args.device)
                optimizer.zero_grad()
                out = model(data)
                F = out['pred']

                U[index, :] = F.data

                cnn_loss, hash_loss, quan_loss = criterion(F, B, S[index, :], sample_index[index])
                cnn_losses.update(cnn_loss.item())
                hash_losses.update(hash_loss.item())
                quan_losses.update(quan_loss.item())
                cnn_loss.backward()
                optimizer.step()
            logger.info('[epoch:{}/{}][cnn_loss:{:.6f}][hash_loss:{:.6f}][quan_loss:{:.6f}]'.format(epoch+1, args.max_epoch,
                        cnn_losses.avg, hash_losses.avg, quan_losses.avg))

        scheduler.step()

        expand_U = torch.zeros(B.shape).to(args.device)
        expand_U[sample_index, :] = U
        B = solve_dcc(B, U, expand_U, S, args.num_queries,num_roll, args.gamma)


        # Total loss

        iter_loss = calc_loss(U, B, S, args.num_queries*num_roll, sample_index, args.gamma)
      
        logger.info('[iter:{}/{}][loss:{:.6f}][iter_time:{:.2f}]'.format(it+1, args.max_iter, iter_loss, time.time()-iter_start))

    # Evaluate

        if (it < step and (it + 1) % args.val_freq == 0) or (it >= step and (it + 1) % 1 == 0):
        # if (it + 1) % 1 == 0:
            query_code = generate_code(model, query_dataloader, code_length, args.device)

            database_code = B[:,:code_length]
            print('database_code.shape:',database_code.shape)
            mAP = evaluate.mean_average_precision(
                query_code.to(args.device),
                database_code.to(args.device),
                query_dataloader.dataset.get_onehot_targets().to(args.device),
                retrieval_targets,
                args.device,
                args.topk,
            )
            if mAP > best_mAP:
                best_mAP = mAP

                ret_path = os.path.join('checkpoints', args.arch,args.info, str(code_length))
                if not os.path.exists(ret_path):
                    os.makedirs(ret_path)
                torch.save(query_code.cpu(), os.path.join(ret_path, 'query_code.t'))
                torch.save(database_code.cpu(), os.path.join(ret_path, 'database_code.t'))
                torch.save(query_dataloader.dataset.get_onehot_targets().cpu(), os.path.join(ret_path, 'query_targets.t'))
                torch.save(retrieval_targets.cpu(), os.path.join(ret_path, 'database_targets.t'))
                torch.save(model.cpu(), os.path.join(ret_path, 'model.t'))
                model = model.to(args.device)
            logger.info('[iter:{}/{}][code_length:{}][mAP:{:.5f}][best_mAP:{:.5f}]'.format(it+1, args.max_iter, code_length, mAP, best_mAP))
    logger.info('[Training time:{:.2f}]'.format(time.time()-start))


    return best_mAP



def solve_dcc(B, U, expand_U, S, code_length,num_roll, gamma):
    """
    Solve DCC problem.
    """
    optim_dim = code_length*num_roll
    Q = (optim_dim * S).t() @ U + gamma * expand_U

    for bit in range(optim_dim):
        q = Q[:, bit]
        u = U[:, bit]
        B_prime = torch.cat((B[:, :bit], B[:, bit+1:]), dim=1)
        U_prime = torch.cat((U[:, :bit], U[:, bit+1:]), dim=1)

        B[:, bit] = (q.t() - B_prime @ U_prime.t() @ u.t()).sign()
    Q = (optim_dim * S).t() @ U + gamma * expand_U

    # for bit in range(code_length):
    #     q = Q[:, bit]
    #     u = U[:, bit]
    #     B_prime = torch.cat((B[:, :bit], B[:, bit+1:code_length]), dim=1)
    #     U_prime = torch.cat((U[:, :bit], U[:, bit+1:code_length]), dim=1)

    #     B[:, bit] = (q.t() - B_prime @ U_prime.t() @ u.t()).sign()

    return B

def calc_loss(U, B, S, code_length, omega, gamma):
    """
    Calculate loss.
    """
    hash_loss = ((code_length * S - U @ B.t()) ** 2).sum()
    quantization_loss = ((U - B[omega, :]) ** 2).sum()
    loss = (hash_loss + gamma * quantization_loss) / (U.shape[0] * B.shape[0])

    return loss.item()


def generate_code(model, dataloader, code_length, device):
    # query_dataloader  wp
    """
    Generate hash code

    Args
        dataloader(torch.utils.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.

    Returns
        code(torch.Tensor): Hash code.
    """

    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, index in dataloader:
            data = data.to(device)
            hash_code = model(data)['pred']
            code[index, :] = hash_code.sign().cpu()

    model.train()
    return code



def count_multihead_attention_flops(m, inputs, outputs):
    seq_len, batch_size, embed_dim = inputs[0].shape
    num_heads = m.num_heads
    head_dim = embed_dim // num_heads
    # Q, K, V 线性变换 FLOPs
    flops = 3 * seq_len * batch_size * embed_dim * embed_dim
    # Scaled dot-product attention FLOPs
    flops += batch_size * num_heads * (seq_len * seq_len * head_dim)
    # 最后线性变换 FLOPs
    flops += seq_len * batch_size * embed_dim * embed_dim
    return flops