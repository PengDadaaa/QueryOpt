import argparse



def load_args(args ,flag):
    args.dataset = 'cub-2011'
    args.root = "/4T/dataset/CUB_200_2011"

    # args.dataset = 'nabirds'
    # args.root = ""

    # args.dataset = 'food101'
    # args.root = ""

    # args.dataset = 'aircraft'
    # args.root = ""

    # args.dataset = 'vegfru'
    # args.root = ""


    args.pretrain = True
    args.num_workers = 4
    #------------
    if flag == "queryopt":#baseline


        args.batch_size = 16
        args.lr = 3e-4
        args.wd = 1e-4
        args.gpu = 0
                
        if args.dataset == 'food101':
            args.max_epoch = 10
            args.max_iter = 120
            args.num_samples = 2000
        elif args.dataset == 'nabirds' or args.dataset == 'vegfru':
            args.max_epoch = 30
            args.max_iter = 50
            args.num_samples = 4000
        else:
            args.max_epoch = 30
            args.max_iter = 40
            args.num_samples = 2000
        args.momen = 0.9
        args.optim = "SGD"
        args.enc_layers = 1
        args.dec_layers = 1
        args.hidden_dim = 384
        args.position_embedding = 'learned'
        args.dropout = 0
        args.code_length = '48'

        args.val_freq = 5

        if args.dataset == 'food101':
            args.lr_step ='90'
        else:
            args.lr_step ='30' 

        args.arch = "query_opt"

        args.info = f"your info"

    return args
