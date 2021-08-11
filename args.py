import argparse

def parse_args():
    """Parse parameters."""
    parser = argparse.ArgumentParser(description='Main pipeline for self-driving vehicles simulation using machine learning.')

    # device
    parser.add_argument('--local_rank',   type=int,   default=-1,           help='DDP parameter, do not modify')
    parser.add_argument('--is_dist',      type=bool,  default=True,       help='whether using multi-GPUs')

    # dataset settings
    parser.add_argument('--classes',      type=int,   default=10,          help='classification categories')
    
    # directory
    parser.add_argument('--ckptroot',     type=str,   default="./save/",          help='path to checkpoint')
    
    # hyperparameters settings
    parser.add_argument('--lr',           type=float, default=1e-3,        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,        help='weight decay (L2 penalty)')
    parser.add_argument('--train_bs',     type=int,   default=32,          help='training batch size')
    parser.add_argument('--valid_bs',     type=int,   default=64,          help='validation batch size')
    parser.add_argument('--num_workers',  type=int,   default=0,           help='# of workers used in dataloader')

    # training settings
    parser.add_argument('--epochs',       type=int,   default=15,          help='number of epochs to train')
    parser.add_argument('--start_epoch',  type=int,   default=0,           help='pre-trained epochs')
    parser.add_argument('--resume',       type=bool,  default=False,        help='whether re-training from ckpt')

    # parse the arguments
    args = parser.parse_args()

    return args
