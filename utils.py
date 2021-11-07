import torch
import os, errno
import logging
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description='PyTorch/torchtext NLI Baseline')
    parser.add_argument('--dataset', '-d', type=str, default='snli')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-len', type=int, default=128)

    parser.add_argument('--embed-dim', type=int, default=128)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--k-dim', type=int, default=32)
    parser.add_argument('--v-dim', type=int, default=32)
    parser.add_argument('--ffn-embed-dim', type=int, default=128)
    parser.add_argument('--N', type=int, default=3)
    parser.add_argument('--M', type=int, default=2)

    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--activation-dropout', type=float, default=0.)
    parser.add_argument('--attention-dropout', type=float, default=0.)

    parser.add_argument('--epochs', type=int, default=200)

    parser.add_argument('--warmup-updates', type=int, default=4000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--min-lr', type=float, default=4e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-3)

    parser.add_argument('--results-dir', type=str, default='results')
    return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
    # --result_dir
    check_folder(os.path.join(args.results_dir, args.dataset))

    # --epoch
    try:
        assert args.epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


def get_device(gpu_no):
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_no)
        return torch.device('cuda:{}'.format(gpu_no))
    else:
        return torch.device('cpu')


def makedirs(name):
    """helper function for python 2 and 3 to call os.makedirs() avoiding an error
    if the directory to be created already exists.s"""

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def get_logger(args, phase):
    logging.basicConfig(level=logging.INFO,
                        filename="{}/{}/{}.log".format(args.results_dir, args.dataset, phase),
                        format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    return logging.getLogger(phase)
