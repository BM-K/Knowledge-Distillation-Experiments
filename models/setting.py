import torch
import random
import logging
import argparse
import numpy as np

logger = logging.getLogger(__name__)


def set_args() -> argparse:
    parser = argparse.ArgumentParser()

    # model hyperparameters
    # LSTM
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--hid_dim', type=int, default=256)

    # Transformer
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--n_heads', type=int, default=8)

    parser.add_argument('--dropout', type=int, default=0.1)  # RNN 0.5
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--lr', type=float, default=0.0002)  # RNN 0.03

    # type of processing
    parser.add_argument('--train_', type=str, default='True')
    parser.add_argument('--test_', type=str, default='True')

    # Data loader parser
    parser.add_argument('--train_data', type=str, default='ratings_train.tsv')
    parser.add_argument('--test_data', type=str, default='ratings_test.tsv')
    parser.add_argument('--valid_data', type=str, default='ratings_valid.tsv')
    parser.add_argument('--path_to_data', type=str, default='./data/')
    parser.add_argument('--path_to_saved_model', type=str, default='./output/.pt')
    parser.add_argument('--saved_teacher_model', type=str, default='./output/lstm_1_layer.pt')
    parser.add_argument('--path_to_save', type=str, default='./output_mv')

    # Distillation
    parser.add_argument('--have_teacher', type=str, default='False')
    parser.add_argument('--alpha', type=float, default=0.1)  # RNN 0.5
    parser.add_argument('--temperature', type=int, default=3)  # RNN 3

    # Device
    parser.add_argument('--device', type=str, default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    args = parser.parse_args()
    return args


def set_logger() -> logger:
    _logger = logging.getLogger()
    formatter = logging.Formatter(
        '[%(levelname)s] %(asctime)s [ %(message)s ] | file::%(filename)s | line::%(lineno)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    _logger.addHandler(stream_handler)
    _logger.setLevel(logging.DEBUG)
    return _logger


def set_seed(args):
    logger.info('Setting Seed')
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    logger.info('Setting Seed Complete')


def print_args(args):
    logger.info('Args configuration')
    for idx, (key, value) in enumerate(args.__dict__.items()):
        if idx == 0 : print("argparse{\n", "\t", key, ":", value)
        elif idx == len(args.__dict__) - 1 : print("\t", key, ":", value, "\n}")
        else : print("\t", key, ":", value)