import logging
import torch.nn as nn
import torch.quantization
import torch.optim as optim
import torch.nn.functional as F
from models.lstm import LSTM, RNN, Low_LSTM
from models.transformer import Transformer
from data.data_loader import get_loader


logger = logging.getLogger(__name__)


def get_loss_func(tokenizer):
    criterion = nn.CrossEntropyLoss()

    return criterion


def get_optim(args, model) -> optim:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    return optimizer


def distilled_loss(config, logits_s, inputs, labels):
    student_loss = config['criterion'](logits_s, labels)

    teacher_outputs = config['teacher'](inputs)
    loss_KD = F.kl_div(F.log_softmax(logits_s / config['args'].temperature, dim=1),
                       F.softmax(teacher_outputs / config['args'].temperature, dim=1), reduction="batchmean")

    loss = (1 - config['args'].alpha) * student_loss + config['args'].alpha * (
            config['args'].temperature ** 2) * loss_KD

    return loss


def processing_model(config, inputs):
    logits = config['model'](inputs)

    return logits


def model_setting(args):
    loader, tokenizer = get_loader(args)

    if args.have_teacher == 'True':

        #model = Low_LSTM(args, tokenizer)
        #teacher = LSTM(args, tokenizer)

        student_transformer_config = {'n_layers': 1, 'n_heads': 2, 'd_model': 128}
        teacher_transformer_config = {'n_layers': 4, 'n_heads': 4, 'd_model': 128}

        model = Transformer(args, tokenizer, student_transformer_config)
        teacher = Transformer(args, tokenizer, teacher_transformer_config)

        model.to(args.device)
        teacher.to(args.device)

    else:
        transformer_config = {'n_layers': args.n_layers, 'n_heads': args.n_heads, 'd_model': args.d_model}

        #model = Low_LSTM(args, tokenizer)
        model = Transformer(args, tokenizer, transformer_config)
        teacher = ''

        model.to(args.device)

    criterion = get_loss_func(tokenizer)
    optimizer = get_optim(args, model)
    criterion.to(args.device)

    config = {'loader': loader,
              'optimizer': optimizer,
              'criterion': criterion,
              'tokenizer': tokenizer,
              'args': args,
              'model': model,
              'teacher': teacher}

    return config