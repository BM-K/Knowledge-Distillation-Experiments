import time
import torch
from tqdm import tqdm
from models.setting import set_args, set_logger, set_seed, print_args
from models.functions import processing_model, model_setting, distilled_loss
from models.utils import save_model, get_lr, epoch_time, cal_acc, print_size_of_model


def system_setting():
    args = set_args()
    print_args(args)
    set_seed(args)

    early_stop_check = [0]
    best_valid_loss = [float('inf')]
    performance_check_objects = {'early_stop_check': early_stop_check,
                                 'best_valid_loss': best_valid_loss}

    return args, performance_check_objects


def train(config) -> (float, float):
    total_loss = 0
    iter_num = 0
    train_acc = 0

    logger.info("Training main")
    for step, batch in enumerate(tqdm(config['loader']['train'])):
        config['optimizer'].zero_grad()

        inputs, labels = batch
        logits = processing_model(config, inputs)

        if config['args'].have_teacher == 'True':
            loss = distilled_loss(config, logits, inputs, labels)
        else:
            loss = config['criterion'](logits, labels)

        loss.backward()

        config['optimizer'].step()

        total_loss += loss
        iter_num += 1

        with torch.no_grad():
            tr_acc, _ = cal_acc(logits, labels)

        train_acc += tr_acc

    return total_loss.data.cpu().numpy() / iter_num,\
           train_acc.data.cpu().numpy() / iter_num,\


def valid(config) -> (float, float):
    total_loss = 0
    iter_num = 0
    valid_acc = 0

    with torch.no_grad():
        for step, batch in enumerate(config['loader']['valid']):
            inputs, labels = batch
            logits = processing_model(config, inputs)

            if config['args'].have_teacher == 'True':
                loss = distilled_loss(config, logits, inputs, labels)
            else:
                loss = config['criterion'](logits, labels)

            total_loss += loss
            iter_num += 1

            with torch.no_grad():
                tr_acc, _ = cal_acc(logits, labels)

            valid_acc += tr_acc

    return total_loss.data.cpu().numpy() / iter_num,\
           valid_acc.cpu().numpy() / iter_num,\


def test(config) -> (float, float, float):
    total_loss = 0
    iter_num = 0
    test_acc = 0
    test_f1_score = 0

    with torch.no_grad():
        for step, batch in enumerate(config['loader']['test']):
            inputs, labels = batch
            logits = processing_model(config, inputs)

            if config['args'].have_teacher == 'True':
                loss = distilled_loss(config, logits, inputs, labels)
            else:
                loss = config['criterion'](logits, labels)

            total_loss += loss
            iter_num += 1

            with torch.no_grad():
                tr_acc, f1_score = cal_acc(logits, labels)

            test_acc += tr_acc
            test_f1_score += f1_score

    return total_loss.data.cpu().numpy() / iter_num, \
           test_acc.cpu().numpy() / iter_num, \
           test_f1_score / iter_num


def main() -> None:
    """
    config is made up of
    dictionary {data loader, optimizer, criterion, scheduler, tokenizer, args, model}
    """
    args, performance_check_objects = system_setting()
    config = model_setting(args)

    if config['args'].have_teacher == 'True':
        config['teacher'].load_state_dict(torch.load(args.saved_teacher_model))
        for p in config['teacher'].parameters():
            p.requires_grad_(False)

    if args.train_ == 'True':
        logger.info('Start Training')

        for epoch in range(args.epochs):
            start_time = time.time()

            config['model'].train()
            train_loss, train_acc,  = train(config)

            config['model'].eval()
            valid_loss, valid_acc,  = valid(config)

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            performance = {'tl': train_loss, 'vl': valid_loss,
                           'tma': train_acc, 'vma': valid_acc,
                           'ep': epoch, 'epm': epoch_mins, 'eps': epoch_secs}

            performance_check_objects['early_stop_check'], performance_check_objects['best_valid_loss'] = \
                save_model(config, performance, performance_check_objects)

    if args.test_ == 'True':
        logger.info("Start Test")

        config['model'].load_state_dict(torch.load(args.path_to_saved_model))
        config['model'].eval()

        test_loss, test_mem_acc, f1 = test(config)
        print(f'\n\t==Test loss: {test_loss:.4f} | Test memory acc: {test_mem_acc:.4f} | Test F1 score: {f1:.4f}==\n')

        print_size_of_model(config['model'])

        if config['args'].have_teacher == 'True':
            print_size_of_model(config['teacher'])


if __name__ == '__main__':
    logger = set_logger()
    main()