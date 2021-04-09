import os
import torch
import logging
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)
writer = SummaryWriter()


def cal_acc(yhat, y):
    with torch.no_grad():
        yhat = yhat.max(dim=-1)[1] # [0]: max value, [1]: index of max value
        acc = (yhat == y).float().mean()
        f1 = f1_score(y.cpu(), yhat.cpu())

    return acc, f1


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


def move_to_device(sample, device):
    if len(sample) == 0:
        return {}

    def _move_to_device(maybe_tensor, device):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.to(device)
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_device(value, device)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_move_to_device(x, device) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return [_move_to_device(x, device) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_device(sample, device)


def get_lr(optimizer):
    return optimizer.state_dict()['param_groups'][0]['lr']


def epoch_time(start_time, end_time) -> (int, int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def save_model(config, cp, pco):
    """
    cp (current performance) has train valid loss and train valid perplexity
    pco (performance_check_objects)
    saved model's name | epoch-{}-loss-{}.pt | in args.path_to_save
    """
    if not os.path.exists(config['args'].path_to_save):
        os.makedirs(config['args'].path_to_save)

    sorted_path = config['args'].path_to_save + '/checkpoint-epoch-{}-loss-{}.pt'.format(str(cp['ep']), round(cp['vl'], 4))

    writer.add_scalars('loss_graph', {'train': cp['tl'], 'valid': cp['vl']}, cp['ep'])
    writer.add_scalars('acc_graph', {'train': cp['tma'], 'valid': cp['vma']}, cp['ep'])

    if cp['ep'] + 1 == config['args'].epochs:
        writer.close()

    if cp['vl'] < pco['best_valid_loss'][0]:
        pco['early_stop_check'] = [0]
        pco['best_valid_loss'][0] = cp['vl']
        torch.save(config['model'].state_dict(), sorted_path)
        print(f'\n\t## SAVE valid_loss: {cp["vl"]:.3f} | valid acc: {cp["vma"]:.3f} ##')

    print(f'\t==Epoch: {cp["ep"] + 1:02} | Epoch Time: {cp["epm"]}m {cp["eps"]}s==')
    print(f'\t==Train Loss: {cp["tl"]:.3f} | Train acc: {cp["tma"]:.3f}==')
    print(f'\t==Valid Loss: {cp["vl"]:.3f} | Valid acc: {cp["vma"]:.3f}==')
    print(f'\t==Epoch latest LR: {get_lr(config["optimizer"]):.9f}==\n')

    return pco['early_stop_check'], pco['best_valid_loss']
