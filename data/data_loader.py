import torch
import logging
from ETRI_tok.tokenization_etri_eojeol import BertTokenizer
from torch.utils.data import DataLoader, Dataset
from models.utils import move_to_device

logger = logging.getLogger(__name__)


class ModelDataLoader(Dataset):
    def __init__(self, file_path, args, tokenizer):
        self.args = args

        self.sentence = []
        self.label = []

        self.bert_tokenizer = tokenizer
        self.vocab_size = len(self.bert_tokenizer)
        self.file_path = file_path

        """
        init token, idx = [CLS], 2
        pad token, idx = [PAD], 0
        unk token, idx = [UNK], 1
        eos token, idx = [EOS], 30797
        """

        sepcial_tokens_dict = {'eos_token': '[EOS]'}
        self.bert_tokenizer.add_special_tokens(sepcial_tokens_dict)

        self.init_token = self.bert_tokenizer.cls_token
        self.pad_token = self.bert_tokenizer.pad_token
        self.unk_token = self.bert_tokenizer.unk_token
        self.eos_token = self.bert_tokenizer.eos_token

        self.init_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.init_token)
        self.pad_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.pad_token)
        self.unk_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.unk_token)
        self.eos_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.eos_token)

    # Load train, valid, test data in args.path_to_data
    def load_data(self, type):

        with open(self.file_path) as file:
            lines = file.readlines()

            for line in lines:
                sentence, label = self.data2tensor(line)
                self.sentence.append(sentence)
                self.label.append(label)

        assert len(self.sentence) == len(self.label)

    """
    Converting text data to tensor &
    expanding length of sentence to args.max_len filled with PAD idx
    """
    def data2tensor(self, line):
        split_data = line.split('\t')
        sentence, label = split_data[0], split_data[1]

        sentence_tokens = [self.init_token_idx] + self.bert_tokenizer.convert_tokens_to_ids(
             self.bert_tokenizer.tokenize(sentence)) + [self.eos_token_idx]

        if len(sentence_tokens) > self.args.max_len:
            sentence_tokens = sentence_tokens[:self.args.max_len - 1]
            sentence_tokens += [self.eos_token_idx]

        for i in range(self.args.max_len - len(sentence_tokens)):sentence_tokens.append(self.pad_token_idx)

        return torch.tensor(sentence_tokens), torch.tensor(int(label))

    def __getitem__(self, index):
        return self.sentence[index].to(self.args.device), self.label[index].to(self.args.device)

    def __len__(self):
        return len(self.label)


# Get train, valid, test data loader and BERT tokenizer
def get_loader(args):
    path_to_train_data = args.path_to_data+'/'+args.train_data
    path_to_valid_data = args.path_to_data+'/'+args.valid_data
    path_to_test_data = args.path_to_data+'/'+args.test_data

    tokenizer = BertTokenizer.from_pretrained('./ETRI_KoBERT/003_bert_eojeol_pytorch/vocab.txt', do_lower_case=False)

    train_iter = ModelDataLoader(path_to_train_data, args, tokenizer)
    valid_iter = ModelDataLoader(path_to_valid_data, args, tokenizer)
    test_iter = ModelDataLoader(path_to_test_data, args, tokenizer)

    train_iter.load_data('train')
    valid_iter.load_data('valid')
    test_iter.load_data('test')

    loader = {'train': DataLoader(dataset=train_iter,
                                  batch_size=args.batch_size,
                                  shuffle=True),
              'valid': DataLoader(dataset=valid_iter,
                                  batch_size=args.batch_size,
                                  shuffle=True),
              'test': DataLoader(dataset=test_iter,
                                 batch_size=args.batch_size,
                                 shuffle=True)}

    return loader, tokenizer


if __name__ == '__main__':
    get_loader('test')