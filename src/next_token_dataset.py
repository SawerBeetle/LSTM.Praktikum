import os

from rouge_score import rouge_scorer
import numpy as np
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

TRAIN_MODE = os.getenv('TRAIN_MODE')
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
MODEL_NAME = os.getenv('MODEL_NAME')

# open the datasets
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.dirname(current_dir)
file_path_train = os.path.join(data_dir, 'data', 'train.txt')
print(15 * '-')
with open(file_path_train, 'r', encoding='utf-8') as file: 
    train = file.read().splitlines()
    train = [line for line in train if line.strip()]
file_path_valid = os.path.join(data_dir, 'data', 'valid.txt')
with open(file_path_valid, 'r', encoding='utf-8') as file: 
    valid = file.read().splitlines()
    valid = [line for line in valid if line.strip()]
file_path_test = os.path.join(data_dir, 'data', 'test.txt')
with open(file_path_test, 'r', encoding='utf-8') as file: 
    test = file.read().splitlines()
    test = [line for line in test if line.strip()]

# add pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
if TRAIN_MODE == 'preliminar': 
    print(tokenizer)

tokenizer.pad_token = tokenizer.eos_token

def tokenize(row):
    return tokenizer.encode(
        row, add_special_tokens=True, return_tensors='pt', 
        )

# класс датасета
class MaskedDataset(Dataset):
    def __init__(self, texts, tokenizer=tokenize, target_mode='single'):
        # the list for pairs, including the start of tokenized text and their end
        self.samples = []

        for line in texts: 
            # tokenize the text
            token_ids = tokenizer(line) 
            # create a context (the known 75% of tokens)
            context = token_ids[0][0:(3 * len(token_ids[0]) // 4)] 
            if target_mode == 'complete': 
                # create a target (the last 25% of tokens which must be reconstructed)
                target = token_ids[0][(3 * len(token_ids[0]) // 4):] 
            elif target_mode == 'single': 
                # create a target (the single token following to first 75% of tokens)
                target = token_ids[0][(3 * len(token_ids[0]) // 4) ]
            # join the 'context' and 'target' as tulpe and add to 'samples'
            self.samples.append((context, target))
           
    def __len__(self):
        # return the length of samples
        return len(self.samples) 

    def __getitem__(self, idx):
        # return the context and target with given number ('idx')
        x, y = self.samples[idx] 
        return {
            'context': x.detach().clone(), 
            'target': y.detach().clone()
        }
    
if TRAIN_MODE == 'preliminar': 
    print('Пример содержимого класса MaskedDataset: ')
    print(MaskedDataset(train, tokenize)[np.random.randint(0, len(train), 1).item()])
    print(15 * '-')

def collate_fn(batch): 
    # список текстов и классов из батча
    contexts = [item['context'] for item in batch]
    targets = torch.stack([item['target'] for item in batch])

    # дополняем тексты в батче padding'ом
    padded_contexts = pad_sequence(contexts, batch_first=True, padding_value=0)

    # lengths = [len(text) for text in texts]
    lengths = torch.tensor([len(text) for text in contexts])
    # считаем маски
    masks = (padded_contexts != 0).long()

    # возвращаем преобразованный батч
    return padded_contexts, masks, lengths, targets

train_tok = MaskedDataset(train, tokenize)

if TRAIN_MODE == 'preliminar': 
    print('Размеры выдачи функции collate_fn (без разделения на батчи): ')
    padded_contexts, masks, lengths, targets = collate_fn(train_tok)
    print(padded_contexts.shape)
    print(masks.shape)
    print(lengths.shape)
    print(targets.shape)
    print(15 * '-')


