import os
import re

import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

MODEL_NAME = 'distilgpt2'
TRAIN_MODE = os.getenv('TRAIN_MODE')
SEED = os.getenv('SEED')

# set data directory
os.chdir('C:/Users/User/Yandex.Disk/DS.projects/LSTM.Praktikum/data')
# read raw text data and save to array
with open('raw_data.txt', 'r', encoding='utf-8') as file:
    raw_data = np.array(file.read().lower().splitlines())

if TRAIN_MODE == 'preliminar': 
    raw_data = raw_data[:1_000]

# define the function for clearing and splitting of data
def split_and_clean(row): 
    # remove the mentions (@*)
    row = re.sub(r'@.*?\s', '', row)
    row = re.sub(r'@.*?\Z', '', row)

    # remove the URLs (http or www)
    row = re.sub(r'www.*?\s', '', row)
    row = re.sub(r'http.*?\s', '', row)
    row = re.sub(r'www.*?\Z', '', row)
    row = re.sub(r'http.*?\Z', '', row)

    # remove emojies ('*...anything')
    row = re.sub(r'\*([^ ]+)\s', '', row)
    row = re.sub(r'\*([^ ]+)\Z', '', row)

    # remove special symbols (&*;)
    row = re.sub(r'&([^ ]+)\;', '', row)

    # remove everything except of letters and numbers
    row = re.sub(r'[^a-z0-9\s]', '', row)

    # substitute the multiple spaces to single ones
    row = re.sub(r'[\s+]', ' ', row)

    # split the strings by spaces
    row = row.split(' ')

    # remove the empty elements from lists
    row = list(filter(None, row))
    
    return(row)

raw_data = list(map(split_and_clean, raw_data))

# check the result of raw data import (deliberabely without seed)
if TRAIN_MODE == 'preliminar': 
    for _ in np.random.randint(0, len(raw_data), 10): 
        print(raw_data[_])

print(f'Количество фраз до удаления слишком коротких: {len(raw_data)}.')
# drop too short phrases
raw_data = [phrase for phrase in raw_data if len(phrase) > 5]

# check the results
print(f'Количество фраз после удаления слишком коротких: {len(raw_data)}.')
if len(min(raw_data, key=len)) < 6: 
    print('Очистка от коротких фраз прошла с ошибкой.')
else: 
    print('Все короткие фразы удалены.')

# save the processed (cleaned) dataset
with open('processed_data.txt', 'w+', encoding='utf-8') as file: 
    for row in raw_data:
        file.write(' '.join(row) + '\n')

# create train, test and valid datasets
train, interhim = train_test_split(raw_data, train_size=0.8, random_state=SEED)
valid, test = train_test_split(interhim, train_size=0.5, random_state=SEED)

del interhim

# check splitting
print(f'В обучающей выборке содержится {len(train)} фраз.')
print(f'В валидационной выборке содержится {len(valid)} фраз.')
print(f'В тестовой выборке содержится {len(test)} фраз.')

# save the datasets to disk
with open('train.txt', 'w+', encoding='utf-8') as file: 
    for row in train:
        file.write(' '.join(row) + '\n')
with open('valid.txt', 'w+', encoding='utf-8') as file: 
    for row in valid:
        file.write(' '.join(row) + '\n')
with open('test.txt', 'w+', encoding='utf-8') as file: 
    for row in test:
        file.write(' '.join(row) + '\n')

# add pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
if TRAIN_MODE == 'preliminar': 
    print(tokenizer)

tokenizer.pad_token = tokenizer.eos_token

# check the results of tokenization
if TRAIN_MODE == 'preliminar': 
    example_phrase = raw_data[np.random.randint(0, len(raw_data), 1).item()]
    print(tokenizer.encode(
        example_phrase, 
        is_split_into_words=True, 
        add_special_tokens=True, 
        return_tensors='pt')
        )
    print('Длина фразы после энкодинга: ', len(tokenizer.encode(
        example_phrase, 
        is_split_into_words=True, 
        add_special_tokens=True, 
        return_tensors='pt')[0])
        )
    print(tokenizer.tokenize(
        example_phrase, 
        is_split_into_words=True, 
        return_tensors='pt'
        )
        )
    print('Длина токенизированной фразы: ', len(tokenizer.tokenize(
        example_phrase, 
        is_split_into_words=True, 
        return_tensors='pt')
        )
    )
    print(example_phrase)
    print('Длина исходной фразы: ', len(example_phrase))

def tokenize(row):
    return tokenizer.encode(
        row, is_split_into_words=True, add_special_tokens=True, return_tensors='pt', 
        )

