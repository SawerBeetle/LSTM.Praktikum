import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from lstm_model import model_lstm
from next_token_dataset import MaskedDataset, collate_fn, tokenize, tokenizer

BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
CHECK_MESSAGES = bool(os.getenv('CHECK_MESSAGES'))
MODEL_NAME = os.getenv('MODEL_NAME')
SAVE_WEIGHT = bool(os.getenv('SAVE_WEIGHT'))
TRAIN_MODE = os.getenv('TRAIN_MODE')

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

# create tokenized datasets
train_tok = MaskedDataset(train, tokenize)
valid_tok = MaskedDataset(valid, tokenize)
test_tok = MaskedDataset(test, tokenize)
test_tok_complete = MaskedDataset(test, tokenize, 'complete')

# create dataloaders
train_dataloader = DataLoader(
    train_tok, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
valid_dataloader = DataLoader(
    valid_tok, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
test_dataloader = DataLoader(
    test_tok, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
test_compl_dataloader = DataLoader(
    test_tok_complete, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

if TRAIN_MODE == 'preliminar': 
    print(f'Количество батчей в train_dataloader: {len(train_dataloader)}')
    print(f'Размер батча равен {BATCH_SIZE}')
    print()
    for x_batch, masks, lengths, y_batch in train_dataloader: 
        print('Содержимое контекстов в батче: ')
        print(x_batch)
        print(f'Размерность тензора с контекстом: {x_batch.shape}')
        print()
        print('Содержимое таргетов в батче: ')
        print(y_batch)
        print(f'Размерность тензора с таргетом: {y_batch.shape}')
        break

# create an optimizer
optimizer = torch.optim.Adam(model_lstm.parameters(), lr=0.001)
# set the cross-entropy loss calculator
criterion = nn.CrossEntropyLoss()

# add pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
if TRAIN_MODE == 'preliminar': 
    print(tokenizer)

# функция замера лосса и accuracy for the single output token
def evaluate_single_token(model, loader):

    # switch model to inference regime
    model.eval()

    # Set initial values of correct predictions and batch size 
    # to calculate accuracy.
    correct, total_batch_size = 0, 0
    # set initial value of loss sum for the given epoch
    sum_loss = 0

    # calculate loss and accuracy for the epoch
    with torch.no_grad():

        # the loop to calculate loss and true predictions for each batch
        for x_batch, masks, lengths, y_batch in loader:

            # выход модели для входа x_batch
            x_output = model.forward(x_batch, lengths)

            # reshape batch output and array with targets
            logits = x_output.flatten(start_dim=1)
            targets = y_batch.reshape(-1)
            # check the reshaping result
            if TRAIN_MODE == 'preliminar' and CHECK_MESSAGES:
                print(f'Размерность таргета после reshape: {targets.shape}')

            # calculate loss
            loss = criterion(logits, targets) 
            # check the loss value
            if TRAIN_MODE == 'preliminar' and CHECK_MESSAGES:
                print(f'Величина потерь: {loss}')

            # get the predicted tokens
            preds = torch.argmax(x_output, dim=1).mode()[0]
            # check the shape of predicted array and predicted tokens
            if TRAIN_MODE == 'preliminar' and CHECK_MESSAGES: 
                print(f'Размерность прогноза: {preds.shape}')
                print(f'Прогноз модели: {preds}')

            # количество верно угаданных токенов (to accuracy calculation)
            correct += (preds == targets).sum().item() 
            # размер батча 
            total_batch_size += targets.size(0) 
            # суммарная функция потерь for the all batches
            sum_loss += loss.item() 
    
    # лосс (mean for all batches, i.e., loss for the epoch) и accuracy
    avg_loss = sum_loss / len(loader)
    accuracy = correct / total_batch_size
    
    return avg_loss, accuracy

# add a dictionary for loss/accuracy visualisation
hist = {'train_loss': [], 'val_loss': [], 'val_acc': []} 

# Основной цикл обучения
if TRAIN_MODE == 'preliminar': 
    n_epochs = 5
else: 
    n_epochs = 1000

if __name__ == '__main__':
    for epoch in range(n_epochs):
        model_lstm.train()
        # initial loss value
        train_loss = 0.
        
        for x_batch, masks, lengths, y_batch in tqdm(train_dataloader, leave=False):

            optimizer.zero_grad() # обнуление градиентов оптимизатора

            # выход модели для входа x_batch
            x_output = model_lstm(x_batch, lengths)
            if TRAIN_MODE == 'preliminar' and CHECK_MESSAGES: 
                print('Размерность выхода модели: ', x_output.shape)

            # convert output to logits
            logits = x_output.flatten(start_dim=1)
            # control the shape of output
            if TRAIN_MODE == 'preliminar' and CHECK_MESSAGES: 
                print('Размерность логитов: ', logits.shape)

            # get targets   
            targets = y_batch.reshape(-1)
            # control the shape of target array
            if TRAIN_MODE == 'preliminar' and CHECK_MESSAGES: 
                print('Размерность таргета после reshape: ', targets.shape)

            # функция потерь
            loss = criterion(logits, targets)

            # расчёт градиентов
            loss.backward() 

            # обновление градиентов
            optimizer.step() 

            # update loss for the given batch
            train_loss += loss.item()

        # calculate mean train loss for the given epoch
        train_loss /= len(train_dataloader)
        # calculate validation loss and accuracy
        val_loss, val_acc = evaluate_single_token(model_lstm, valid_dataloader)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Accuracy: {val_acc:.2%}")

        # add loss/accuracy data to histogram dictionary
        hist['train_loss'].append(train_loss)
        hist['val_loss'].append(val_loss)
        hist['val_acc'].append(val_acc)

        epochs = range(n_epochs)

    # Потеря
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist['val_loss'], marker='x', label="Val Loss")
    plt.plot(epochs, hist['train_loss'], marker='o', label="Train Loss",)
    plt.title("Сравнение кривых потерь")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Точность
    plt.subplot(1, 2, 2)
    plt.plot(range(n_epochs), hist['val_acc'], marker='o', label="Val Acc")
    plt.title("Сравнение точности на валидации")
    plt.xlabel("Эпоха")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# save model weights
if SAVE_WEIGHT:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.dirname(current_dir)
    file_path_weights = os.path.join(data_dir, 'models', 'lstm_model_weights.pth')
    torch.save(model_lstm.state_dict(), file_path_weights)