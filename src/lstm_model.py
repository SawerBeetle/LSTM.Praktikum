import os

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import AutoTokenizer

TRAIN_MODE = os.getenv('TRAIN_MODE')
CHECK_MESSAGES = bool(os.getenv('CHECK_MESSAGES'))
MODEL_NAME = os.getenv('MODEL_NAME')

# add pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
if TRAIN_MODE == 'preliminar': 
    print(tokenizer)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256):
        super().__init__()

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)

        # out_dim for sum
        # NB: change if summ will be changed to concatenation
        out_dim = hidden_dim 

        # output linear layer
        self.fc = nn.Linear(out_dim, vocab_size)

    def forward(self, x, lengths): 
        # embed the text
        emb = self.embedding(x) 
        emb = self.dropout(emb)
        if TRAIN_MODE == 'preliminar' and CHECK_MESSAGES: 
            print('Размерность эмбеддинга: ', emb.shape)
        pack = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        # get the output of recurrent layer ('out')
        out, _ = self.rnn(pack) 
        out, _ = pad_packed_sequence(out, batch_first=True)
        # retain last four token only
        out = out[:, -4:, :]
        if TRAIN_MODE == 'preliminar' and CHECK_MESSAGES: 
            print('Размерность выходных данных RNN после pad_packed_sequence: ', out.shape)

        # скрытые состояния <MASK> токена 
        # после двух проходов двунаправленной сети
        hidden_forward = out[:, :, :out.size(2) // 2]
        hidden_backward = out[:, :, out.size(2) // 2:]

        # агрегация скрытых состояний в зависимости от self.combine
        hidden_agg = hidden_forward + hidden_backward
        if TRAIN_MODE == 'preliminar' and CHECK_MESSAGES: 
            print('Размерность выхода скрытых слоёв: ', hidden_agg.shape)
        
        hidden_agg = self.dropout(hidden_agg)
        linear_out = self.fc(hidden_agg)
        if TRAIN_MODE == 'preliminar' and CHECK_MESSAGES: 
            print('Размерность выхода линейного слоя: ', linear_out.shape)

        # return the last predicted token
        return linear_out[:, -1:, :]

# create an exemplar of LSTM model
model_lstm = LSTMClassifier(vocab_size=tokenizer.vocab_size)

