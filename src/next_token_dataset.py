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

if TRAIN_MODE == 'preliminar': 
    print('Пример содержимого класса MaskedDataset: ')
    print(MaskedDataset(train, tokenize)[np.random.randint(0, len(train), 1).item()])

# create tokenized datasets
train_tok = MaskedDataset(train, tokenize)
valid_tok = MaskedDataset(valid, tokenize)
test_tok = MaskedDataset(test, tokenize)
test_tok_complete = MaskedDataset(test, tokenize, 'complete')

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

if TRAIN_MODE == 'preliminar': 
    print('Размеры выдачи функции collate_fn (без разделения на батчи): ')
    padded_contexts, masks, lengths, targets = collate_fn(train_tok)
    print(padded_contexts.shape)
    print(masks.shape)
    print(lengths.shape)
    print(targets.shape)

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

