# Основной цикл обучения
if TRAIN_MODE == 'preliminar': 
    n_epochs = 5
else: 
    n_epochs = 1000

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

# switch the model to inference mode
model_lstm.eval()

# set the empty list of completed texts
completed_texts = []
# set the empty list for predicted parts of texts
output_texts = []


# set the time of the start of execution 
start = datetime.now()

# set the initial counter value (to print only n predictions, but not all)
counter = 0

for i in tqdm(range(len(test_tok_complete))):
    # initialize text for completion
    input_text = test_tok_complete.__getitem__(i)['context']
    # Set the arbitrary non-zero predicted integer (instead of token)
    # to prevent stop while loop before end of completion
    preds_int = 1000
    
    # set count of predicted tokens
    n_tokens = 0

    # Complete the text while model output was not zero 
    # or below 200 words (max length of tweet). 
    while(preds_int != 0 and len(input_text) < 200): 
        
        # get the token predicted by input part of phrase
        # get logits
        x_output = model_lstm(
            # convert text and length to appropriate shape
            input_text.unsqueeze(0), 
            torch.tensor(input_text.shape).view(1)
            )
        # reshape logits array (it needs when test the code, else commented)
        # logits = x_output.flatten(start_dim=1)
        # print(logits)

        # get logit for the last token
        last_token_logits = x_output[:, -1, :] 

        # Because of the model picks the most probable logit and give us the equal result 
        # for each analyzed tweet, we should use the 'temperature sampling' to make the 
        # predictions more diverse. 
        # set the temperature
        temperature = 0.7 
        # calculate the tensor of probabilities of the tokens
        probs = torch.softmax(last_token_logits / temperature, dim=-1)

        # get random token based on `probs`
        preds = torch.multinomial(probs, num_samples=1)[0]

        # get modal value of logits
        preds = torch.multinomial(probs, num_samples=1)[0]
        # print(preds)

        # get value to check stop condition of the 'while' loop
        preds_int = preds.item()

        # add the token to 'input_text', i.e., to phrase to completion
        input_text = torch.cat(
            (input_text, preds.to(input_text.dtype).view(-1)), 
            dim=0
            )
        # print(input_text)
        
        # Create ('if') or update ('else') the predicted part 
        # of the text to ROUGE calculation. 
        if n_tokens == 0:
            output_text = preds
            n_tokens += 1
        else: 
            output_text = torch.cat(
                (output_text, preds.to(output_text.dtype).view(-1)), 
                dim=0
            )

    # decode the completed text
    input_text = tokenizer.decode(input_text, skip_special_tokens=True)
    output_text = tokenizer.decode(output_text, skip_special_tokens=True)
    
    # print completed text
    if counter < 5:
        tqdm.write(f'Входящий текст и дополнение № {counter + 1}')
        tqdm.write(input_text)
        tqdm.write(input_text + output_text)
        counter += 1
        tqdm.write(15 * '-')

    # add the decoded text to another completed texts from the batch
    completed_texts.append(list(filter(None, input_text.split(' '))))
    output_texts.append(list(output_text.split(' ')))


# print the time of execution
finish = datetime.now() - start
mean_time = finish.total_seconds() / len(test_tok_complete)
print(f'Время дополнения одной фразы самописной RNN равно {mean_time:.3f} секунд.')

# the empty list for target phrases
output_targets = []

# fill the 'output_targets'
for _ in range(len(test_tok_complete)):
    # get the tokinized target
    output_target = test_tok_complete.__getitem__(_)['target']
    # decode the target
    output_target = tokenizer.decode(output_target, skip_special_tokens=True)
    # add the target to 'output_targets'
    output_targets.append(list(filter(None, output_target.split(' '))))

# initiate the metric values by zeros
rouge1 = 0.
rouge2 = 0.

# calculate the sum of ROUGE1 and ROUGE2 for the whole data set
for _ in range(len(output_texts)): 
    rouge1 += metric_scorer.score(
        ' '.join(output_texts[_]), 
        ' '.join(output_targets[_]))['rouge1'].fmeasure
    rouge2 += metric_scorer.score(
        ' '.join(output_texts[_]), 
        ' '.join(output_targets[_]))['rouge2'].fmeasure

# calculate the mean ROUGE1 and ROUGE2 and display them
print('Метрики качества самописной сети: ')
print()
print(f'ROUGE1 = {(rouge1 / len(output_texts)):.4f} | ROUGE2 = {(rouge2 / len(output_texts)):.4f}') 

