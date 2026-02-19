# set the ROUGE scorer
metric_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)

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
