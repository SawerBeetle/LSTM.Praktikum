generator = pipeline("text-generation", model="distilgpt2", tokenizer=tokenizer)
logging.set_verbosity_error() 
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
    input_tokens = test_tok_complete[i]['context']
    
    # get the length of input tokens sequence
    input_len = len(input_tokens)

    # decode the input text
    input_text = tokenizer.decode(input_tokens, skip_special_tokens=True)

    # set the maximal length of the phrase and number of new tokens
    max_total_limit = 200
    new_tokens_limit = max(1, max_total_limit - input_len)

    # generate the new text
    output_text = generator(
        input_text, 
        max_new_tokens=new_tokens_limit,
        max_length=None, 
        do_sample=True, 
        truncation=True, 
        top_k=50, 
        pad_token_id=tokenizer.eos_token_id
        )
    
    # print completed text
    if counter < 5:
        tqdm.write(f'Входящий текст и сгенерированный текст № {counter + 1}')
        tqdm.write(input_text)
        tqdm.write(output_text[0]['generated_text'])
        counter += 1
        tqdm.write(15 * '-')
        
    # add the decoded text to another completed texts from the batch
    completed_texts.append(output_text[0]['generated_text'].split(' '))
    output_texts.append(
        output_text[0]['generated_text'][len(input_text):].split(' ')
        )
    
# print the time of execution
finish = datetime.now() - start
mean_time = finish.total_seconds() / len(test_tok_complete)
print(f'Время дополнения одной фразы с помощью distilgpt2 равно {mean_time:.3f} секунд.')

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
print('Метрики качества distilgpt2: ')
print()
print(f'ROUGE1 = {(rouge1 / len(output_texts)):.4f} | ROUGE2 = {(rouge2 / len(output_texts)):.4f}') 

