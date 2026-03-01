# -*- coding: utf-8 -*-

import os
import torch
from datasets import Dataset
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from preprocessing import SeparateData
from dataset import CreateDataCollator
from load import LoadTokenizer_and_Model
from metrics import ComputeMetrics
from test import Test
import torch.multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mp.set_start_method('spawn', force=True)

translation_corpus_dir = output_corpus_name = 'translation_corpus'
translation_corpus_path = '../' + translation_corpus_dir + '/'
save_dir_path = 'saved_model'
load_dir_path = 'loaded_data'
prefix_dictionary = {
    'ja-en': 'translate Japanese to English: ',
    'en-ja': 'translate English to Japanese: ',
    'en-ru': 'translate English to Russian: ',
    'ru-en': 'translate Russian to English: ',
    'ja-ru': 'translate Japanese to Russian: ',
    'ru-ja': 'translate Russian to Japanese: '
}
max_length = 128
batch_size = 8
epoch = 20
max_cp = 5
learning_rate = 3e-4


while True:
  print('Select language pair. (1 to 3)')
  print('1. Japanese to English')
  print('2. English to Russian')
  print('3. Japanese to Russian')
  input_language = input()
  if input_language == '1':
    language_pair = 'ja-en'
    break
  elif input_language == '2':
    language_pair = 'en-ru'
    break
  elif input_language == '3':
    language_pair = 'ja-ru'
    break
  else:
    print('Invalid input.')
    continue

corpus_text_files = [file for file in os.listdir(translation_corpus_path) if file.endswith('.txt')]

tokenizer, model = LoadTokenizer_and_Model(save_dir_path="../"+save_dir_path, load_dir_path="../"+load_dir_path, \
                                           max_length=max_length, max_cp=max_cp, device=device)

def ReturnTokenizer():
    return tokenizer
def ReturnMaxlength():
    return max_length
def ReturnDevice():
    return device


train_data, dev_data, test_data, reversed_test_data = SeparateData(language_pair, prefix_dictionary, corpus_text_files, translation_corpus_path)

train_data = [{'src': pair.split('\t')[0].strip(), 'tgt': pair.split('\t')[1].strip()} for pair in train_data]
dev_data = [{'src': pair.split('\t')[0].strip(), 'tgt': pair.split('\t')[1].strip()} for pair in dev_data]
test_data = [{'src': pair.split('\t')[0].strip(), 'tgt': pair.split('\t')[1].strip()} for pair in test_data]
reversed_test_data = [{'src': pair.split('\t')[0].strip(), 'tgt': pair.split('\t')[1].strip()} for pair in reversed_test_data]

train_dataset = Dataset.from_list(train_data).with_format("torch")
dev_dataset = Dataset.from_list(dev_data).with_format("torch")
test_dataset = Dataset.from_list(test_data).with_format("torch")
reversed_test_dataset = Dataset.from_list(reversed_test_data).with_format("torch")


data_collator_fn = CreateDataCollator

training_args = Seq2SeqTrainingArguments(
    output_dir= "../" + save_dir_path,
    overwrite_output_dir=True,
    num_train_epochs=epoch,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="steps",
    save_strategy="steps",
    learning_rate=learning_rate,
    weight_decay=0.01,
    eval_steps=5000,
    save_steps=1000,
    warmup_steps=1000,
    gradient_accumulation_steps=4,
    save_total_limit=100,
    prediction_loss_only=True,
    fp16=True,
    fp16_full_eval=True,
    dataloader_pin_memory=False,
    #dataloader_num_workers=1,
    remove_unused_columns=False,
    metric_for_best_model='loss',
    greater_is_better=False
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=ComputeMetrics,
    data_collator=data_collator_fn,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset
)

torch.cuda.empty_cache()

trainer.train()
trainer.save_model("../" + save_dir_path)

Test(test_dataset, tokenizer, model, device)