# -*- coding: utf-8 -*-

import torch
from train import ReturnTokenizer, ReturnMaxlength, ReturnDevice

tokenizer = ReturnTokenizer()
max_length = ReturnMaxlength()
device = ReturnDevice()

def CreateDataCollator(batch):
  inputs = {}

  input_ids_list = []
  attention_mask_list = []
  labels_list = []

  for texts in batch:
    input = {}
    src = texts['src'].strip()
    tgt = texts['tgt'].strip()

    tokenized_src = tokenizer(src, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt', add_special_tokens=True)
    tokenized_tgt = tokenizer(tgt, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt', add_special_tokens=True)


    input['input_ids'] = tokenized_src['input_ids'].squeeze().to(device)
    input['attention_mask'] = tokenized_src['attention_mask'].squeeze().to(device)

    input_ids_list.append(input['input_ids'])
    attention_mask_list.append(input['attention_mask'])

    input['labels'] = tokenized_tgt['input_ids'].squeeze().to(device)

    input['labels'][input['labels'] == tokenizer.pad_token_id] = -100
    input['labels'][input['labels'] == tokenizer.unk_token_id] = -100

    labels_list.append(input['labels'])

  inputs['input_ids'] = torch.stack(input_ids_list)
  inputs['attention_mask'] = torch.stack(attention_mask_list)
  inputs['labels'] = torch.stack(labels_list)

  return inputs

