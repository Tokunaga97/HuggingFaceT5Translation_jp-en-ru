# -*- coding: utf-8 -*-

import re, os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


translation_corpus_dir = output_corpus_name = 'translation_corpus'
translation_corpus_path = '../' + translation_corpus_dir + '/'
save_dir_path = 'saved_model'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

save_dir = os.listdir('../' + save_dir_path)
cp_str = r'checkpoint\-([0-9]+)'
cp_list = [file for file in save_dir if re.match(cp_str, file)]
cp_dic = {cp:float(os.path.getctime('../' + save_dir_path + r'/' + cp)) for cp in cp_list}
sorted_cp_dic = sorted(cp_dic.items(), key=lambda x:x[1])
newest_cp_str = sorted_cp_dic[-1][0]

tokenizer = T5Tokenizer.from_pretrained('../' + save_dir_path + r'/' + newest_cp_str)
model = T5ForConditionalGeneration.from_pretrained('../' + save_dir_path + r'/' + newest_cp_str)
model = model.to(device)

while True:
  print('翻訳元の言語と翻訳先の言語を選択してください。(半角数字)')
  print('1. 日本語 → 英語')
  print('2. 英語 → 日本語')
  print('3. 英語 → ロシア語')
  print('4. ロシア語 → 英語')
  language_prefix_number = input().strip()
  if language_prefix_number == '1':
    language_prefix = 'translate Japanese to English: '
    break
  elif language_prefix_number == '2':
    language_prefix = 'translate English to Japanese: '
    break
  elif language_prefix_number == '3':
    language_prefix = 'translate English to Russian: '
    break
  elif language_prefix_number == '4':
    language_prefix = 'translate Russian to English: '
    break
  else:
    continue

print('翻訳したい文章を入力してください。')
inputs = input()
inputs = language_prefix + inputs
tokenized_inputs = tokenizer(inputs, return_tensors='pt')
input_ids = tokenized_inputs['input_ids'].to(device)
attention_mask = tokenized_inputs['attention_mask'].to(device)

outputs = model.generate(input_ids, attention_mask=attention_mask)

translated_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)

print('\n翻訳結果：')
print(translated_sentence)

