# -*- coding: utf-8 -*-

from random import shuffle

def SeparateData(language_pair, prefix_dictionary, corpus_text_files, translation_corpus_path, ratio=0.9):
  all_sentences_list_train = []
  all_sentences_list_dev = []
  all_sentences_list_test = []
  reversed_all_sentences_list_test = []
  reversed_language_pair = language_pair.split('-')[::-1]
  reversed_language_pair = '-'.join(reversed_language_pair).strip()

  for text_file in corpus_text_files:
    pair_part = text_file.split('.')[-3]
    if (language_pair == pair_part) or (reversed_language_pair == pair_part):
      text_file_path = translation_corpus_path + text_file

      with open(text_file_path, 'r', encoding='utf-8') as f:
        all_pairs = f.readlines()
      all_pairs = [pair for pair in all_pairs if len(pair.split('\t')) == 2]
      data_added_prefix = [prefix_dictionary[pair_part] + pair.strip() for pair in all_pairs]
      train_data = data_added_prefix[:int(len(all_pairs) * ratio)]
      dev_data = data_added_prefix[int(len(all_pairs) * ratio):int(len(all_pairs) * (ratio + 0.05))]
      test_data = data_added_prefix[int(len(all_pairs) * (ratio + 0.05)):]

      all_sentences_list_train.extend(train_data)
      all_sentences_list_dev.extend(dev_data)
      if pair_part == language_pair:
        all_sentences_list_test.extend(test_data)
      elif pair_part == reversed_language_pair:
        reversed_all_sentences_list_test.extend(test_data)
    else:
      continue
  shuffle(all_sentences_list_train)
  shuffle(all_sentences_list_dev)

  return all_sentences_list_train, all_sentences_list_dev, all_sentences_list_test, reversed_all_sentences_list_test

