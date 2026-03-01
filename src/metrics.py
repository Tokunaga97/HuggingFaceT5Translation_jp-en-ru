# -*- coding: utf-8 -*-

from train import ReturnTokenizer

tokenizer = ReturnTokenizer()

def ComputeMetrics(pred):
  from nltk.translate import bleu_score, ribes_score  
  predictions, labels = pred
  decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
  decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

  decoded_preds = [pred.strip() for pred in decoded_preds]
  decoded_labels = [[label.strip()] for label in decoded_labels]

  bleu_score = bleu_score.corpus_bleu(decoded_labels, decoded_preds)
  ribes_score = ribes_score.corpus_ribes(decoded_labels, decoded_preds)

  return {'bleu': bleu_score, 'ribes': ribes_score}