# -*- coding: utf-8 -*-


def Test(test_dataset, tokenizer, model, device):
    from nltk.translate import bleu_score, ribes_score 
    bleu_smoothing = bleu_score.SmoothingFunction().method7
    ribes_smoothing = ribes_score.SmoothingFunction()
    bleu_scores = []
    ribes_scores = []
    for sentence in test_dataset:
      inputs = sentence['src'].strip()
      labels = sentence['tgt'].strip()
      tokenized_inputs = tokenizer(inputs, return_tensors='pt')
    
      input_ids = tokenized_inputs['input_ids'].to(device)
      attention_mask = tokenized_inputs['attention_mask'].to(device)
    
      outputs = model.generate(input_ids, attention_mask=attention_mask)
    
      bleu_score = bleu_score.sentence_bleu([labels], tokenizer.decode(outputs[0], skip_special_tokens=True), smoothing_function=bleu_smoothing)
      ribes_score = ribes_score.sentence_ribes([labels], tokenizer.decode(outputs[0], skip_special_tokens=True), smoothing_function=ribes_smoothing)
      bleu_scores.append(bleu_score)
      ribes_scores.append(ribes_score)
      print('入力文： '+inputs)
      print('出力文： '+tokenizer.decode(outputs[0], skip_special_tokens=True))
      print('正解文： '+labels)
      print('BLEU： '+round(bleu_score, 4))
      print('RIBES： '+round(ribes_score, 4))
    
    print('----------------------------------------')
    
    average_bleu_score = round(sum(bleu_scores) / len(bleu_scores), 4)
    average_ribes_score = round(sum(ribes_scores) / len(ribes_scores), 4)
    print('平均BLEU値'+str(average_bleu_score)+'\n'+'平均RIBES値'+str(average_ribes_score))