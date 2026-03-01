# -*- coding: utf-8 -*-

import os, re, shutil
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration

def LoadTokenizer_and_Model(save_dir_path, load_dir_path, max_length, max_cp, device):
    if len(os.listdir(save_dir_path)) == 0:
      #モデル保存用ディレクトリに何も存在しなければ、初期化する。
      tokenizer = T5Tokenizer.from_pretrained(load_dir_path, padding='max_length', truncation=True, max_length=max_length, keep_accents=True)
      config = T5Config.from_pretrained(load_dir_path)
      config.vocab_size = tokenizer.vocab_size + 1
      config.decoder_start_token_id = tokenizer.pad_token_id
      model = T5ForConditionalGeneration(config)
      model.resize_token_embeddings(config.vocab_size)
    else:
      #そうでなければ、学習させた tokenizer と model をロードする。
      save_dir = os.listdir(save_dir_path)
      cp_str = r'checkpoint\-([0-9]+)'
      cp_list = [file for file in save_dir if re.match(cp_str, file)]
      cp_dic = {cp:float(os.path.getctime(save_dir_path + r'/' + cp)) for cp in cp_list}
      sorted_cp_dic = sorted(cp_dic.items(), key=lambda x:x[1])
      #checkpointが閾値max_cpを超えたら、既存checkpointを古い順に消す（あくまでもゴミ箱への移動）。
      #すなわち、モデルのセーブ用ディレクトリ内にはのcheckpointの数をmax_cp以内しか存在しないようにする。
      if len(sorted_cp_dic) > max_cp:
        for i in range(len(sorted_cp_dic) - max_cp):
          shutil.rmtree(save_dir_path + r'/' + sorted_cp_dic[i][0])
      #最新のcheckpointをロード
      newest_cp_str = sorted_cp_dic[-1][0]
    
      tokenizer = T5Tokenizer.from_pretrained(save_dir_path + r'/' + newest_cp_str)
      model = T5ForConditionalGeneration.from_pretrained(save_dir_path + r'/' + newest_cp_str)
    
    model = model.to(device)
    
    return tokenizer, model