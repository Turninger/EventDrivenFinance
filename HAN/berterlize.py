import pickle, argparse, os, sys
from sklearn.metrics import accuracy_score
import numpy as np
import random
import torch
import torch.nn as nn
import torch.functional as F
import pickle
import re
import pandas as pd
import numpy as np
import datetime
import time 
from transformers import BertTokenizer
#对输入的文本文件进行处理，包括文本清理和使用预训练的BERT模型对每个句子进行编码，然后提取模型的池化输出作为语料库的表示
def berterlize(file):
    # given a txt file having all news 
    f = open(file,encoding='utf-8')
    txt = f.read()
    txt = re.sub(r'(\(image_address="https|image_address="http)?:\/\/(\w|\.|\/|\?|\=|\&|\%\\)*\b', '', txt)
    txt = re.sub(r'\\u[a-zA-Z0-9]*\b', '', txt)
    txt = txt.strip().rstrip()
    txt = txt.split('\n')
    f.close()
    corpus = []
    for sentence in txt:
        inputs = tokenizer(sentence, return_tensors="pt")
        # length greater than 512
        if inputs.input_ids.shape[1] <= 512:
            outputs = model(**inputs)
        else:
            inputs['input_ids'] = inputs['input_ids'][0,:512].reshape((1,-1))
            inputs['token_type_ids'] = inputs['token_type_ids'][0,:512].reshape((1,-1))
            inputs['attention_mask'] = inputs['attention_mask'][0,:512].reshape((1,-1))
            outputs = model(**inputs)

        last_hidden_states = outputs.pooler_output
        corpus.append(last_hidden_states.detach().numpy().reshape(-1))
    return np.array(corpus)
#遍历给定文件夹中的每个文本文件，并对每个文件调用 berterlize 函数。然后，它将处理后的表示保存到指定的输出目录中。
def process(stockFolder):
    print(f"Berterlizing {stockFolder}...\n")
    if stockFolder[:2] != 'sz' and stockFolder[:2] != 'sh':
        return
    for txtFile in os.listdir(dataPath + '/stock_news/' + stockFolder):
        tmp = berterlize(dataPath + '/stock_news/' + stockFolder + '/' + txtFile)
        if not os.path.exists(dataPath + '/stockNewsVec_Bert/' + stockFolder):
            os.makedirs(dataPath + '/stockNewsVec_Bert/' + stockFolder)
        np.savetxt(dataPath + '/stockNewsVec_Bert/' + stockFolder + '/' + txtFile, tmp, delimiter=",")
    pass

from multiprocessing import Pool
from transformers import BertTokenizer, BertModel
import torch

dataPath = 'D:/stockdata_and_code/HAN'

tokenizer = BertTokenizer.from_pretrained('D:/stockdata_and_code/HAN/ChineseWord2Vec/FinBERT/')
model = BertModel.from_pretrained('D:/stockdata_and_code/HAN/ChineseWord2Vec/FinBERT/', return_dict=True)

id_list = os.listdir(dataPath + '/stock_news/')
from tqdm import tqdm
for i in tqdm(range(len(id_list))):
    stockFolder = id_list[i]
    process(stockFolder)