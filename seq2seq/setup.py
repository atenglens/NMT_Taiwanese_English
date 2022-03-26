import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 1234
np.random.seed(SEED)

tailo_txt = open('parallel_corpus/bible.tw', encoding='utf-8').read().split('\n')
eng_txt = open('parallel_corpus/bible.en', encoding='utf-8').read().split('\n')

# tailo_txt = open('data_preprocessing/pretrain_chinese/data/data.zh', encoding='utf-8').read().split('\n')
# eng_txt = open('data_preprocessing/pretrain_chinese/data/data.en', encoding='utf-8').read().split('\n')

raw_data = {'Tailo': [line for line in tailo_txt],
            'English': [line for line in eng_txt]}

df = pd.DataFrame(raw_data, columns=['Tailo', 'English'])

train, test = train_test_split(df, test_size=0.2)
valid, test = train_test_split(test, test_size=0.5)

train.to_csv('train.csv', index=False)
valid.to_csv('valid.csv', index=False)
test.to_csv('test.csv', index=False)
