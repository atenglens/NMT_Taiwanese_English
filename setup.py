import torch, random, spacy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchtext.legacy.data import Field, TabularDataset, BucketIterator


SEED = 1234
np.random.seed(SEED)

tailo_txt = open('bible.tw', encoding='utf-8').read().split('\n')
eng_txt = open('bible.en', encoding='utf-8').read().split('\n')

raw_data = {'Tailo': [line for line in tailo_txt],
            'English': [line for line in eng_txt]}

df = pd.DataFrame(raw_data, columns=['Tailo', 'English'])

train, test = train_test_split(df, test_size=0.2)
valid, test = train_test_split(test, test_size=0.5)

train.to_csv('train.csv', index=False)
valid.to_csv('valid.csv', index=False)
test.to_csv('test.csv', index=False)

spacy_en = spacy.load('en_core_web_sm')

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_tw(text):
    """
    Tokenizes Taiwanese text on spaces and returns reversed sequence.
    """
    return text.replace('-', ' ').split()[::-1]

def get_fields():
    src_tw = Field(tokenize = tokenize_tw, init_token = '<sos>', eos_token = '<eos>', lower = True)
    trg_en = Field(tokenize = tokenize_en, init_token = '<sos>', eos_token = '<eos>', lower = True)
    return src_tw, trg_en

src_tw, trg_en = get_fields()

fields = {'Tailo': ('src', src_tw), 'English': ('trg', trg_en)}

def get_data(train="train.csv", valid="valid.csv", test="test.csv"):
    train_data, valid_data, test_data = TabularDataset.splits(
        path='',
        train='train.csv',
        validation='valid.csv',
        test='test.csv',
        format='csv',
        fields=fields)
    return train_data, valid_data, test_data

def get_iterators(train_data, valid_data, test_data, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src_tw.build_vocab(train_data, min_freq = 2)
    trg_en.build_vocab(train_data, min_freq = 2)
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size = batch_size,
        sort_key=lambda x: len(x.src),
        sort_within_batch=False,
        device = device)
    return train_iterator, valid_iterator, test_iterator, src_tw, trg_en
