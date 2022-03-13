import torch
import numpy as np
from datasets import load_dataset
from transformers import BartTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

data_files = {"train": "train.csv", "validation": "valid.csv", "test": "test.csv"}
dataset = load_dataset('csv', data_files=data_files)

train = load_dataset(dataset, split='train')

# generator that will yield batches of 1,000 texts to train tokenizer
def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]


tokenizer_en = BartTokenizerFast.from_pretrained("facebook/bart-base")

tokenizer_tw = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

pre_tokenizer_tw = pre_tokenizers.Sequence([Whitespace(), CharDelimiterSplit('-')])
tokenizer_tw.pre_tokenizer = pre_tokenizer_tw
