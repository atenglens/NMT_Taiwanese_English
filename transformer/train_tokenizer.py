# import torch
# import numpy as np
# from datasets import load_dataset
from transformers import BartTokenizerFast
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE, Unigram
from tokenizers.trainers import BpeTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace, CharDelimiterSplit

# data_files = {"train": "train.csv", "validation": "valid.csv", "test": "test.csv"}
# dataset = load_dataset('csv', data_files=data_files)
#
# train = load_dataset(dataset, split='train')


# tokenizer_en = BartTokenizerFast.from_pretrained("facebook/bart-base")
tokenizer_en = BartTokenizerFast.from_pretrained("facebook/bart-base")



# tokenizer_tw = Tokenizer(BPE(unk_token="[UNK]"))
# trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer_tw = Tokenizer(Unigram())
trainer = UnigramTrainer(special_tokens=["[unk]", "[pad]", "[/s]"])

pre_tokenizer_tw = Whitespace() # pre_tokenizers.Sequence([Whitespace(), CharDelimiterSplit('-')])
tokenizer_tw.pre_tokenizer = pre_tokenizer_tw

bible_tw = ['data/bible.tw']
tokenizer_tw.train(bible_tw, trainer)

tokenizer_tw.save("data/uni_tokenizer_tw.json")

test_tokenizer_tw = Tokenizer.from_file("data/uni_tokenizer_tw.json")

output = test_tokenizer_tw.encode("Ū ê-hng, ū tsá-khí, sī tē-saⁿ ji̍t.")

print(output.tokens)
