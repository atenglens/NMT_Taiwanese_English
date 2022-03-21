import torch, math, sacrebleu, sys
import torch.nn as nn
from training_functions import evaluate
from build_model import build_model
from tokenizer import get_data, get_orig_data, get_iterators
from random import randrange
from tokenizer import tokenize_tw
# from utils import translate_sentence

BATCH_SIZE = int(sys.argv[1])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data, valid_data, test_data = get_data()
og_train_data, og_valid_data, og_test_data = get_orig_data()
_, _, test_iterator, src_tw, trg_en = get_iterators(train_data, valid_data, test_data, BATCH_SIZE)
PAD_IDX = trg_en.vocab.stoi[trg_en.pad_token] # ignore padding index when calculating loss
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

model = build_model(len(src_tw.vocab), len(trg_en.vocab))

# running on entire test data takes a while
score = bleu(test_data[1:100], model, src_tw, trg_en, device)
print(f"Bleu score {score * 100:.2f}")
