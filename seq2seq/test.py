import torch, math, sacrebleu, sys
import torch.nn as nn
from training_functions import evaluate
from build_model import build_model
from tokenizer import get_data, get_orig_data, get_iterators
from random import randrange
from tokenizer import tokenize_tw
# from utils import translate_sentence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data, valid_data, test_data = get_data()

model = build_model(len(src_tw.vocab), len(trg_en.vocab))

# running on entire test data takes a while
score = bleu(test_data[1:100], model, src_tw, trg_en, device)
print(f"Bleu score {score * 100:.2f}")
