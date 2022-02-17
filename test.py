import torch, math
import torch.nn as nn
from training_functions import evaluate
from build_model import build_model
from setup import get_data, get_iterators

input_dim = 28
output_dim = 37
TRG_PAD_IDX = 1
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
train_data, valid_data, test_data = get_data()
_, _, test_iterator, src_tw, trg_en = get_iterators(train_data, valid_data, test_data)
model = build_model(len(src_tw.vocab), len(trg_en.vocab))
model.load_state_dict(torch.load('translation-model.pt'))


test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
