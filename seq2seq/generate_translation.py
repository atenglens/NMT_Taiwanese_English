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

# model.load_state_dict(torch.load('seq2seq_model_epoch30.pt'))

example_idx = randrange(len(og_valid_data.examples))
example = valid_data.examples[example_idx]
og_example = og_valid_data.examples[example_idx]
orig_source = ' '.join(og_example.src_orig)
print('ORIG SOURCE: ', orig_source)
orig_target = ' '.join(og_example.trg_orig)
print('ORIG TARGET: ', orig_target)
preprocessed_source = ' '.join(example.src[::-1])
print('TOKENIZED SOURCE: ', preprocessed_source)
preprocessed_target = ' '.join(example.trg)
refs = example.trg
print('TOKENIZED TARGET: ', preprocessed_target)

src_tensor = src_tw.process([example.src]).to(device)
trg_tensor = trg_en.process([example.trg]).to(device)
# print(trg_tensor.shape)

model.eval()
with torch.no_grad():
    outputs = model(src_tensor, trg_tensor, teacher_forcing_ratio=0)

output_idx = outputs[1:].squeeze(1).argmax(1)
itos: A list of token strings indexed by their numerical identifiers.
predicted_translation = ' '.join([trg_en.vocab.itos[idx] for idx in output_idx])
print('TRANSLATION: ', predicted_translation)

# do not evaluate on test set until end of project
# test_loss = evaluate(model, test_iterator, criterion)
#
# print(f'Test Loss: {test_loss:.3f}')
