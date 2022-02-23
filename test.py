import torch, math
import torch.nn as nn
from training_functions import evaluate
from build_model import build_model
from setup import get_data, get_iterators
from torchtext.data.metrics import bleu_score

input_dim = 28
output_dim = 37
TRG_PAD_IDX = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
train_data, valid_data, test_data = get_data()
_, _, test_iterator, src_tw, trg_en = get_iterators(train_data, valid_data, test_data)
model = build_model(len(src_tw.vocab), len(trg_en.vocab))
model.load_state_dict(torch.load('translation-model.pt'))

example_idx = 0
example = train_data.examples[example_idx]
print('SOURCE: ', ' '.join(example.src))
target_translation = ' '.join(example.trg)
print('TARGET: ', target_translation)

src_tensor = src_tw.process([example.src]).to(device)
trg_tensor = trg_en.process([example.trg]).to(device)
# print(trg_tensor.shape)

model.eval()
with torch.no_grad():
    outputs = model(src_tensor, trg_tensor, teacher_forcing_ratio=0)

# print(outputs.shape)

output_idx = outputs[1:].squeeze(1).argmax(1)
# itos: A list of token strings indexed by their numerical identifiers.
predicted_translation = ' '.join([trg_en.vocab.itos[idx] for idx in output_idx])
print('TRANSLATION: ', predicted_translation)
# print('BLEU SCORE: ', bleu_score(predicted_translation, target_translation))
# test_loss = evaluate(model, test_iterator, criterion)
#
# print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
