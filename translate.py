import torch
from setup import get_data, get_iterators
from build_model import build_model
from torchtext.data.metrics import bleu_score

train_data, valid_data, test_data = get_data()
train_iterator, valid_iterator, _, src_tw, trg_en = get_iterators(train_data, valid_data, test_data)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_model(len(src_tw.vocab), len(trg_en.vocab))
# model.load_state_dict(torch.load('translation-model.pt'))

example_idx = 0
example = train_data.examples[example_idx]
print('SOURCE: ', ' '.join(example.src))
target_translation = example.trg
print(target_translation)
print('TARGET: ', ' '.join(target_translation))

src_tensor = src_tw.process([example.src]).to(device)
trg_tensor = trg_en.process([example.trg]).to(device)

model.eval()
with torch.no_grad():
    outputs = model(src_tensor, trg_tensor, teacher_forcing_ratio=0)

output_idx = outputs[1:].squeeze(1).argmax(1)
predicted_translation = [trg_en.vocab.itos[idx] for idx in output_idx]
print('TRANSLATION: ', ' '.join(predicted_translation))
print(predicted_translation)
print('BLEU SCORE: ', bleu_score([predicted_translation], [[target_translation]]))
