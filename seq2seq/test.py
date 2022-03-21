import torch, math, sacrebleu, sys
import torch.nn as nn
from training_functions import evaluate
from build_model import build_model
from tokenizer import get_data, get_orig_data, get_iterators
from random import randrange
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

# src_tensor = src_tw.process([example.src]).to(device)
# trg_tensor = trg_en.process([example.trg]).to(device)
# print(trg_tensor.shape)

# model.eval()
# with torch.no_grad():
#     outputs = model(src_tensor, trg_tensor, teacher_forcing_ratio=0)
def translate_sentence(model, sentence, src_tw, trg_en, device, max_length=100):
    tokens = tokenize_tw(sentence)

    # print(tokens)

    # sys.exit()
    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, src_tw.init_token)
    tokens.append(src_tw.eos_token)

    # Go through each src_tw token and convert to an index
    text_to_indices = [src_tw.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor)

    outputs = [trg_en.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == trg_en.vocab.stoi["<eos>"]:
            break

    translated_sentence = [trg_en.vocab.itos[idx] for idx in outputs]

    # remove start token
    return translated_sentence[1:]

sentence = ("I ê tshàn-lān tshin-tshiūnn kng; tuì I ê tshiú ū tshut kng-suàⁿ, tī hia I ê lîng-li̍k khǹg-leh.")
predicted_translation = translate_sentence(model, sentence, src_tw, trg_en, device)

# output_idx = outputs[1:].squeeze(1).argmax(1)
# itos: A list of token strings indexed by their numerical identifiers.
# predicted_translation = ' '.join([trg_en.vocab.itos[idx] for idx in output_idx])
print('TRANSLATION: ', predicted_translation)

# preds = [trg_en.vocab.itos[idx] for idx in output_idx]
# preds = ['i', 'praise', 'and', 'worship', 'Jesus', 'for', 'who', 'He', 'is']
# refs = [['i', 'seek', 'and', 'worship', 'Jesus', 'for', 'who', 'He', 'is']]
# bleu = sacrebleu.corpus_bleu(preds, refs)
# print("BLEU: ", bleu.score)

# do not evaluate on test set until end of project
# test_loss = evaluate(model, test_iterator, criterion)
#
# print(f'Test Loss: {test_loss:.3f}')
