import torch, spacy, sys
from torchtext.data.metrics import bleu_score
from tokenizer import tokenize_tw

def translate_sentence(model, sentence, src_tw, trg_en, device, max_length=100):
    tokens = tokenize_tw(sentence)

    # Add start and end tokens
    tokens.insert(0, src_tw.init_token)
    tokens.append(src_tw.eos_token)

    text_to_indices = [src_tw.vocab.stoi[token] for token in tokens]
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    with torch.no_grad():
        outputs_encoder, hiddens, cells = model.encoder(sentence_tensor)

    outputs = [trg_en.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hiddens, cells = model.decoder(
                previous_word, outputs_encoder, hiddens, cells
            )
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # stop predicting when end tag is predicted
        if output.argmax(1).item() == trg_en.vocab.stoi["<eos>"]:
            break

    translated_sentence = [trg_en.vocab.itos[idx] for idx in outputs]

    # remove start token
    return translated_sentence[1:]


def bleu(data, model, src_tw, trg_en, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, src_tw, trg_en, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
