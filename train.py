import torch, random, math, time, sys
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from tqdm import tqdm
from setup import get_data, get_iterators
from seq2seq import Encoder, Decoder, Seq2Seq
from training_functions import init_weights, count_parameters, epoch_time
from build_model import build_model

SEED = 1234
BATCH_SIZE = int(sys.argv[1])
N_EPOCHS = int(sys.argv[2])

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

train_data, valid_data, test_data = get_data()
train_iterator, valid_iterator, _, src_tw, trg_en = get_iterators(train_data, valid_data, test_data, BATCH_SIZE=128)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# TRG_PAD_IDX = trg_en.vocab.stoi[trg_en.pad_token]
# print(TRG_PAD_IDX)
print(f"Unique tokens in source (tw) vocabulary: {len(src_tw.vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(trg_en.vocab)}")
model = build_model(len(src_tw.vocab), len(trg_en.vocab))
print(f'The model has {count_parameters(model):,} trainable parameters')


optimizer = optim.Adam(model.parameters())

TRG_PAD_IDX = trg_en.vocab.stoi[trg_en.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)


def train(model, iterator, optimizer, criterion, clip):

    model.train()
    epoch_loss = 0

    # len(list(iterator)) = 198
    for i, batch in tqdm(enumerate(iterator), total=int(math.ceil(len(train_data) / BATCH_SIZE))):
        src = batch.src.to(device)
        trg = batch.trg.to(device)

        optimizer.zero_grad()

        output = model(src, trg)

        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss = (epoch_loss * i + loss.item()) / (i+1)

    return epoch_loss

def evaluate(model, iterator, criterion):

    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src.to(device)
            trg = batch.trg.to(device)

            output = model(src, trg, 0) #turn off teacher forcing

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS=30):

    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), f'seq2seq_6model_epoch{epoch+1}.pt')

    print(f'Epoch: {epoch+1:02d} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}')
