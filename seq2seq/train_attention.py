import torch, sys
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from seq2seq_attention import Encoder, Decoder, Seq2Seq
from setup import get_data, get_iterators
from training_functions import count_parameters, epoch_time

### We're ready to define everything we need for training our Seq2Seq model ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = False
save_model = True
CURRENT_EPOCH = 0

# Training hyperparameters
# learning_rate = 3e-4
BATCH_SIZE = int(sys.argv[1])
N_EPOCHS = int(sys.argv[2])

train_data, valid_data, test_data = get_data()
train_iterator, valid_iterator, test_iterator, src_tw, trg_en = get_iterators(train_data, valid_data, test_data, BATCH_SIZE)

print(f"Unique tokens in source (tw) vocabulary: {len(src_tw.vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(trg_en.vocab)}")
# start training from checkpoint
# model.load_state_dict(torch.load(f'seq2seq_model_epoch{CURRENT_EPOCH}.pt'))

# Model hyperparameters
input_size_encoder = len(src_tw.vocab)
input_size_decoder = len(trg_en.vocab)
output_size = len(trg_en.vocab)
encoder_embedding_size = 256 # 300
decoder_embedding_size = 256 # 300
hidden_size = 512 # 1024
num_layers = 2
enc_dropout = 0.1
dec_dropout = 0.1

# Tensorboard to get nice loss plot
writer = SummaryWriter(f"runs/loss_plot")
step = 0

encoder_net = Encoder(
    input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout
).to(device)

decoder_net = Decoder(
    input_size_decoder,
    output_size,
    decoder_embedding_size,
    hidden_size,
    num_layers,
    dec_dropout,
).to(device)

model = Seq2Seq(encoder_net, decoder_net, device).to(device)
print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters())

PAD_IDX = trg_en.vocab.stoi[trg_en.pad_token] # ignore padding index when calculating loss
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)


def train(model, iterator, optimizer, criterion, clip):

    model.train()
    epoch_loss = 0

    # len(list(iterator)) = 198
    for i, batch in tqdm(enumerate(iterator), total=int(math.ceil(len(train_data) / BATCH_SIZE))):
        src = batch.src.to(device)
        trg = batch.trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg)

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

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

            output = model(src, trg, 0) # turn off teacher forcing
            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

CLIP = 1

best_valid_loss = float('inf')
best_train_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if train_loss < best_train_loss:
        best_train_loss = train_loss
        torch.save(model.state_dict(), f'seq2seq_model_epoch{epoch+1+CURRENT_EPOCH}.pt')
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), f'min_valid_epoch{epoch+1}.pt')

    print(f'Epoch: {epoch+1+CURRENT_EPOCH:02d} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}')


# pad_idx = trg_en.vocab.stoi["<pad>"]
# criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
#
# if load_model:
#     load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)
#
# # sentence = (
# #     "ein boot mit mehreren männern darauf wird von einem großen"
# #     "pferdegespann ans ufer gezogen."
# # )
#
# for epoch in range(num_epochs):
#     print(f"[Epoch {epoch} / {num_epochs}]")
#
#     if save_model:
#         checkpoint = {
#             "state_dict": model.state_dict(),
#             "optimizer": optimizer.state_dict(),
#         }
#         save_checkpoint(checkpoint)
#
#     model.eval()
#
#     # translated_sentence = translate_sentence(
#     #     model, sentence, src_tw, trg_en, device, max_length=50
#     # )
#     #
#     # print(f"Translated example sentence: \n {translated_sentence}")
#
#     model.train()
#
#     for batch_idx, batch in enumerate(train_iterator):
#         # Get input and targets and get to cuda
#         inp_data = batch.src.to(device)
#         target = batch.trg.to(device)
#
#         # Forward prop
#         output = model(inp_data, target)
#
#         # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
#         # doesn't take input in that form. For example if we have MNIST we want to have
#         # output to be: (N, 10) and targets just (N). Here we can view it in a similar
#         # way that we have output_words * batch_size that we want to send in into
#         # our cost function, so we need to do some reshapin. While we're at it
#         # Let's also remove the start token while we're at it
#         output = output[1:].reshape(-1, output.shape[2])
#         target = target[1:].reshape(-1)
#
#         optimizer.zero_grad()
#         loss = criterion(output, target)
#
#         # Back prop
#         loss.backward()
#
#         # Clip to avoid exploding gradient issues, makes sure grads are
#         # within a healthy range
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
#
#         # Gradient descent step
#         optimizer.step()
#
#         # Plot to tensorboard
#         writer.add_scalar("Training loss", loss, global_step=step)
#         step += 1

# running on entire test data takes a while
score = bleu(test_data[1:100], model, src_tw, trg_en, device)
print(f"Bleu score {score * 100:.2f}")
