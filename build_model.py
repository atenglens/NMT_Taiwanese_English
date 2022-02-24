import torch
from seq2seq import Encoder, Decoder, Seq2Seq
from torchtext.legacy.data import BucketIterator
from training_functions import init_weights

def build_model(input_dim, output_dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    INPUT_DIM, OUTPUT_DIM = input_dim, output_dim
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 8
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    model = Seq2Seq(enc, dec, device).to(device)

    model.apply(init_weights)
    return model
