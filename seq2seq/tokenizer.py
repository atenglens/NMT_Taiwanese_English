import torch, spacy, re
from torchtext.legacy.data import Field, TabularDataset, BucketIterator

spacy_en = spacy.load('en_core_web_sm')
#spacy_zh = spacy.load('zh_core_web_lg')

# def tokenize_en(text):
#     specialChars = ".,:;?![]\"()"
#     for specialChar in specialChars:
#         text = text.replace(specialChar, '')
#     text = text.replace('-', ' ')
#     return text.split()

def tokenize_en(text):
    specialChars = ",:;?![]\"()"
    for specialChar in specialChars:
        text = text.replace(specialChar, '')
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def tokenize_tw(text):
    """
    Tokenizes Taiwanese text on spaces and returns reversed sequence.
    """
    # specialChars = "—[]()"
    specialChars = ",:;?![]\"()"
    for specialChar in specialChars:
        text = text.replace(specialChar, '')
    text = text.replace('-', ' ')
    isolate_periods = re.split(r'([.])', text)
    strip = [x.lower().strip() for x in isolate_periods]
    return strip[::-1]

# def tokenize_tw(text):
#     """
#     Tokenizes Taiwanese text on spaces and returns reversed sequence.
#     """
#     # specialChars = "—.,:;?![]\"\'()"
#     # for specialChar in specialChars:
#     #     text = text.replace(specialChar, '')
#     text = text.replace('-', ' ')
#     return text.split()[::-1]

def get_fields():
    src_tw = Field(tokenize = tokenize_tw, init_token = '<sos>', eos_token = '<eos>', lower = True)
    trg_en = Field(tokenize = tokenize_en, init_token = '<sos>', eos_token = '<eos>', lower = True)
    src_orig = Field(init_token = '<sos>', eos_token = '<eos>', lower = True)
    trg_orig = Field(init_token = '<sos>', eos_token = '<eos>', lower = True)
    return src_tw, trg_en, src_orig, trg_orig

src_tw, trg_en, src_orig, trg_orig = get_fields()

fields = {'Tailo': ('src', src_tw), 'English': ('trg', trg_en)}
orig_fields =  {'Tailo': ('src_orig', src_orig), 'English': ('trg_orig', trg_orig)}

def get_data(train="train.csv", valid="valid.csv", test="test.csv"):
    train_data, valid_data, test_data = TabularDataset.splits(
        path='',
        train='train.csv',
        validation='valid.csv',
        test='test.csv',
        format='csv',
        fields=fields)
    return train_data, valid_data, test_data

def get_orig_data(train="train.csv", valid="valid.csv", test="test.csv"):
    train_data, valid_data, test_data = TabularDataset.splits(
        path='',
        train='train.csv',
        validation='valid.csv',
        test='test.csv',
        format='csv',
        fields=orig_fields)
    return train_data, valid_data, test_data

def get_iterators(train_data, valid_data, test_data, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src_tw.build_vocab(train_data, min_freq = 2)
    trg_en.build_vocab(train_data, min_freq = 2)
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size = batch_size,
        sort_key=lambda x: len(x.src),
        sort_within_batch=False,
        device = device)
    return train_iterator, valid_iterator, test_iterator, src_tw, trg_en
