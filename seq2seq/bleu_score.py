###### @atenglens ######
# Adapted from https://blog.machinetranslation.io/compute-bleu-score/
########################

import sacrebleu
from sacremoses import MosesDetokenizer

md = MosesDetokenizer(lang='en')


# Open the test dataset human translation file and detokenize the references
refs = []

with open("target_translation.txt") as test:
    for line in test:
        line = line.strip().split()
        line = md.detokenize(line)
        refs.append(line)

# print("TARGET: ", refs[0])

refs = [refs]  # Yes, it is a list of list(s) as required by sacreBLEU


# Open the translation file by the NMT model and detokenize the predictions
preds = []

with open("predicted_translation.txt") as pred:
    for line in pred:
        line = line.strip().split()
        line = md.detokenize(line)
        preds.append(line)

# print("PREDICTED: ", preds[0])


# Calculate and print the BLEU score
bleu = sacrebleu.corpus_bleu(preds, refs)
print(bleu.score)
