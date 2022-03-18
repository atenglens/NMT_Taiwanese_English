import csv
from collections import defaultdict
import numpy as np

with open('bible_names_en.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    names_en = []

    for row in csv_reader:
        names_en.append(str(np.array(row).squeeze()))
        line_count += 1

# names_en = ['Aaron', 'Israel']
with open('data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    name_mapping = defaultdict(set)

    for row in csv_reader:
        possible_translations = []
        for name in names_en:
            if not name_mapping.get(name):
                if name in row[1].split():
                    for word in row[0].split():
                        if word.startswith(name[0]):
                            name_mapping[name].add(word)
        line_count += 1
    # print(dict(name_mapping))

print('{')
for item in dict(name_mapping).items():
    print(f'\"{item[0]}\": {item[1]},')
print('}')
