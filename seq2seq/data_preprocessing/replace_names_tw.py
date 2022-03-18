import ast, csv

file = open("bible_names_mapping.txt", "r")
contents = file.read()
name_mapping = ast.literal_eval(contents)
file.close()

with open("data_preprocessing/bible_original.tw", "r") as reader:
    read_lines = reader.readlines()

writer = open("parallel_corpus/bible.tw", "w")
for row in read_lines:
    for tw in name_mapping.values():
        if tw in row:
            unhyphenated = tw.replace("-", "")
            row = row.replace(tw, unhyphenated)
    writer.write(row)
writer.close()
