import ast, csv

file = open("bible_names_mapping.txt", "r")
contents = file.read()
name_mapping = ast.literal_eval(contents)
file.close()

with open('train_original.csv', "r") as read_file, open('train.csv', "w") as write_file:
    csv_writer = csv.writer(write_file, delimiter=',')
    csv_reader = csv.reader(read_file, delimiter=',')

    line_count = 0
    for row in csv_reader:
        for tw in name_mapping.values():
            if tw in row[0]:
                unhyphenated = tw.replace("-", "")
                row[0] = row[0].replace(tw, unhyphenated)
        csv_writer.writerow([row[0], row[1]])
        line_count += 1

# selected_tw_index = 0
# for en, tw in name_mapping.items():
#     if len(tw) == 1:
#         name_mapping[en] = list(name_mapping.get(en))[0]
#     else:
#         print(en + ": " + str(tw))
#         while True:
#             try:
#                 selected_tw_index = int(input("Manual Selection: "))
#                 if selected_tw_index == 0:
#                     name_mapping[en] = input("Enter name: ")
#                     break
#                 elif selected_tw_index < len(tw)+1:
#                     name_mapping[en] = list(name_mapping.get(en))[selected_tw_index-1]
#                     break
#             except:
#                 selected_tw_index = int(input("Manual Selection: "))
#                 continue
#
# print(name_mapping)
# file = open("bible_names_mapping.txt", "w")
# file.write('{\n')
# for item in dict(name_mapping).items():
#     file.write(f'\"{item[0]}\": \"{item[1]}\",\n')
# file.write('}')
# file.close()
