import ast

file = open("search_names_tw.txt", "r")
contents = file.read()
name_mapping = ast.literal_eval(contents)
file.close()

selected_tw_index = 0
for en, tw in name_mapping.items():
    if len(tw) == 1:
        name_mapping[en] = list(name_mapping.get(en))[0]
    else:
        print(en + ": " + str(tw))
        while True:
            try:
                selected_tw_index = int(input("Manual Selection: "))
                if selected_tw_index == 0:
                    name_mapping[en] = input("Enter name: ")
                    break
                elif selected_tw_index < len(tw)+1:
                    name_mapping[en] = list(name_mapping.get(en))[selected_tw_index-1]
                    break
            except:
                selected_tw_index = int(input("Manual Selection: "))
                continue

print(name_mapping)
file = open("bible_names_mapping.txt", "w")
file.write('{\n')
for item in dict(name_mapping).items():
    file.write(f'\"{item[0]}\": \"{item[1]}\",\n')
file.write('}')
file.close()
