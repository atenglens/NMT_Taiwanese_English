file_en = open("data.en", "a")  # append mode
file_zh = open("data.zh", "a")  # append mode

file = open("Bi-News.txt", "r")
count = 0

while True:
    count += 1

    # Get next line from file
    line = file.readline()
    if count % 2 == 1:
        file_en.write(line)
    else:
        file_zh.write(line)

    # if line is empty
    # end of file is reached
    if not line:
        break

file_en.close()
file_zh.close()
file.close()
