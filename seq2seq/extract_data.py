import sys, csv, requests, http.client, ast
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
http.client._MAXHEADERS = 1000

file = open("bible_books.txt", "r")
contents = file.read()
bible_dict = ast.literal_eval(contents)
file.close()

url_format="https://bible.fhl.net/new/read.php?VERSION19=sgebklcl&VERSION20=web&strongflag=1&TABFLAG=1&TAIU=2&chineses={book}&chap={chapter}&submit1=閱讀"


for book, chapter in bible_dict.items():
    for chap in range(1, chapter+1):
        url = url_format.format(book=book, chapter=chap)

        # Extract the raw HTML content
        html = requests.get(url)
        html.encoding = "utf-8"
        html_content = html.text

        # Parse the html content
        soup = BeautifulSoup(html_content, "lxml")
        table = soup.find("table", attrs={"border": "1"})
        # Remove titles in bold
        for bold in table.find_all(['h3', 'h4']):
            bold.decompose()

        # Use Pandas to manipulate data
        df = pd.read_html(str(table))[0]
        df.replace('"', '', regex=True, inplace=True)
        df.replace('  ', ' ', regex=True, inplace=True)
        df.drop(0, inplace=True)
        df.drop([df.columns[0], df.columns[3]], axis=1, inplace=True)
        df_eng = df.copy(deep=True)
        df_eng.drop(df_eng.columns[0], axis=1, inplace=True)
        df.drop(df.columns[1], axis=1, inplace=True)

        # Export data to text files
        np_tailo = df.to_numpy()
        np_eng = df_eng.to_numpy()
        with open('bible.tw','a', encoding='utf-8') as file:
            np.savetxt(file, np_tailo, fmt="%s")
        with open('bible.en','a') as f:
            np.savetxt(f, np_eng, fmt="%s")

        try:
            html = requests.get(url)
        except requests.exceptions.RequestException as e:
            raise SystemExit(e)
