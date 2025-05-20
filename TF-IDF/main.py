import os
import re
import glob
import shutil
import subprocess
import platform

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords as nltk_stopwords
from pymystem3 import Mystem


# Mystem()  # would download mystem to ~/.local/bin
MYSTEM_PATH = os.path.expanduser(os.path.join("~", ".local", "bin", "mystem.exe" if platform.system() == "Windows" else "mystem"))

OUTPUT_ROWS_NUM = 100

SRC_FOLDER = "plays_tei"

PROPER_NOUNS_FOLDER = os.path.join("tartu", "proper_nouns")
PROPER_NOUNS_BY_PLAY_FOLDER = os.path.join("tartu", "proper_nouns", "by_play")

EXTRACT_CAST_ITEM_BY_PLAY_FOLDER = os.path.join("tartu", "cast_item", "by_play")

EXTRACT_SPEECH_FOLDER = os.path.join("tartu", "01_extract_speech")
STRIP_PUNCTUATION_FOLDER = os.path.join("tartu", "02_strip_punctuation")
REMOVE_PROPER_NOUNS_FOLDER = os.path.join("tartu", "03_remove_proper_nouns")
LEMMATIZE_FOLDER = os.path.join("tartu", "04_lemmatize")
GROUPED_FOLDER = os.path.join("tartu", "05_speech_grouped")

XLSX_PATH = os.path.join("tartu", "tf_idf.xlsx")

PLAY_GROUPS_MAPPING = {
     'Аттестат зрелости.xml.txt': 'Childrens-plays.txt',
     'Борьба без линии фронта.xml.txt': 'Historical-plays.txt',
     'В одном городе.xml.txt': 'Factory-plays.txt',
     'Вас вызывает Таймыр.xml.txt': 'Prohibited-plays.txt',
     'Васек Трубачев и его товарищи.xml.txt': 'Childrens-plays.txt',
     'Великая сила.xml.txt': 'Patriotic-plays.txt',
     'Великий государь.xml.txt': 'Great-Past-plays.txt',
     'Вперед, отважные!.xml.txt': 'Childrens-plays.txt',
     'Вынужденная посадка.xml.txt': 'Prohibited-plays.txt',
     'Высота 88.xml.txt': 'Childrens-plays.txt',
     'Гастелло.xml.txt': 'War-plays.txt',
     'Глина и фарфор.xml.txt': 'Factory-plays.txt',
     'Голос Америки.xml.txt': 'West-plays.txt',
     'Два друга.xml.txt': 'Childrens-plays.txt',
     'Ее друзья.xml.txt': 'Childrens-plays.txt',
     'Жизнь в цитадели.xml.txt': 'Intelligentsia-plays.txt',
     'За тех, кто в море.xml.txt': 'War-plays.txt',
     'Заговор обреченных.xml.txt': 'Historical-plays.txt',
     'Зеленая улица.xml.txt': 'Patriotic-plays.txt',
     'Илья Головин.xml.txt': 'Intelligentsia-plays.txt',
     'Калиновая роща.xml.txt': 'Kolhoz-plays.txt',
     'Канун грозы.xml.txt': 'Historical-plays.txt',
     'Красный галстук.xml.txt': 'Childrens-plays.txt',
     'Макар Дубрава.xml.txt': 'Factory-plays.txt',
     'Московский характер.xml.txt': 'Factory-plays.txt',
     'На новой земле.xml.txt': 'Kolhoz-plays.txt',
     'Незабываемый 1919.xml.txt': 'Historical-plays.txt',
     'Новые времена.xml.txt': 'Kolhoz-plays.txt',
     'Окно в лесу.xml.txt': 'Prohibited-plays.txt',
     'Особое задание.xml.txt': 'Childrens-plays.txt',
     'Павлик Морозов.xml.txt': 'Childrens-plays.txt',
     'Письмо в редакцию.xml.txt': 'Childrens-plays.txt',
     'Победители.xml.txt': 'War-plays.txt',
     'Потопленные камни.xml.txt': 'Historical-plays.txt',
     'Поют жаворонки.xml.txt': 'Kolhoz-plays.txt',
     'Рассвет над Москвой.xml.txt': 'Factory-plays.txt',
     'Русский вопрос.xml.txt': 'West-plays.txt',
     'Свадьба с приданым.xml.txt': 'Kolhoz-plays.txt',
     'Семейный совет.xml.txt': 'Childrens-plays.txt',
     'Семья Аллана.xml.txt': 'Kolhoz-plays.txt',
     'Снежок.xml.txt': 'Childrens-plays.txt',
     'Совесть.xml.txt': 'Factory-plays.txt',
     'Старые друзья.xml.txt': 'War-plays.txt',
     'Суворовцы.xml.txt': 'Childrens-plays.txt',
     'Третья молодость.xml.txt': 'Patriotic-plays.txt',
     'У лесного озера.xml.txt': 'Childrens-plays.txt',
     'Флаг адмирала.xml.txt': 'Great-Past-plays.txt',
     'Хлеб наш насущный.xml.txt': 'Kolhoz-plays.txt',
     'Чрезвычайный закон.xml.txt': 'Prohibited-plays.txt',
     'Чужая тень.xml.txt': 'Patriotic-plays.txt',
     'Я хочу домой.xml.txt': 'Childrens-plays.txt',
}

# TODO: use xml reader/parser
def extract_speech():
    for root, dirs, files in os.walk(SRC_FOLDER):
        for filename in files:
            if filename.endswith('.xml'):
                src_filepath = os.path.join(root, filename)
                dst_filepath = os.path.join(EXTRACT_SPEECH_FOLDER, filename) + '.txt'

                print('extract_speech', src_filepath, '->', dst_filepath)

                with open(src_filepath, 'r', encoding='utf-8') as file:
                    speech = []

                    for line in file:
                        line = re.sub('\s*<stage>.*?</stage>', '', line)
                        line = re.sub('\s*<note.*?>.*?</note>', '', line)

                        match = re.search('<p>(.+?)</p>', line)

                        if match:
                            speech.append(match.group(1))

                        match = re.search('<l>(.+?)</l>', line)

                        if match:
                            speech.append(match.group(1))

                    with open(dst_filepath, 'w', encoding='utf-8') as output:
                        output.writelines(line + '\n' for line in speech)


def __extract_cast_item():
    for root, dirs, files in os.walk(SRC_FOLDER):
        for filename in files:
            if filename.endswith('.xml'):
                src_filepath = os.path.join(root, filename)
                dst_filepath = os.path.join(EXTRACT_CAST_ITEM_BY_PLAY_FOLDER, filename) + '.txt'

                print('__extract_cast_item', src_filepath, '->', dst_filepath)

                with open(src_filepath, 'r', encoding='utf-8') as file:
                    cast_items = []

                    for line in file:
                        match = re.search('<castItem>(.+?)</castItem>', line)

                        if match:
                            cast_items.append(match.group(1))

                    with open(dst_filepath, 'w', encoding='utf-8') as output:
                        output.writelines(line + '\n' for line in cast_items)


def __extract_capitalized_words():
    for filename in os.listdir(EXTRACT_SPEECH_FOLDER):
        if filename.endswith('.xml.txt'):
            src_filepath = os.path.join(EXTRACT_SPEECH_FOLDER, filename)
            dst_filepath = os.path.join(PROPER_NOUNS_BY_PLAY_FOLDER, filename)

            print('__extract_capitalized_words', src_filepath, '->', dst_filepath)

            content = Path(src_filepath).read_text(encoding='utf-8')
            words = re.findall(r'\b[A-ZА-Я]\S+\b', content)

            Path(dst_filepath).write_text('\n'.join(sorted(set(words))), encoding='utf-8')

    tmp1_filepath = os.path.join(PROPER_NOUNS_BY_PLAY_FOLDER, "1.txt")  # because windows cmd cannot unicode
    tmp2_filepath = os.path.join(PROPER_NOUNS_BY_PLAY_FOLDER, "2.txt")  # because windows cmd cannot unicode

    for filename in os.listdir(PROPER_NOUNS_BY_PLAY_FOLDER):
        if filename.endswith('.xml.txt'):
            filepath = os.path.join(PROPER_NOUNS_BY_PLAY_FOLDER, filename)
            print('__extract_capitalized_words mystem', filepath)
            shutil.copyfile(filepath, tmp1_filepath)
            subprocess.run([MYSTEM_PATH, "-ci", tmp1_filepath, tmp2_filepath], check=True, capture_output=True)
            shutil.copyfile(tmp2_filepath, filepath)

    if os.path.isfile(tmp1_filepath):
        os.remove(tmp1_filepath)

    if os.path.isfile(tmp2_filepath):
        os.remove(tmp2_filepath)

    for filename in os.listdir(PROPER_NOUNS_BY_PLAY_FOLDER):
        if filename.endswith('.xml.txt'):
            filepath = os.path.join(PROPER_NOUNS_BY_PLAY_FOLDER, filename)

            print('__extract_capitalized_words filter', filepath)

            name_lines = []
            other_lines = []

            content = Path(filepath).read_text(encoding='utf-8')
            lines = content.split('\n')

            for line in lines:
                if 'имя' in line or 'фам' in line or 'отч' in line:
                    name_lines.append(line)  # or: filename.ljust(45) + ' | ' + line
                else:
                    other_lines.append(line)

            lines = sorted(name_lines) + [''] * 3 + other_lines

            Path(filepath).write_text('\n'.join(lines), encoding='utf-8')


def strip_punctuation():
    for filename in os.listdir(EXTRACT_SPEECH_FOLDER):
        if filename.endswith('.xml.txt'):
            src_filepath = os.path.join(EXTRACT_SPEECH_FOLDER, filename)
            dst_filepath = os.path.join(STRIP_PUNCTUATION_FOLDER, filename)

            print('strip_punctuation', src_filepath, '->', dst_filepath)

            content = Path(src_filepath).read_text(encoding='utf-8')
            content = re.sub('[^\w\s-]', '', content)  # remove everything except word characters (a-z, A-Z, 0-9, _) and whitespaces
            content = re.sub('\s+[-–—]\s+', ' ', content)  # remove hypen, em, en dashes surrounded by whitespace -- replace them with single space
            content = re.sub(' {2,}', ' ', content)  # replace multiple spaces with one
            content = re.sub('\n +', '\n', content)  # replace spaces at the beginning of lines
            Path(dst_filepath).write_text(content, encoding='utf-8')


def remove_proper_nouns():
    proper_nouns = set()

    for filename in os.listdir(PROPER_NOUNS_FOLDER):
        if filename.endswith('.txt'):
            filepath = os.path.join(PROPER_NOUNS_FOLDER, filename)
            content = Path(filepath).read_text(encoding='utf-8')
            words = re.sub(r'{.*?}', '', content).split('\n')
            proper_nouns.update(words)

    for filename in os.listdir(STRIP_PUNCTUATION_FOLDER):
        if filename.endswith('.xml.txt'):
            src_filepath = os.path.join(STRIP_PUNCTUATION_FOLDER, filename)
            dst_filepath = os.path.join(REMOVE_PROPER_NOUNS_FOLDER, filename)

            print('remove proper nouns', src_filepath, '->', dst_filepath)

            lines = Path(src_filepath).read_text(encoding='utf-8')
            lines = lines.split('\n')
            lines = [line.split(' ') for line in lines]
            lines = [[word for word in words if word not in proper_nouns] for words in lines]  # or: lines = [[word if word not in proper_nouns else 'ИМЯ' for word in words] for words in lines]
            lines = [' '.join(words) for words in lines]
            lines = '\n'.join(lines)
            Path(dst_filepath).write_text(lines, encoding='utf-8')


def lemmatize():
    tmp1_filepath = os.path.join(LEMMATIZE_FOLDER, "1.txt")  # because windows cmd cannot unicode
    tmp2_filepath = os.path.join(LEMMATIZE_FOLDER, "2.txt")  # because windows cmd cannot unicode

    for filename in os.listdir(REMOVE_PROPER_NOUNS_FOLDER):
        if filename.endswith('.xml.txt'):
            src_filepath = os.path.join(REMOVE_PROPER_NOUNS_FOLDER, filename)
            dst_filepath = os.path.join(LEMMATIZE_FOLDER, filename)

            print('lemmatize', src_filepath, '->', dst_filepath)

            shutil.copyfile(src_filepath, tmp1_filepath)
            subprocess.run([MYSTEM_PATH, "-lcd", tmp1_filepath, tmp2_filepath], check=True, capture_output=True)
            shutil.copyfile(tmp2_filepath, dst_filepath)

    if os.path.isfile(tmp1_filepath):
        os.remove(tmp1_filepath)

    if os.path.isfile(tmp2_filepath):
        os.remove(tmp2_filepath)

    for filename in os.listdir(LEMMATIZE_FOLDER):
        if filename.endswith('.xml.txt'):
            filepath = os.path.join(LEMMATIZE_FOLDER, filename)
            file = Path(filepath)
            content = file.read_text(encoding='utf-8')
            content = re.sub('[^\w\s-]', '', content)  # remove everything except word characters (a-z, A-Z, 0-9, _) and whitespaces
            file.write_text(content, encoding='utf-8')


def group_plays():
    groups = {}

    for play_filename, group_filename in PLAY_GROUPS_MAPPING.items():
        groups[group_filename] = groups.get(group_filename, [])
        groups[group_filename].append(play_filename)

    print('group_plays')

    for group_filename in groups.keys():
        play_filepaths = [os.path.join(LEMMATIZE_FOLDER, play_filename) for play_filename in groups[group_filename]]
        content = [Path(filepath).read_text(encoding='utf-8') for filepath in play_filepaths]
        content = '\n'.join(content)

        group_filepath = os.path.join(GROUPED_FOLDER, group_filename)
        Path(group_filepath).write_text(content, encoding='utf-8')


def calculate_tf_idf():
    text_files = glob.glob(f"{GROUPED_FOLDER}/*.txt")
    text_files = [Path(file_path) for file_path in text_files]
    text_titles = [path.stem for path in text_files]
    text_files = [path.read_text(encoding='utf-8') for path in text_files]

    stopwords = nltk_stopwords.words('russian')
    tfidf_vectorizer = TfidfVectorizer(
        input='content',
        sublinear_tf=False,
        smooth_idf=False,
        norm=None,
        use_idf=True,
        stop_words=stopwords,
        preprocessor=None,
        token_pattern=r"(?u)[\w-]+",  # or: r"(?u)\S\S+"
        max_df=0.9,
        min_df=1
    )
    X = tfidf_vectorizer.fit_transform(text_files)

    print('calculate_tf_idf', len(tfidf_vectorizer.get_feature_names_out()))

    tfidf_df = pd.DataFrame(X.toarray(), index=text_titles, columns=tfidf_vectorizer.get_feature_names_out())
    tfidf_df = tfidf_df.stack().reset_index()
    tfidf_df = tfidf_df.rename(columns={0: 'tf_idf', 'level_0': 'document', 'level_1': 'term', 'level_2': 'term'})

    result_df = tfidf_df.sort_values(by=['document', 'tf_idf'], ascending=[True, False]).groupby(['document']).head(OUTPUT_ROWS_NUM)
    result_df.to_excel(XLSX_PATH)

    # with pd.option_context('display.max_rows', OUTPUT_ROWS_NUM):
    #     print(result_df)


def main():
    extract_speech()
    strip_punctuation()
    remove_proper_nouns()
    lemmatize()
    group_plays()
    calculate_tf_idf()


if __name__ == '__main__':
    main()