import csv
import os
import re
import pandas
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tika import parser
import constants
from constants import *
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer

factory = StemmerFactory()
stemmer = factory.create_stemmer()

INPUT_FILE = "./tweets.csv"
OUTPUT_FILE = "./output.csv"

count_vectorizer = CountVectorizer(
    stop_words=stopwords.words('indonesian')
)

tfidf_transformer = TfidfTransformer(
)

tfidf_vectorizer = TfidfVectorizer(
    stop_words=stopwords.words('indonesian')
)

def clean_text(input_text: str) -> str:
    # Filtering, menghapus semua karakter non teks
    input_text = re.sub(r'\W', ' ', input_text)

    # Menghapus semua karakter tunggal pada bagian tengah teks
    input_text = re.sub(r'\s+[a-zA-Z]\s+', ' ', input_text)

    # Menghapus semua karakter tunggal pada awal teks
    input_text = re.sub(r'\^[a-zA-Z]\s+', ' ', input_text)

    # Mengganti spasi berurutan dengan ' '
    input_text = re.sub(r'\s+', ' ', input_text, flags=re.I)

    # Case folding
    input_text = input_text.lower()

    # Stemming
    input_text = stemmer.stem(input_text)
    return input_text

def normalize_target(target_text: str) -> str:
    lower_first_char = target_text[0].lower()
    
    if (lower_first_char not in ['h', 'f']):
        raise ValueError("Incorrect value for target: {}".format(target_text))

    return lower_first_char

input_data_frame = pandas.read_csv(
    INPUT_FILE,
    header=None,
    skiprows=[0]
)

input_data_frame[DATA_KEY] = input_data_frame[INPUT_DATA_INDEX].apply(
    clean_text
)

input_data_frame[TARGET_KEY] = input_data_frame[INPUT_TARGET_INDEX].apply(
    normalize_target
)

input_data_frame[
    [DATA_KEY, TARGET_KEY]
].to_csv(
    PREPROCESSED_DATA_FILE
)