import re

import pandas
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

from constants import *
from util import chunks

factory = StemmerFactory()
stemmer = factory.create_stemmer()

INPUT_FILE = "./tweets.csv"
OUTPUT_FILE = "./output.csv"

pandas.DataFrame(
    stopwords_chunks
).to_excel("stopwords.xlsx")

def filter(input_text: str) -> str:
    input_text = re.sub(r'\W', ' ', input_text)
    input_text = re.sub(r'\s+[a-zA-Z]\s+', ' ', input_text)
    input_text = re.sub(r'\^[a-zA-Z]\s+', ' ', input_text)
    input_text = re.sub(r'\s+', ' ', input_text, flags=re.I)
    return input_text

def case_fold(input_text: str) -> str:
    return input_text.lower()

def stem(input_text: str) -> str:
    return stemmer.stem(input_text)

def remove_stop_words(input_text: str) -> str:
    tokens = input_text.split(" ")
    cleaned_tokens = [token for token in tokens if token not in stopwords.words('indonesian')]
    return " ".join(cleaned_tokens)

def clean_text(input_text: str) -> str:
    input_text = filter(input_text)
    input_text = case_fold(input_text)
    input_text = stem(input_text)
    input_text = remove_stop_words(input_text)
    return input_text



def normalize_target(target_text: str) -> str:
    lower_first_char = target_text[0].lower()

    if lower_first_char not in ['h', 'f']:
        raise ValueError("Incorrect value for target: {}".format(target_text))

    return lower_first_char


input_data_frame = pandas.read_csv(
    INPUT_FILE,
    header=None,
    skiprows=[0]
)


# Report preprocessing steps
PREPORT_KEY_STEP = "Langkah"
PREPORT_KEY_DESC = "Deskripsi"
preprocessing_report = []

report_text = input_data_frame[0][0]
preprocessing_report.append({
    PREPORT_KEY_STEP: "Teks Awal",
    PREPORT_KEY_DESC: report_text
})

report_text = filter(report_text)
preprocessing_report.append({
    PREPORT_KEY_STEP: "Filter",
    PREPORT_KEY_DESC: report_text
})

report_text = case_fold(report_text)
preprocessing_report.append({
    PREPORT_KEY_STEP: "Case Folding",
    PREPORT_KEY_DESC: report_text
})

report_text = stem(report_text)
preprocessing_report.append({
    PREPORT_KEY_STEP: "Stemming",
    PREPORT_KEY_DESC: report_text
})

report_text = remove_stop_words(report_text)
preprocessing_report.append({
    PREPORT_KEY_STEP: "Stop Words Removal",
    PREPORT_KEY_DESC: report_text
})

pandas.DataFrame(
    preprocessing_report
).to_excel("preprocessing_report.xlsx")



input_data_frame.to_excel("tweets.xlsx")

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
