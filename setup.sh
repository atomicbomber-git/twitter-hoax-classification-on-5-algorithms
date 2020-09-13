#!/bin/bash

python3 -m virtualenv -p python3 env
source ./env/bin/activate

pip install -r requirements.txt

python -m nltk.downloader stopwords punkt --dir=./env/nltk_data