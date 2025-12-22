# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 09:27:55 2025
Latent Dirichlet Allocation (LDA) Example
Source: https://medium.com/analytics-vidhya/topic-modeling-using-gensim-lda-in-python-48eaa2344920
Also used to examine text for burnout inventories.

Output NEEDS TO BE: synth_comments.csv
Header: USER_ID, RECORD_DATA, TOPIC_1, TOPIC_2, ....., TOPIC_N
@author: seboe
"""

#%% ---- Optional: auto-install missing packages (safe for .py scripts) ----
import sys, subprocess

def ensure(pkg, spec=None):
    try:
        __import__(pkg)
    except ImportError:
        to_install = spec or pkg
        subprocess.check_call([sys.executable, "-m", "pip", "install", to_install])

# Install core deps if missing (comment these three ensures out if you prefer manual installs)
ensure("gensim", 'gensim>=4.3,<5')
ensure("spacy", 'spacy>=3.7,<4')
ensure("pyLDAvis", 'pyLDAvis==3.4.1')

#%% ---- Imports ----
import nltk
nltk.download('stopwords')

import re
import numpy as np
import pandas as pd
from pprint import pprint
import os, json

# Gensim (v4+)
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel

# spaCy for preprocessing
import spacy

# Ensure spaCy model is present
try:
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
except OSError:
    # auto-download small English model once
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

# Plotting
import matplotlib.pyplot as plt

# pyLDAvis (use adapter for gensim>=4)
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

#%% ---- DATA SOURCE CONFIG ----
from pathlib import Path

DATA_SOURCE = "csv"   # one of: "csv", "jsonl", "txt_folder", "newsgroups_url"

# A) CSV (synthetic data)
# CSV_PATH = r"C:\Users\seboe\Documents\Professional\Out of Our Minds\synthetic_burnout_longitudinal_100x10.csv"
CSV_PATH = r"X:\ting\shared_ting\Scott\OOM\synthetic_burnout_longitudinal_100x10.csv"
CSV_TEXT_COL = "text"

# B) JSONL (one object per line with key 'text')
JSONL_PATH = r"C:\path\to\your.jsonl"
JSONL_TEXT_KEY = "text"

# C) TXT folder (each .txt file is one document)
TXT_FOLDER = r"C:\path\to\txt_folder"

# D) Fallback URL (the old 20 Newsgroups JSON)
NEWSGROUPS_URL = "https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json"

#%% ---- LOAD DATA  ----
def load_from_csv(path, text_col="text"):
    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise ValueError(f"CSV missing required text column: {text_col}")
    return df[text_col].astype(str).tolist(), df

def load_from_jsonl(path, text_key="text"):
    texts = []
    metas = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(str(obj.get(text_key, "")))
            metas.append(obj)
    return texts, pd.DataFrame(metas)

def load_from_txt_folder(folder):
    texts, names = [], []
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith(".txt"):
            with open(os.path.join(folder, fn), "r", encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())
                names.append(fn)
    meta = pd.DataFrame({"filename": names})
    return texts, meta

def load_from_newsgroups(url):
    df = pd.read_json(url)
    if "content" not in df.columns:
        raise ValueError("Unexpected schema from newsgroups URL")
    return df["content"].astype(str).tolist(), df

# Choose source
if DATA_SOURCE == "csv":
    assert Path(CSV_PATH).exists(), f"CSV not found: {CSV_PATH}"
    raw_texts, df_meta = load_from_csv(CSV_PATH, text_col=CSV_TEXT_COL)
elif DATA_SOURCE == "jsonl":
    assert Path(JSONL_PATH).exists(), f"JSONL not found: {JSONL_PATH}"
    raw_texts, df_meta = load_from_jsonl(JSONL_PATH, text_key=JSONL_TEXT_KEY)
elif DATA_SOURCE == "txt_folder":
    assert Path(TXT_FOLDER).exists(), f"Folder not found: {TXT_FOLDER}"
    raw_texts, df_meta = load_from_txt_folder(TXT_FOLDER)
elif DATA_SOURCE == "newsgroups_url":
    raw_texts, df_meta = load_from_newsgroups(NEWSGROUPS_URL)
else:
    raise ValueError("Unknown DATA_SOURCE")
print(f"Loaded {len(raw_texts)} documents from {DATA_SOURCE}.")

#%% ---- PREPROCESS (clean, tokenize, ngrams, lemmatize) ----
# Regex cleanup
def basic_clean(texts):
    out = []
    for sent in texts:
        s = re.sub(r'\S*@\S*\s?', '', sent)   # remove emails
        s = re.sub(r'\s+', ' ', s)            # collapse whitespace
        s = re.sub(r"\'", "", s)              # strip single quotes
        out.append(s)
    return out

clean_texts = basic_clean(raw_texts)

# Tokenize
def to_words(texts):
    return [gensim.utils.simple_preprocess(str(t), deacc=True) for t in texts]

data_words = to_words(clean_texts)

# Build ngrams on current corpus (don’t reuse old models across very different corpora)
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words.update([
    'from','subject','re','edu','use','m','s','ve','sc','c',
    'rsa','pgp','ax','pp','tc','nhl','faq','gmt','ms_window','dn',
    'km','lemieux','spec','baud','yehuda','enviroleague','_t','_','tel_fax'
])

def remove_stopwords(texts):
    return [[w for w in doc if w not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

# FAST lemmatization with nlp.pipe
def lemmatize_pipe(texts_tokens, allowed_postags=('NOUN','ADJ','VERB','ADV'), batch_size=200, n_process=1):
    texts_out = []
    for doc in nlp.pipe((" ".join(toks) for toks in texts_tokens),
                        batch_size=batch_size, n_process=n_process, disable=['parser','ner']):
        texts_out.append([tok.lemma_ for tok in doc if tok.pos_ in allowed_postags])
    return texts_out

# Pipeline
data_nostop = remove_stopwords(data_words)
data_bi = make_bigrams(data_nostop)
data_tri = make_trigrams(data_bi)  # optional; you can also skip trigrams

data_lemmatized = lemmatize_pipe(data_tri, allowed_postags=('NOUN','ADJ','VERB','ADV'))
print("Example lemmatized doc:", data_lemmatized[0][:20])

#%% ---- DICTIONARY & CORPUS ----
id2word = corpora.Dictionary(data_lemmatized)
# Optional filtering to drop very rare/common tokens:
# id2word.filter_extremes(no_below=5, no_above=0.5, keep_n=100000)

texts = data_lemmatized
corpus = [id2word.doc2bow(text) for text in texts]

pprint(corpus[:1])
print([[ (id2word[i], f) for (i,f) in corpus[0] ]][:1])

#%% ---- LDA TRAIN ----
np.seterrcall(None)
np.seterr(all='warn')

K = 12  # Num Topics -- can tune this for speed
NChunk = 200 #chunk size -- can tune this for speed
NPasses = 8 #number of passes -- can tune this for speed
lda_model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    num_topics=K,
    random_state=100,
    update_every=1,
    chunksize=200,
    passes=8,
    alpha='auto',
    per_word_topics=True
)

print("\nTop words per topic:")
for i, topic in lda_model.print_topics(num_topics=K, num_words=10):
    print(f"Topic {i}: {topic}")

print('\nPerplexity:', lda_model.log_perplexity(corpus))

coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score:', coherence_lda)

#%% ---- VISUALIZATION ----
vis = gensimvis.prepare(lda_model, corpus, id2word)
out_html = Path.cwd() / "lda_topics_vis.html"
pyLDAvis.save_html(vis, str(out_html))
print(f"Saved pyLDAvis HTML to: {out_html}")

#%% ---- INFER TOPIC MIXTURES FOR SIMULATED CSV ----
def infer_doc_topics_dense(model, bows, minimum_probability=0.0):
    K = model.num_topics
    mat = np.zeros((len(bows), K), dtype=float)
    for i, bow in enumerate(bows):
        for k, p in model.get_document_topics(bow, minimum_probability=minimum_probability):
            mat[i, k] = p
    return mat

theta = infer_doc_topics_dense(lda_model, corpus, minimum_probability=0.0)

# If synthetic CSV was used:
if DATA_SOURCE == "csv" and "person_id" in df_meta.columns and "timepoint" in df_meta.columns:
    for k in range(theta.shape[1]):
        df_meta[f"topic_{k}"] = theta[:, k]
    # quick check: mean topic prevalence by time
    by_time = df_meta.groupby("timepoint")[[f"topic_{k}" for k in range(theta.shape[1])]].mean()
    print("\nMean topic prevalence by time (first 5 rows):")
    print(by_time.head())
