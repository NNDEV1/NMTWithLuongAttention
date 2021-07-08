import tensorflow as tf
import os
import contractions
import tensorflow as tf
import pandas as pd
import numpy as np

import time
import rich
from rich.progress import track
import spacy


#Preprocessing Text
class preprocess_text():

    def __init__(self):
        pass
    
    def remove_pattern(self, text, pattern= r'[^a-zA-Z0-9.!?, ]', replace_with= ""):
        return re.sub(pattern, replace_with, text)
    
    def tokenize_sent(self, text, nlp):
        doc= nlp(text)
        return [sent.text for sent in doc.sents]
    
    def tokenize_words(self, text, nlp):
        doc= nlp(text)
        return " ".join(tok.text for tok in doc)
    
    def expand_contractions(self, text):

        return contractions.fix(text)
        
    def do_lemmatization(self, text, nlp):
        doc= nlp(text)
        return ' '.join(tok.lemma_ if tok.lemma_ != "-PRON-" else tok.text for tok in doc)
        
    def add_sos_eos(self, text, sos= False, eos= False):
        if (sos and eos):
            return "<sos> " + text + " <eos>" 
        if eos:
            return text + " <eos>"
        if sos:
            return "<sos> " + text
        return text
        
    def remove_accents(self, text):

        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('UTF-8', 'ignore')

def call_preprocessing(df_col, nlp_en= True, lower_= True, remove_pattern_= False, tokenize_words_= False,
               expand_contractions_= False, do_lemmatization_= False,
               sos= False, eos= False, remove_accents_= False):
    
    nlp= spacy.load('en_core_web_sm') if nlp_en else spacy.load('de_core_news_sm')
    prep= preprocess_text()
    
    if expand_contractions_:
        df_col= df_col.map(lambda text: prep.expand_contractions(text))
        
    if remove_accents_:
        df_col= df_col.map(lambda text: prep.remove_accents(text))
        
    if do_lemmatization_:
        df_col= df_col.map(lambda text: prep.do_lemmatization(text, nlp))
        
    if tokenize_words_:
        df_col= df_col.map(lambda text: prep.tokenize_words(text, nlp))
        
    if remove_pattern_:
        df_col= df_col.map(lambda text: prep.remove_pattern_(text))
    
    if eos or sos:
        df_col= df_col.map(lambda text: prep.add_sos_eos(text, sos, eos))
        

    if lower_:
        df_col= df_col.map(lambda text: text.lower())
    return df_col

def tokenizer(df_col, nlp_en= True):
    vocab= set()
    _= [[vocab.update([tok]) for tok in text.split(" ")] for text in df_col]

    if not nlp_en:
        vocab.update(["<sos>"])
        vocab.update(["<eos>"])

    tokenize= dict(zip(vocab, range(1, 1+len(vocab))))
    detokenize= dict(zip(range(1, 1+len(vocab)), vocab))
    return tokenize, detokenize, len(vocab)

def padding(txt_toks, max_len):
    curr_ls= txt_toks.split(" ")
    len_ls= len(curr_ls)
    _= [curr_ls.append("<pad>") for i in range(max_len-len_ls) if len(curr_ls)<max_len]
    return " ".join(curr_ls)

def make_minibatches(df, col1= 'rev_eng_tok', col2= 'teach_force_tok', col3= 'target_tok'):
    enc_seq= np.array([df[col1].values[i] for i in range(len(df[col1]))])
    enc_seq= tf.data.Dataset.from_tensor_slices(enc_seq).batch(params.batch_size)

    teach_force_seq= np.array([df[col2].values[i] for i in range(len(df[col2]))])
    teach_force_seq= tf.data.Dataset.from_tensor_slices(teach_force_seq).batch(params.batch_size)

    y= np.array([df[col3].values[i] for i in range(len(df[col3]))])
    y= tf.data.Dataset.from_tensor_slices(y).batch(params.batch_size)
    return enc_seq, teach_force_seq, y
