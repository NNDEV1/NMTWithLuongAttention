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

from config import params
from preprocess import *
from train import restore_checkpoint
from model.encoder import Encoder
from model.decoder import Decoder

df = pd.read_csv('/content/eng2ger.csv')

tokenize_eng, detokenize_eng, params.len_eng = tokenizer(df['eng_input'], True)
tokenize_ger, detokenize_ger, params.len_ger = tokenizer(df['ger_input'], False)

def make_prediction(txt, params, greedy=False, random_sampling=True, beam_search=False):

    nlp = spacy.load('en_core_web_sm')
    txt = contractions.fix(txt)

    x = tf.expand_dims(tf.constant([tokenize_eng[tok.text.lower()] for tok in nlp(txt)]), 0)

    #Comment if error comes up ##################
    
    encoder = Encoder(params)
    decoder = Decoder(params)

    restore_checkpoint(params, encoder, decoder)
    
    #############################################

    dec_inp = tf.reshape(tokenize_ger['<sos>'], (1,1))
    final_tok, i = '<sos>', 0

    sent, att = [], []
    enc_seq, hidden1, hidden2 = encoder(x)

    while final_tok != '<eos>':

        ypred, hidden1, hidden2, attention_weights = decoder(enc_seq, dec_inp, hidden1, hidden2)

        if random_sampling:
            idx = tf.random.categorical(ypred[:, 0, :], num_samples= 1)

        elif greedy:
            idx = tf.argmax(ypred[:, 0, :], axis= -1)

        elif beam_search:
            pass

        sent.append(detokenize_ger[tf.squeeze(idx).numpy()])

        att.append(attention_weights)
        dec_inp = idx
        if i == 10:
            break
        else:
            i += 1
    return " ".join(sent), att

txt = input('Type anything: ')
sent, att = make_prediction(txt, params)
print('[bold blue]' + sent)
