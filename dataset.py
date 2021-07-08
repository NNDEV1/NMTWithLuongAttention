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

df = pd.read_csv('/content/eng2ger.csv')

tokenize_eng, detokenize_eng, len_eng= tokenizer(df['eng_input'], True)
tokenize_ger, detokenize_ger, len_ger= tokenizer(df['ger_input'], False)

tokenize_eng['<pad>'] = 0
detokenize_eng[0] = "<pad>"
tokenize_ger["<pad>"] = 0
detokenize_ger[0] = "<pad>"


num_samples = df.shape[0]
eng_vocab = len_eng + 1
ger_vocab = len_ger + 1 


df['eng_input'] = df['eng_input'].map(lambda txt: padding(txt, params.en_max_len))
df['ger_input'] = df['ger_input'].map(lambda txt: padding(txt, params.dec_max_len))
df['ger_target'] = df['ger_target'].map(lambda txt: padding(txt, params.dec_max_len))

df['eng_tok'] = df['eng_input'].map(lambda txt: [tokenize_eng[tok] for tok in txt.split(' ')])

df['teach_force_tok'] = df['ger_input'].map(lambda txt: [tokenize_ger[tok] for tok in txt.split(' ')])
df['target_tok'] = df['ger_target'].map(lambda txt: [tokenize_ger[tok] for tok in txt.split(' ')])

df['rev_eng_tok'] = df['eng_tok'].map(lambda ls: ls[:: -1])

enc_seq, teach_force_seq, y = make_minibatches(df, col1='rev_eng_tok', col2='teach_force_tok', col3='target_tok')

