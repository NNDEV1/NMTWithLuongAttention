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

from attention import LuongAttention

class Decoder(tf.keras.Model):
    def __init__(self, params):

        super(Decoder, self).__init__()

        self.embed = tf.keras.layers.Embedding(input_dim=params.ger_vocab,
                                               output_dim=params.embed_size)
        
        self.gru1 = tf.keras.layers.GRU(params.gru_units, kernel_initializer='glorot_normal',
                                        return_sequences=True, return_state=True)
        
        self.gru2 = tf.keras.layers.GRU(params.gru_units, kernel_initializer='glorot_normal',
                                        return_sequences=True, return_state=True)
        
        self.attention = LuongAttention(params)

        self.fc = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(params.ger_vocab))

    def call(self, enc_seq, teach_force_seq, init_hidden1, init_hidden2):

        x = self.embed(teach_force_seq)

        output_seq1, hidden1 = self.gru1(x, initial_state=init_hidden1)

        output_seq2, hidden2 = self.gru2(output_seq1, initial_state=init_hidden2)

        context_vector, attention_weights = self.attention(enc_seq, output_seq2)

        x = tf.concat([output_seq2, tf.expand_dims(context_vector, 1)], axis= -1)

        x = tf.nn.tanh(x)

        y = self.fc(x)

        return y, hidden1, hidden2, attention_weights
      
      
