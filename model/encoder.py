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

class Encoder(tf.keras.Model):
    def __init__(self, params):
        super(Encoder, self).__init__()

        self.embed = tf.keras.layers.Embedding(input_dim=params.eng_vocab,
                                               output_dim=params.embed_size)
        
        self.gru1 = tf.keras.layers.GRU(params.gru_units, kernel_initializer='glorot_normal',
                                        return_sequences=True, return_state=True)
        
        self.gru2 = tf.keras.layers.GRU(params.gru_units, kernel_initializer='glorot_normal',
                                        return_sequences=True, return_state=True)
        
    def call(self, input_seq):

        x = self.embed(input_seq)

        output_seq1, hidden1 = self.gru1(x)

        output_seq2, hidden2 = self.gru2(output_seq1)

        return output_seq2, hidden1, hidden2
