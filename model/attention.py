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


class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, params):
        super(LuongAttention, self).__init__()

        self.tdfc = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(params.gru_units))

    def call(self, en_seq, dec_out):
        scores = tf.keras.backend.batch_dot(self.tdfc(en_seq), dec_out, axes=(2, 2))

        attention_weights = tf.nn.softmax(scores, axis=1)

        mul = en_seq * attention_weights

        context_vector = tf.reduce_mean(mul, axis=1)


        return context_vector, attention_weights
      
      
