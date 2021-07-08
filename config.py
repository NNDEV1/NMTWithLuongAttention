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

# Constants

class params():
    pass

params.batch_size = 64

params.embed_size = 300
params.gru_units = 128
params.learning_rate = .001
params.optimizer = tf.keras.optimizers.Adam(params.learning_rate, clipvalue=1)
params.epochs = 100


params.num_samples = 30000 
params.eng_vocab = 5776
params.ger_vocab = 8960
params.dec_max_len = 17
params.en_max_len = 20
