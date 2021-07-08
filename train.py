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

from model.encoder import Encoder
from model.decoder import Decoder
from config import params
from preprocess import *
from dataset import enc_seq, teach_force_seq, y

def loss(y, ypred, sce):

    loss_ = sce(y, ypred)

    mask = tf.cast(tf.not_equal(y, 0), tf.float32)

    loss_ = mask * loss_

    return tf.reduce_mean(loss_)
  
@tf.function
def train_step(params, x, ger_inp, ger_out, encoder, decoder, sce):
    with tf.GradientTape() as tape:

        tot_loss = 0

        enc_seq, hidden1, hidden2 = encoder(x)

        for i in range(params.dec_max_len):

            dec_inp = tf.expand_dims(ger_inp[:, i], axis=1)

            ypred, hidden1, hidden2, attention_weights = decoder(enc_seq, dec_inp, hidden1, hidden2)

            timestep_loss = loss(tf.expand_dims(ger_out[:, i], 1), ypred, sce)

            tot_loss += timestep_loss

        avg_timestep_loss = tot_loss/params.dec_max_len

    total_vars = encoder.trainable_variables + decoder.trainable_variables

    grads = tape.gradient(avg_timestep_loss, total_vars)
    params.optimizer.apply_gradients(zip(grads, total_vars))

    return grads, avg_timestep_loss
  
def save_checkpoints(params, encoder, decoder):
    checkpoint_dir = '/content/model_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    ckpt = tf.train.Checkpoint(optimizer=params.optimizer,
                               encoder=encoder,
                               decoder=decoder)
    ckpt.save(file_prefix=checkpoint_prefix)

def restore_checkpoint(params, encoder, decoder):
    checkpoint_dir = '/content/model_checkpoints'
    ckpt= tf.train.Checkpoint(optimizer=params.optimizer,
                              encoder=encoder,
                              decoder=decoder)
    ckpt.restore(tf.train.latest_checkpoint(checkpoint_dir))
    

encoder = Encoder(params)
decoder = Decoder(params)

def train():

    sce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    
    start = time.time()
    avg_loss = []

    for e in track(range(0, params.epochs)):

        losses = []

        st = time.time()

        for enc_seq_batch, teach_force_seq_batch, y_batch in zip(enc_seq, teach_force_seq, y):

            grads, loss = train_step(params, enc_seq_batch, teach_force_seq_batch, y_batch, encoder, decoder, sce)

            losses.append(loss.numpy())

        avg_loss.append(np.mean(losses))

        print(f'EPOCH - {e+1} ---- LOSS - {np.mean(losses)} ---- TIME - {time.time()- st}')

    save_checkpoints(params, encoder, decoder)
    print(f'total time taken: {time.time()-start}')

    return grads, avg_loss

grads, avg_loss = train()
