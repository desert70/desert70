# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 16:51:24 2021

@author: Bastien
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import uuid
import cv2

# physical_device = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_device[0], True)

NUM_EPOCH = 1

directory = Path(r'C:\Users\Bastien\Documents\Python Scripts\GAN')

gen_path = directory.joinpath('generator_weights.ckpt')
disc_path = directory.joinpath('discriminator_weights.ckpt')


dataset = keras.preprocessing.image_dataset_from_directory(
    directory=directory,label_mode=None,image_size=(64,64),color_mode='rgb',batch_size=32,shuffle=True
    ).map(lambda x: x/255.0)


def discriminator(disc_path=''):
    S = keras.Input(shape=(64,64,3))
    h0 = layers.Conv2D(64,kernel_size=4,strides=2,padding='same',activation=layers.LeakyReLU(alpha=0.002))(S)
    h1 = layers.Conv2D(128,kernel_size=4,strides=2,padding='same',activation=layers.LeakyReLU(alpha=0.002))(h0)
    h2 = layers.Conv2D(128,kernel_size=4,strides=2,padding='same',activation=layers.LeakyReLU(alpha=0.002))(h1)
    f0 = layers.Flatten()(h2)
    drop0 = layers.Dropout(0.2)(f0)
    real_prob = layers.Dense(1,activation='sigmoid')(drop0)
    
    model = keras.Model(inputs=[S], outputs=[real_prob])
    
    if os.path.exists(disc_path):
        model.load_weights(disc_path)
    
    return model
    
def generator(latent_dim, gen_path=''):
    S = keras.Input(shape=(latent_dim))
    d0 = layers.Dense(8*8*latent_dim)(S)
    r0 = layers.Reshape((8,8,128))(d0)
    inv_h0 = layers.Conv2DTranspose(128,kernel_size=4,strides=2,padding='same',activation=layers.LeakyReLU(alpha=0.002))(r0)
    inv_h1 = layers.Conv2DTranspose(256,kernel_size=4,strides=2,padding='same',activation=layers.LeakyReLU(alpha=0.002))(inv_h0)
    inv_h2 = layers.Conv2DTranspose(512,kernel_size=4,strides=2,padding='same',activation=layers.LeakyReLU(alpha=0.002))(inv_h1)
    fake_image = layers.Conv2D(3,kernel_size=5,padding='same',activation='sigmoid')(inv_h2)
    
    model = keras.Model(inputs=[S], outputs=[fake_image])
    
    if os.path.exists(gen_path):
        model.load_weights(gen_path)
    
    return model

def train(dataset,NUM_EPOCH,gen_path,disc_path):
    disc = discriminator(disc_path)
    
    print(disc.summary())
    
    latent_dim = 128
    
    gen = generator(latent_dim,gen_path)
    
    print(gen.summary())
    
    opt_gen = keras.optimizers.Adam(1e-4)
    opt_disc = keras.optimizers.Adam(1e-4)
    
    loss_func = keras.losses.BinaryCrossentropy()
    
    ######### Custom Training Loop ###########
    
    for epoch in range(NUM_EPOCH):
        for idx, real in enumerate(tqdm(dataset)):
            batch_size = real.shape[0]
            random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
            
            fake = gen(random_latent_vectors)
            
            with tf.GradientTape() as disc_tape:    
    ######## Train Discriminator: max y * log(D(real)) + (1 - y) * (log(1 - D(G(z)))) where G(z) is fake image #########
                loss_disc_real = loss_func(tf.ones((batch_size, 1)), disc(real))  # y is set to 1 so we are only evaluating log(D(real))
                loss_disc_fake = loss_func(tf.zeros((batch_size, 1)), disc(fake))  # y is set to 0 so we are only evaluating log(1 - D(G(z)))
                
                loss_disc = (loss_disc_real + loss_disc_fake) / 2
    
            grads = disc_tape.gradient(loss_disc, disc.trainable_weights)
            opt_disc.apply_gradients(zip(grads, disc.trainable_weights))
    
    
            with tf.GradientTape() as gen_tape:
    ######## Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))) #########
                fake = gen(random_latent_vectors)            
                output = disc(fake)
                loss_gen = loss_func(tf.ones((batch_size, 1)), output)
                
            grads = gen_tape.gradient(loss_gen, gen.trainable_weights)
            opt_gen.apply_gradients(zip(grads, gen.trainable_weights))
    
    disc.save_weights(disc_path)
    gen.save_weights(gen_path)
    
def generate_images(num_images,latent_dim,gen_path='',output_folder=''):
    gen = generator(latent_dim,gen_path)
    
    random_latent_vectors = tf.random.normal(shape=(num_images, latent_dim))
    
    fake = gen(random_latent_vectors).numpy()
    
    height = np.shape(fake)[1]
    width = np.shape(fake)[2]
    
    print(fake)
    
    if os.path.exists(output_folder):
        for img in fake:
            img[:,:,0] = np.ones([height,width])*64/255.0
            img[:,:,1] = np.ones([height,width])*128/255.0
            img[:,:,2] = np.ones([height,width])*192/255.0
            
            cv2.imwrite(output_folder + '\\' + str(uuid.uuid4()) + '.jpg', img)
    else:
        for img in fake:
            plt.imshow(img)
    
    
if __name__ == '__main__':
    train(dataset,NUM_EPOCH,gen_path,disc_path)
    generate_images(1,128,gen_path,r'C:\Users\Bastien\Documents\Python Scripts\GAN\Fake Dogs')



















    