# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 21:20:03 2021

@author: Bastien
"""
from os import listdir
from PIL import Image
import os
import tensorflow as tf
import numpy as np

directory = 'C:\\Users\\Bastien\\Documents\\Python Scripts\\GAN\\Real Dogs\\Dog\\'
for filename in listdir(directory):
  if filename.endswith(".jpg"):
    try:
        img = Image.open(directory+filename) # open the image file
        img.verify() # verify that it is, in fact an image
        
        if img is not None:
            if len(img.getbands()) != 3:
                img.close()
                os.remove(directory+filename)
                print(directory+filename,' did not have enough channels')
    except (IOError, SyntaxError) as e:
        print('Bad file:', filename) # print out the names of corrupt files
        
num_skipped = 0
total = 0

for fname in os.listdir(directory):
    if fname.endswith(".jpg"):
        fpath = os.path.join(directory, fname)
        try:
            total += 1
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()
    
        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)
