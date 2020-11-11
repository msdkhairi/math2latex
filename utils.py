import numpy as np
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
import os
import glob

def load_data():
  
  train_size = len(glob.glob('Dataset/images/images_train/*.png'))
  validation_size = len(glob.glob('Dataset/images/images_validation/*.png'))
  test_size = len(glob.glob('Dataset/images/images_test/*.png'))
    
  train_images = [cv.imread('Dataset/images/images_train/' + str(i) + '.png', 0) 
                  for i in range(train_size)]
  train_images = np.stack(train_images)
  
  validation_images = [cv.imread('Dataset/images/images_validation/' + str(i) + '.png', 0) 
                  for i in range(validation_size)]
  validation_images = np.stack(validation_images)
  
  test_images = [cv.imread('Dataset/images/images_test/' + str(i) + '.png', 0) 
                  for i in range(test_size)]
  test_images = np.stack(test_images)
  
  with open('Dataset/formulas/train_formulas.txt') as f:
    train_formulas = [line.split() for line in f]
    
  with open('Dataset/formulas/validation_formulas.txt') as f:
    validation_formulas = [line.split() for line in f]
  
  return train_images, train_formulas, validation_images, validation_formulas, test_images


def max_len_formulas(all_formulas): 
  return max([len(formula) for formulas in all_formulas for formula in formulas])


def get_fixed_length_formulas(formulas, extra_tokens, max_len):
  
  bof, pad, eof = extra_tokens
  num_formul = len(formulas)
  result = []
  
  for i, theformula in enumerate(formulas):
    formula = theformula.copy()
    # padding
    for _ in range(max_len - len(formula)):
      formula.append(pad)
    # add bof and eof
    formula.insert(0, bof)
    formula.append(eof)
    result.append(formula)
  return result


def to_tensor_normalize(img):
  img = tf.convert_to_tensor(img, dtype=tf.float32)
  return (img - 127.5) / 127.5

def generate_batch(batch_size, b, 
                   train_images=train_images, train_formulas=train_formulas):
  
  img = to_tensor_normalize(train_images[b*batch_size:(b+1)*batch_size])
  img = tf.expand_dims(img, -1)
  
  target = tf.convert_to_tensor(train_formulas[b*batch_size:(b+1)*batch_size])
  
  return img, target