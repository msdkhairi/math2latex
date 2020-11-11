import numpy as np
import tensorflow as tf

class CNN(tf.keras.Model):
  
  def __init__(self, filters):
    super(CNN, self).__init__()
    
        
    # c:64, k:(3,3), s:(1,1), p:(1,1) po:(2,2), s:(2,2), p(2,2)
    self.conv1 = tf.keras.layers.Conv2D(filters=filters, 
                                        kernel_size=3, 
                                        strides=1,
                                        padding='same',
                                        activation='relu')
    self.pool1 = tf.keras.layers.MaxPool2D(pool_size=2, 
                                           strides=2, 
                                           padding='same')
    
    # c:128, k:(3,3), s:(1,1), p:(1,1) po:(2,2), s:(2,2), p:(0,0)
    self.conv2 = tf.keras.layers.Conv2D(filters=filters*2, 
                                        kernel_size=3, 
                                        strides=1,
                                        padding='same',
                                        activation='relu')
    self.pool2 = tf.keras.layers.MaxPool2D(pool_size=2, 
                                           strides=2, 
                                           padding='valid')
    
    # c:256, k:(3,3), s:(1,1), p:(1,1), bn -
    self.conv3 = tf.keras.layers.Conv2D(filters=filters*4, 
                                        kernel_size=3, 
                                        strides=1,
                                        padding='same',
                                        activation='relu')
    self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)


    # c:256, k:(3,3), s:(1,1), p:(1,1) po:(2,1), s:(2,1), p(0,0)
    self.conv4 = tf.keras.layers.Conv2D(filters=filters*4, 
                                        kernel_size=3, 
                                        strides=1,
                                        padding='same',
                                        activation='relu')
    self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 1), 
                                           strides=(2, 1), 
                                           padding='valid')


    # c:512, k:(3,3), s:(1,1), p:(1,1), bn po:(1,2), s:(1,2), p:(0,0)
    self.conv5 = tf.keras.layers.Conv2D(filters=filters*8, 
                                        kernel_size=3, 
                                        strides=1,
                                        padding='same',
                                        activation='relu')
    self.pool4 = tf.keras.layers.MaxPool2D(pool_size=(1, 2), 
                                           strides=(1, 2), 
                                           padding='valid')

    # c:512, k:(3,3), s:(1,1), p:(0,0), bn -
    self.conv6 = tf.keras.layers.Conv2D(filters=filters*8, 
                                        kernel_size=3, 
                                        strides=1,
                                        padding='valid',
                                        activation='relu')
    self.bn2 = tf.keras.layers.BatchNormalization(axis=-1)


    
  def call(self, inputs):
    
    
    output = inputs
    for layer in self.layers:
      output = layer(output)
      
    return output
      

class Encoder(tf.keras.Model):
  
  def __init__(self, enc_units):
    super(Encoder, self).__init__()
    
    self.gru = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(units=enc_units,
                            return_sequences=True, 
                            return_state=True))
    
  def call(self, inputs, hidden=None):
    
    # inputs (batch, height, width, channels)
    height = inputs.shape[1]
    
    outputs = []
    for h in range(height):
      output, _, _ = self.gru(inputs[:,h])
      outputs.append(output)
      
    output = tf.stack(outputs)
    output = tf.transpose(output, [1, 0, 2, 3])
    
    return output
      


class Attention(tf.keras.Model):
  
  def __init__(self, units):
    super(Attention, self).__init__()
    
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)
    
  def call(self, dec_hidden, enc_output):
    
    dec_hidden = tf.expand_dims(dec_hidden, 1)
    dec_hidden = tf.expand_dims(dec_hidden, 1)
        
    score = self.V(tf.keras.activations.tanh(
        self.W1(enc_output) + self.W2(dec_hidden)))
    
    attention_weights = tf.keras.activations.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * enc_output
    context_vector = tf.reduce_sum(context_vector, axis=[1, 2])

    return context_vector, attention_weights


class Decoder(tf.keras.Model):
  
  def __init__(self, dec_units, embed_size, vocab_size):
    super(Decoder, self).__init__()
    
    self.embedding = tf.keras.layers.Embedding(
        input_dim=vocab_size, 
        output_dim=embed_size)
    
    self.gru = tf.keras.layers.GRU(units=dec_units, 
                      return_sequences=True, return_state=True)
    
    self.fc = tf.keras.layers.Dense(vocab_size)
    
    self.attention = Attention(dec_units)
    
  def call(self, x, dec_state, enc_output, training=True):
    
    context_vector, attention_weights = self.attention(dec_state, enc_output)
    
    x = self.embedding(x)
    
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    
    output, state = self.gru(x)

    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    output = self.fc(output)
    
    
    return output, state, attention_weights
    

