import tensorflow as tf
from tensorflow.keras import layers, Model
from tf2tlstmcell import TLSTMCell


class ModelConstruct(Model):
    def __init__(self, output_dim):
        super(ModelConstruct, self).__init__()
        self.tlstmcell = TLSTMCell(128, time_input=True)
        self.rnn = layers.RNN(self.tlstmcell, return_sequences=True)
        self.dense = layers.Dense(64)
        self.do = layers.Dropout(0.1)
        self.densef = layers.Dense(output_dim)
        
    @tf.function
    def call(self, inputs, training=None):
        input, time, position = inputs
        position = tf.cast(tf.expand_dims(position, axis=-1), dtype=tf.int32)
        input = tf.concat([tf.expand_dims(time, axis=-1), input], axis=-1)
        x = self.rnn(input)
        x = tf.gather_nd(x, position, batch_dims=1)
        x = self.dense(x)
        x = tf.nn.gelu(x)
        x = self.do(x, training=training)
        x = self.densef(x)
        x = tf.nn.sigmoid(x)
        return x
      
