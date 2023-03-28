# TLSTM-with-LRP-tf2-keras
TLSTM original (tensorflow 1.2 & python 2.7): https://github.com/illidanlab/T-LSTM \
LRP for LSTM original: https://github.com/ArrasL/LRP_for_LSTM

### Required
This code is tested with python==3.8, tensorflow-gpu==2.7, and cuda/cudnn 11.2/8.1
- Python 3.X
- Tensorflow 2.X
- CUDA 10.X or 11.X for gpu settings depends on your hardware

### How to use
Since this code is only a cell of TLSTM, you need an RNN layer in order to use it.
```python3
# ex) multi-label classification
import tensorflow as tf
from tensorflow.keras import layers, Model
from tf2tlstmcell import TLSTMCell
from nplrp import LRPforTLSTM

# TLSTM
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
model = ModelConstruct(...)

# LRP for TLSTM
lrp = LRPforTLSTM(model=model)
lrp.set_input(...)
lrp.lrp(...)
```
Note that this is a time-aware LSTM thus additional time input needs to be prepared and the time_input option needs to be set to `True`. It may work as a vanilla LSTM otherwise.

### Input format
input shape: `(batch_size, seq_len, dim)`\
time input shape: `(batch_size, seq_len, 1)`
```
input = tf.concat([time_input, input], axis=-1)
```
Refer to the code for further details.
