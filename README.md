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
```
import tensorflow as tf
from tf2tlstmcell import TLSTMCell

tlstm_layer = tf.keras.layers.RNN(TLSTMCell(64, time_input=True))
```
Note that this is a time-aware LSTM thus additional time input needs to be prepared and the time_input option needs to be set to `True`. It may work as a vanilla LSTM otherwise.

### Input format
input shape: `(batch_size, seq_len, dim)`\
time input shape: `(batch_size, seq_len, 1)`
```
input = tf.concat([time_input, input], axis=-1)
```
Refer to the code for further details.
