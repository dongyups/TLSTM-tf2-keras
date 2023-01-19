# TLSTM-tf2-keras
original: https://github.com/illidanlab/T-LSTM
### Required
This code is tested with python==3.8, tensorflow-gpu==2.7, and cuda/cudnn 11.2/8.1
- Python 3.X
- Tensorflow 2.X
- CUDA 10.X or 11.X for gpu settings depends on your hardware

### how to use
Since this code is only a cell of TLSTM, you need an RNN layer in order to use it.
```
import tensorflow as tf
from tf2tlstmcell import TLSTMCell

tlstm_layer = tf.keras.layers.RNN(TLSTMCell(64, time_input=True))
```
Note that this is a time-aware LSTM thus additional time input needs to be prepared and the option needs to be set to `True`.\
It may work as a vanilla LSTM otherwise.\
Refer to the comments in the code for further details.
