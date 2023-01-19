# reference: https://github.com/illidanlab/T-LSTM

import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable()
class TLSTMCell(layers.Layer):
    """
    Since this is a cell of T-LSTM, you may use it with RNN layer.
    Make sure your inputs have a shape of (batch, seq_len, dim) and your time_inputs have a shape of (batch, seq_len, 1).
    They must be concatenated as a shape of (batch, seq_len, 1 + dim) with the order of 'time' and 'input' (time first).
    """
    def __init__(self, units, time_input=True, **kwargs):
        self.units = units
        self.time_input = time_input
        self.state_size = [tf.TensorShape([units]), tf.TensorShape([units])]
        self.output_size = [tf.TensorShape([units]), tf.TensorShape([units])]
        self.kernel_initializer = tf.random_normal_initializer(0.0, 0.1) # default setting for T-LSTM is RandomNormal(mean=0., stddev=0.1)
        self.recurrent_initializer = tf.random_normal_initializer(0.0, 0.1)
        self.bias_initializer = tf.constant_initializer(1.0) # default setting for T-LSTM is tf.constant_initializer(1.0)
        super(TLSTMCell, self).__init__(**kwargs)


    def build(self, input_shape):
        if self.time_input is False:
            # Elapsed Time Input is NOT given. Activating Vanilla LSTM cells.
            input_dim = int(input_shape[-1])
        elif self.time_input is True:
            # Elapsed Time Input is given. Activating T-LSTM cells.
            input_dim = int(input_shape[-1] - 1)

        # i
        self.kernel_i = self.add_weight(
            shape=(input_dim, self.units),
            name='kernel_i',
            initializer=self.kernel_initializer,
            trainable=True)
        self.recurrent_kernel_i = self.add_weight(
            shape=(self.units, self.units),
            name='recurrent_kernel_i',
            initializer=self.recurrent_initializer,
            trainable=True)
        self.bias_i = self.add_weight(
            shape=(self.units),
            name='bias_i',
            initializer=self.bias_initializer,
            trainable=True)

        # f
        self.kernel_f = self.add_weight(
            shape=(input_dim, self.units),
            name='kernel_f',
            initializer=self.kernel_initializer,
            trainable=True)
        self.recurrent_kernel_f = self.add_weight(
            shape=(self.units, self.units),
            name='recurrent_kernel_f',
            initializer=self.recurrent_initializer,
            trainable=True)
        self.bias_f = self.add_weight(
            shape=(self.units),
            name='bias_f',
            initializer=self.bias_initializer,
            trainable=True)

        # g
        self.kernel_g = self.add_weight(
            shape=(input_dim, self.units),
            name='kernel_g',
            initializer=self.kernel_initializer,
            trainable=True)
        self.recurrent_kernel_g = self.add_weight(
            shape=(self.units, self.units),
            name='recurrent_kernel_g',
            initializer=self.recurrent_initializer,
            trainable=True)
        self.bias_g = self.add_weight(
            shape=(self.units),
            name='bias_g',
            initializer=self.bias_initializer,
            trainable=True)

        # o
        self.kernel_o = self.add_weight(
            shape=(input_dim, self.units),
            name='kernel_o',
            initializer=self.recurrent_initializer,
            trainable=True)
        self.recurrent_kernel_o = self.add_weight(
            shape=(self.units, self.units),
            name='recurrent_kernel_o',
            initializer=self.recurrent_initializer,
            trainable=True)
        self.bias_o = self.add_weight(
            shape=(self.units),
            name='bias_o',
            initializer=self.bias_initializer,
            trainable=True)

        # Time elapsed part
        # W_h(hidden), b(bias) for the time elapsed part.
        self.kernel_time = self.add_weight(
            shape=(self.units, self.units),
            name='kernel_time',
            initializer=self.kernel_initializer,
            trainable=True)
        self.bias_time = self.add_weight(
            shape=(self.units),
            name='bias_time',
            initializer=self.bias_initializer,
            trainable=True)


    def _comput_gate_state(self, z0, z1, z2, z3, c_tm1):
        # input gate (i)
        i = tf.nn.sigmoid(z0)

        # forget gate (f)
        f = tf.nn.sigmoid(z1)

        # cell state (g & c)
        g = tf.nn.tanh(z2)
        c = f * c_tm1 + g * i

        # output gate (o)
        o = tf.nn.sigmoid(z3)
        return c, o


    def call(self, inputs, states):
        # h_tm1, previous memory state
        h_tm1 = states[0]

        # c_tm1, previous carry state
        c_tm1 = states[1]

        # Vanilla LSTM
        if self.time_input is False:
            pass
        # T-LSTM
        elif self.time_input is True:
            # Detach time_input from the inputs where the 1+dim is separated into 1 and dim
            time_input = tf.expand_dims(inputs[:, 0], axis=-1) # First dim of the inputs (batch, 1)
            inputs = inputs[:, 1:] # (batch, dim)

            # Modified map_elapse_time part. Map elapse time in days or months
            c1 = tf.constant(1.0, dtype=inputs.dtype)
            c2 = tf.math.exp(c1) # 2.7183

            T = tf.math.divide(c1, tf.math.log(time_input + c2), name='Log_elpase_time')
            Ones = tf.ones([1, self.units], dtype=inputs.dtype)
            T = tf.matmul(T, Ones)

            # Decompose the previous cell if there is a elapse time
            C_ST = tf.nn.tanh(tf.matmul(c_tm1, self.kernel_time) + self.bias_time)
            C_ST_dis = tf.math.multiply(T, C_ST)

            # If T is 0, then the weight is one
            c_tm1 = c_tm1 - C_ST + C_ST_dis

        # input gate (i): (X_t * W_x) + (h_tm1 * W_h) + b
        z0 = tf.matmul(inputs, self.kernel_i)
        z0 += tf.matmul(h_tm1, self.recurrent_kernel_i)
        z0 = tf.nn.bias_add(z0, self.bias_i)

        # forget gate (f)
        z1 = tf.matmul(inputs, self.kernel_f)
        z1 += tf.matmul(h_tm1, self.recurrent_kernel_f)
        z1 = tf.nn.bias_add(z1, self.bias_f)

        # cell state (g & c)
        z2 = tf.matmul(inputs, self.kernel_g)
        z2 += tf.matmul(h_tm1, self.recurrent_kernel_g)
        z2 = tf.nn.bias_add(z2, self.bias_g)

        # output gate (o)
        z3 = tf.matmul(inputs, self.kernel_o)
        z3 += tf.matmul(h_tm1, self.recurrent_kernel_o)
        z3 = tf.nn.bias_add(z3, self.bias_o)

        # LSTM Calculation
        # c_t, o
        c, o = self._comput_gate_state(z0, z1, z2, z3, c_tm1)

        # h_t
        h = o * tf.nn.sigmoid(c)

        # return them
        hidden = (h)
        hiddencarray = (h,c)
        return hidden, hiddencarray


    def get_config(self):
        return {"units": self.units, "time_input": self.time_input}
