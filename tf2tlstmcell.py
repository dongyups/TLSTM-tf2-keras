### reference: https://github.com/illidanlab/T-LSTM

import tensorflow as tf
from tensorflow.keras import layers

# @tf.keras.utils.register_keras_serializable()
class TLSTMCell(layers.Layer):
    """
    Since this is a cell of T-LSTM, you may use it with RNN layer.
    Make sure your inputs have a shape of (batch, seq_len, dim) and your time_inputs have a shape of (batch, seq_len, 1).
    They must be concatenated as a shape of (batch, seq_len, 1 + dim) with the order of 'time' and 'input' (time first).
    """
    def __init__(
        self, 
        units, 
        time_input=True, 
        # default initializer settings for tf2keras LSTMCell are respectively: "glorot_uniform", "orthogonal", "zeros"
        kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
        recurrent_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
        bias_initializer = tf.keras.initializers.Constant(value=1.0),
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        # omitted: dropout, recurrent_dropout
        **kwargs
    ):
        super(TLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.time_input = time_input
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.recurrent_initializer = tf.keras.initializers.get(recurrent_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer=tf.keras.regularizers.get(kernel_regularizer)
        self.recurrent_regularizer=tf.keras.regularizers.get(recurrent_regularizer)
        self.bias_regularizer=tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint=tf.keras.constraints.get(kernel_constraint)
        self.recurrent_constraint=tf.keras.constraints.get(recurrent_constraint)
        self.bias_constraint=tf.keras.constraints.get(bias_constraint)
        self.state_size = [self.units, self.units]
        self.output_size = self.units


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
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True)
        self.recurrent_kernel_i = self.add_weight(
            shape=(self.units, self.units),
            name='recurrent_kernel_i',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            trainable=True)
        self.bias_i = self.add_weight(
            shape=(self.units),
            name='bias_i',
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            trainable=True)

        # f
        self.kernel_f = self.add_weight(
            shape=(input_dim, self.units),
            name='kernel_f',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True)
        self.recurrent_kernel_f = self.add_weight(
            shape=(self.units, self.units),
            name='recurrent_kernel_f',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            trainable=True)
        self.bias_f = self.add_weight(
            shape=(self.units),
            name='bias_f',
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            trainable=True)

        # g
        self.kernel_g = self.add_weight(
            shape=(input_dim, self.units),
            name='kernel_g',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True)
        self.recurrent_kernel_g = self.add_weight(
            shape=(self.units, self.units),
            name='recurrent_kernel_g',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            trainable=True)
        self.bias_g = self.add_weight(
            shape=(self.units),
            name='bias_g',
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            trainable=True)

        # o
        self.kernel_o = self.add_weight(
            shape=(input_dim, self.units),
            name='kernel_o',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True)
        self.recurrent_kernel_o = self.add_weight(
            shape=(self.units, self.units),
            name='recurrent_kernel_o',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            trainable=True)
        self.bias_o = self.add_weight(
            shape=(self.units),
            name='bias_o',
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
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
        config = {
            "units": self.units,
            "time_input": self.time_input,
            "kernel_initializer": tf.keras.initializers.serialize(self.kernel_initializer),
            "recurrent_initializer": tf.keras.initializers.serialize(self.recurrent_initializer),
            "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer),
            "recurrent_regularizer": tf.keras.regularizers.serialize(self.recurrent_regularizer),
            "bias_regularizer": tf.keras.regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": tf.keras.constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": tf.keras.constraints.serialize(self.recurrent_constraint),
            "bias_constraint": tf.keras.constraints.serialize(self.bias_constraint),
        }
        config.update(config_for_enable_caching_device(self))
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return list(
            generate_zero_filled_state_for_cell(
                self, inputs, batch_size, dtype
            )
        )



### reference: https://github.com/keras-team/keras/blob/v2.11.0/keras/layers/rnn/rnn_utils.py
def generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype):
    if inputs is not None:
        batch_size = tf.shape(inputs)[0]
        dtype = inputs.dtype
    return generate_zero_filled_state(batch_size, cell.state_size, dtype)


def generate_zero_filled_state(batch_size_tensor, state_size, dtype):
    """Generate a zero filled tensor with shape [batch_size, state_size]."""
    if batch_size_tensor is None or dtype is None:
        raise ValueError(
            "batch_size and dtype cannot be None while constructing initial "
            f"state. Received: batch_size={batch_size_tensor}, dtype={dtype}"
        )

    def create_zeros(unnested_state_size):
        flat_dims = tf.TensorShape(unnested_state_size).as_list()
        init_state_size = [batch_size_tensor] + flat_dims
        return tf.zeros(init_state_size, dtype=dtype)

    if tf.nest.is_nested(state_size):
        return tf.nest.map_structure(create_zeros, state_size)
    else:
        return create_zeros(state_size)


def config_for_enable_caching_device(rnn_cell):
    """Return the dict config for RNN cell wrt to enable_caching_device field.
    Since enable_caching_device is a internal implementation detail for speed up
    the RNN variable read when running on the multi remote worker setting, we
    don't want this config to be serialized constantly in the JSON. We will only
    serialize this field when a none default value is used to create the cell.
    Args:
      rnn_cell: the RNN cell for serialize.
    Returns:
      A dict which contains the JSON config for enable_caching_device value or
      empty dict if the enable_caching_device value is same as the default
      value.
    """
    default_enable_caching_device = (
        tf.compat.v1.executing_eagerly_outside_functions()
    )
    if rnn_cell._enable_caching_device != default_enable_caching_device:
        return {"enable_caching_device": rnn_cell._enable_caching_device}
    return {}
