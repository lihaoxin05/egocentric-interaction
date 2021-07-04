import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

def _linear(args,
            output_size,
            bias,
            scope_suffix = None,
            bias_initializer=init_ops.constant_initializer(0.0),
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_AVG', uniform=False),
            regularizer_scale=0.0):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    scope_suffix: suffix of variable names.
    bias_initializer: starting value to initialize the bias.
    kernel_initializer: starting value to initialize the weight.
    regularizer_scale: weight for kernel regularization.
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        _WEIGHTS_VARIABLE_NAME + '_%s'%scope_suffix, [total_arg_size, output_size],
        dtype=dtype,
        initializer=kernel_initializer,
        regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale))
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      biases = vs.get_variable(
          _BIAS_VARIABLE_NAME + '_%s'%scope_suffix, [output_size],
          dtype=dtype,
          initializer=bias_initializer)
    return nn_ops.bias_add(res, biases)

class InteractiveLSTMCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_units, forget_bias=1.0,
                   state_is_tuple=True, activation=None, regularizer_scale=0.0, reuse=None):

        super(InteractiveLSTMCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh
        self._regularizer_scale = regularizer_scale

    @property
    def state_size(self):
        return (tf.nn.rnn_cell.LSTMStateTuple(2 * self._num_units, 2 * self._num_units)
                if self._state_is_tuple else 4 * self._num_units)

    @property
    def output_size(self):
        return 2 * self._num_units

    def call(self, inputs, state):
        sigmoid = math_ops.sigmoid
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
          c, h = state
        else:
          c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)

        c1, c2 = array_ops.split(value=c, num_or_size_splits=2, axis=1)
        h1, h2 = array_ops.split(value=h, num_or_size_splits=2, axis=1)
        
        inputs1, inputs2 = array_ops.split(value=inputs, num_or_size_splits=2, axis=1)
        
        concat1 = _linear([tf.concat([inputs1, h2], axis=-1), h1], 4 * self._num_units, True, 3, regularizer_scale=self._regularizer_scale)
        concat2 = _linear([tf.concat([inputs2, h1], axis=-1), h2], 4 * self._num_units, True, 4, regularizer_scale=self._regularizer_scale)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i1, j1, f1, o1 = array_ops.split(value=concat1, num_or_size_splits=4, axis=1)
        i2, j2, f2, o2 = array_ops.split(value=concat2, num_or_size_splits=4, axis=1)

        new_c1 = (c1 * sigmoid(f1 + self._forget_bias) + sigmoid(i1) * self._activation(j1))
        new_h1 = self._activation(new_c1) * sigmoid(o1)
        new_c2 = (c2 * sigmoid(f2 + self._forget_bias) + sigmoid(i2) * self._activation(j2))
        new_h2 = self._activation(new_c2) * sigmoid(o2)
        new_c = array_ops.concat([new_c1, new_c2], 1)
        new_h = array_ops.concat([new_h1, new_h2], 1)

        if self._state_is_tuple:
          new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
        else:
          new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state

