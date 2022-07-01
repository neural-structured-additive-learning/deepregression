from keras import activations
from keras import backend
from keras import constraints
from keras import initializers
from keras import regularizers
from keras import utils
from keras import layers
import functools
from tensorflow.python.ops import nn_ops
import tensorflow as tf
import keras

class SparseConv(layers.convolutional.Conv):
    def __init__(self,
                 rank,
                filters,
                kernel_size,
                lam=None,
                position_sparsity=-1, # use slide(start,end-1) for multiple indices,
                depth=2,
                strides=1,
                padding='valid',
                data_format=None,
                dilation_rate=1,
                groups=1,
                activation=None,
                use_bias=True,
                multfac_initiliazer=initializers.Ones(),
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                multfac_regularizer=None,
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                trainable=True,
                name=None,
                conv_op=None,
                **kwargs):
        super(SparseConv, self).__init__(
            rank = rank,
            filters = filters,
            kernel_size = kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name,
            conv_op=conv_op,
            **kwargs
    )

        self.lam = lam
        self.multfac_initializer = multfac_initiliazer
        self.position_sparsity = position_sparsity 
        self.depth = depth

        if multfac_regularizer is None and kernel_regularizer is None and lam is not None:
            # blow up penalty corresponding to depth of factorization (AM-GM)
            self.multfac_regularizer = tf.keras.regularizers.L2((self.depth-1)*self.lam)
            self.kernel_regularizer = tf.keras.regularizers.L2(self.lam)
        else:
            self.multfac_regularizer = multfac_regularizer
            
    def _validate_init(self):
        if self.filters is not None and self.filters % self.groups != 0:
            raise ValueError(
              'The number of filters must be evenly divisible by the number of '
              'groups. Received: groups={}, filters={}'.format(
              self.groups, self.filters))

        if not all(self.kernel_size):
            raise ValueError('The argument `kernel_size` cannot contain 0(s). '
                       'Received: %s' % (self.kernel_size,))

        if not all(self.strides):
            raise ValueError('The argument `strides` cannot contains 0(s). '
                       'Received: %s' % (self.strides,))

        if (self.padding == 'causal' and not isinstance(self,
                                                    (Conv1D, SeparableConv1D))):
            raise ValueError('Causal padding is only supported for `Conv1D`'
                       'and `SeparableConv1D`.')
    
    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        if input_channel % self.groups != 0:
            raise ValueError(
              'The number of input channels must be evenly divisible by the number '
              'of groups. Received groups={}, but the input has {} channels '
              '(full input shape is {}).'.format(self.groups, input_channel,
                                                 input_shape))
        kernel_shape = self.kernel_size + (input_channel // self.groups,
                                           self.filters)
        
        # create 1s with same length as kernel_shape tuple                                   
        multfac_shape = [1]*len(kernel_shape)
        # overwrite those that should be overparameterized
        multfac_shape[self.position_sparsity] = kernel_shape[self.position_sparsity]

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
    
        self.multfac = self.add_weight(
            name='multfac',
            shape=tuple(multfac_shape), 
            initializer=self.multfac_initializer,
            regularizer=self.multfac_regularizer,
            constraint=None,
            trainable=True,
            dtype=self.dtype
        )
    
        if self.use_bias:
            self.bias = self.add_weight(
              name='bias',
              shape=(self.filters,),
              initializer=self.bias_initializer,
              regularizer=self.bias_regularizer,
              constraint=self.bias_constraint,
              trainable=True,
              dtype=self.dtype)
        else:
            self.bias = None
        channel_axis = self._get_channel_axis()
        self.input_spec = keras.layers.InputSpec(min_ndim=self.rank + 2,
                                    axes={channel_axis: input_channel})
        # Convert Keras formats to TF native formats.
        if self.padding == 'causal':
            tf_padding = 'VALID'  # Causal padding handled in `call`.
        elif isinstance(self.padding, str):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding
        tf_dilations = list(self.dilation_rate)
        tf_strides = list(self.strides)

        tf_op_name = self.__class__.__name__
        if tf_op_name == 'Conv1D':
            tf_op_name = 'conv1d'  # Backwards compat.

        self._convolution_op = functools.partial(
            nn_ops.convolution_v2,
            strides=tf_strides,
            padding=tf_padding,
            dilations=tf_dilations,
            data_format=self._tf_data_format,
            name=tf_op_name)
        self.built = True

#     def convolution_op(self, inputs, kernel, multfac):
#         if self.padding == 'causal':
#             tf_padding = 'VALID'  # Causal padding handled in `call`.
#         elif isinstance(self.padding, str):
#             tf_padding = self.padding.upper()
#         else:
#             tf_padding = self.padding

#         return tf.nn.convolution(
#             inputs,
#             self.multiply(kernel, multfac),
#             strides=list(self.strides),
#             padding=tf_padding,
#             dilations=list(self.dilation_rate),
#             data_format=self._tf_data_format,
#             name=self.__class__.__name__)

    def call(self, inputs):
        input_shape = inputs.shape

        if self._is_causal:  # Apply causal padding to inputs for Conv1D.
            inputs = tf.pad(inputs, self._compute_causal_padding(inputs))

        outputs = self._convolution_op(inputs, tf.multiply(self.kernel, 
                tf.pow(x = self.multfac, y = (self.depth-1))
                ))

        if self.use_bias:
            output_rank = outputs.shape.rank
            if self.rank == 1 and self._channels_first:
                # nn.bias_add does not accept a 1D input tensor.
                bias = tf.reshape(self.bias, (1, self.filters, 1))
                outputs += bias
            else:
                # Handle multiple batch dimensions.
                if output_rank is not None and output_rank > 2 + self.rank:

                    def _apply_fn(o):
                        return tf.nn.bias_add(o, self.bias, data_format=self._tf_data_format)

                    outputs = utils.conv_utils.squeeze_batch_dims(
                          outputs, _apply_fn, inner_rank=self.rank + 1)
                else:
                    outputs = tf.nn.bias_add(
                      outputs, self.bias, data_format=self._tf_data_format)

        if not tf.executing_eagerly():
          # Infer the static output shape:
            out_shape = self.compute_output_shape(input_shape)
            outputs.set_shape(out_shape)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def _spatial_output_shape(self, spatial_input_shape):
        return [
            utils.conv_utils.conv_output_length(
                length,
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            for i, length in enumerate(spatial_input_shape)
        ]

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        batch_rank = len(input_shape) - self.rank - 1
        try:
            if self.data_format == 'channels_last':
                return tf.TensorShape(
                    input_shape[:batch_rank] +
                    self._spatial_output_shape(input_shape[batch_rank:-1]) +
                    [self.filters])
            else:
                return tf.TensorShape(
                    input_shape[:batch_rank] + [self.filters] +
                    self._spatial_output_shape(input_shape[batch_rank + 1:]))

        except ValueError:
            raise ValueError(
              f'One of the dimensions in the output is <= 0 '
              f'due to downsampling in {self.name}. Consider '
              f'increasing the input size. '
              f'Received input shape {input_shape} which would produce '
              f'output shape with a zero or negative value in a '
              f'dimension.')

    def _recreate_conv_op(self, inputs):  # pylint: disable=unused-argument
        return False

    def get_config(self):
        config = {
            'filters':
                self.filters,
            'kernel_size':
                self.kernel_size,
            'strides':
                self.strides,
            'padding':
                self.padding,
            'data_format':
                self.data_format,
            'dilation_rate':
                self.dilation_rate,
            'groups':
                self.groups,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'lam':
                self.lam,
            'position_sparsity':
                self.position_sparsity,
            'depth':
                self.depth,
            'multfac_initializer':
                self.multfac_initializer,
            'multfac_regularizer':
                self.multfac_regularizer
        }
        base_config = super(layers.convolutional.Conv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _compute_causal_padding(self, inputs):
        """Calculates padding for 'causal' option for 1-d conv layers."""
        left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
        if getattr(inputs.shape, 'ndims', None) is None:
            batch_rank = 1
        else:
            batch_rank = len(inputs.shape) - 2
        if self.data_format == 'channels_last':
            causal_padding = [[0, 0]] * batch_rank + [[left_pad, 0], [0, 0]]
        else:
            causal_padding = [[0, 0]] * batch_rank + [[0, 0], [left_pad, 0]]
        return causal_padding

    def _get_channel_axis(self):
        if self.data_format == 'channels_first':
            return -1 - self.rank
        else:
            return -1

    def _get_input_channel(self, input_shape):
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs should be defined. '
                           f'The input_shape received is {input_shape}, '
                           f'where axis {channel_axis} (0-based) '
                           'is the channel dimension, which found to be `None`.')
        return int(input_shape[channel_axis])

    def _get_padding_op(self):
        if self.padding == 'causal':
            op_padding = 'valid'
        else:
            op_padding = self.padding
        if not isinstance(op_padding, (list, tuple)):
            op_padding = op_padding.upper()
        return op_padding


class SparseConv2D(SparseConv):

    def __init__(self,
               filters,
               kernel_size,
               lam=None,
               position_sparsity=-1,
               depth = 2,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               groups=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               multfac_initiliazer=initializers.Ones(),
               bias_initializer='zeros',
               kernel_regularizer=None,
               multfac_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
        super(SparseConv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            lam=lam,
            position_sparsity=position_sparsity,
            depth=depth,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            multfac_initiliazer=initializers.get(multfac_initiliazer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            multfac_regularizer=regularizers.get(multfac_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)
