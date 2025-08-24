# convlasso.py — Keras 3 public-API version 

from keras import activations, constraints, initializers, regularizers, utils, layers
import tensorflow as tf


def _to_tuple(x, rank, name):
    if isinstance(x, int):
        return (x,) * rank
    if isinstance(x, (list, tuple)) and len(x) == rank:
        return tuple(int(v) for v in x)
    raise ValueError(f"`{name}` must be int or tuple/list of length {rank}. Got: {x}")


def _tf_data_format(rank, data_format):
    """Return TF data_format expected by tf.nn.convolution and tf.nn.bias_add."""
    if data_format not in {None, "channels_last", "channels_first"}:
        raise ValueError(f"data_format must be None, 'channels_last', or 'channels_first'. Got: {data_format}")
    channels_last = (data_format in (None, "channels_last"))
    if rank == 1:
        return "NWC" if channels_last else "NCW"
    if rank == 2:
        return "NHWC" if channels_last else "NCHW"
    if rank == 3:
        return "NDHWC" if channels_last else "NCDHW"
    raise ValueError(f"Unsupported rank: {rank}")


class SparseConv(layers.Layer):
    """
    Factorized sparse convolution with multiplicative factor `multfac` applied to the kernel.
    Uses public Keras 3 API and tf.nn.convolution.
    """

    def __init__(
        self,
        rank,
        filters,
        kernel_size,
        lam=None,
        position_sparsity=-1,   # index in kernel shape to over-parameterize
        depth=2,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        groups=1,
        activation=None,
        use_bias=True,
        multfac_initiliazer=initializers.Ones(),
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        multfac_regularizer=None,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        **kwargs,
    ):
        super().__init__(trainable=trainable, name=name, activity_regularizer=activity_regularizer, **kwargs)
        self.rank = int(rank)
        self.filters = int(filters) if filters is not None else None
        self.kernel_size = _to_tuple(kernel_size, self.rank, "kernel_size")
        self.strides = _to_tuple(strides, self.rank, "strides")
        self.dilation_rate = _to_tuple(dilation_rate, self.rank, "dilation_rate")
        self.padding = padding
        self.data_format = data_format
        self.groups = int(groups)
        self.activation = activations.get(activation)
        self.use_bias = bool(use_bias)

        # regularization and init
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.multfac_initializer = initializers.get(multfac_initiliazer)
        self.multfac_regularizer = regularizers.get(multfac_regularizer)

        self.lam = lam
        self.position_sparsity = int(position_sparsity)
        self.depth = int(depth)

        # If explicit regularizers are not provided, derive from lam.
        if self.multfac_regularizer is None and self.kernel_regularizer is None and self.lam is not None:
            # AM-GM style split: multfac gets (depth-1)*lam, kernel gets lam
            self.multfac_regularizer = tf.keras.regularizers.L2((self.depth - 1) * self.lam)
            self.kernel_regularizer = tf.keras.regularizers.L2(self.lam)

        # simple validation
        if self.filters is not None and self.filters % self.groups != 0:
            raise ValueError(f"`filters` must be divisible by `groups`. Got filters={self.filters}, groups={self.groups}")
        if not all(self.kernel_size):
            raise ValueError(f"`kernel_size` cannot contain 0. Got {self.kernel_size}")
        if not all(self.strides):
            raise ValueError(f"`strides` cannot contain 0. Got {self.strides}")
        if self.padding == "causal" and self.rank != 1:
            raise ValueError("Causal padding is only supported for rank==1.")

        self._built_dynamic = False
        self.input_spec = None
        self.kernel = None
        self.multfac = None
        self.bias = None

    # Convenience flags
    @property
    def _is_causal(self):
        return self.rank == 1 and self.padding == "causal"

    @property
    def _channels_first(self):
        return self.data_format == "channels_first"

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        channel_axis = -1 - self.rank if self._channels_first else -1
        in_ch = input_shape[channel_axis]
        if in_ch is None:
            raise ValueError("The channel dimension of the inputs must be defined.")
        in_ch = int(in_ch)
        if in_ch % self.groups != 0:
            raise ValueError(
                f"Input channels must be divisible by groups. Got in_ch={in_ch}, groups={self.groups}"
            )

        # Kernel shape: spatial dims + (in_ch/groups, filters)
        kernel_shape = tuple(self.kernel_size) + (in_ch // self.groups, self.filters)

        # multfac shape: broadcast across all dims except one sparsified index
        multfac_shape = [1] * len(kernel_shape)
        ps = self.position_sparsity
        if ps < 0:
            ps = len(kernel_shape) + ps
        if ps < 0 or ps >= len(kernel_shape):
            raise ValueError(f"position_sparsity out of range for kernel_shape {kernel_shape}. Got {self.position_sparsity}")
        multfac_shape[ps] = kernel_shape[ps]

        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
        )
        self.multfac = self.add_weight(
            name="multfac",
            shape=tuple(multfac_shape),
            initializer=self.multfac_initializer,
            regularizer=self.multfac_regularizer,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
            )

        # Keras InputSpec for channel count
        self.input_spec = layers.InputSpec(
            min_ndim=self.rank + 2,
            axes={channel_axis: in_ch},
        )

        self._tf_data_format = _tf_data_format(self.rank, self.data_format)
        self._built_dynamic = True
        super().build(input_shape)

    def _compute_causal_padding(self, inputs):
        # Only rank==1
        left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
        batch_rank = len(inputs.shape) - 2 if getattr(inputs.shape, "ndims", None) is not None else 1
        if self.data_format in (None, "channels_last"):
            return [[0, 0]] * batch_rank + [[left_pad, 0], [0, 0]]
        return [[0, 0]] * batch_rank + [[0, 0], [left_pad, 0]]

    def _op_padding(self):
        # tf.nn.convolution expects "VALID" or "SAME" or explicit paddings
        if self.padding == "causal":
            return "VALID"
        if not isinstance(self.padding, (list, tuple)):
            return str(self.padding).upper()
        # explicit paddings are not used here
        return self.padding

    def call(self, inputs):
        x = inputs
        if self._is_causal:
            x = tf.pad(x, self._compute_causal_padding(x))

        # effective kernel
        eff_kernel = self.kernel * tf.pow(tf.abs(self.multfac), self.depth - 1)

        y = tf.nn.convolution(
            x,
            eff_kernel,
            strides=list(self.strides),
            padding=self._op_padding(),
            dilations=list(self.dilation_rate),
            data_format=self._tf_data_format,
            feature_group_count=self.groups,
        )

        if self.use_bias:
            # tf.nn.bias_add expects "N..C" or "NC.."
            y = tf.nn.bias_add(y, self.bias, data_format=self._tf_data_format)

        if not tf.executing_eagerly():
            y.set_shape(self.compute_output_shape(tf.TensorShape(inputs.shape)))

        if self.activation is not None:
            y = self.activation(y)
        return y

    def _spatial_output_shape(self, spatial_input_shape):
        return [
            utils.conv_utils.conv_output_length(
                length,
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i],
            )
            for i, length in enumerate(spatial_input_shape)
        ]

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        batch_rank = len(input_shape) - self.rank - 1
        try:
            if self.data_format in (None, "channels_last"):
                return tf.TensorShape(
                    input_shape[:batch_rank]
                    + self._spatial_output_shape(input_shape[batch_rank:-1])
                    + [self.filters]
                )
            else:
                return tf.TensorShape(
                    input_shape[:batch_rank]
                    + [self.filters]
                    + self._spatial_output_shape(input_shape[batch_rank + 1 :])
                )
        except ValueError:
            raise ValueError(
                f"Invalid output dimensions due to downsampling in {self.name}. "
                f"Input shape {input_shape} would produce a nonpositive dimension."
            )

    def get_config(self):
        config = {
            "rank": self.rank,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "groups": self.groups,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "lam": self.lam,
            "position_sparsity": self.position_sparsity,
            "depth": self.depth,
            "multfac_initializer": initializers.serialize(self.multfac_initializer),
            "multfac_regularizer": regularizers.serialize(self.multfac_regularizer),
        }
        base = super().get_config()
        base.update(config)
        return base


class SparseConv2D(SparseConv):
    def __init__(
        self,
        filters,
        kernel_size,
        lam=None,
        position_sparsity=-1,
        depth=2,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        multfac_initiliazer=initializers.Ones(),
        bias_initializer="zeros",
        kernel_regularizer=None,
        multfac_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
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
            **kwargs,
        )
