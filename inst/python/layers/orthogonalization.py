import tensorflow as tf

def orthog_tf(Y, X):
    Q = tf.linalg.qr(X, full_matrices=False, name="QR").q
    X_XtXinv_Xt = tf.linalg.matmul(Q, tf.linalg.matrix_transpose(Q))
    return(tf.subtract(Y, tf.linalg.matmul(X_XtXinv_Xt, Y)))

class Orthogonalization(tf.keras.layers.Layer):
    def __init__(self, deactivate_at_test = True, **kwargs):
        self.deactivate_at_test = deactivate_at_test
        super(Orthogonalization, self).__init__(**kwargs)

    def call(self, Y, X, training=None):
        if not self.deactivate_at_test:
            return orthog_tf(Y, X)
        if training:
            return orthog_tf(Y, X)
        else:
            return Y
