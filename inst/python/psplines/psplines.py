import tensorflow as tf
import numpy as np
import math
from tensorflow import keras
import tensorflow.keras.regularizers as regularizers

### Helper functions matrix algebra
def tf_crossprod(a,b):
    return(tf.matmul(tf.transpose(a), b))
def tf_incross(a,b):
    return(tf.matmul(tf_crossprod(a,b),a))
def tf_unitvec(n,j):
    return(tf.transpose(tf.eye(n)[None,j,:]))
def tf_operator_multiply(scalar, operator):
    return(operator.matmul(tf.linalg.diag(tf.repeat(scalar, operator.shape[0]))))
def vecmatvec(a, B, c=None, sparse_mat = False):
    if c is None:
        c = a
    #return(tf.matmul(tf.transpose(a),tf.linalg.matvec(B, tf.squeeze(c, [1]), a_is_sparse = sparse_mat)))
    return(tf.keras.backend.sum(tf.keras.backend.batch_dot(a, tf.keras.backend.dot(B, c), axes=1)))

### Stuff for smoothing
def lambda_times_P(lambdas, Plist):
    # doing a shitty trafo from LinOp -> Tensor -> LinOp because TF does not
    # support LinOp multiplication with no type conversion (?)
    return([tf.linalg.LinearOperatorFullMatrix((tf.multiply(lambdas[i],
                                                            tf.cast(Plist[i].to_dense(), dtype="float32")))) 
            for i in range(lambdas.shape[0])])

def weight_decay(vecOld, vecNew, rate = 0.01):
    return(vecOld*(1-rate) + vecNew*rate)

def update_lambda(S, I, weights, lambdas, mask, constdiv = 0, constinv = 0):
    lambdas = tf.exp(lambdas)
    # set_trace()
    S_lambda = tf.linalg.LinearOperatorBlockDiag(lambda_times_P(lambdas, S)).to_dense()
    def calcHinv(x):
        return(tf.linalg.solve(I + S_lambda + tf.diag(tf.ones(I.shape[0])*constinv), x))
    new_lambdas = tf.random.normal([1,1])
    for j in range(lambdas.shape[0]):
        if(mask[j]==0):
            new_lambdas = tf.concat([new_lambdas, lambdas[j,:]], axis = 0)
        else:        
            p_j = S[j].shape[0] # tf.rank(S[j]) #
            unitvec = tf_unitvec(lambdas.shape[0], j)
            S_j = tf.linalg.LinearOperatorBlockDiag(lambda_times_P(unitvec, S)).to_dense()
#         p_j = tf.rank(S_j) 
            Hinv = calcHinv(S_j)
            tracePart = tf.linalg.trace(Hinv)
            bTSb = tf_incross(weights, S_j)
            bTSb += tf.constant(constdiv)
            new_lambdas = tf.concat([new_lambdas, (p_j + tracePart) * lambdas[j,:] / bTSb], axis = 0)

    return(new_lambdas[1:(lambdas.shape[0]+1),:], calcHinv)

### Convenience functions
def get_specific_weight(string_to_match, weights, index=True, invert=False):

    indices = []
    for j in range(len(string_to_match)):
        
        this_indices = [string_to_match[j] in weights[i].name for i in range(len(weights))]       
        # set_trace()
    
        if(len(this_indices)>0):
            wh = np.where(this_indices)[0][0]
            indices = np.append(indices,wh)
        
    if(len(indices)==0):
        return([])
    
    indices = [int(li) for li in indices.tolist()]
    
    if invert:
        indices = list(set(list(range(len(weights)))).difference(set(set(indices))))
    
    if index:
        return(indices)
    else:
        return(weights[indices])
    
def get_specific_layer(string_to_match, layers, index=True, invert=False):
#     set_trace()
    indices = []
    for j in range(len(string_to_match)):
        this_indices = [string_to_match[j] in layer for layer.name in layers]

        if(len(this_indices)>0):
            wh = np.where(this_indices)[0][0]
            indices = np.append(indices,wh)
                
    indices = [int(li) for li in indices.tolist()]
                
    if invert:
        indices = list(set(list(range(len(weights)))).difference(set(set(indices))))
        
    if(len(indices)==0):
        return([])
    
    if index:
        return(wh)
    else:
        return(layers[wh])    
    

def layer_spline(P, units, name, trainable = True, kernel_initializer = "glorot_uniform"):
    return(tf.keras.layers.Dense(units = units, name = name, use_bias=False, kernel_regularizer = squaredPenalty(P, 1), trainable = trainable, kernel_initializer = kernel_initializer))

def layer_splineVC(P, units, name, trainable = True, kernel_initializer = "glorot_uniform"):
    return(tf.keras.layers.Dense(units = units, name = name, use_bias=False, kernel_regularizer = squaredPenaltyVC(P, 1), trainable = trainable, kernel_initializer = kernel_initializer))

class squaredPenalty(regularizers.Regularizer):

    def __init__(self, P, strength):
        self.strength = strength
        self.P = P

    def __call__(self, x):
        return self.strength * tf.reduce_sum(vecmatvec(x, tf.cast(self.P, dtype="float32"), sparse_mat = True))

    def get_config(self):
        return {'strength': self.strength, 'P': self.P}

class squaredPenaltyVC(regularizers.Regularizer):

    def __init__(self, P, strength, nlev):
        self.strength = strength
        self.P = P
        self.nlev = nlev

    def __call__(self, x):
        x_splitted = tf.split(x, self.nlev)
        pen = 0
        for x_k in x_splitted:
            pen += tf.reduce_sum(vecmatvec(x_k, tf.cast(self.P, dtype="float32"), sparse_mat = True))
        return self.strength * pen

    def get_config(self):
        return {'strength': self.strength, 'P': self.P}
    
class PenLinear(tf.keras.layers.Layer):
    def __init__(self, units, lambdas, mask, P, n, nr):
        super(PenLinear, self).__init__()
        self.units = units
        self.lambdas = tf.Variable(lambdas, name = "lambda" + str(nr))
        self.mask = mask
        self.P = P
        self.n = n

    def get_penalty(self, x=None):
        lambdas = self.calc_lambda_mask()
        lP = lambda_times_P(lambdas, self.P)
        bigP = tf.linalg.LinearOperatorBlockDiag(lP).to_dense() / self.n
        lambdaJ = tf_incross(self.w, bigP)
        return(tf.reshape(lambdaJ,[]))

    def calc_lambda_mask(self):
        return(tf.math.multiply(tf.exp(self.lambdas), self.mask))
        
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True
        )

    def get_config(self):
        return({"name": self.name})

    def call(self, inputs):
        return tf.matmul(inputs, self.w)

def get_masks(mod):
    masks = []
    for layer in mod.layers:
        if 'pen_layer' in layer.name:
            masks.append(layer.mask)
    return(masks)
    
def exp_decay(x, fac = 1):
    return(x * np.exp(-fac))
    
def build_kerasGAM(fac = 0.01, lr_scheduler = None, avg_over_past = False):


    class kerasGAM(keras.models.Model):

        fac_update = tf.constant(fac)
        
        def train_step(self, data):
            x, y = data

            with tf.GradientTape() as t2:
                with tf.GradientTape() as t1:
                    y_pred = self(x, training=True)  # Forward pass
                    # Compute our own loss
                    loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
#                 H = tf.hessians(loss, y_pred)

                # Compute gradients
                    trainable_vars = self.trainable_variables
                    beta_index = get_specific_weight(["pen_linear"], trainable_vars)
                    lambda_index = get_specific_weight(["lambda"], trainable_vars)
                    # not_lambda_index = get_specific_weight(["lambda"], trainable_vars, invert = True)
                    betas = trainable_vars[beta_index[0]]
                    lambdas = trainable_vars[lambda_index[0]]
                
                gradients = t1.gradient(loss, trainable_vars)

            # ====================================
#             gradients_not_lambda = gradients[not_lambda_index]
                gradients_betas = gradients[beta_index[0]]
            
            
            H = tf.reshape(tf.stack(t2.jacobian(gradients_betas, betas)), [betas.shape[0],betas.shape[0]])
            
            update = update_lambda(Plist, H, betas, lambdas, get_masks(self))
            phi = self.compiled_loss(y, y_pred) / (y.shape[0] - tf.linalg.trace(update[1](tf_crossprod(x,x))))
            
            if lr_scheduler is not None:
                fac_update = lr_scheduler(fac_update)                                
            
            lambdas.assign(phi*update[0]*fac_update + (1-fac_update)*lambdas)      

            betas.assign(betas-update[1](gradients_betas))
            # ====================================

            # Compute our own metrics
            # loss_tracker.update_state(loss)
            #  return {"loss": loss_tracker.result()}
            
            self.compiled_metrics.update_state(y, y_pred)
            return {m.name: m.result() for m in self.metrics}

    return(kerasGAM)   
