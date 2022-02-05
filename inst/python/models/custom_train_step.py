import tensorflow as tf
import numpy as np
import math
from tensorflow import keras
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop

def build_customKeras(custom_update = None):

	if(custom_update is None):
	# this is just for comparison reasons with default keras routine

		class customKeras(keras.models.Model):
		
			def train_step(self, data):
				data = data_adapter.expand_1d(data)
				x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
				# Run forward pass.
				with backprop.GradientTape() as tape:
					y_pred = self(x, training=True)
					loss = self.compiled_loss(
					  y, y_pred, sample_weight, regularization_losses=self.losses)
			 	# Run backwards pass with custom minimization
				# grads = tape.gradient(loss, self.trainable_variables)
				# self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
				grads_and_vars = self.optimizer._compute_gradients(
					loss, var_list=self.trainable_variables, grad_loss=None, tape=tape)
				self.optimizer.apply_gradients(grads_and_vars)
				self.compiled_metrics.update_state(y, y_pred, sample_weight)
				# Collect metrics to return
				return_metrics = {}
				for metric in self.metrics:
					result = metric.result()
					if isinstance(result, dict):
						return_metrics.update(result)
					else:
						return_metrics[metric.name] = result
				return return_metrics
				
	else:
	
		raise RuntimeError("Not implemented yet.")

	
	return(customKeras)
