"""
Localise faults in offline for any faults
"""
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import tensorflow.keras.backend as K
from tqdm import tqdm
import lstm_layer
import utils.model_util as model_util
import os

# "divided by zeros" is handleded afterward
np.seterr(divide='ignore', invalid='ignore')

def get_target_weights(model, path_to_keras_model, indices_to_target = None):
	"""
	return indices to weight layers denoted by indices_to_target, or return all trainable layers
	"""
	import re
	targeting_clname_pattns = ['Dense*', 'Conv*', '.*LSTM*'] #if not target_all else None
	is_target = lambda clname,targets: (targets is None) or any([bool(re.match(t,clname)) for t in targets])
		
	if model is None:
		assert path_to_keras_model is not None
		model = load_model(path_to_keras_model, compile=False)

	target_weights = {} # key = layer index, value: [weight value, layer name]
	if indices_to_target is not None:
		num_layers = len(model.layers)
		indices_to_target = [idx if idx >= 0 else num_layers + idx for idx in indices_to_target]

		for i, layer in enumerate(model.layers):
			if i in indices_to_target:
				ws = layer.get_weights()
				assert len(ws) > 0, "the target layer doesn't have weight"
				#target_weights[i] = ws[0] # we will target only the weight, and not the bias
				target_weights[i] = [ws[0], type(layer).__name__]
	else:
		for i, layer in enumerate(model.layers):
			class_name = type(layer).__name__
			if is_target(class_name, targeting_clname_pattns): 
				ws = layer.get_weights()
				if len(ws): # has weight
					if model_util.is_FC(class_name) or model_util.is_C2D(class_name):
						target_weights[i] = [ws[0], type(layer).__name__]
					elif model_util.is_LSTM(class_name): 
						# for LSTM, even without bias, a fault can be in the weights of the kernel 
						# or the recurrent kernel (hidden state handling)
						assert len(ws) == 3, ws
						# index 0: for the kernel, index 1: for the recurrent kernel
						target_weights[i] = [ws[:-1], type(layer).__name__] 
					else:
						print ("{} not supported yet".format(class_name))
						assert False

	return target_weights


def compute_gradient_to_output(path_to_keras_model, 
idx_to_target_layer, X, 
by_batch = False, on_weight = False, wo_reset = False):
	"""
	compute gradients normalisesd and averaged for a given input X
	on_weight = False -> on output of idx_to_target_layer'th layer
	"""
	from sklearn.preprocessing import Normalizer
	from collections.abc import Iterable
	norm_scaler = Normalizer(norm = "l1")
		
	model = load_model(path_to_keras_model, compile = False)
	
	# Create intermediate models
	intermediate_model = tf.keras.Model(
		inputs=model.input,
		outputs=[model.layers[idx_to_target_layer].output, model.output]
	)

	# since this might cause OOM error, divide them 
	num = X.shape[0]
	if by_batch:
		batch_size = 64 
		num_split = int(np.round(num/batch_size))
		if num_split == 0:
			num_split = 1
		chunks = np.array_split(np.arange(num), num_split)
	else:
		chunks = [np.arange(num)]

	if not on_weight:	
		grad_shape = tuple([num] + [int(v) for v in model.layers[idx_to_target_layer].output.shape[1:]])
		gradient = np.zeros(grad_shape)
		for chunk in chunks:
			x_chunk = tf.convert_to_tensor(X[chunk])
			with tf.GradientTape(persistent=True) as tape:
				tape.watch(x_chunk)
				intermediate_output, final_output = intermediate_model(x_chunk)
				# Sum across all dimensions except batch to get a scalar per example
				summed_output = tf.reduce_sum(final_output, axis=list(range(1, len(final_output.shape))))
			_gradient = tape.gradient(summed_output, intermediate_output)
			gradient[chunk] = _gradient.numpy()
			del tape
	
		gradient = np.abs(gradient)
		reshaped_gradient = gradient.reshape(gradient.shape[0],-1) # flatten
		norm_gradient = norm_scaler.fit_transform(reshaped_gradient) # normalised
		mean_gradient = np.mean(norm_gradient, axis = 0) # compute mean for a given input
		ret_gradient = mean_gradient.reshape(gradient.shape[1:]) # reshape to the orignal shape
		if not wo_reset:
			reset_keras([gradient])
		return ret_gradient
	else: # on a weight variable
		gradients = []
		for chunk in chunks:
			x_chunk = tf.convert_to_tensor(X[chunk])
			with tf.GradientTape() as tape:
				tape.watch(x_chunk)
				_, final_output = intermediate_model(x_chunk)
				summed_output = tf.reduce_sum(final_output, axis=list(range(1, len(final_output.shape))))
			_gradients = tape.gradient(summed_output, model.layers[idx_to_target_layer].weights[:-1])
			if len(gradients) == 0:
				gradients = _gradients 
			else:
				for i in range(len(_gradients)):
					gradients[i] += _gradients[i]
		ret_gradients = list(map(lambda x: np.abs(x.numpy()), gradients))
		if not wo_reset:
			reset_keras(gradients)

		if len(ret_gradients) == 0:
			return ret_gradients[0]
		else:
			return ret_gradients


def compute_gradient_per_sample(path_to_keras_model, idx_to_target_layer, X, y, 
loss_func = 'categorical_cross_entropy', **kwargs):
	"""
	计算每个样本对每个权重的梯度
	返回: [样本数, 权重行, 权重列] 的梯度数组
	"""
	import tensorflow as tf
	from tensorflow.keras.models import load_model
	
	model = load_model(path_to_keras_model, compile = False)
	targets = model.layers[idx_to_target_layer].weights[:-1]
	
	gradients_per_sample = []
	
	for sample_idx in range(len(X)):
		# 处理单个样本
		x_sample = tf.convert_to_tensor(X[sample_idx:sample_idx+1])
		y_sample = tf.convert_to_tensor(y[sample_idx:sample_idx+1])
		
		with tf.GradientTape(persistent=True) as tape:
			tape.watch(x_sample)
			predictions = model(x_sample)
			if loss_func == 'categorical_cross_entropy':
				loss = tf.nn.softmax_cross_entropy_with_logits(
					labels = y_sample,
					logits = predictions)
			elif loss_func == 'binary_crossentropy':
				if 'name' in kwargs.keys():
					kwargs.pop("name")
				loss = tf.keras.losses.binary_crossentropy(y_sample, predictions)
				loss.__dict__.update(kwargs)
			elif loss_func in ['mean_squared_error', 'mse']:
				if 'name' in kwargs.keys():
					kwargs.pop("name")
				loss = tf.keras.losses.MeanSquaredError()(y_sample, predictions)
				loss.__dict__.update(kwargs)
			else:
				print (loss_func)
				print ("{} not supported yet".format(loss_func))
				assert False
			loss = tf.reduce_mean(loss)
		
		_gradients = tape.gradient(loss, targets)
		# 只取第一个目标权重的梯度（通常是主要的权重矩阵）
		sample_grad = _gradients[0].numpy()
		gradients_per_sample.append(sample_grad)
		del tape
	
	reset_keras()
	return gradients_per_sample


def compute_gradient_to_loss(path_to_keras_model, idx_to_target_layer, X, y, 
by_batch = False, wo_reset = False, loss_func = 'categorical_cross_entropy', **kwargs):
	"""
	compute gradients for the loss. 
	kwargs contains the key-word argumenets required for the loss funation
	"""
	model = load_model(path_to_keras_model, compile = False)
	targets = model.layers[idx_to_target_layer].weights[:-1]

	# since this might cause OOM error, divide them 
	num = X.shape[0]
	if by_batch:
		batch_size = 64
		num_split = int(np.round(num/batch_size))
		if num_split == 0:
			num_split += 1
		chunks = np.array_split(np.arange(num), num_split)
	else:
		chunks = [np.arange(num)]
	
	gradients = [[] for _ in range(len(targets))]
	for chunk in chunks:
		x_chunk = tf.convert_to_tensor(X[chunk])
		y_chunk = tf.convert_to_tensor(y[chunk])
		with tf.GradientTape(persistent=True) as tape:
			tape.watch(x_chunk)
			predictions = model(x_chunk)
			if loss_func == 'categorical_cross_entropy':
				loss = tf.nn.softmax_cross_entropy_with_logits(
					labels = y_chunk,
					logits = predictions)
			elif loss_func == 'binary_crossentropy':
				if 'name' in kwargs.keys():
					kwargs.pop("name")
				loss = tf.keras.losses.binary_crossentropy(y_chunk, predictions)
				loss.__dict__.update(kwargs)
			elif loss_func in ['mean_squared_error', 'mse']:
				if 'name' in kwargs.keys():
					kwargs.pop("name")
				loss = tf.keras.losses.MeanSquaredError()(y_chunk, predictions)
				loss.__dict__.update(kwargs)
			else:
				print (loss_func)
				print ("{} not supported yet".format(loss_func))
				assert False
			loss = tf.reduce_mean(loss)
		_gradients = tape.gradient(loss, targets)
		for i,_gradient in enumerate(_gradients):
			gradients[i].append(_gradient.numpy())
		del tape

	for i, gradients_p_chunk in enumerate(gradients):
		gradients[i] = np.abs(np.sum(np.asarray(gradients_p_chunk), axis = 0)) # combine

	if not wo_reset:
		reset_keras(gradients)
	return gradients[0] if len(gradients) == 1 else gradients


def reset_keras(delete_list = None, frac = 1):
	if delete_list is None:
		K.clear_session()
	else:
		import gc
		K.clear_session()
		try:
			for d in delete_list:
				del d
		except:
			pass
		gc.collect()

def sample_input_for_loc_by_rd(
	indices_to_chgd, 
	indices_to_unchgd,
	predictions = None, ys = None):
	"""
	"""
	num_chgd = len(indices_to_chgd)
	if num_chgd >= len(indices_to_unchgd): # no need to do any sampling
		return indices_to_chgd, indices_to_unchgd
	
	if predictions is None and ys is None:
		sampled_indices_to_unchgd = np.random.choice(indices_to_unchgd, num_chgd, replace = False)	
		return indices_to_chgd, sampled_indices_to_unchgd
	else:
		_, sampled_indices_to_unchgd = sample_input_for_loc_sophis(
			indices_to_chgd, 
			indices_to_unchgd, 
			predictions, ys)

		return indices_to_chgd, sampled_indices_to_unchgd


def sample_input_for_loc_sophis(
	indices_to_chgd, 
	indices_to_unchgd, 
	predictions, ys):
	"""
	prediction -> model ouput. Right before outputing as the final classification result 
		from 0~len(indices_to_unchgd)-1, the results of unchagned
		from len(indices_to_unchgd)~end, the results of changed 
	sample the indices to changed and unchanged behaviour later used for localisation 
	"""
	if len(ys.shape) > 1 and ys.shape[-1] > 1:
		pred_labels = np.argmax(predictions, axis = 1)
		y_labels = np.argmax(ys, axis = 1) 
	else:
		pred_labels = np.round(predictions).flatten()
		y_labels = ys

	_indices = np.zeros(len(indices_to_unchgd) + len(indices_to_chgd))
	_indices[:len(indices_to_unchgd)] = indices_to_unchgd
	_indices[len(indices_to_unchgd):] = indices_to_chgd
	
	###  checking  ###
	_indices_to_unchgd = np.where(pred_labels == y_labels)[0]; _indices_to_unchgd.sort()
	indices_to_unchgd = np.asarray(indices_to_unchgd); indices_to_unchgd.sort()
	_indices_to_chgd = np.where(pred_labels != y_labels)[0]; _indices_to_chgd.sort()
	indices_to_chgd = np.asarray(indices_to_chgd); indices_to_chgd.sort()

	assert all(indices_to_unchgd == _indices[_indices_to_unchgd])
	assert all(indices_to_chgd == np.sort(_indices[_indices_to_chgd]))
	###  checking end  ###

	# here, only the labels of the ys (original labels) are considered
	uniq_labels = np.unique(y_labels[_indices_to_unchgd]); uniq_labels.sort()
	grouped_by_label = {uniq_label:[] for uniq_label in uniq_labels}
	for idx in _indices_to_unchgd:	
		pred_label = pred_labels[idx]
		grouped_by_label[pred_label].append(idx)

	num_unchgd = len(indices_to_unchgd)
	num_chgd = len(indices_to_chgd)
	sampled_indices_to_unchgd = []
	num_total_sampled = 0
	for _,vs in grouped_by_label.items():
		num_sample = int(np.round(num_chgd * len(vs)/num_unchgd))
		if num_sample <= 0:
			num_sample = 1
		
		if num_sample > len(vs):
			num_sample = len(vs)

		sampled_indices_to_unchgd.extend(list(np.random.choice(vs, num_sample, replace = False)))
		num_total_sampled += num_sample

	#print ("Total number of sampled: {}".format(num_total_sampled))
	return indices_to_chgd, sampled_indices_to_unchgd


def compute_FI_and_GL(
	X, y,
	indices_to_target,
	target_weights,
	is_multi_label = True, 
	path_to_keras_model = None):
	"""
	compute FL and GL for the given inputs
	"""

	## Now, start localisation !!! ##
	from sklearn.preprocessing import Normalizer
	from collections.abc import Iterable
	norm_scaler = Normalizer(norm = "l1")
	total_cands = {}
	FIs = None; grad_scndcr = None

	#t0 = time.time()
	## slice inputs
	target_X = X[indices_to_target]
	target_y = y[indices_to_target]
	
	# get loss func
	loss_func = model_util.get_loss_func(is_multi_label = is_multi_label)
	model = None
	for idx_to_tl, vs in target_weights.items():
		t1 = time.time()
		t_w, lname = vs
		model = load_model(path_to_keras_model, compile = False)
		if idx_to_tl == 0: 
			# meaning the model doesn't specify the input layer explicitly
			prev_output = target_X
		else:
			prev_output = model.layers[idx_to_tl - 1].output
		layer_config = model.layers[idx_to_tl].get_config() 

		if model_util.is_FC(lname):
			from_front = []
			if idx_to_tl == 0 or idx_to_tl - 1 == 0:
				prev_output = target_X
			else:
				t_model = Model(inputs = model.input, outputs = model.layers[idx_to_tl - 1].output)
				prev_output = t_model.predict(target_X)
			if len(prev_output.shape) == 3:
				prev_output = prev_output.reshape(prev_output.shape[0], prev_output.shape[-1])
			
			for idx in tqdm(range(t_w.shape[-1])):
				assert int(prev_output.shape[-1]) == t_w.shape[0], "{} vs {}".format(
					int(prev_output.shape[-1]), t_w.shape[0])
					
				output = np.multiply(prev_output, t_w[:,idx]) # -> shape = prev_output.shape
				output = np.abs(output)
				output = norm_scaler.fit_transform(output) 
				output = np.mean(output, axis = 0)
				from_front.append(output) 
			
			from_front = np.asarray(from_front)
			from_front = from_front.T
			from_behind = compute_gradient_to_output(path_to_keras_model, idx_to_tl, target_X)
			#print ("shape", from_front.shape, from_behind.shape)
			FIs = from_front * from_behind
			############ FI end #########

			# Gradient
			grad_scndcr = compute_gradient_to_loss(
				path_to_keras_model, idx_to_tl, target_X, target_y, loss_func=loss_func)
			# G end
		elif model_util.is_C2D(lname):
			is_channel_first = layer_config['data_format'] == 'channels_first'
			if idx_to_tl == 0 or idx_to_tl - 1 == 0:
				prev_output_v = target_X
			else:
				t_model = Model(inputs = model.input, outputs = model.layers[idx_to_tl - 1].output)
				prev_output_v = t_model.predict(target_X)
			tr_prev_output_v = np.moveaxis(prev_output_v, [1,2,3],[3,1,2]) if is_channel_first else prev_output_v

			kernel_shape = t_w.shape[:2] 
			strides = layer_config['strides']
			padding_type =  layer_config['padding']
			if padding_type == 'valid':
				paddings = [0,0]
			else:
				if padding_type == 'same':
					#P = ((S-1)*W-S+F)/2
					true_ws_shape = [t_w.shape[0], t_w.shape[-1]] # Channel_in, Channel_out
					paddings = [int(((strides[i]-1)*true_ws_shape[i]-strides[i]+kernel_shape[i])/2) for i in range(2)]
				elif not isinstance(padding_type, str) and isinstance(padding_type, Iterable): # explicit paddings given
					paddings = list(padding_type)
					if len(paddings) == 1:
						paddings = [paddings[0], paddings[0]]
				else:
					print ("padding type: {} not supported".format(padding_type))
					paddings = [0,0]
					assert False

				# add padding
				if is_channel_first:
					paddings_per_axis = [[0,0], [0,0], [paddings[0], paddings[0]], [paddings[1], paddings[1]]]
				else:
					paddings_per_axis = [[0,0], [paddings[0], paddings[0]], [paddings[1], paddings[1]], [0,0]]
				
				tr_prev_output_v = np.pad(tr_prev_output_v, paddings_per_axis, 
					mode = 'constant', constant_values = 0) # zero-padding

			if is_channel_first:
				num_kernels = int(prev_output.shape[1]) # Channel_in
			else: # channels_last
				assert layer_config['data_format'] == 'channels_last', layer_config['data_format']
				num_kernels = int(prev_output.shape[-1]) # Channel_in
			assert num_kernels == t_w.shape[2], "{} vs {}".format(num_kernels, t_w.shape[2])
			#print ("t_w***", t_w.shape)

			# H x W				
			if is_channel_first:
				# the last two (front two are # of inputs and # of kernels (Channel_in))
				input_shape = [int(v) for v in prev_output.shape[2:]] 
			else:
				input_shape = [int(v) for v in prev_output.shape[1:-1]]

			# (W1−F+2P)/S+1, W1 = input volumne , F = kernel, P = padding 
			n_mv_0 = int((input_shape[0] - kernel_shape[0] + 2 * paddings[0])/strides[0] + 1) # H_out
			n_mv_1 = int((input_shape[1] - kernel_shape[1] + 2 * paddings[1])/strides[1] + 1) # W_out

			n_output_channel = t_w.shape[-1]  # Channel_out
			from_front = []
			# move axis for easier computation
			for idx_ol in tqdm(range(n_output_channel)): # t_w.shape[-1]
				for i in range(n_mv_0): # H
					for j in range(n_mv_1): # W
						curr_prev_output_slice = tr_prev_output_v[:,i*strides[0]:i*strides[0]+kernel_shape[0],:,:]
						curr_prev_output_slice = curr_prev_output_slice[:,:,j*strides[1]:j*strides[1]+kernel_shape[1],:]
						output = curr_prev_output_slice * t_w[:,:,:,idx_ol] 
						sum_output = np.sum(np.abs(output))
						output = output/sum_output
						sum_output = np.nan_to_num(output, posinf = 0.)
						output = np.mean(output, axis = 0) 
						from_front.append(output)
			
			from_front = np.asarray(from_front)
			#from_front.shape: [Channel_out * n_mv_0 * n_mv_1, F1, F2, Channel_in]
			if is_channel_first:
				from_front = from_front.reshape(
					(n_output_channel,n_mv_0,n_mv_1,kernel_shape[0],kernel_shape[1],int(prev_output.shape[1])))
			else: # channels_last
				from_front = from_front.reshape(
					(n_mv_0,n_mv_1,n_output_channel,kernel_shape[0],kernel_shape[1],int(prev_output.shape[-1])))

			# [F1,F2,Channel_in, Channel_out, n_mv_0, n_mv_1] 
			# 	or [F1,F2,Channel_in, n_mv_0, n_mv_1,Channel_out]
			from_front = np.moveaxis(from_front, [0,1,2], [3,4,5])
			# [Channel_out, H_out(n_mv_0), W_out(n_mv_1)]
			from_behind = compute_gradient_to_output(path_to_keras_model, idx_to_tl, target_X, by_batch = True) 
			
			#t1 = time.time()
			# [F1,F2,Channel_in, Channel_out, n_mv_0, n_mv_1] (channels_firs) 
			# or [F1,F2,Channel_in,n_mv_0, n_mv_1,Channel_out] (channels_last)
			FIs = from_front * from_behind 
			#t2 = time.time()
			#print ('Time for multiplying front and behind results: {}'.format(t2 - t1))
			#FIs = np.mean(np.mean(FIs, axis = -1), axis = -1) # [F1, F2, Channel_in, Channel_out]
			if is_channel_first:
				FIs = np.sum(np.sum(FIs, axis = -1), axis = -1) # [F1, F2, Channel_in, Channel_out]
			else:
				FIs = np.sum(np.sum(FIs, axis = -2), axis = -2) # [F1, F2, Channel_in, Channel_out] 
			#t3 = time.time()
			#print ('Time for computing mean for FIs: {}'.format(t3 - t2))
			## Gradient
			# will be [F1, F2, Channel_in, Channel_out]
			grad_scndcr = compute_gradient_to_loss(
				path_to_keras_model, idx_to_tl, target_X, target_y, by_batch = True, loss_func = loss_func)

		elif model_util.is_LSTM(lname): #
			from scipy.special import expit as sigmoid
			num_weights = 2 
			assert len(t_w) == num_weights, t_w
			# t_w_kernel: 
			# (input_feature_size, 4 * num_units). t_w_recurr_kernel: (num_units, 4 * num_units)
			t_w_kernel, t_w_recurr_kernel = t_w 
			
			# get the previous output, which will be the input of the lstm
			if model_util.is_Input(type(model.layers[idx_to_tl - 1]).__name__):
				prev_output = target_X
			else:
				# shape = (batch_size, time_steps, input_feature_size)
				t_model = Model(inputs = model.input, outputs = model.layers[idx_to_tl - 1].output) 
				prev_output = t_model.predict(target_X)
		
			assert len(prev_output.shape) == 3, prev_output.shape
			num_features = prev_output.shape[-1] # the dimension of features that will be processed by the model
			
			num_units = t_w_recurr_kernel.shape[0] 
			assert t_w_kernel.shape[0] == num_features, "{} (kernel) vs {} (input)".format(t_w_kernel.shape[0], num_features)

			# hidden state and cell state sequences computation
			# generate a temporary model that only contains the target lstm layer 
			# but with the modification to return sequences of hidden and cell states
			temp_lstm_layer_inst = lstm_layer.LSTM_Layer(model.layers[idx_to_tl])
			hstates_sequence, cell_states_sequence = temp_lstm_layer_inst.gen_lstm_layer_from_another(prev_output)
			init_hstates, init_cell_states = lstm_layer.LSTM_Layer.get_initial_state(model.layers[idx_to_tl])
			if init_hstates is None: 
				init_hstates = np.zeros((len(target_X), num_units)) 
			if init_cell_states is None:
				# shape = (batch_size, num_units)
				init_cell_states = np.zeros((len(target_X), num_units)) 
		
			# shape = (batch_size, time_steps + 1, num_units)
			hstates_sequence = np.insert(hstates_sequence, 0, init_hstates, axis = 1)
			# shape = (batch_size, time_steps + 1, num_units)
			cell_states_sequence = np.insert(cell_states_sequence, 0, init_cell_states, axis = 1)
			bias = model.layers[idx_to_tl].get_weights()[-1] # shape = (4 * num_units,)
			indices_to_each_gates = np.array_split(np.arange(num_units * 4), 4)

			## prepare all the intermediate outputs and the variables that will be used later
			idx_to_input_gate = 0
			idx_to_forget_gate = 1
			idx_to_cand_gate = 2
			idx_to_output_gate = 3
			
			# for kenerl, weight shape = (input_feature_size, num_units) 
			# and for recurrent, (num_units, num_units), bias (num_units)
			# and the shape of all the intermedidate outpu is "(batch_size, time_step, num_units)"
		
			# input 
			t_w_kernel_I = t_w_kernel[:, indices_to_each_gates[idx_to_input_gate]] 
			t_w_recurr_kernel_I = t_w_recurr_kernel[:, indices_to_each_gates[idx_to_input_gate]] 
			bias_I = bias[indices_to_each_gates[idx_to_input_gate]]
			I = sigmoid(np.dot(prev_output, t_w_kernel_I) + np.dot(hstates_sequence[:,:-1,:], t_w_recurr_kernel_I) + bias_I)

			# forget
			t_w_kernel_F = t_w_kernel[:, indices_to_each_gates[idx_to_forget_gate]]
			t_w_recurr_kernel_F = t_w_recurr_kernel[:, indices_to_each_gates[idx_to_forget_gate]]
			bias_F = bias[indices_to_each_gates[idx_to_forget_gate]]
			F = sigmoid(np.dot(prev_output, t_w_kernel_F) + np.dot(hstates_sequence[:,:-1,:], t_w_recurr_kernel_F) + bias_F) 

			# cand
			t_w_kernel_C = t_w_kernel[:, indices_to_each_gates[idx_to_cand_gate]]
			t_w_recurr_kernel_C = t_w_recurr_kernel[:, indices_to_each_gates[idx_to_cand_gate]]
			bias_C = bias[indices_to_each_gates[idx_to_cand_gate]]
			C = np.tanh(np.dot(prev_output, t_w_kernel_C) + np.dot(hstates_sequence[:,:-1,:], t_w_recurr_kernel_C) + bias_C)

			# output
			t_w_kernel_O = t_w_kernel[:, indices_to_each_gates[idx_to_output_gate]]
			t_w_recurr_kernel_O = t_w_recurr_kernel[:, indices_to_each_gates[idx_to_output_gate]]
			bias_O = bias[indices_to_each_gates[idx_to_output_gate]]
			# shape = (batch_size, time_steps, num_units)
			O = sigmoid(np.dot(prev_output, t_w_kernel_O) + np.dot(hstates_sequence[:,:-1,:], t_w_recurr_kernel_O) + bias_O)

			# set arguments to compute forward impact for the neural weights from these four gates
			t_w_kernels = {
				'input':t_w_kernel_I, 'forget':t_w_kernel_F, 
				'cand':t_w_kernel_C, 'output':t_w_kernel_O}
			t_w_recurr_kernels = {
				'input':t_w_recurr_kernel_I, 'forget':t_w_recurr_kernel_F, 
				'cand':t_w_recurr_kernel_C, 'output':t_w_recurr_kernel_O}

			consts = {}
			consts['input'] = get_constants('input', F, I, C, O, cell_states_sequence)
			consts['forget'] = get_constants('forget', F, I, C, O, cell_states_sequence)
			consts['cand'] = get_constants('cand', F, I, C, O, cell_states_sequence)
			consts['output'] = get_constants('output', F, I, C, O, cell_states_sequence)

			# from_front's shape = (num_units, (num_features + num_units) * 4)
			# gate_orders = ['input', 'forget', 'cand', 'output']
			from_front, gate_orders  = lstm_local_front_FI_for_target_all(
				prev_output, hstates_sequence[:,:-1,:], num_units, 
				t_w_kernels, t_w_recurr_kernels, consts)

			from_front = from_front.T # ((num_features + num_units) * 4, num_units)
			N_k_rk_w = int(from_front.shape[0]/4)
			assert N_k_rk_w == num_features + num_units, "{} vs {}".format(N_k_rk_w, num_features + num_units)
			
			## from behind
			from_behind = compute_gradient_to_output(
				path_to_keras_model, idx_to_tl, target_X, by_batch = True) # shape = (num_units,)

			#t1 = time.time()
			# shape = (N_k_rk_w, num_units) 
			FIs_combined = from_front * from_behind
			#print ("Shape", from_behind.shape, FIs_combined.shape)
			#t2 = time.time()
			#print ('Time for multiplying front and behind results: {}'.format(t2 - t1))
			
			# reshaping
			FIs_kernel = np.zeros(t_w_kernel.shape) # t_w_kernel's shape (num_features, num_units * 4)
			FIs_recurr_kernel = np.zeros(t_w_recurr_kernel.shape) # t_w_recurr_kernel's shape (num_units, num_units * 4)
			# from (4 * N_k_rk_w, num_units) to 4 * (N_k_rk_w, num_units)
			for i, FI_p_gate in enumerate(np.array_split(FIs_combined, 4, axis = 0)): 
				# FI_p_gate's shape = (N_k_rk_w, num_units) 
				# 	-> will divided into (num_features, num_units) & (num_units, num_units)
				# local indices that will split FI_p_gate (shape = (N_k_rk_w, num_units))
				# since we append the weights in order of a kernel weight and a recurrent kernel weight
				indices_to_features = np.arange(num_features)
				indices_to_units = np.arange(num_units) + num_features
				#FIs_kernel[indices_to_features + (i * N_k_rk_w)] 
				# = FI_p_gate[indices_to_features] # shape = (num_features, num_units)
				#FIs_recurr_kernel[indices_to_units + (i * N_k_rk_w)] 
				# = FI_p_gate[indices_to_units] # shape = (num_units, num_units)
				FIs_kernel[:, i * num_units:(i+1) * num_units] = FI_p_gate[indices_to_features] # shape = (num_features, num_units)
				FIs_recurr_kernel[:, i * num_units:(i+1) * num_units] = FI_p_gate[indices_to_units] # shape = (num_units, num_units)

			#t3 =time.time()
			FIs = [FIs_kernel, FIs_recurr_kernel] # [(num_features, num_units*4), (num_units, num_units*4)]
			#print ('Time for formatting: {}'.format(t3 - t2))
			
			## Gradient
			grad_scndcr = compute_gradient_to_loss(
				path_to_keras_model, idx_to_tl, target_X, target_y, by_batch = True, loss_func = loss_func)

		else:
			print ("Currenlty not supported: {}. (shoulde be filtered before)".format(lname))		
			import sys; sys.exit()

		#t2 = time.time()
		#print ("Time for computing cost for the {} layer: {}".format(idx_to_tl, t2 - t1))
		if not model_util.is_LSTM(target_weights[idx_to_tl][1]): # only one weight variable to process 
			pairs = np.asarray([grad_scndcr.flatten(), FIs.flatten()]).T
			total_cands[idx_to_tl] = {'shape':FIs.shape, 'costs':pairs}
		else: # currently, all of them go into here
			total_cands[idx_to_tl] = {'shape':[], 'costs':[]}
			pairs = []
			for _FIs, _grad_scndcr in zip(FIs, grad_scndcr):
				pairs = np.asarray([_grad_scndcr.flatten(), _FIs.flatten()]).T
				total_cands[idx_to_tl]['shape'].append(_FIs.shape)
				total_cands[idx_to_tl]['costs'].append(pairs)

	#t3 = time.time()
	#print ("Time for computing total costs: {}".format(t3 - t0))
	return total_cands


def compute_output_per_w(x, h, t_w_kernel, t_w_recurr_kernel, const, with_norm = False): 
	"""
	A slice for a single neuron (unit or lstm cell)
	x = (batch_size, time_steps, num_features)
	h = (batch_size, time_steps, num_units)
	t_w_kernel = (num_features,)
	t_w_recurr_kernel = (num_units,)
	consts = (batch_size, time_steps) -> the value that is multiplied in the final state computation
	Return the product of the multiplication of weights and input for each unit (i.e., each LSTM cell)
	-> meaning the direct front impact computed per neural weights 
	"""
	from sklearn.preprocessing import Normalizer
	norm_scaler = Normalizer(norm = "l1")

	# here, "multiply" instead of dot, b/c we want to get the output of each neural weight (not the final one)
	out_kernel = x * t_w_kernel #np.multiply(x, t_w_kernel) # shape = (batch_size, time_steps, num_features)
	if h is not None: # None -> we will skip the weights for hiddens states
		out_recurr_kernel = h * t_w_recurr_kernel # shape = (batch_size, time_steps, num_units)
		out = np.append(out_kernel, out_recurr_kernel, axis = -1) # shape:(batch_size,time_steps,(num_features+num_units))
	else:
		out = out_kernel # shape = (batch_size, time_steps, num_features)

	# normalise
	out = np.abs(out)
	if with_norm: 
		original_shape = out.shape
		out = norm_scaler.fit_transform(out.flatten().reshape(1,-1)).reshape(-1,)
		out = out.reshape(original_shape)

	# N = num_features or num_features + num_units
	out = np.einsum('ijk,ij->ijk', out, const) # shape = (batch_size, time_steps, N) 
	return out


def get_constants(gate, F, I, C, O, cell_states):
	"""
	"""
	if gate == 'input':
		return np.multiply(O, np.divide(
			C, cell_states[:,1:,:], 
			out = np.zeros_like(C), where = cell_states[:,1:,:] != 0))
	elif gate == 'forget':
		return np.multiply(O, np.divide(
			cell_states[:,:-1,:], cell_states[:,1:,:], 
			out = np.zeros_like(cell_states[:,:-1,:]), where = cell_states[:,1:,:] != 0))
	elif gate == 'cand':
		return np.multiply(O, np.divide(
			I, cell_states[:,1:,:], 
			out = np.zeros_like(C), where = cell_states[:,1:,:] != 0))
	else: # output
		return np.tanh(cell_states[:,1:,:])
	

def lstm_local_front_FI_for_target_all(
	x, h, num_units, 
	t_w_kernels, t_w_recurr_kernels, consts, 
	gate_orders = ['input', 'forget', 'cand', 'output']):
	"""
	x = previous output
	h = hidden state (-> should be computed using the layer)
	t_w_kernels / t_w_recurr_kernels / consts: 
		a group of neural weights that should be taken into account when measuring the impact.
		arg consts is the corresponding group of constants that will be multiplied 
		to each nueral weight's output, respectively
	"""
	from sklearn.preprocessing import Normalizer
	norm_scaler = Normalizer(norm = "l1")
	from_front = []
	for idx_to_unit in tqdm(range(num_units)):
		out_combined = None
		for gate in gate_orders:
			# out's shape, (batch_size, time_steps, (num_features + num_units))	
			# since the weights of each gate are added with the weights of the other gates, normalise later
			out = compute_output_per_w(
				x, h,
				t_w_kernels[gate][:,idx_to_unit],
				t_w_recurr_kernels[gate][:,idx_to_unit],
				consts[gate][...,idx_to_unit],
				with_norm = False)

			if out_combined is None:
				out_combined = out 
			else:
				out_combined = np.append(out_combined, out, axis = -1)

		# the shape of out_combined => 
		# 	(batch_size, time_steps, 4 * (num_features + num_units)) (since this is per unit)
		# here, keep in mind that we have to use a scaler on the current out_combined 
		# (for instance, divide by the final output (the last hidden state won't work here anymore, 
		# as the summation of the current value differs from the original due to 
		# the absence of act and the scaling in the middle, etc.)
		original_shape = out_combined.shape
		# normalised
		scaled_out_combined = norm_scaler.fit_transform(np.abs(out_combined).flatten().reshape(1,-1)) 
		scaled_out_combined = scaled_out_combined.reshape(original_shape) 
		# mean out_combined's shape: ((num_features + num_units) * 4,) 
		# for each neural weight, the average over both time step and the batch 
		avg_scaled_out_combined = np.mean(
			scaled_out_combined.reshape(-1, scaled_out_combined.shape[-1]), axis = 0) 
		from_front.append(avg_scaled_out_combined)

	# from_front's shape = (num_units, (num_features + num_units) * 4)
	from_front = np.asarray(from_front)
	print ("For lstm's front part of FI: {}".format(from_front.shape))
	return from_front, gate_orders


def localise_by_chgd_unchgd(
	X, y,
	indices_to_chgd,
	indices_to_unchgd,
	target_weights,
	path_to_keras_model = None,
	is_multi_label = True):
	"""
	Find those likely to be highly influential to the changed behaviour 
	while less influential to the unchanged behaviour
	"""
	from collections.abc import Iterable
	#loc_start_time = time.time()
	#print ("Layers to inspect", list(target_weights.keys()))
	# compute FI and GL with changed inputs
	#target_weights = {k:target_weights[k] for k in [2]}
	total_cands_chgd = compute_FI_and_GL(
		X, y,
		indices_to_chgd,
		target_weights,
		is_multi_label = is_multi_label,
		path_to_keras_model = path_to_keras_model)

	# compute FI and GL with unchanged inputs
	total_cands_unchgd = compute_FI_and_GL(
		X, y,
		indices_to_unchgd,
		target_weights,
		is_multi_label = is_multi_label,
		path_to_keras_model = path_to_keras_model)

	indices_to_tl = list(total_cands_chgd.keys()) 
	costs_and_keys = []; indices_to_nodes = []
	shapes = {}
	for idx_to_tl in tqdm(indices_to_tl):
		if not model_util.is_LSTM(target_weights[idx_to_tl][1]): # we have only one weight to process
			#assert not isinstance(
			#	total_cands_unchgd[idx_to_tl]['shape'], Iterable), 
			# 	type(total_cands_unchgd[idx_to_tl]['shape'])
			cost_from_chgd = total_cands_chgd[idx_to_tl]['costs']
			cost_from_unchgd = total_cands_unchgd[idx_to_tl]['costs']
			## key: more influential to changed behaviour and less influential to unchanged behaviour
			costs_combined = cost_from_chgd/(1. + cost_from_unchgd) # shape = (N,2)
			#costs_combined = cost_from_chgd
			shapes[idx_to_tl] = total_cands_chgd[idx_to_tl]['shape']

			for i,c in enumerate(costs_combined):
				costs_and_keys.append(([idx_to_tl, i], c))
				indices_to_nodes.append([idx_to_tl, np.unravel_index(i, shapes[idx_to_tl])])
		else: # 
			#assert isinstance(
			#	total_cands_unchgd[idx_to_tl]['shape'], Iterable), 
			#	type(total_cands_unchgd[idx_to_tl]['shape'])
			num = len(total_cands_unchgd[idx_to_tl]['shape'])
			shapes[idx_to_tl] = []
			for idx_to_pair in range(num):
				cost_from_chgd = total_cands_chgd[idx_to_tl]['costs'][idx_to_pair]
				cost_from_unchgd = total_cands_unchgd[idx_to_tl]['costs'][idx_to_pair]
				costs_combined = cost_from_chgd/(1. + cost_from_unchgd) # shape = (N,2)
				shapes[idx_to_tl].append(total_cands_chgd[idx_to_tl]['shape'][idx_to_pair])

				for i,c in enumerate(costs_combined):
					costs_and_keys.append(([(idx_to_tl, idx_to_pair), i], c))
					indices_to_nodes.append(
						[(idx_to_tl, idx_to_pair), np.unravel_index(i, shapes[idx_to_tl][idx_to_pair])])

	costs = np.asarray([vs[1] for vs in costs_and_keys])
	#t4 = time.time()
	_costs = costs.copy()
	is_efficient = np.arange(costs.shape[0])
	next_point_index = 0 # Next index in the is_efficient array to search for
	while next_point_index < len(_costs):
		nondominated_point_mask = np.any(_costs > _costs[next_point_index], axis=1)
		nondominated_point_mask[next_point_index] = True
		is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
		_costs = _costs[nondominated_point_mask]
		next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1	

	pareto_front = [tuple(v) for v in np.asarray(indices_to_nodes, dtype = object)[is_efficient]]
	#t5 = time.time()
	#print ("Time for computing the pareto front: {}".format(t5 - t4))
	#loc_end_time = time.time()
	#print ("Time for total localisation: {}".format(loc_end_time - loc_start_time))
	return pareto_front, costs_and_keys


def localise_by_gradient(
	X, y,
	indices_to_chgd,
	indices_to_unchgd,
	target_weights,
	path_to_keras_model = None, 
	is_multi_label = True):
	"""
	localise using chgd & unchgd
	"""
	from collections.abc import Iterable
	
	total_cands = {}
	# set loss func
	loss_func = model_util.get_loss_func(is_multi_label = is_multi_label)
	## slice inputs
	for idx_to_tl, vs in target_weights.items():
		t_w, lname = vs
		#print ("targeting layer {} ({})".format(idx_to_tl, lname))
		if model_util.is_C2D(lname) or model_util.is_FC(lname): # either FC or C2D
			# for changed inputs
			grad_scndcr_for_chgd = compute_gradient_to_loss(
				path_to_keras_model, idx_to_tl, X[indices_to_chgd], y[indices_to_chgd], 
				loss_func = loss_func, by_batch = True)
			# for unchanged inputs
			grad_scndcr_for_unchgd = compute_gradient_to_loss(
				path_to_keras_model, idx_to_tl, X[indices_to_unchgd], y[indices_to_unchgd], 
				loss_func = loss_func, by_batch = True)

			assert t_w.shape == grad_scndcr_for_chgd.shape, "{} vs {}".format(t_w.shape, grad_scndcr_for_chgd.shape)
			total_cands[idx_to_tl] = {
				'shape':grad_scndcr_for_chgd.shape, 
				'costs':grad_scndcr_for_chgd.flatten()/(1.+grad_scndcr_for_unchgd.flatten())}
		elif model_util.is_LSTM(lname):
			# for changed inputs
			grad_scndcr_for_chgd = compute_gradient_to_loss(
				path_to_keras_model, idx_to_tl,
				X[indices_to_chgd], y[indices_to_chgd], 
				loss_func = loss_func, by_batch = True)
			# for unchanged inptus
			grad_scndcr_for_unchgd = compute_gradient_to_loss(
				path_to_keras_model, idx_to_tl, 
				X[indices_to_unchgd], y[indices_to_unchgd], 
				loss_func = loss_func, by_batch = True)

			# check the shape of kernel (index = 0) and recurrent kernel (index =1) weights
			assert t_w[0].shape == grad_scndcr_for_chgd[0].shape, "{} vs {}".format(t_w[0].shape, grad_scndcr_for_chgd[0].shape)
			assert t_w[1].shape == grad_scndcr_for_chgd[1].shape, "{} vs {}".format(t_w[1].shape, grad_scndcr_for_chgd[1].shape)

			# generate total candidates
			total_cands[idx_to_tl] = {'shape':[], 'costs':[]}
			for _grad_scndr_chgd, _grad_scndr_unchgd in zip(grad_scndcr_for_chgd, grad_scndcr_for_unchgd):
				#_grad_scndr_chgd & _grad_scndr_unchgd -> can be for either kernel or recurrent kernel
				_costs = _grad_scndr_chgd.flatten()/(1. + _grad_scndr_unchgd.flatten())
				total_cands[idx_to_tl]['shape'].append(_grad_scndr_chgd.shape)
				total_cands[idx_to_tl]['costs'].append(_costs)
		else:
			print ("{} not supported yet".format(lname))
			assert False

	indices_to_tl = list(total_cands.keys())
	costs_and_keys = []
	for idx_to_tl in indices_to_tl:
		if not model_util.is_LSTM(target_weights[idx_to_tl][1]):
			for local_i,c in enumerate(total_cands[idx_to_tl]['costs']):
				cost_and_key = ([idx_to_tl, np.unravel_index(local_i, total_cands[idx_to_tl]['shape'])], c) 
				costs_and_keys.append(cost_and_key)
		else:
			num = len(total_cands[idx_to_tl]['shape'])
			for idx_to_w in range(num):
				for local_i, c in enumerate(total_cands[idx_to_tl]['costs'][idx_to_w]):
					cost_and_key = (
						[(idx_to_tl, idx_to_w), 
						np.unravel_index(local_i, total_cands[idx_to_tl]['shape'][idx_to_w])], c) 
					costs_and_keys.append(cost_and_key)

	sorted_costs_and_keys = sorted(costs_and_keys, key = lambda vs:vs[1], reverse = True)
	return sorted_costs_and_keys


def localise_by_random_selection(number_of_place_to_fix, target_weights):
	"""
	randomly select places to fix
	"""
	from collections.abc import Iterable

	total_indices = []
	for idx_to_tl, vs in target_weights.items():
		t_w, lname = vs
		if not model_util.is_LSTM(lname):
			l_indices = list(np.ndindex(t_w.shape))
			total_indices.extend(list(zip([idx_to_tl] * len(l_indices), l_indices)))
		else: # to handle the layers with more than one weights (e.g., LSTM)
			for idx_to_w, a_t_w in enumerate(t_w):
				l_indices = list(np.ndindex(a_t_w.shape))
				total_indices.extend(list(zip([(idx_to_tl, idx_to_w)] * len(l_indices), l_indices)))

	np.random.shuffle(total_indices)
	if number_of_place_to_fix > 0 and number_of_place_to_fix < len(total_indices):
		selected_indices = np.random.choice(
			np.arange(len(total_indices)), number_of_place_to_fix, replace = False)
		indices_to_places_to_fix = [total_indices[idx] for idx in selected_indices]
	else:
		indices_to_places_to_fix = total_indices

	return indices_to_places_to_fix


def is_correct(pred, true_label, is_multi_label):
	"""
	判断预测是否正确
	"""
	if is_multi_label:
		# 分类任务：比较预测类别和真实类别
		pred_class = np.argmax(pred)
		if len(true_label.shape) > 1:
			true_class = np.argmax(true_label)
		else:
			# 安全地处理数组
			true_class = int(np.array(true_label).flatten()[0])
		return pred_class == true_class
	else:
		# 二分类任务
		pred_class = 1 if pred > 0.5 else 0
		true_class = int(np.array(true_label).flatten()[0])
		return pred_class == true_class


def localise_by_weighted_sbfl(
	X, y,
	indices_to_chgd,
	indices_to_unchgd,
	target_weights,
	path_to_keras_model = None, 
	is_multi_label = True):
	"""
	基于加权SBFL的权重定位算法（仅分类任务）
	使用GUIDER方式：直接区分成功/失败，不区分changed/unchanged
	"""
	from collections.abc import Iterable
	import numpy as np
	
	print("Debug - 进入weighted_sbfl函数")
	print(f"Debug - X shape: {X.shape}")
	print(f"Debug - y shape: {y.shape}")
	print(f"Debug - target_weights keys: {list(target_weights.keys())}")
	
	# 写入文件确认函数被调用
	with open('/tmp/debug_weighted_sbfl.txt', 'w') as f:
		f.write("weighted_sbfl函数被调用了\n")
		f.write(f"X shape: {X.shape}\n")
		f.write(f"y shape: {y.shape}\n")
	
	# 对所有样本计算成功/失败和置信度
	eva = []  # 成功/失败
	confidence = []
	
	# 优化：只加载一次模型
	import tensorflow as tf
	model = tf.keras.models.load_model(path_to_keras_model)
	
	print("Debug - 开始计算样本成功/失败状态...")
	
	for i in range(len(X)):
		# 直接使用已加载的模型
		sample = X[i].reshape(1, *X[i].shape)
		pred = model.predict(sample, verbose=0)[0]
		
		is_correct_pred = is_correct(pred, y[i], is_multi_label)
		if is_correct_pred:
			eva.append(1)  # 成功
		else:
			eva.append(0)  # 失败
		confidence.append(calculate_confidence(pred, y[i], is_multi_label))
		
		# 调试前几个样本
		if i < 3:
			true_class = np.argmax(y[i]) if len(y[i].shape) > 1 else int(np.array(y[i]).flatten()[0])
			pred_class = np.argmax(pred)
			debug_msg = f"样本{i}: 真实={true_class}, 预测={pred_class}, 正确={is_correct_pred}\n"
			with open('/tmp/debug_weighted_sbfl.txt', 'a') as f:
				f.write(debug_msg)
	
	print(f"Debug - 计算完成，样本数: {len(X)}")
	
	# 归一化置信度
	min_conf = np.min(confidence)
	max_conf = np.max(confidence)
	if min_conf != max_conf:
		confidence = [(c - min_conf) / (max_conf - min_conf) for c in confidence]
	else:
		confidence = [0] * len(confidence)
	
	# 计算总权重
	total_pass = sum(1 + confidence[i] for i in range(len(X)) if eva[i] == 1) + 1e-4
	total_fail = sum(1 + confidence[i] for i in range(len(X)) if eva[i] == 0) + 1e-4
	
	# 添加调试信息
	debug_info = f"Debug - 样本总数: {len(X)}\n"
	debug_info += f"Debug - 成功样本数: {sum(eva)}\n"
	debug_info += f"Debug - 失败样本数: {len(X) - sum(eva)}\n"
	debug_info += f"Debug - total_pass: {total_pass:.6f}\n"
	debug_info += f"Debug - total_fail: {total_fail:.6f}\n"
	print(debug_info)
	
	# 写入文件
	with open('/tmp/debug_weighted_sbfl.txt', 'a') as f:
		f.write(debug_info)
	
	# 对每个权重层计算可疑度
	total_cands = {}
	for idx_to_tl, vs in target_weights.items():
		t_w, lname = vs
		
		if model_util.is_C2D(lname) or model_util.is_FC(lname):
			# 计算所有样本对每个权重的梯度
			grad_per_sample = compute_gradient_per_sample(
				path_to_keras_model, idx_to_tl, X, y, 
				loss_func = model_util.get_loss_func(is_multi_label = is_multi_label))
			
			# 计算每个权重的可疑度
			suspiciousness = np.zeros_like(grad_per_sample[0])  # 使用第一个样本的形状
			for i in range(suspiciousness.shape[0]):
				for j in range(suspiciousness.shape[1]):
					passed = 0
					failed = 0
					for k in range(len(X)):
						# 获取第k个样本对权重(i,j)的梯度
						sample_grad = grad_per_sample[k][i, j]
						if abs(sample_grad) > 1e-6:  # 权重参与
							weight_participation = abs(sample_grad)  # 权重参与强度
							sample_confidence = confidence[k]        # 样本置信度
							if eva[k] == 1:  # 成功
								passed += weight_participation * (1 + sample_confidence)
							else:  # 失败
								failed += weight_participation * (1 + sample_confidence)
					
					# 使用SBFL公式计算可疑度
					suspiciousness[i, j] = calculate_weighted_suspiciousness(
						failed, passed, total_fail, total_pass)
					
					# 添加调试信息（只打印前几个权重）
					if i < 3 and j < 3:
						print(f"Debug - 权重({i},{j}): failed={failed:.6f}, passed={passed:.6f}, suspiciousness={suspiciousness[i, j]:.6f}")
			
			total_cands[idx_to_tl] = {
				'shape': suspiciousness.shape,
				'costs': suspiciousness.flatten()
			}
			
		elif model_util.is_LSTM(lname):
			# LSTM层的处理 - 暂时跳过，因为LSTM的梯度计算更复杂
			print(f"LSTM layer {lname} not supported in weighted SBFL yet")
			continue
		else:
			print(f"{lname} not supported yet")
			assert False
	
	# 构建结果
	indices_to_tl = list(total_cands.keys())
	costs_and_keys = []
	for idx_to_tl in indices_to_tl:
		if not model_util.is_LSTM(target_weights[idx_to_tl][1]):
			for local_i, c in enumerate(total_cands[idx_to_tl]['costs']):
				# 将(row, col)转换为BL格式的index
				row_col = np.unravel_index(local_i, total_cands[idx_to_tl]['shape'])
				if len(row_col) == 2:  # 2D权重矩阵
					bl_index = row_col[0] * total_cands[idx_to_tl]['shape'][1] + row_col[1]
				else:  # 1D权重向量
					bl_index = row_col[0]
				cost_and_key = ([idx_to_tl, bl_index], c) 
				costs_and_keys.append(cost_and_key)
		else:
			num = len(total_cands[idx_to_tl]['shape'])
			for idx_to_w in range(num):
				for local_i, c in enumerate(total_cands[idx_to_tl]['costs'][idx_to_w]):
					# 将(row, col)转换为BL格式的index
					row_col = np.unravel_index(local_i, total_cands[idx_to_tl]['shape'][idx_to_w])
					if len(row_col) == 2:  # 2D权重矩阵
						bl_index = row_col[0] * total_cands[idx_to_tl]['shape'][idx_to_w][1] + row_col[1]
					else:  # 1D权重向量
						bl_index = row_col[0]
					cost_and_key = ([(idx_to_tl, idx_to_w), bl_index], c) 
					costs_and_keys.append(cost_and_key)
	
	sorted_costs_and_keys = sorted(costs_and_keys, key = lambda vs: vs[1], reverse = True)
	
	# 提取权重位置
	indices_to_places_to_fix = [item[0] for item in sorted_costs_and_keys]
	
	return indices_to_places_to_fix, sorted_costs_and_keys


def calculate_confidence(pred, true_label, is_multi_label):
	"""
	计算分类任务的预测置信度
	"""
	if np.isscalar(pred):
		return pred
	else:
		if np.isnan(np.max(pred)):
			return 0
		else:
			return np.max(pred)


def calculate_weighted_suspiciousness(failed, passed, total_fail, total_pass):
	"""
	使用加权SBFL公式计算可疑度
	这里使用Tarantula公式，您可以选择其他公式
	"""
	if failed == 0 and passed == 0:
		return 0
	
	# Tarantula公式
	failed_ratio = failed / total_fail
	passed_ratio = passed / total_pass
	
	if failed_ratio + passed_ratio == 0:
		return 0
	
	return failed_ratio / (failed_ratio + passed_ratio)
	

def localise_by_FI_only(
	X, y,
	indices_to_chgd,
	indices_to_unchgd,
	target_weights,
	path_to_keras_model = None,
	is_multi_label = True):
	"""
	仅使用FI (Feature Importance) 进行故障定位
	类似GL算法的结构：分别计算changed和unchanged的FI，然后计算比值
	"""
	from collections.abc import Iterable
	
	total_cands = {}
	# set loss func (虽然FI不直接用loss，但compute_FI_and_GL需要)
	loss_func = model_util.get_loss_func(is_multi_label = is_multi_label)
	
	## slice inputs
	for idx_to_tl, vs in target_weights.items():
		t_w, lname = vs
		#print ("targeting layer {} ({})".format(idx_to_tl, lname))
		if model_util.is_C2D(lname) or model_util.is_FC(lname): # either FC or C2D
			# 计算changed样本的FI和GL
			fi_and_gl_for_chgd = compute_FI_and_GL(
				X, y, indices_to_chgd, {idx_to_tl: vs},
				is_multi_label = is_multi_label,
				path_to_keras_model = path_to_keras_model)
			
			# 计算unchanged样本的FI和GL  
			fi_and_gl_for_unchgd = compute_FI_and_GL(
				X, y, indices_to_unchgd, {idx_to_tl: vs},
				is_multi_label = is_multi_label, 
				path_to_keras_model = path_to_keras_model)
			
			# 提取FI分数（索引1为FI，索引0为GL）
			fi_chgd = fi_and_gl_for_chgd[idx_to_tl]['costs'][:, 1]  # FI for changed
			fi_unchgd = fi_and_gl_for_unchgd[idx_to_tl]['costs'][:, 1]  # FI for unchanged
			
			assert t_w.shape == fi_and_gl_for_chgd[idx_to_tl]['shape'], "{} vs {}".format(t_w.shape, fi_and_gl_for_chgd[idx_to_tl]['shape'])
			
			# 计算FI比值：FI_changed / (1 + FI_unchanged)
			total_cands[idx_to_tl] = {
				'shape': fi_and_gl_for_chgd[idx_to_tl]['shape'], 
				'costs': fi_chgd/(1. + fi_unchgd)}
				
		elif model_util.is_LSTM(lname):
			# LSTM层的处理
			fi_and_gl_for_chgd = compute_FI_and_GL(
				X, y, indices_to_chgd, {idx_to_tl: vs},
				is_multi_label = is_multi_label,
				path_to_keras_model = path_to_keras_model)
			
			fi_and_gl_for_unchgd = compute_FI_and_GL(
				X, y, indices_to_unchgd, {idx_to_tl: vs},
				is_multi_label = is_multi_label,
				path_to_keras_model = path_to_keras_model)
			
			# LSTM有多个权重矩阵
			num = len(fi_and_gl_for_chgd[idx_to_tl]['shape'])
			total_cands[idx_to_tl] = {'shape': [], 'costs': []}
			
			for idx_to_pair in range(num):
				fi_chgd = fi_and_gl_for_chgd[idx_to_tl]['costs'][idx_to_pair][:, 1]
				fi_unchgd = fi_and_gl_for_unchgd[idx_to_tl]['costs'][idx_to_pair][:, 1]
				
				fi_costs = fi_chgd/(1. + fi_unchgd)
				total_cands[idx_to_tl]['shape'].append(fi_and_gl_for_chgd[idx_to_tl]['shape'][idx_to_pair])
				total_cands[idx_to_tl]['costs'].append(fi_costs)
		else:
			print ("{} not supported yet".format(lname))
			assert False

	# 生成排序结果（完全类似GL算法的结构）
	indices_to_tl = list(total_cands.keys())
	costs_and_keys = []
	for idx_to_tl in indices_to_tl:
		if not model_util.is_LSTM(target_weights[idx_to_tl][1]):
			for local_i,c in enumerate(total_cands[idx_to_tl]['costs']):
				cost_and_key = ([idx_to_tl, np.unravel_index(local_i, total_cands[idx_to_tl]['shape'])], c) 
				costs_and_keys.append(cost_and_key)
		else:
			num = len(total_cands[idx_to_tl]['shape'])
			for idx_to_w in range(num):
				for local_i, c in enumerate(total_cands[idx_to_tl]['costs'][idx_to_w]):
					cost_and_key = (
						[(idx_to_tl, idx_to_w), 
						np.unravel_index(local_i, total_cands[idx_to_tl]['shape'][idx_to_w])], c) 
					costs_and_keys.append(cost_and_key)

	sorted_costs_and_keys = sorted(costs_and_keys, key = lambda vs:vs[1], reverse = True)
	return sorted_costs_and_keys
	

def localise_by_FI_SBFL(
	X, y, indices_to_chgd, indices_to_unchgd, target_weights,
	path_to_keras_model=None, is_multi_label=True, 
	fi_threshold_percentile=96):  # 使用百分位数
	
	import numpy as np
	import tensorflow as tf
	from tqdm import tqdm
	
	# 1. 预计算所有样本的FI (一次性)
	all_indices = list(indices_to_chgd) + list(indices_to_unchgd)
	all_sample_fi = {}
	
	print("预计算所有样本的FI值...")
	for sample_idx in tqdm(all_indices):
		all_sample_fi[sample_idx] = compute_FI_and_GL(
			X, y, [sample_idx], target_weights, 
			is_multi_label=is_multi_label,
			path_to_keras_model=path_to_keras_model)
	
	# 2. 统一失败定义 (建议使用分类正确性)
	model = tf.keras.models.load_model(path_to_keras_model)
	failed_samples = []
	passed_samples = []
	
	for idx in all_indices:
		sample = X[idx].reshape(1, *X[idx].shape)
		pred = model.predict(sample, verbose=0)[0]
		if is_correct(pred, y[idx], is_multi_label):
			passed_samples.append(idx)
		else:
			failed_samples.append(idx)
	
	print(f"  失败样本数: {len(failed_samples)}, 成功样本数: {len(passed_samples)}")
	
	# 3. 动态确定FI阈值
	
	# 方案A: 全局阈值 (所有层统一) - CURRENTLY ACTIVE
	print("计算全局FI阈值...")
	all_fi_values = []
	for idx_to_tl, vs in target_weights.items():
		for sample_idx in all_indices:
			fi_costs = all_sample_fi[sample_idx][idx_to_tl]['costs']
			if not model_util.is_LSTM(vs[1]):
				all_fi_values.extend(fi_costs[:, 1])  # FI值在索引1
			else:
				for matrix_costs in fi_costs:
					all_fi_values.extend(matrix_costs[:, 1])
	
	fi_threshold = np.percentile(all_fi_values, fi_threshold_percentile)
	print(f"  FI阈值 (第{fi_threshold_percentile}百分位): {fi_threshold:.8f}")
	
	# 调试信息：显示实际FI值分布
	all_fi_array = np.array(all_fi_values)
	zero_count = np.sum(all_fi_array == 0)
	print(f"  FI值统计: 总数={len(all_fi_array)}, 零值={zero_count} ({zero_count/len(all_fi_array)*100:.1f}%)")
	print(f"  FI值范围: [{np.min(all_fi_array):.8f}, {np.max(all_fi_array):.8f}]")
	if len(all_fi_array) > 0:
		percentiles = [25, 50, 75, 90]
		for p in percentiles:
			print(f"  第{p}百分位: {np.percentile(all_fi_array, p):.8f}")
	
	# 方案B: 层内阈值 (每层独立) - 75%百分位 (COMMENTED OUT)
	# print("计算层内FI阈值...")
	# layer_thresholds = {}
	# 
	# for idx_to_tl, vs in target_weights.items():
	#     # 收集该层的所有FI值
	#     layer_fi_values = []
	#     for sample_idx in all_indices:
	#         fi_costs = all_sample_fi[sample_idx][idx_to_tl]['costs']
	#         if not model_util.is_LSTM(vs[1]):
	#             layer_fi_values.extend(fi_costs[:, 1])  # FI值在索引1
	#         else:
	#             for matrix_costs in fi_costs:
	#                 layer_fi_values.extend(matrix_costs[:, 1])
	#     
	#     # 计算该层的阈值
	#     layer_threshold = np.percentile(layer_fi_values, 75)  # 使用75%阈值
	#     layer_thresholds[idx_to_tl] = layer_threshold
	#     
	#     # 调试信息
	#     layer_fi_array = np.array(layer_fi_values)
	#     zero_count = np.sum(layer_fi_array == 0)
	#     print(f"  层{idx_to_tl}: 阈值={layer_threshold:.8f}, 零值={zero_count}/{len(layer_fi_array)*100:.1f}%)")
	#     print(f"    范围=[{np.min(layer_fi_array):.8f}, {np.max(layer_fi_array):.8f}]")
	
	# 4. 计算SBFL (现在很快，因为FI已预计算)
	total_cands = {}
	
	for idx_to_tl, vs in target_weights.items():
		t_w, lname = vs
		
		if model_util.is_C2D(lname) or model_util.is_FC(lname):
			# 获取层形状
			layer_shape = all_sample_fi[all_indices[0]][idx_to_tl]['shape']
			num_weights = np.prod(layer_shape)
			
			# 初始化可疑度数组 (添加DStar和Jaccard)
			suspiciousness_ochiai = np.zeros(num_weights)
			suspiciousness_tarantula = np.zeros(num_weights)
			suspiciousness_dstar = np.zeros(num_weights)
			suspiciousness_jaccard = np.zeros(num_weights)
			
			print(f"  计算层{idx_to_tl}的{num_weights}个权重的SBFL可疑度...")
			
			for weight_idx in range(num_weights):
				ef = 0  # executed & failed
				ep = 0  # executed & passed  
				nf = 0  # not executed & failed
				np_val = 0  # not executed & passed
				
				# 遍历所有样本，使用预计算的FI值
				for sample_idx in all_indices:
					# 获取该权重在该样本上的FI值
					weight_fi_value = all_sample_fi[sample_idx][idx_to_tl]['costs'][weight_idx, 1]
					
					# 判断权重是否在该样本上参与 (使用全局阈值)
					weight_participates = weight_fi_value > fi_threshold
					
					# 判断该样本分类是否正确
					sample_correct = sample_idx in passed_samples
					
					# 累积计算ef, ep, nf, np
					if weight_participates and not sample_correct:
						ef += 1
					elif weight_participates and sample_correct:
						ep += 1
					elif not weight_participates and not sample_correct:
						nf += 1
					elif not weight_participates and sample_correct:
						np_val += 1
				
				# 计算四种SBFL公式的可疑度
				
				# 1. Ochiai公式: ef / sqrt((ef + ep) * (ef + nf))
				total_failed = ef + nf
				total_executed = ef + ep
				if total_failed == 0 or total_executed == 0:
					suspiciousness_ochiai[weight_idx] = 0
				else:
					suspiciousness_ochiai[weight_idx] = ef / np.sqrt(total_executed * total_failed)
				
				# 2. DStar公式: ef² / (ep + nf) - 激进筛选，重度惩罚
				if ep + nf == 0:
					suspiciousness_dstar[weight_idx] = ef * ef if ef > 0 else 0
				else:
					suspiciousness_dstar[weight_idx] = (ef * ef) / (ep + nf)
				
				# 3. Jaccard公式: ef / (ef + ep + nf) - 纯度导向
				if ef + ep + nf == 0:
					suspiciousness_jaccard[weight_idx] = 0
				else:
					suspiciousness_jaccard[weight_idx] = ef / (ef + ep + nf)
				
				# 计算Tarantula公式可疑度
				if ef + nf == 0:
					suspiciousness_tarantula[weight_idx] = 0
				elif ep + np_val == 0:
					suspiciousness_tarantula[weight_idx] = 1.0 if ef > 0 else 0
				else:
					failed_ratio = ef / (ef + nf)
					passed_ratio = ep / (ep + np_val)
					denominator = failed_ratio + passed_ratio
					if denominator == 0:
						suspiciousness_tarantula[weight_idx] = 0
					else:
						suspiciousness_tarantula[weight_idx] = failed_ratio / denominator
			
			total_cands[idx_to_tl] = {
				'shape': layer_shape,
				'costs_ochiai': suspiciousness_ochiai,
				'costs_tarantula': suspiciousness_tarantula,
				'costs_dstar': suspiciousness_dstar,
				'costs_jaccard': suspiciousness_jaccard
			}
			
			# 统计参与情况
			participating_count = np.sum(suspiciousness_ochiai > 0)
			print(f"  层{idx_to_tl}: {participating_count}/{num_weights}个权重有非零分数")
			
		elif model_util.is_LSTM(lname):
			# LSTM层处理
			layer_shapes = all_sample_fi[all_indices[0]][idx_to_tl]['shape']
			num_matrices = len(layer_shapes)
			
			total_cands[idx_to_tl] = {'shape': [], 'costs_ochiai': [], 'costs_tarantula': [], 'costs_dstar': [], 'costs_jaccard': []}
			
			for matrix_idx in range(num_matrices):
				matrix_shape = layer_shapes[matrix_idx]
				num_weights = np.prod(matrix_shape)
				
				suspiciousness_ochiai = np.zeros(num_weights)
				suspiciousness_tarantula = np.zeros(num_weights)
				suspiciousness_dstar = np.zeros(num_weights)
				suspiciousness_jaccard = np.zeros(num_weights)
				
				print(f"  计算层{idx_to_tl}矩阵{matrix_idx}的{num_weights}个权重...")
				
				for weight_idx in range(num_weights):
					ef = 0; ep = 0; nf = 0; np_val = 0
					
					for sample_idx in all_indices:
						weight_fi_value = all_sample_fi[sample_idx][idx_to_tl]['costs'][matrix_idx][weight_idx, 1]
						# 使用全局阈值
						weight_participates = weight_fi_value > fi_threshold
						sample_correct = sample_idx in passed_samples
						
						if weight_participates and not sample_correct: ef += 1
						elif weight_participates and sample_correct: ep += 1
						elif not weight_participates and not sample_correct: nf += 1
						elif not weight_participates and sample_correct: np_val += 1
					
					# 计算四种SBFL公式的可疑度
					
					# 1. Ochiai公式
					total_failed = ef + nf
					total_executed = ef + ep
					if total_failed == 0 or total_executed == 0:
						suspiciousness_ochiai[weight_idx] = 0
					else:
						suspiciousness_ochiai[weight_idx] = ef / np.sqrt(total_executed * total_failed)
					
					# 2. DStar公式: ef² / (ep + nf)
					if ep + nf == 0:
						suspiciousness_dstar[weight_idx] = ef * ef if ef > 0 else 0
					else:
						suspiciousness_dstar[weight_idx] = (ef * ef) / (ep + nf)
					
					# 3. Jaccard公式: ef / (ef + ep + nf)
					if ef + ep + nf == 0:
						suspiciousness_jaccard[weight_idx] = 0
					else:
						suspiciousness_jaccard[weight_idx] = ef / (ef + ep + nf)
					
					# 4. Tarantula公式
					if ef + nf == 0:
						suspiciousness_tarantula[weight_idx] = 0
					elif ep + np_val == 0:
						suspiciousness_tarantula[weight_idx] = 1.0 if ef > 0 else 0
					else:
						failed_ratio = ef / (ef + nf)
						passed_ratio = ep / (ep + np_val)
						denominator = failed_ratio + passed_ratio
						if denominator == 0:
							suspiciousness_tarantula[weight_idx] = 0
						else:
							suspiciousness_tarantula[weight_idx] = failed_ratio / denominator
				
				total_cands[idx_to_tl]['shape'].append(matrix_shape)
				total_cands[idx_to_tl]['costs_ochiai'].append(suspiciousness_ochiai)
				total_cands[idx_to_tl]['costs_tarantula'].append(suspiciousness_tarantula)
				total_cands[idx_to_tl]['costs_dstar'].append(suspiciousness_dstar)
				total_cands[idx_to_tl]['costs_jaccard'].append(suspiciousness_jaccard)
		else:
			print ("{} not supported yet".format(lname))
			assert False

	# 5. 生成并排序结果
	indices_to_tl = list(total_cands.keys())
	costs_and_keys_ochiai = []
	costs_and_keys_tarantula = []
	costs_and_keys_dstar = []
	costs_and_keys_jaccard = []
	
	for idx_to_tl in indices_to_tl:
		if not model_util.is_LSTM(target_weights[idx_to_tl][1]):
			# 普通层
			for local_i, (cost_ochiai, cost_tarantula, cost_dstar, cost_jaccard) in enumerate(zip(
				total_cands[idx_to_tl]['costs_ochiai'], 
				total_cands[idx_to_tl]['costs_tarantula'],
				total_cands[idx_to_tl]['costs_dstar'],
				total_cands[idx_to_tl]['costs_jaccard'])):
				
				weight_coords = [idx_to_tl, np.unravel_index(local_i, total_cands[idx_to_tl]['shape'])]
				
				cost_and_key_ochiai = (weight_coords, cost_ochiai)
				cost_and_key_tarantula = (weight_coords, cost_tarantula)
				cost_and_key_dstar = (weight_coords, cost_dstar)
				cost_and_key_jaccard = (weight_coords, cost_jaccard)
				
				costs_and_keys_ochiai.append(cost_and_key_ochiai)
				costs_and_keys_tarantula.append(cost_and_key_tarantula)
				costs_and_keys_dstar.append(cost_and_key_dstar)
				costs_and_keys_jaccard.append(cost_and_key_jaccard)
		else:
			# LSTM层
			num_matrices = len(total_cands[idx_to_tl]['shape'])
			for matrix_idx in range(num_matrices):
				for local_i, (cost_ochiai, cost_tarantula, cost_dstar, cost_jaccard) in enumerate(zip(
					total_cands[idx_to_tl]['costs_ochiai'][matrix_idx],
					total_cands[idx_to_tl]['costs_tarantula'][matrix_idx],
					total_cands[idx_to_tl]['costs_dstar'][matrix_idx],
					total_cands[idx_to_tl]['costs_jaccard'][matrix_idx])):
					
					weight_coords = [(idx_to_tl, matrix_idx), np.unravel_index(local_i, total_cands[idx_to_tl]['shape'][matrix_idx])]
					
					cost_and_key_ochiai = (weight_coords, cost_ochiai)
					cost_and_key_tarantula = (weight_coords, cost_tarantula)
					cost_and_key_dstar = (weight_coords, cost_dstar)
					cost_and_key_jaccard = (weight_coords, cost_jaccard)
					
					costs_and_keys_ochiai.append(cost_and_key_ochiai)
					costs_and_keys_tarantula.append(cost_and_key_tarantula)
					costs_and_keys_dstar.append(cost_and_key_dstar)
					costs_and_keys_jaccard.append(cost_and_key_jaccard)

	# 排序结果
	sorted_costs_and_keys_ochiai = sorted(costs_and_keys_ochiai, key = lambda vs:vs[1], reverse = True)
	sorted_costs_and_keys_tarantula = sorted(costs_and_keys_tarantula, key = lambda vs:vs[1], reverse = True)
	sorted_costs_and_keys_dstar = sorted(costs_and_keys_dstar, key = lambda vs:vs[1], reverse = True)
	sorted_costs_and_keys_jaccard = sorted(costs_and_keys_jaccard, key = lambda vs:vs[1], reverse = True)
	
	print(f"  最终结果: 四种SBFL公式都生成了{len(sorted_costs_and_keys_ochiai)}个权重排名")
	
	return {
		'ochiai': sorted_costs_and_keys_ochiai,
		'tarantula': sorted_costs_and_keys_tarantula,
		'dstar': sorted_costs_and_keys_dstar,
		'jaccard': sorted_costs_and_keys_jaccard
	}


# 标准化处理：对每层的FI值进行z-score标准化，解决不同层FI值范围差异问题
# 公平竞争：通过标准化使得不同层的权重可以公平竞争
# 双公式支持：同时输出Ochiai和Tarantula两种SBFL公式结果
# 详细统计：提供了层级和全局的参与统计信息
def localise_by_FI_SBFL_standardized(
	X, y, indices_to_chgd, indices_to_unchgd, target_weights,
	path_to_keras_model=None, is_multi_label=True,
	standardized_threshold=1.0):
	"""
	FI-SBFL方法的标准化版本 - 解决层间FI值范围差异问题
	
	核心改进:
	1. 对每层的FI值进行标准化: (fi - mean) / std
	2. 使用标准化后的阈值判断权重参与
	3. 实现层间公平竞争
	
	参数:
	- standardized_threshold: 标准化后的参与阈值，默认1.0 (即1个标准差以上)
	"""
	
	import numpy as np
	import tensorflow as tf
	from tqdm import tqdm

	print(f"🔧 FI-SBFL标准化版本 (阈值={standardized_threshold}σ)")

	# 1. 预计算所有样本的FI (一次性)
	all_indices = list(indices_to_chgd) + list(indices_to_unchgd)
	all_sample_fi = {}
	print("预计算所有样本的FI值...")
	for sample_idx in tqdm(all_indices):
		all_sample_fi[sample_idx] = compute_FI_and_GL(
			X, y, [sample_idx], target_weights,
			is_multi_label=is_multi_label,
			path_to_keras_model=path_to_keras_model)

	# 2. 统一失败定义 (分类正确性)
	model = tf.keras.models.load_model(path_to_keras_model)
	failed_samples = []
	passed_samples = []
	for idx in all_indices:
		sample = X[idx].reshape(1, *X[idx].shape)
		pred = model.predict(sample, verbose=0)[0]
		if is_correct(pred, y[idx], is_multi_label):
			passed_samples.append(idx)
		else:
			failed_samples.append(idx)
	print(f"  失败样本数: {len(failed_samples)}, 成功样本数: {len(passed_samples)}")

	# 3. 计算每层FI值的标准化参数
	print("计算层间FI标准化参数...")
	layer_standardization = {}
	
	for idx_to_tl, vs in target_weights.items():
		layer_fi_values = []
		for sample_idx in all_indices:
			fi_costs = all_sample_fi[sample_idx][idx_to_tl]['costs']
			if not model_util.is_LSTM(vs[1]):
				layer_fi_values.extend(fi_costs[:, 1])
			else:
				for matrix_costs in fi_costs:
					layer_fi_values.extend(matrix_costs[:, 1])
		
		layer_fi_array = np.array(layer_fi_values)
		# 只计算非零值的均值和标准差
		nonzero_fi = layer_fi_array[layer_fi_array > 0]
		
		if len(nonzero_fi) > 0:
			layer_mean = np.mean(nonzero_fi)
			layer_std = np.std(nonzero_fi)
			if layer_std == 0:
				layer_std = 1e-8  # 避免除零
		else:
			layer_mean = 0
			layer_std = 1e-8
		
		layer_standardization[idx_to_tl] = {
			'mean': layer_mean,
			'std': layer_std,
			'nonzero_count': len(nonzero_fi),
			'total_count': len(layer_fi_array)
		}
		
		zero_count = np.sum(layer_fi_array == 0)
		print(f"  层{idx_to_tl}: 均值={layer_mean:.6f}, 标准差={layer_std:.6f}")
		print(f"    非零值={len(nonzero_fi)}/{len(layer_fi_array)} ({len(nonzero_fi)/len(layer_fi_array)*100:.1f}%)")
		print(f"    范围=[{np.min(layer_fi_array):.6f}, {np.max(layer_fi_array):.6f}]")

	# 4. 计算标准化SBFL
	total_cands = {}
	total_participating_weights = 0
	
	for idx_to_tl, vs in target_weights.items():
		t_w, lname = vs
		layer_stats = layer_standardization[idx_to_tl]
		
		if model_util.is_C2D(lname) or model_util.is_FC(lname):
			layer_shape = all_sample_fi[all_indices[0]][idx_to_tl]['shape']
			num_weights = np.prod(layer_shape)
			suspiciousness_ochiai = np.zeros(num_weights)
			suspiciousness_tarantula = np.zeros(num_weights)
			suspiciousness_dstar = np.zeros(num_weights)
			suspiciousness_jaccard = np.zeros(num_weights)
			
			print(f"  计算层{idx_to_tl}的{num_weights}个权重的标准化SBFL...")
			layer_participating = 0
			
			for weight_idx in range(num_weights):
				ef = 0; ep = 0; nf = 0; np_val = 0
				
				for sample_idx in all_indices:
					weight_fi_value = all_sample_fi[sample_idx][idx_to_tl]['costs'][weight_idx, 1]
					
					# 标准化FI值
					if weight_fi_value > 0:
						standardized_fi = (weight_fi_value - layer_stats['mean']) / layer_stats['std']
					else:
						standardized_fi = -layer_stats['mean'] / layer_stats['std']  # 零值的标准化
					
					# 判断权重是否参与 (基于标准化值)
					weight_participates = standardized_fi > standardized_threshold
					sample_correct = sample_idx in passed_samples
					
					if weight_participates and not sample_correct: ef += 1
					elif weight_participates and sample_correct: ep += 1
					elif not weight_participates and not sample_correct: nf += 1
					elif not weight_participates and sample_correct: np_val += 1
				
				# 计算Ochiai可疑度
				if ef + ep > 0 and ef + nf > 0:
					suspiciousness_ochiai[weight_idx] = ef / np.sqrt((ef + ep) * (ef + nf))
				else:
					suspiciousness_ochiai[weight_idx] = 0.0
				
				# 计算Tarantula可疑度
				if ef + nf > 0 and ep + np_val > 0:
					failed_rate = ef / (ef + nf)
					passed_rate = ep / (ep + np_val)
					if failed_rate + passed_rate > 0:
						suspiciousness_tarantula[weight_idx] = failed_rate / (failed_rate + passed_rate)
					else:
						suspiciousness_tarantula[weight_idx] = 0.0
				else:
					suspiciousness_tarantula[weight_idx] = 0.0
				
				# 计算DStar可疑度
				if ep + nf > 0:
					suspiciousness_dstar[weight_idx] = (ef * ef) / (ep + nf)
				else:
					suspiciousness_dstar[weight_idx] = 0.0
				
				# 计算Jaccard可疑度
				if ef + ep + nf > 0:
					suspiciousness_jaccard[weight_idx] = ef / (ef + ep + nf)
				else:
					suspiciousness_jaccard[weight_idx] = 0.0
				
				# 统计参与的权重数
				if ef + ep > 0:
					layer_participating += 1
			
			total_cands[idx_to_tl] = {
				'shape': layer_shape,
				'costs_ochiai': suspiciousness_ochiai,
				'costs_tarantula': suspiciousness_tarantula,
				'costs_dstar': suspiciousness_dstar,
				'costs_jaccard': suspiciousness_jaccard
			}
			
			total_participating_weights += layer_participating
			print(f"    层{idx_to_tl}参与权重: {layer_participating}/{num_weights} ({layer_participating/num_weights*100:.1f}%)")
		
		elif model_util.is_LSTM(lname):
			layer_shapes = all_sample_fi[all_indices[0]][idx_to_tl]['shape']
			num_matrices = len(layer_shapes)
			total_cands[idx_to_tl] = {'shape': [], 'costs_ochiai': [], 'costs_tarantula': [], 'costs_dstar': [], 'costs_jaccard': []}
			
			for matrix_idx in range(num_matrices):
				matrix_shape = layer_shapes[matrix_idx]
				num_weights = np.prod(matrix_shape)
				suspiciousness_ochiai = np.zeros(num_weights)
				suspiciousness_tarantula = np.zeros(num_weights)
				suspiciousness_dstar = np.zeros(num_weights)
				suspiciousness_jaccard = np.zeros(num_weights)
				
				print(f"  计算层{idx_to_tl}矩阵{matrix_idx}的{num_weights}个权重...")
				layer_participating = 0
				
				for weight_idx in range(num_weights):
					ef = 0; ep = 0; nf = 0; np_val = 0
					
					for sample_idx in all_indices:
						weight_fi_value = all_sample_fi[sample_idx][idx_to_tl]['costs'][matrix_idx][weight_idx, 1]
						
						# 标准化FI值
						if weight_fi_value > 0:
							standardized_fi = (weight_fi_value - layer_stats['mean']) / layer_stats['std']
						else:
							standardized_fi = -layer_stats['mean'] / layer_stats['std']
						
						# 判断权重是否参与 (基于标准化值)
						weight_participates = standardized_fi > standardized_threshold
						sample_correct = sample_idx in passed_samples
						
						if weight_participates and not sample_correct: ef += 1
						elif weight_participates and sample_correct: ep += 1
						elif not weight_participates and not sample_correct: nf += 1
						elif not weight_participates and sample_correct: np_val += 1
					
					# 计算Ochiai可疑度
					if ef + ep > 0 and ef + nf > 0:
						suspiciousness_ochiai[weight_idx] = ef / np.sqrt((ef + ep) * (ef + nf))
					else:
						suspiciousness_ochiai[weight_idx] = 0.0
					
					# 计算Tarantula可疑度
					if ef + nf > 0 and ep + np_val > 0:
						failed_rate = ef / (ef + nf)
						passed_rate = ep / (ep + np_val)
						if failed_rate + passed_rate > 0:
							suspiciousness_tarantula[weight_idx] = failed_rate / (failed_rate + passed_rate)
						else:
							suspiciousness_tarantula[weight_idx] = 0.0
					else:
						suspiciousness_tarantula[weight_idx] = 0.0
					
					# 计算DStar可疑度
					if ep + nf > 0:
						suspiciousness_dstar[weight_idx] = (ef * ef) / (ep + nf)
					else:
						suspiciousness_dstar[weight_idx] = 0.0
					
					# 计算Jaccard可疑度
					if ef + ep + nf > 0:
						suspiciousness_jaccard[weight_idx] = ef / (ef + ep + nf)
					else:
						suspiciousness_jaccard[weight_idx] = 0.0
					
					# 统计参与的权重数
					if ef + ep > 0:
						layer_participating += 1
				
				total_cands[idx_to_tl]['shape'].append(matrix_shape)
				total_cands[idx_to_tl]['costs_ochiai'].append(suspiciousness_ochiai)
				total_cands[idx_to_tl]['costs_tarantula'].append(suspiciousness_tarantula)
				total_cands[idx_to_tl]['costs_dstar'].append(suspiciousness_dstar)
				total_cands[idx_to_tl]['costs_jaccard'].append(suspiciousness_jaccard)
				
				total_participating_weights += layer_participating
				print(f"    层{idx_to_tl}矩阵{matrix_idx}参与权重: {layer_participating}/{num_weights} ({layer_participating/num_weights*100:.1f}%)")
		
		else:
			print ("{} not supported yet".format(lname))
			assert False

	# 5. 生成并排序结果
	costs_and_keys_ochiai = []
	costs_and_keys_tarantula = []
	costs_and_keys_dstar = []
	costs_and_keys_jaccard = []
	
	for idx_to_tl, vs in total_cands.items():
		if not model_util.is_LSTM(target_weights[idx_to_tl][1]):
			for i, (ochiai_score, tarantula_score, dstar_score, jaccard_score) in enumerate(zip(vs['costs_ochiai'], vs['costs_tarantula'], vs['costs_dstar'], vs['costs_jaccard'])):
				cost_and_key_ochiai = ([idx_to_tl, i], ochiai_score)
				cost_and_key_tarantula = ([idx_to_tl, i], tarantula_score)
				cost_and_key_dstar = ([idx_to_tl, i], dstar_score)
				cost_and_key_jaccard = ([idx_to_tl, i], jaccard_score)
				costs_and_keys_ochiai.append(cost_and_key_ochiai)
				costs_and_keys_tarantula.append(cost_and_key_tarantula)
				costs_and_keys_dstar.append(cost_and_key_dstar)
				costs_and_keys_jaccard.append(cost_and_key_jaccard)
		else:
			for matrix_idx, (ochiai_matrix, tarantula_matrix, dstar_matrix, jaccard_matrix) in enumerate(zip(vs['costs_ochiai'], vs['costs_tarantula'], vs['costs_dstar'], vs['costs_jaccard'])):
				for i, (ochiai_score, tarantula_score, dstar_score, jaccard_score) in enumerate(zip(ochiai_matrix, tarantula_matrix, dstar_matrix, jaccard_matrix)):
					cost_and_key_ochiai = ([(idx_to_tl, matrix_idx), i], ochiai_score)
					cost_and_key_tarantula = ([(idx_to_tl, matrix_idx), i], tarantula_score)
					cost_and_key_dstar = ([(idx_to_tl, matrix_idx), i], dstar_score)
					cost_and_key_jaccard = ([(idx_to_tl, matrix_idx), i], jaccard_score)
					costs_and_keys_ochiai.append(cost_and_key_ochiai)
					costs_and_keys_tarantula.append(cost_and_key_tarantula)
					costs_and_keys_dstar.append(cost_and_key_dstar)
					costs_and_keys_jaccard.append(cost_and_key_jaccard)

	# 排序结果
	sorted_costs_and_keys_ochiai = sorted(costs_and_keys_ochiai, key = lambda vs:vs[1], reverse = True)
	sorted_costs_and_keys_tarantula = sorted(costs_and_keys_tarantula, key = lambda vs:vs[1], reverse = True)
	sorted_costs_and_keys_dstar = sorted(costs_and_keys_dstar, key = lambda vs:vs[1], reverse = True)
	sorted_costs_and_keys_jaccard = sorted(costs_and_keys_jaccard, key = lambda vs:vs[1], reverse = True)
	
	total_weights = sum(np.prod(vs['shape']) if not model_util.is_LSTM(target_weights[k][1]) 
					else sum(np.prod(s) for s in vs['shape']) 
					for k, vs in total_cands.items())
	
	print(f"  📊 标准化结果统计:")
	print(f"    参与权重: {total_participating_weights}/{total_weights} ({total_participating_weights/total_weights*100:.1f}%)")
	print(f"    最终结果: 四个公式都生成了{len(sorted_costs_and_keys_ochiai)}个权重排名")
	
	return {
		'ochiai': sorted_costs_and_keys_ochiai,
		'tarantula': sorted_costs_and_keys_tarantula,
		'dstar': sorted_costs_and_keys_dstar,
		'jaccard': sorted_costs_and_keys_jaccard
	}

def localise_by_FI_SBFL_continuous(
	X, y, indices_to_chgd, indices_to_unchgd, target_weights,
	path_to_keras_model=None, is_multi_label=True):
	"""
	FI-SBFL 连续值版本 - 使用FI值作为连续权重而非二元参与判断
	
	正确的实现思路:
	1. 逐样本计算FI值
	2. 逐样本判断分类正确性（预测 vs 真实标签）
	3. 根据分类正确性累加FI值到ef_continuous或ep_continuous
	"""
	
	import numpy as np
	import tensorflow as tf
	from tqdm import tqdm
	
	print("🚀 FI-SBFL连续值方法: 逐样本计算FI并按分类正确性累加")
	
	# 合并所有样本索引
	all_indices = list(indices_to_chgd) + list(indices_to_unchgd)
	print(f"  使用样本数: {len(all_indices)} (changed: {len(indices_to_chgd)}, unchanged: {len(indices_to_unchgd)})")
	
	# 载入模型
	model = tf.keras.models.load_model(path_to_keras_model, compile=False)
	
	# 预计算所有样本的FI值和分类正确性
	print("逐样本计算FI值和分类正确性...")
	sample_data = {}  # {sample_idx: {'fi_values': {weight_key: fi_value}, 'is_correct': bool}}
	
	for sample_idx in tqdm(all_indices, desc="处理样本"):
		sample_x = X[sample_idx:sample_idx+1]
		sample_y = y[sample_idx:sample_idx+1]
		
		# 1. 计算该样本的FI值
		single_sample_indices = np.array([sample_idx])
		fi_and_gl_results = compute_FI_and_GL(X, y, single_sample_indices, target_weights,
											is_multi_label=is_multi_label, 
											path_to_keras_model=path_to_keras_model)
		
		# 提取FI值
		sample_fi_dict = {}
		for layer_idx, layer_results in fi_and_gl_results.items():
			costs = layer_results['costs']  # shape = (N, 2)，第0列是GL，第1列是FI
			shape = layer_results['shape']
			
			for weight_flat_idx, (gl_score, fi_score) in enumerate(costs):
				weight_multidim_idx = np.unravel_index(weight_flat_idx, shape)
				weight_key = (layer_idx, weight_multidim_idx)
				sample_fi_dict[weight_key] = fi_score
		
		# 2. 判断该样本分类是否正确
		pred = model.predict(sample_x, verbose=0)
		sample_is_correct = is_correct(pred, sample_y, is_multi_label)
		
		sample_data[sample_idx] = {
			'fi_values': sample_fi_dict,
			'is_correct': sample_is_correct
		}
	
	# 统计分类正确性
	correct_count = sum(1 for data in sample_data.values() if data['is_correct'])
	wrong_count = len(all_indices) - correct_count
	print(f"  分类统计: 正确{correct_count}个, 错误{wrong_count}个")
	
	# 计算连续SBFL指标
	print("计算连续SBFL可疑度...")
	
	# 收集所有权重键
	all_weight_keys = set()
	for data in sample_data.values():
		all_weight_keys.update(data['fi_values'].keys())
	all_weight_keys = list(all_weight_keys)
	
	# 为每个权重计算连续SBFL分数
	all_scores = {'ochiai': [], 'tarantula': [], 'dstar': [], 'jaccard': []}
	
	# 用于快速索引的权重索引表
	weight_index = {wk: i for i, wk in enumerate(all_weight_keys)}
	num_weights = len(all_weight_keys)
	
	# 初始化连续版的 ef/ep/nf/np
	ef_arr = np.zeros(num_weights, dtype=float)
	ep_arr = np.zeros(num_weights, dtype=float)
	nf_arr = np.zeros(num_weights, dtype=float)
	np_arr = np.zeros(num_weights, dtype=float)
	
	# 改进累加口径：FI>0 视为执行；执行则 ef/ep 累加原始FI；未执行则 nf/np 记1
	for sample_idx in all_indices:
		sample_info = sample_data[sample_idx]
		fi_map = sample_info['fi_values']
		is_ok = sample_info['is_correct']
		for i, wk in enumerate(all_weight_keys):
			fi_val = fi_map.get(wk, 0.0)
			if fi_val > 0.0:
				if is_ok:
					ep_arr[i] += fi_val
				else:
					ef_arr[i] += fi_val
			else:
				if is_ok:
					np_arr[i] += 1.0
				else:
					nf_arr[i] += 1.0

	# 计算四种SBFL公式
	ef = ef_arr; ep = ep_arr; nf = nf_arr; np_val = np_arr

	# Ochiai: ef / sqrt((ef + ep) * (ef + nf))
	denominator_ochiai = np.sqrt((ef + ep) * (ef + nf))
	score_ochiai = np.divide(ef, denominator_ochiai, out=np.zeros_like(ef), where=denominator_ochiai>0)

	# Tarantula: ef/(ef+nf) / (ef/(ef+nf) + ep/(ep+np))
	ef_rate = np.divide(ef, ef + nf, out=np.zeros_like(ef), where=(ef+nf)>0)
	ep_rate = np.divide(ep, ep + np_val, out=np.zeros_like(ep), where=(ep+np_val)>0)
	denom_tarantula = ef_rate + ep_rate
	score_tarantula = np.divide(ef_rate, denom_tarantula, out=np.zeros_like(ef_rate), where=denom_tarantula>0)

	# DStar: ef^2 / (ep + nf)
	denom_dstar = ep + nf
	score_dstar = np.divide(ef*ef, denom_dstar, out=np.zeros_like(ef), where=denom_dstar>0)

	# Jaccard: ef / (ef + ep + nf)
	denom_jaccard = ef + ep + nf
	score_jaccard = np.divide(ef, denom_jaccard, out=np.zeros_like(ef), where=denom_jaccard>0)

	# 创建排序结果
	results = {}
	for formula_name, scores in [
		('ochiai', score_ochiai),
		('tarantula', score_tarantula),
		('dstar', score_dstar),
		('jaccard', score_jaccard),
	]:
		sorted_indices = np.argsort(scores)[::-1]
		sorted_results = [(all_weight_keys[i], float(scores[i])) for i in sorted_indices]
		results[formula_name] = sorted_results
		# 统计非零分数数量
		nonzero_count = int(np.sum(scores > 0))
		print(f"  {formula_name.capitalize()}: {nonzero_count}/{len(scores)}个权重有非零分数")

	print(f"  最终结果: 四种连续SBFL公式都生成了{len(all_weight_keys)}个权重排名")

	return results


def localise_by_FI_only_confidence(
	X, y,
	indices_to_chgd,
	indices_to_unchgd,
	target_weights,
	path_to_keras_model = None,
	is_multi_label = True):
	"""
	置信度加权的FI-only故障定位
	对每个样本计算 fi_score * (1 + confidence)，然后聚合
	"""
	from collections.abc import Iterable
	import tensorflow as tf
	import numpy as np
	
	total_cands = {}
	# set loss func (虽然FI不直接用loss，但compute_FI_and_GL需要)
	loss_func = model_util.get_loss_func(is_multi_label = is_multi_label)
	
	# 加载模型计算置信度
	model = tf.keras.models.load_model(path_to_keras_model, compile=False)
	
	## slice inputs
	for idx_to_tl, vs in target_weights.items():
		t_w, lname = vs
		print(f"targeting layer {idx_to_tl} ({lname})")
		
		if model_util.is_C2D(lname) or model_util.is_FC(lname): # either FC or C2D
			# 计算changed样本的置信度加权FI
			weighted_fi_chgd = np.zeros(t_w.size)
			print(f"Processing {len(indices_to_chgd)} changed samples...")
			
			for sample_idx in indices_to_chgd:
				# 计算单个样本的FI
				fi_and_gl_single = compute_FI_and_GL(
					X, y, [sample_idx], {idx_to_tl: vs},
					is_multi_label = is_multi_label,
					path_to_keras_model = path_to_keras_model)
				
				# 计算这个样本的置信度
				pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
				# 使用 margin (top1 - top2) 代替 max probability
				if pred.size >= 2:
					# 计算 margin: top1 - top2
					top2 = np.partition(pred, -2)[-2:]
					confidence = top2.max() - top2.min()
				else:
					# 如果只有一个输出，使用原来的最大概率
					# 使用 margin (top1 - top2) 代替 max probability
					if pred.size >= 2:
						# 计算 margin: top1 - top2
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						# 如果只有一个输出，使用原来的最大概率
						confidence = np.max(pred)
				
				# FI * (1 + confidence)
				fi_single = fi_and_gl_single[idx_to_tl]['costs'][:, 1]
				weighted_fi_chgd += fi_single * (1.0 + confidence)
			
			# 计算unchanged样本的置信度加权FI
			weighted_fi_unchgd = np.zeros(t_w.size)
			print(f"Processing {len(indices_to_unchgd)} unchanged samples...")
			
			for sample_idx in indices_to_unchgd:
				# 计算单个样本的FI
				fi_and_gl_single = compute_FI_and_GL(
					X, y, [sample_idx], {idx_to_tl: vs},
					is_multi_label = is_multi_label,
					path_to_keras_model = path_to_keras_model)
				
				# 计算这个样本的置信度
				pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
				# 使用 margin (top1 - top2) 代替 max probability
				if pred.size >= 2:
					# 计算 margin: top1 - top2
					top2 = np.partition(pred, -2)[-2:]
					confidence = top2.max() - top2.min()
				else:
					# 如果只有一个输出，使用原来的最大概率
					# 使用 margin (top1 - top2) 代替 max probability
					if pred.size >= 2:
						# 计算 margin: top1 - top2
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						# 如果只有一个输出，使用原来的最大概率
						confidence = np.max(pred)
				
				# FI * (1 + confidence)
				fi_single = fi_and_gl_single[idx_to_tl]['costs'][:, 1]
				weighted_fi_unchgd += fi_single * (1.0 + confidence)
			
			# 计算置信度加权的FI比值：weighted_FI_changed / (1 + weighted_FI_unchanged)
			final_scores = weighted_fi_chgd / (1.0 + weighted_fi_unchgd)
			
			total_cands[idx_to_tl] = {
				'shape': t_w.shape, 
				'costs': final_scores}
				
		elif model_util.is_LSTM(lname):
			# LSTM层的处理
			num_matrices = len(t_w)  # LSTM通常有多个权重矩阵
			total_cands[idx_to_tl] = {'shape': [], 'costs': []}
			
			for matrix_idx in range(num_matrices):
				matrix_shape = t_w[matrix_idx].shape
				matrix_size = t_w[matrix_idx].size
				
				# 计算changed样本的置信度加权FI
				weighted_fi_chgd = np.zeros(matrix_size)
				
				for sample_idx in indices_to_chgd:
					# 计算单个样本的FI
					fi_and_gl_single = compute_FI_and_GL(
						X, y, [sample_idx], {idx_to_tl: vs},
						is_multi_label = is_multi_label,
						path_to_keras_model = path_to_keras_model)
					
					# 计算这个样本的置信度
					pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
					# 使用 margin (top1 - top2) 代替 max probability
					if pred.size >= 2:
						# 计算 margin: top1 - top2
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						# 如果只有一个输出，使用原来的最大概率
						confidence = np.max(pred)
					
					# FI * (1 + confidence)
					fi_single = fi_and_gl_single[idx_to_tl]['costs'][matrix_idx][:, 1]
					weighted_fi_chgd += fi_single * (1.0 + confidence)
				
				# 计算unchanged样本的置信度加权FI
				weighted_fi_unchgd = np.zeros(matrix_size)
				
				for sample_idx in indices_to_unchgd:
					# 计算单个样本的FI
					fi_and_gl_single = compute_FI_and_GL(
						X, y, [sample_idx], {idx_to_tl: vs},
						is_multi_label = is_multi_label,
						path_to_keras_model = path_to_keras_model)
					
					# 计算这个样本的置信度
					pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
					# 使用 margin (top1 - top2) 代替 max probability
					if pred.size >= 2:
						# 计算 margin: top1 - top2
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						# 如果只有一个输出，使用原来的最大概率
						confidence = np.max(pred)
					
					# FI * (1 + confidence)
					fi_single = fi_and_gl_single[idx_to_tl]['costs'][matrix_idx][:, 1]
					weighted_fi_unchgd += fi_single * (1.0 + confidence)
				
				# 计算这个矩阵的最终分数
				final_scores = weighted_fi_chgd / (1.0 + weighted_fi_unchgd)
				
				total_cands[idx_to_tl]['shape'].append(matrix_shape)
				total_cands[idx_to_tl]['costs'].append(final_scores)
		else:
			print ("{} not supported yet".format(lname))
			assert False

	# 生成排序结果（完全类似GL算法的结构）
	indices_to_tl = list(total_cands.keys())
	costs_and_keys = []
	for idx_to_tl in indices_to_tl:
		if not model_util.is_LSTM(target_weights[idx_to_tl][1]):
			for local_i,c in enumerate(total_cands[idx_to_tl]['costs']):
				cost_and_key = ([idx_to_tl, np.unravel_index(local_i, total_cands[idx_to_tl]['shape'])], c) 
				costs_and_keys.append(cost_and_key)
		else:
			num = len(total_cands[idx_to_tl]['shape'])
			for idx_to_w in range(num):
				for local_i, c in enumerate(total_cands[idx_to_tl]['costs'][idx_to_w]):
					cost_and_key = (
						[(idx_to_tl, idx_to_w), 
						np.unravel_index(local_i, total_cands[idx_to_tl]['shape'][idx_to_w])], c) 
					costs_and_keys.append(cost_and_key)

	sorted_costs_and_keys = sorted(costs_and_keys, key = lambda vs:vs[1], reverse = True)
	print(f"✅ FI-only-confidence完成，共{len(sorted_costs_and_keys)}个权重")
	return sorted_costs_and_keys

def localise_by_FI_SBFL_topk(
	X, y, indices_to_chgd, indices_to_unchgd, target_weights,
	path_to_keras_model=None, is_multi_label=True,
	top_k=3, per_layer=True):
	"""
	FI-SBFL per-sample top-k 版本：
	- 逐样本计算该样本的FI
	- 对每个样本，按层（per_layer=True）为每层选FI最高的top_k个权重作为"执行"；否则全局选top_k
	- 用真实分类正确性统计 ef/ep/nf/np（标准离散SBFL计数）
	- 返回四种公式的排序结果
	"""
	import numpy as np
	import tensorflow as tf
	from tqdm import tqdm

	print(f"🚀 FI-SBFL top-k 方法: per_sample top_k={top_k}, per_layer={per_layer}")

	# 样本集合
	all_indices = list(indices_to_chgd) + list(indices_to_unchgd)
	print(f"  使用样本数: {len(all_indices)} (changed: {len(indices_to_chgd)}, unchanged: {len(indices_to_unchgd)})")

	# 加载模型
	model = tf.keras.models.load_model(path_to_keras_model, compile=False)

	# 为第一次获取权重空间（all_weight_keys）准备：
	first_idx = np.array([all_indices[0]]) if len(all_indices)>0 else np.array([0])
	first_res = compute_FI_and_GL(X, y, first_idx, target_weights,
								is_multi_label=is_multi_label,
								path_to_keras_model=path_to_keras_model)
	# 权重键空间（固定顺序）
	all_weight_keys = []
	layer_to_shape = {}
	for layer_idx, layer_results in first_res.items():
		shape = layer_results['shape']
		layer_to_shape[layer_idx] = shape
		for flat_idx in range(np.prod(shape)):
			all_weight_keys.append((layer_idx, np.unravel_index(flat_idx, shape)))

	num_weights = len(all_weight_keys)
	ef = np.zeros(num_weights, dtype=float)
	ep = np.zeros(num_weights, dtype=float)
	nf = np.zeros(num_weights, dtype=float)
	np_val = np.zeros(num_weights, dtype=float)

	# 快速索引
	weight_index = {wk: i for i, wk in enumerate(all_weight_keys)}

	# 逐样本处理
	for sample_idx in tqdm(all_indices, desc="per-sample top-k"):
		sample_x = X[sample_idx:sample_idx+1]
		sample_y = y[sample_idx:sample_idx+1]
		# 分类是否正确
		pred = model.predict(sample_x, verbose=0)
		ok = is_correct(pred, sample_y, is_multi_label)

		# 该样本的FI
		res = compute_FI_and_GL(X, y, np.array([sample_idx]), target_weights,
								is_multi_label=is_multi_label,
								path_to_keras_model=path_to_keras_model)

		# 构造参与集合
		participating = set()
		if per_layer:
			for layer_idx, layer_results in res.items():
				costs = layer_results['costs']  # (N,2)
				if costs.size == 0:
					continue
				fi_vec = costs[:, 1]  # 第1列是FI
				# 选top_k
				k = min(top_k, fi_vec.shape[0])
				if k <= 0:
					continue
				topk_idx = np.argpartition(-fi_vec, kth=k-1)[:k]
				shape = layer_results['shape']
				for flat in topk_idx:
					wk = (layer_idx, np.unravel_index(int(flat), shape))
					participating.add(wk)
		else:
			# 全局top-k
			glob_keys = []
			glob_scores = []
			for layer_idx, layer_results in res.items():
				fi_vec = layer_results['costs'][:, 1]
				shape = layer_results['shape']
				for flat, score in enumerate(fi_vec):
					glob_keys.append((layer_idx, np.unravel_index(int(flat), shape)))
					glob_scores.append(score)
			if len(glob_scores) > 0:
				glob_scores = np.array(glob_scores)
				k = min(top_k, len(glob_scores))
				topk_idx = np.argpartition(-glob_scores, kth=k-1)[:k]
				for idx in topk_idx:
					participating.add(glob_keys[int(idx)])

		# 用标准计数更新ef/ep/nf/np
		if ok:
			for i, wk in enumerate(all_weight_keys):
				if wk in participating:
					ep[i] += 1
				else:
					np_val[i] += 1
		else:
			for i, wk in enumerate(all_weight_keys):
				if wk in participating:
					ef[i] += 1
				else:
					nf[i] += 1

	# 计算四种公式
	results = {}
	# Ochiai
	denom_ochiai = np.sqrt((ef + ep) * (ef + nf))
	score_ochiai = np.divide(ef, denom_ochiai, out=np.zeros_like(ef), where=denom_ochiai>0)
	# Tarantula
	ef_rate = np.divide(ef, ef + nf, out=np.zeros_like(ef), where=(ef+nf)>0)
	ep_rate = np.divide(ep, ep + np_val, out=np.zeros_like(ep), where=(ep+np_val)>0)
	denom_tarantula = ef_rate + ep_rate
	score_tarantula = np.divide(ef_rate, denom_tarantula, out=np.zeros_like(ef_rate), where=denom_tarantula>0)
	# DStar
	denom_dstar = ep + nf
	score_dstar = np.divide(ef*ef, denom_dstar, out=np.zeros_like(ef), where=denom_dstar>0)
	# Jaccard
	denom_jaccard = ef + ep + nf
	score_jaccard = np.divide(ef, denom_jaccard, out=np.zeros_like(ef), where=denom_jaccard>0)

	for name, scores in [
		('ochiai', score_ochiai),
		('tarantula', score_tarantula),
		('dstar', score_dstar),
		('jaccard', score_jaccard),
	]:
		order = np.argsort(scores)[::-1]
		results[name] = [(all_weight_keys[i], float(scores[i])) for i in order]
		nonzero = int(np.sum(scores > 0))
		print(f"  {name.capitalize()}: {nonzero}/{num_weights}个权重有非零分数")

	print(f"  最终结果: 四种SBFL公式都生成了{num_weights}个权重排名")
	return results


def localise_by_FI_only_confidence(
	X, y,
	indices_to_chgd,
	indices_to_unchgd,
	target_weights,
	path_to_keras_model = None,
	is_multi_label = True):
	"""
	置信度加权的FI-only故障定位
	对每个样本计算 fi_score * (1 + confidence)，然后聚合
	"""
	from collections.abc import Iterable
	import tensorflow as tf
	import numpy as np
	
	total_cands = {}
	# set loss func (虽然FI不直接用loss，但compute_FI_and_GL需要)
	loss_func = model_util.get_loss_func(is_multi_label = is_multi_label)
	
	# 加载模型计算置信度
	model = tf.keras.models.load_model(path_to_keras_model, compile=False)
	
	## slice inputs
	for idx_to_tl, vs in target_weights.items():
		t_w, lname = vs
		print(f"targeting layer {idx_to_tl} ({lname})")
		
		if model_util.is_C2D(lname) or model_util.is_FC(lname): # either FC or C2D
			# 计算changed样本的置信度加权FI
			weighted_fi_chgd = np.zeros(t_w.size)
			print(f"Processing {len(indices_to_chgd)} changed samples...")
			
			for sample_idx in indices_to_chgd:
				# 计算单个样本的FI
				fi_and_gl_single = compute_FI_and_GL(
					X, y, [sample_idx], {idx_to_tl: vs},
					is_multi_label = is_multi_label,
					path_to_keras_model = path_to_keras_model)
				
				# 计算这个样本的置信度
				pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
				# 使用 margin (top1 - top2) 代替 max probability
				if pred.size >= 2:
					# 计算 margin: top1 - top2
					top2 = np.partition(pred, -2)[-2:]
					confidence = top2.max() - top2.min()
				else:
					# 如果只有一个输出，使用原来的最大概率
					# 使用 margin (top1 - top2) 代替 max probability
					if pred.size >= 2:
						# 计算 margin: top1 - top2
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						# 如果只有一个输出，使用原来的最大概率
						confidence = np.max(pred)
				
				# FI * (1 + confidence)
				fi_single = fi_and_gl_single[idx_to_tl]['costs'][:, 1]
				weighted_fi_chgd += fi_single * (1.0 + confidence)
			
			# 计算unchanged样本的置信度加权FI
			weighted_fi_unchgd = np.zeros(t_w.size)
			print(f"Processing {len(indices_to_unchgd)} unchanged samples...")
			
			for sample_idx in indices_to_unchgd:
				# 计算单个样本的FI
				fi_and_gl_single = compute_FI_and_GL(
					X, y, [sample_idx], {idx_to_tl: vs},
					is_multi_label = is_multi_label,
					path_to_keras_model = path_to_keras_model)
				
				# 计算这个样本的置信度
				pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
				# 使用 margin (top1 - top2) 代替 max probability
				if pred.size >= 2:
					# 计算 margin: top1 - top2
					top2 = np.partition(pred, -2)[-2:]
					confidence = top2.max() - top2.min()
				else:
					# 如果只有一个输出，使用原来的最大概率
					# 使用 margin (top1 - top2) 代替 max probability
					if pred.size >= 2:
						# 计算 margin: top1 - top2
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						# 如果只有一个输出，使用原来的最大概率
						confidence = np.max(pred)
				
				# FI * (1 + confidence)
				fi_single = fi_and_gl_single[idx_to_tl]['costs'][:, 1]
				weighted_fi_unchgd += fi_single * (1.0 + confidence)
			
			# 计算置信度加权的FI比值：weighted_FI_changed / (1 + weighted_FI_unchanged)
			final_scores = weighted_fi_chgd / (1.0 + weighted_fi_unchgd)
			
			total_cands[idx_to_tl] = {
				'shape': t_w.shape, 
				'costs': final_scores}
				
		elif model_util.is_LSTM(lname):
			# LSTM层的处理
			num_matrices = len(t_w)  # LSTM通常有多个权重矩阵
			total_cands[idx_to_tl] = {'shape': [], 'costs': []}
			
			for matrix_idx in range(num_matrices):
				matrix_shape = t_w[matrix_idx].shape
				matrix_size = t_w[matrix_idx].size
				
				# 计算changed样本的置信度加权FI
				weighted_fi_chgd = np.zeros(matrix_size)
				
				for sample_idx in indices_to_chgd:
					# 计算单个样本的FI
					fi_and_gl_single = compute_FI_and_GL(
						X, y, [sample_idx], {idx_to_tl: vs},
						is_multi_label = is_multi_label,
						path_to_keras_model = path_to_keras_model)
					
					# 计算这个样本的置信度
					pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
					# 使用 margin (top1 - top2) 代替 max probability
					if pred.size >= 2:
						# 计算 margin: top1 - top2
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						# 如果只有一个输出，使用原来的最大概率
						confidence = np.max(pred)
					
					# FI * (1 + confidence)
					fi_single = fi_and_gl_single[idx_to_tl]['costs'][matrix_idx][:, 1]
					weighted_fi_chgd += fi_single * (1.0 + confidence)
				
				# 计算unchanged样本的置信度加权FI
				weighted_fi_unchgd = np.zeros(matrix_size)
				
				for sample_idx in indices_to_unchgd:
					# 计算单个样本的FI
					fi_and_gl_single = compute_FI_and_GL(
						X, y, [sample_idx], {idx_to_tl: vs},
						is_multi_label = is_multi_label,
						path_to_keras_model = path_to_keras_model)
					
					# 计算这个样本的置信度
					pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
					# 使用 margin (top1 - top2) 代替 max probability
					if pred.size >= 2:
						# 计算 margin: top1 - top2
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						# 如果只有一个输出，使用原来的最大概率
						confidence = np.max(pred)
					
					# FI * (1 + confidence)
					fi_single = fi_and_gl_single[idx_to_tl]['costs'][matrix_idx][:, 1]
					weighted_fi_unchgd += fi_single * (1.0 + confidence)
				
				# 计算这个矩阵的最终分数
				final_scores = weighted_fi_chgd / (1.0 + weighted_fi_unchgd)
				
				total_cands[idx_to_tl]['shape'].append(matrix_shape)
				total_cands[idx_to_tl]['costs'].append(final_scores)
		else:
			print ("{} not supported yet".format(lname))
			assert False

	# 生成排序结果（完全类似GL算法的结构）
	indices_to_tl = list(total_cands.keys())
	costs_and_keys = []
	for idx_to_tl in indices_to_tl:
		if not model_util.is_LSTM(target_weights[idx_to_tl][1]):
			for local_i,c in enumerate(total_cands[idx_to_tl]['costs']):
				cost_and_key = ([idx_to_tl, np.unravel_index(local_i, total_cands[idx_to_tl]['shape'])], c) 
				costs_and_keys.append(cost_and_key)
		else:
			num = len(total_cands[idx_to_tl]['shape'])
			for idx_to_w in range(num):
				for local_i, c in enumerate(total_cands[idx_to_tl]['costs'][idx_to_w]):
					cost_and_key = (
						[(idx_to_tl, idx_to_w), 
						np.unravel_index(local_i, total_cands[idx_to_tl]['shape'][idx_to_w])], c) 
					costs_and_keys.append(cost_and_key)

	sorted_costs_and_keys = sorted(costs_and_keys, key = lambda vs:vs[1], reverse = True)
	print(f"✅ FI-only-confidence完成，共{len(sorted_costs_and_keys)}个权重")
	return sorted_costs_and_keys

def localise_by_FI_SBFL_cont_exec(
	X, y, indices_to_chgd, indices_to_unchgd, target_weights,
	path_to_keras_model=None, is_multi_label=True,
	fi_eps=0.0, use_fi_weight=True,
	exec_percentile=None, per_layer=True):
	"""
	FI-SBFL 连续值（执行判定）版本：
	- 逐样本计算FI
	- 执行判定：
	* 若 exec_percentile 不为 None，则每样本、按层用该百分位阈值选执行集合（例如99表示Top 1%）
	* 否则使用 fi_eps 判断 fi > fi_eps 为执行
	- ef/ep：若执行，用FI作为权重累加（use_fi_weight=True），否则按1计数（可选）
	- nf/np：若未执行，各加1（不做样本归一化、不用补数）
	- 返回四种公式排序
	"""
	import numpy as np
	import tensorflow as tf
	from tqdm import tqdm

	print(f"🚀 FI-SBFL 连续执行版: fi_eps={fi_eps}, use_fi_weight={use_fi_weight}, exec_percentile={exec_percentile}, per_layer={per_layer}")

	all_indices = list(indices_to_chgd) + list(indices_to_unchgd)
	print(f"  使用样本数: {len(all_indices)} (changed: {len(indices_to_chgd)}, unchanged: {len(indices_to_unchgd)})")

	model = tf.keras.models.load_model(path_to_keras_model, compile=False)

	# 确定权重空间
	init_idx = np.array([all_indices[0]]) if len(all_indices) > 0 else np.array([0])
	init_res = compute_FI_and_GL(X, y, init_idx, target_weights,
								is_multi_label=is_multi_label,
								path_to_keras_model=path_to_keras_model)
	all_weight_keys = []
	for layer_idx, layer_results in init_res.items():
		shape = layer_results['shape']
		for flat_idx in range(np.prod(shape)):
			all_weight_keys.append((layer_idx, np.unravel_index(int(flat_idx), shape)))
	num_weights = len(all_weight_keys)

	ef = np.zeros(num_weights, dtype=float)
	ep = np.zeros(num_weights, dtype=float)
	nf = np.zeros(num_weights, dtype=float)
	np_val = np.zeros(num_weights, dtype=float)

	# 逐样本
	for sample_idx in tqdm(all_indices, desc="per-sample exec"):
		sample_x = X[sample_idx:sample_idx+1]
		sample_y = y[sample_idx:sample_idx+1]
		pred = model.predict(sample_x, verbose=0)
		ok = is_correct(pred, sample_y, is_multi_label)

		res = compute_FI_and_GL(X, y, np.array([sample_idx]), target_weights,
								is_multi_label=is_multi_label,
								path_to_keras_model=path_to_keras_model)
		# 遍历所有权重（与all_weight_keys顺序一致：按层、flat顺序）
		wi = 0
		for layer_idx, layer_results in res.items():
			fi_vec = layer_results['costs'][:, 1]
			shape = layer_results['shape']
			# 计算该层执行掩码
			if exec_percentile is not None and per_layer and fi_vec.size > 0:
				thr = np.percentile(fi_vec, exec_percentile)
				exec_mask = fi_vec >= thr
			else:
				exec_mask = fi_vec > fi_eps
			# 累加
			for flat in range(fi_vec.shape[0]):
				fi = float(fi_vec[flat])
				executed = bool(exec_mask[flat])
				if executed:
					if use_fi_weight:
						if ok:
							ep[wi] += fi
						else:
							ef[wi] += fi
					else:
						if ok:
							ep[wi] += 1.0
						else:
							ef[wi] += 1.0
				else:
					if ok:
						np_val[wi] += 1.0
					else:
						nf[wi] += 1.0
				wi += 1

	# 计算四种SBFL公式
	results = {}
	# Ochiai
	denom_ochiai = np.sqrt((ef + ep) * (ef + nf))
	score_ochiai = np.divide(ef, denom_ochiai, out=np.zeros_like(ef), where=denom_ochiai>0)
	# Tarantula
	ef_rate = np.divide(ef, ef + nf, out=np.zeros_like(ef), where=(ef+nf)>0)
	ep_rate = np.divide(ep, ep + np_val, out=np.zeros_like(ep), where=(ep+np_val)>0)
	denom_tarantula = ef_rate + ep_rate
	score_tarantula = np.divide(ef_rate, denom_tarantula, out=np.zeros_like(ef_rate), where=denom_tarantula>0)
	# DStar
	denom_dstar = ep + nf
	score_dstar = np.divide(ef*ef, denom_dstar, out=np.zeros_like(ef), where=denom_dstar>0)
	# Jaccard
	denom_jaccard = ef + ep + nf
	score_jaccard = np.divide(ef, denom_jaccard, out=np.zeros_like(ef), where=denom_jaccard>0)

	for name, scores in [
		('ochiai', score_ochiai),
		('tarantula', score_tarantula),
		('dstar', score_dstar),
		('jaccard', score_jaccard),
	]:
		order = np.argsort(scores)[::-1]
		results[name] = [(all_weight_keys[i], float(scores[i])) for i in order]
		nonzero = int(np.sum(scores > 0))
		print(f"  {name.capitalize()}: {nonzero}/{num_weights}个权重有非零分数")

	print(f"  最终结果: 四种SBFL公式都生成了{num_weights}个权重排名")
	return results


def localise_by_FI_only_confidence(
	X, y,
	indices_to_chgd,
	indices_to_unchgd,
	target_weights,
	path_to_keras_model = None,
	is_multi_label = True):
	"""
	置信度加权的FI-only故障定位
	对每个样本计算 fi_score * (1 + confidence)，然后聚合
	"""
	from collections.abc import Iterable
	import tensorflow as tf
	import numpy as np
	
	total_cands = {}
	# set loss func (虽然FI不直接用loss，但compute_FI_and_GL需要)
	loss_func = model_util.get_loss_func(is_multi_label = is_multi_label)
	
	# 加载模型计算置信度
	model = tf.keras.models.load_model(path_to_keras_model, compile=False)
	
	## slice inputs
	for idx_to_tl, vs in target_weights.items():
		t_w, lname = vs
		print(f"targeting layer {idx_to_tl} ({lname})")
		
		if model_util.is_C2D(lname) or model_util.is_FC(lname): # either FC or C2D
			# 计算changed样本的置信度加权FI
			weighted_fi_chgd = np.zeros(t_w.size)
			print(f"Processing {len(indices_to_chgd)} changed samples...")
			
			for sample_idx in indices_to_chgd:
				# 计算单个样本的FI
				fi_and_gl_single = compute_FI_and_GL(
					X, y, [sample_idx], {idx_to_tl: vs},
					is_multi_label = is_multi_label,
					path_to_keras_model = path_to_keras_model)
				
				# 计算这个样本的置信度
				pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
				# 使用 margin (top1 - top2) 代替 max probability
				if pred.size >= 2:
					# 计算 margin: top1 - top2
					top2 = np.partition(pred, -2)[-2:]
					confidence = top2.max() - top2.min()
				else:
					# 如果只有一个输出，使用原来的最大概率
					# 使用 margin (top1 - top2) 代替 max probability
					if pred.size >= 2:
						# 计算 margin: top1 - top2
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						# 如果只有一个输出，使用原来的最大概率
						confidence = np.max(pred)
				
				# FI * (1 + confidence)
				fi_single = fi_and_gl_single[idx_to_tl]['costs'][:, 1]
				weighted_fi_chgd += fi_single * (1.0 + confidence)
			
			# 计算unchanged样本的置信度加权FI
			weighted_fi_unchgd = np.zeros(t_w.size)
			print(f"Processing {len(indices_to_unchgd)} unchanged samples...")
			
			for sample_idx in indices_to_unchgd:
				# 计算单个样本的FI
				fi_and_gl_single = compute_FI_and_GL(
					X, y, [sample_idx], {idx_to_tl: vs},
					is_multi_label = is_multi_label,
					path_to_keras_model = path_to_keras_model)
				
				# 计算这个样本的置信度
				pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
				# 使用 margin (top1 - top2) 代替 max probability
				if pred.size >= 2:
					# 计算 margin: top1 - top2
					top2 = np.partition(pred, -2)[-2:]
					confidence = top2.max() - top2.min()
				else:
					# 如果只有一个输出，使用原来的最大概率
					# 使用 margin (top1 - top2) 代替 max probability
					if pred.size >= 2:
						# 计算 margin: top1 - top2
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						# 如果只有一个输出，使用原来的最大概率
						confidence = np.max(pred)
				
				# FI * (1 + confidence)
				fi_single = fi_and_gl_single[idx_to_tl]['costs'][:, 1]
				weighted_fi_unchgd += fi_single * (1.0 + confidence)
			
			# 计算置信度加权的FI比值：weighted_FI_changed / (1 + weighted_FI_unchanged)
			final_scores = weighted_fi_chgd / (1.0 + weighted_fi_unchgd)
			
			total_cands[idx_to_tl] = {
				'shape': t_w.shape, 
				'costs': final_scores}
				
		elif model_util.is_LSTM(lname):
			# LSTM层的处理
			num_matrices = len(t_w)  # LSTM通常有多个权重矩阵
			total_cands[idx_to_tl] = {'shape': [], 'costs': []}
			
			for matrix_idx in range(num_matrices):
				matrix_shape = t_w[matrix_idx].shape
				matrix_size = t_w[matrix_idx].size
				
				# 计算changed样本的置信度加权FI
				weighted_fi_chgd = np.zeros(matrix_size)
				
				for sample_idx in indices_to_chgd:
					# 计算单个样本的FI
					fi_and_gl_single = compute_FI_and_GL(
						X, y, [sample_idx], {idx_to_tl: vs},
						is_multi_label = is_multi_label,
						path_to_keras_model = path_to_keras_model)
					
					# 计算这个样本的置信度
					pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
					# 使用 margin (top1 - top2) 代替 max probability
					if pred.size >= 2:
						# 计算 margin: top1 - top2
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						# 如果只有一个输出，使用原来的最大概率
						confidence = np.max(pred)
					
					# FI * (1 + confidence)
					fi_single = fi_and_gl_single[idx_to_tl]['costs'][matrix_idx][:, 1]
					weighted_fi_chgd += fi_single * (1.0 + confidence)
				
				# 计算unchanged样本的置信度加权FI
				weighted_fi_unchgd = np.zeros(matrix_size)
				
				for sample_idx in indices_to_unchgd:
					# 计算单个样本的FI
					fi_and_gl_single = compute_FI_and_GL(
						X, y, [sample_idx], {idx_to_tl: vs},
						is_multi_label = is_multi_label,
						path_to_keras_model = path_to_keras_model)
					
					# 计算这个样本的置信度
					pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
					# 使用 margin (top1 - top2) 代替 max probability
					if pred.size >= 2:
						# 计算 margin: top1 - top2
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						# 如果只有一个输出，使用原来的最大概率
						confidence = np.max(pred)
					
					# FI * (1 + confidence)
					fi_single = fi_and_gl_single[idx_to_tl]['costs'][matrix_idx][:, 1]
					weighted_fi_unchgd += fi_single * (1.0 + confidence)
				
				# 计算这个矩阵的最终分数
				final_scores = weighted_fi_chgd / (1.0 + weighted_fi_unchgd)
				
				total_cands[idx_to_tl]['shape'].append(matrix_shape)
				total_cands[idx_to_tl]['costs'].append(final_scores)
		else:
			print ("{} not supported yet".format(lname))
			assert False

	# 生成排序结果（完全类似GL算法的结构）
	indices_to_tl = list(total_cands.keys())
	costs_and_keys = []
	for idx_to_tl in indices_to_tl:
		if not model_util.is_LSTM(target_weights[idx_to_tl][1]):
			for local_i,c in enumerate(total_cands[idx_to_tl]['costs']):
				cost_and_key = ([idx_to_tl, np.unravel_index(local_i, total_cands[idx_to_tl]['shape'])], c) 
				costs_and_keys.append(cost_and_key)
		else:
			num = len(total_cands[idx_to_tl]['shape'])
			for idx_to_w in range(num):
				for local_i, c in enumerate(total_cands[idx_to_tl]['costs'][idx_to_w]):
					cost_and_key = (
						[(idx_to_tl, idx_to_w), 
						np.unravel_index(local_i, total_cands[idx_to_tl]['shape'][idx_to_w])], c) 
					costs_and_keys.append(cost_and_key)

	sorted_costs_and_keys = sorted(costs_and_keys, key = lambda vs:vs[1], reverse = True)
	print(f"✅ FI-only-confidence完成，共{len(sorted_costs_and_keys)}个权重")
	return sorted_costs_and_keys

def localise_by_FI_SBFL_confidence(
	X, y, indices_to_chgd, indices_to_unchgd, target_weights,
	path_to_keras_model=None, is_multi_label=True,
	fi_threshold_percentile=96):
	"""
	FI-SBFL (confidence-weighted):
	- 执行判定：使用全局FI百分位阈值（默认96%）
	- 成败：由分类正确性决定
	- 样本权重：对每个样本取模型输出最大概率，做min-max归一化到[0,1]得到W(Ti)
	加权方案可通过环境变量`CONF_WEIGHT_SCHEME`配置：
		* 'all_add1' (默认): ef/ep/nf/np 均使用 1+W(Ti)
		* 'all': ef/ep/nf/np 均使用 W(Ti)
		* 'fail_inv_add1': 仅失败样本使用 1+(1-W(Ti))，通过/未执行按1计数
		* 'fail_inv': 仅失败样本使用 (1-W(Ti))，通过/未执行按1计数
		* 'fail_add1': 仅失败样本使用 1+W(Ti)，通过/未执行按1计数
		* 'fail': 仅失败样本使用 W(Ti)，通过/未执行按1计数
	无匹配时回退到 'all_add1'.
	- 计数：按所选方案对 ef/ep/nf/np 加权累加（加权谱统计）
	- 输出四种SBFL公式（Ochiai、Tarantula、DStar、Jaccard）
	"""
	import numpy as np
	import tensorflow as tf
	from tqdm import tqdm
	import os

	weight_scheme = os.environ.get('CONF_WEIGHT_SCHEME', 'all_add1').strip().lower()
	score_metric = os.environ.get('CONF_SCORE_METRIC', 'maxprob').strip().lower()  # 'maxprob' or 'margin'
	print(f"🚀 FI-SBFL 置信度加权版: fi_threshold_percentile={fi_threshold_percentile}, scheme={weight_scheme}, metric={score_metric}")

	# 采样集合
	all_indices = list(indices_to_chgd) + list(indices_to_unchgd)
	print(f"  使用样本数: {len(all_indices)} (changed: {len(indices_to_chgd)}, unchanged: {len(indices_to_unchgd)})")

	# 加载模型
	model = tf.keras.models.load_model(path_to_keras_model, compile=False)

	# 1) 逐样本获取预测最大概率（置信度），以及正确性
	sample_scores = []  # 根据 metric 选择的分数（maxprob 或 margin）
	sample_correct = {}
	for sample_idx in all_indices:
		sx = X[sample_idx:sample_idx+1]
		sy = y[sample_idx:sample_idx+1]
		pred = model.predict(sx, verbose=0)
		# 计算分数
		if pred.ndim > 1:
			vec = np.ravel(pred)
		else:
			vec = np.array([float(pred)])
		if score_metric == 'margin':
			if vec.size >= 2:
				top2 = np.partition(vec, -2)[-2:]
				score = float(top2.max() - top2.min())
			else:
				score = float(vec.max())
		else:  # 'maxprob'
			score = float(vec.max())
		sample_scores.append(score)
		sample_correct[sample_idx] = bool(is_correct(pred, sy, is_multi_label))

	# min-max 归一化到[0,1]
	max_probs = np.asarray(sample_scores, dtype=float)
	mm_min, mm_max = float(np.min(max_probs)), float(np.max(max_probs))
	denom = mm_max - mm_min if (mm_max - mm_min) > 1e-12 else 1.0
	norm_W = (max_probs - mm_min) / denom
	# 根据方案计算不同通道的样本权重
	sample_weights_fail = {}
	sample_weights_pass = {}
	sample_weights_notexec = {}
	for i, idx in enumerate(all_indices):
		c = float(norm_W[i])  # in [0,1]
		if weight_scheme == 'all':
			wf = c; wp = c; wne = c
		elif weight_scheme == 'fail_inv_add1':
			wf = 1.0 + (1.0 - c); wp = 1.0; wne = 1.0
		elif weight_scheme == 'fail_inv':
			wf = (1.0 - c); wp = 1.0; wne = 1.0
		elif weight_scheme == 'fail_add1':
			wf = 1.0 + c; wp = 1.0; wne = 1.0
		elif weight_scheme == 'fail':
			wf = c; wp = 1.0; wne = 1.0
		else:  # 'all_add1'
			wf = 1.0 + c; wp = 1.0 + c; wne = 1.0 + c
		sample_weights_fail[idx] = float(wf)
		sample_weights_pass[idx] = float(wp)
		sample_weights_notexec[idx] = float(wne)

	# 2) 预计算所有样本的FI（用于参与判定）
	print("预计算所有样本的FI值用于全局阈值...")
	all_fi_values_list = []  # 收集所有FI用于阈值
	all_fi_per_sample = {}   # {sample_idx: { (layer, coord): fi_value }}

	for sample_idx in tqdm(all_indices, desc="computing FI"):
		fi_res = compute_FI_and_GL(X, y, np.array([sample_idx]), target_weights,
								is_multi_label=is_multi_label,
								path_to_keras_model=path_to_keras_model)
		fi_map = {}
		for layer_idx, layer_results in fi_res.items():
			costs = layer_results['costs']  # (N,2) [GL, FI]
			shape = layer_results['shape']
			for flat_i, pair in enumerate(costs):
				fi_val = float(pair[1])
				coord = np.unravel_index(flat_i, shape)
				key = (layer_idx, coord)
				fi_map[key] = fi_val
				all_fi_values_list.append(fi_val)
		all_fi_per_sample[sample_idx] = fi_map

	all_fi_values = np.asarray(all_fi_values_list, dtype=float)
	fi_thr = float(np.percentile(all_fi_values, fi_threshold_percentile)) if all_fi_values.size > 0 else 0.0
	print(f"  全局FI阈值（{fi_threshold_percentile}%）：{fi_thr:.8f}")

	# 3) 构建权重键空间（与compute_FI_and_GL顺序一致：按层+flat）
	init_idx = np.array([all_indices[0]]) if len(all_indices) > 0 else np.array([0])
	init_res = compute_FI_and_GL(X, y, init_idx, target_weights,
								is_multi_label=is_multi_label,
								path_to_keras_model=path_to_keras_model)
	all_weight_keys = []
	for layer_idx, layer_results in init_res.items():
		shape = layer_results['shape']
		for flat_i in range(int(np.prod(shape))):
			coord = np.unravel_index(flat_i, shape)
			all_weight_keys.append((layer_idx, coord))

	num_weights = len(all_weight_keys)
	ef = np.zeros(num_weights, dtype=float)
	ep = np.zeros(num_weights, dtype=float)
	nf = np.zeros(num_weights, dtype=float)
	np_val = np.zeros(num_weights, dtype=float)

	# 4) 逐权重、逐样本进行加权谱统计
	print("计算加权SBFL计数...")
	key_index = {wk: i for i, wk in enumerate(all_weight_keys)}
	for sample_idx in all_indices:
		fi_map = all_fi_per_sample[sample_idx]
		w_fail = sample_weights_fail[sample_idx]
		w_pass = sample_weights_pass[sample_idx]
		w_notexec = sample_weights_notexec[sample_idx]
		ok = sample_correct[sample_idx]
		for wk, i in key_index.items():
			fi_val = fi_map.get(wk, 0.0)
			executed = fi_val > fi_thr
			if executed:
				if ok:
					ep[i] += w_pass
				else:
					ef[i] += w_fail
			else:
				if ok:
					np_val[i] += w_notexec
				else:
					nf[i] += w_notexec

	# 5) 四种SBFL公式
	results = {}
	# Ochiai
	denom_ochiai = np.sqrt((ef + ep) * (ef + nf))
	score_ochiai = np.divide(ef, denom_ochiai, out=np.zeros_like(ef), where=denom_ochiai > 0)
	# Tarantula
	ef_rate = np.divide(ef, ef + nf, out=np.zeros_like(ef), where=(ef + nf) > 0)
	ep_rate = np.divide(ep, ep + np_val, out=np.zeros_like(ep), where=(ep + np_val) > 0)
	denom_tarantula = ef_rate + ep_rate
	score_tarantula = np.divide(ef_rate, denom_tarantula, out=np.zeros_like(ef_rate), where=denom_tarantula > 0)
	# DStar
	denom_dstar = ep + nf
	score_dstar = np.divide(ef * ef, denom_dstar, out=np.zeros_like(ef), where=denom_dstar > 0)
	# Jaccard
	denom_jaccard = ef + ep + nf
	score_jaccard = np.divide(ef, denom_jaccard, out=np.zeros_like(ef), where=denom_jaccard > 0)

	for name, scores in [
		('ochiai', score_ochiai),
		('tarantula', score_tarantula),
		('dstar', score_dstar),
		('jaccard', score_jaccard),
	]:
		order = np.argsort(scores)[::-1]
		results[name] = [(all_weight_keys[i], float(scores[i])) for i in order]
		nonzero = int(np.sum(scores > 0))
		print(f"  {name.upper()}: 非零={nonzero}/{num_weights}")

	print(f"  最终结果: 生成{num_weights}个权重排名（四公式）")
	return results


def localise_by_FI_only_confidence(
	X, y,
	indices_to_chgd,
	indices_to_unchgd,
	target_weights,
	path_to_keras_model = None,
	is_multi_label = True):
	"""
	置信度加权的FI-only故障定位
	对每个样本计算 fi_score * (1 + confidence)，然后聚合
	"""
	from collections.abc import Iterable
	import tensorflow as tf
	import numpy as np
	
	total_cands = {}
	# set loss func (虽然FI不直接用loss，但compute_FI_and_GL需要)
	loss_func = model_util.get_loss_func(is_multi_label = is_multi_label)
	
	# 加载模型计算置信度
	model = tf.keras.models.load_model(path_to_keras_model, compile=False)
	
	## slice inputs
	for idx_to_tl, vs in target_weights.items():
		t_w, lname = vs
		print(f"targeting layer {idx_to_tl} ({lname})")
		
		if model_util.is_C2D(lname) or model_util.is_FC(lname): # either FC or C2D
			# 计算changed样本的置信度加权FI
			weighted_fi_chgd = np.zeros(t_w.size)
			print(f"Processing {len(indices_to_chgd)} changed samples...")
			
			for sample_idx in indices_to_chgd:
				# 计算单个样本的FI
				fi_and_gl_single = compute_FI_and_GL(
					X, y, [sample_idx], {idx_to_tl: vs},
					is_multi_label = is_multi_label,
					path_to_keras_model = path_to_keras_model)
				
				# 计算这个样本的置信度
				pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
				# 使用 margin (top1 - top2) 代替 max probability
				if pred.size >= 2:
					# 计算 margin: top1 - top2
					top2 = np.partition(pred, -2)[-2:]
					confidence = top2.max() - top2.min()
				else:
					# 如果只有一个输出，使用原来的最大概率
					# 使用 margin (top1 - top2) 代替 max probability
					if pred.size >= 2:
						# 计算 margin: top1 - top2
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						# 如果只有一个输出，使用原来的最大概率
						confidence = np.max(pred)
				
				# FI * (1 + confidence)
				fi_single = fi_and_gl_single[idx_to_tl]['costs'][:, 1]
				weighted_fi_chgd += fi_single * (1.0 + confidence)
			
			# 计算unchanged样本的置信度加权FI
			weighted_fi_unchgd = np.zeros(t_w.size)
			print(f"Processing {len(indices_to_unchgd)} unchanged samples...")
			
			for sample_idx in indices_to_unchgd:
				# 计算单个样本的FI
				fi_and_gl_single = compute_FI_and_GL(
					X, y, [sample_idx], {idx_to_tl: vs},
					is_multi_label = is_multi_label,
					path_to_keras_model = path_to_keras_model)
				
				# 计算这个样本的置信度
				pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
				# 使用 margin (top1 - top2) 代替 max probability
				if pred.size >= 2:
					# 计算 margin: top1 - top2
					top2 = np.partition(pred, -2)[-2:]
					confidence = top2.max() - top2.min()
				else:
					# 如果只有一个输出，使用原来的最大概率
					# 使用 margin (top1 - top2) 代替 max probability
					if pred.size >= 2:
						# 计算 margin: top1 - top2
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						# 如果只有一个输出，使用原来的最大概率
						confidence = np.max(pred)
				
				# FI * (1 + confidence)
				fi_single = fi_and_gl_single[idx_to_tl]['costs'][:, 1]
				weighted_fi_unchgd += fi_single * (1.0 + confidence)
			
			# 计算置信度加权的FI比值：weighted_FI_changed / (1 + weighted_FI_unchanged)
			final_scores = weighted_fi_chgd / (1.0 + weighted_fi_unchgd)
			
			total_cands[idx_to_tl] = {
				'shape': t_w.shape, 
				'costs': final_scores}
				
		elif model_util.is_LSTM(lname):
			# LSTM层的处理
			num_matrices = len(t_w)  # LSTM通常有多个权重矩阵
			total_cands[idx_to_tl] = {'shape': [], 'costs': []}
			
			for matrix_idx in range(num_matrices):
				matrix_shape = t_w[matrix_idx].shape
				matrix_size = t_w[matrix_idx].size
				
				# 计算changed样本的置信度加权FI
				weighted_fi_chgd = np.zeros(matrix_size)
				
				for sample_idx in indices_to_chgd:
					# 计算单个样本的FI
					fi_and_gl_single = compute_FI_and_GL(
						X, y, [sample_idx], {idx_to_tl: vs},
						is_multi_label = is_multi_label,
						path_to_keras_model = path_to_keras_model)
					
					# 计算这个样本的置信度
					pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
					# 使用 margin (top1 - top2) 代替 max probability
					if pred.size >= 2:
						# 计算 margin: top1 - top2
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						# 如果只有一个输出，使用原来的最大概率
						confidence = np.max(pred)
					
					# FI * (1 + confidence)
					fi_single = fi_and_gl_single[idx_to_tl]['costs'][matrix_idx][:, 1]
					weighted_fi_chgd += fi_single * (1.0 + confidence)
				
				# 计算unchanged样本的置信度加权FI
				weighted_fi_unchgd = np.zeros(matrix_size)
				
				for sample_idx in indices_to_unchgd:
					# 计算单个样本的FI
					fi_and_gl_single = compute_FI_and_GL(
						X, y, [sample_idx], {idx_to_tl: vs},
						is_multi_label = is_multi_label,
						path_to_keras_model = path_to_keras_model)
					
					# 计算这个样本的置信度
					pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
					# 使用 margin (top1 - top2) 代替 max probability
					if pred.size >= 2:
						# 计算 margin: top1 - top2
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						# 如果只有一个输出，使用原来的最大概率
						confidence = np.max(pred)
					
					# FI * (1 + confidence)
					fi_single = fi_and_gl_single[idx_to_tl]['costs'][matrix_idx][:, 1]
					weighted_fi_unchgd += fi_single * (1.0 + confidence)
				
				# 计算这个矩阵的最终分数
				final_scores = weighted_fi_chgd / (1.0 + weighted_fi_unchgd)
				
				total_cands[idx_to_tl]['shape'].append(matrix_shape)
				total_cands[idx_to_tl]['costs'].append(final_scores)
		else:
			print ("{} not supported yet".format(lname))
			assert False

	# 生成排序结果（完全类似GL算法的结构）
	indices_to_tl = list(total_cands.keys())
	costs_and_keys = []
	for idx_to_tl in indices_to_tl:
		if not model_util.is_LSTM(target_weights[idx_to_tl][1]):
			for local_i,c in enumerate(total_cands[idx_to_tl]['costs']):
				cost_and_key = ([idx_to_tl, np.unravel_index(local_i, total_cands[idx_to_tl]['shape'])], c) 
				costs_and_keys.append(cost_and_key)
		else:
			num = len(total_cands[idx_to_tl]['shape'])
			for idx_to_w in range(num):
				for local_i, c in enumerate(total_cands[idx_to_tl]['costs'][idx_to_w]):
					cost_and_key = (
						[(idx_to_tl, idx_to_w), 
						np.unravel_index(local_i, total_cands[idx_to_tl]['shape'][idx_to_w])], c) 
					costs_and_keys.append(cost_and_key)

	sorted_costs_and_keys = sorted(costs_and_keys, key = lambda vs:vs[1], reverse = True)
	print(f"✅ FI-only-confidence完成，共{len(sorted_costs_and_keys)}个权重")
	return sorted_costs_and_keys

def localise_by_FI_SBFL_confidence_balanced(
	X, y, indices_to_chgd, indices_to_unchgd, target_weights,
	path_to_keras_model=None, is_multi_label=True,
	fi_threshold_percentile=96, alpha=1.0):
	"""
	FI-SBFL (confidence-weighted, balanced):
	- 执行：全局FI百分位阈值（默认96%）
	- 成败：由分类正确性决定
	- 对称加权：
		执行且失败:  w_fail_raw = 1 + alpha * c
		执行且通过:  w_pass_raw = 1 + alpha * (1 - c)
		未执行:       w_notexec = 1
	其中 c 为样本最大预测概率的 min-max 归一化到[0,1]
	- 分组归一：对失败/通过两组分别除以各自均值，使组均值=1，消除总量不公平
	- 输出四种SBFL公式
	"""
	import numpy as np
	import tensorflow as tf
	from tqdm import tqdm

	print(f"🚀 FI-SBFL 置信度加权（对称+分组归一）: fi_threshold_percentile={fi_threshold_percentile}, alpha={alpha}")

	all_indices = list(indices_to_chgd) + list(indices_to_unchgd)
	print(f"  使用样本数: {len(all_indices)} (changed: {len(indices_to_chgd)}, unchanged: {len(indices_to_unchgd)})")

	model = tf.keras.models.load_model(path_to_keras_model, compile=False)

	# 1) 置信度与正确性
	max_probs = []
	sample_correct = {}
	for sample_idx in all_indices:
		sx = X[sample_idx:sample_idx+1]
		sy = y[sample_idx:sample_idx+1]
		pred = model.predict(sx, verbose=0)
		mprob = float(np.max(pred)) if pred.ndim > 0 else float(pred)
		max_probs.append(mprob)
		sample_correct[sample_idx] = bool(is_correct(pred, sy, is_multi_label))

	max_probs = np.asarray(max_probs, dtype=float)
	mm_min, mm_max = float(np.min(max_probs)), float(np.max(max_probs))
	denom = mm_max - mm_min if (mm_max - mm_min) > 1e-12 else 1.0
	c_arr = (max_probs - mm_min) / denom  # in [0,1]

	# 原始权重
	w_fail_raw = 1.0 + alpha * c_arr
	w_pass_raw = 1.0 + alpha * (1.0 - c_arr)

	# 分组均值（避免空集）
	wrong_mask = np.array([not sample_correct[idx] for idx in all_indices], dtype=bool)
	correct_mask = ~wrong_mask
	mu_fail = float(np.mean(w_fail_raw[wrong_mask])) if np.any(wrong_mask) else 1.0
	mu_pass = float(np.mean(w_pass_raw[correct_mask])) if np.any(correct_mask) else 1.0

	# 归一后的权重映射
	sample_weights_fail = {}
	sample_weights_pass = {}
	sample_weights_notexec = {}
	for pos, idx in enumerate(all_indices):
		sample_weights_fail[idx] = float(w_fail_raw[pos] / mu_fail)
		sample_weights_pass[idx] = float(w_pass_raw[pos] / mu_pass)
		sample_weights_notexec[idx] = 1.0

	# 2) 预计算FI
	print("预计算所有样本的FI值用于全局阈值...")
	all_fi_values_list = []
	all_fi_per_sample = {}
	for sample_idx in tqdm(all_indices, desc="computing FI"):
		fi_res = compute_FI_and_GL(X, y, np.array([sample_idx]), target_weights,
								is_multi_label=is_multi_label,
								path_to_keras_model=path_to_keras_model)
		fi_map = {}
		for layer_idx, layer_results in fi_res.items():
			costs = layer_results['costs']
			shape = layer_results['shape']
			for flat_i, pair in enumerate(costs):
				fi_val = float(pair[1])
				coord = np.unravel_index(flat_i, shape)
				fi_map[(layer_idx, coord)] = fi_val
				all_fi_values_list.append(fi_val)
		all_fi_per_sample[sample_idx] = fi_map

	all_fi_values = np.asarray(all_fi_values_list, dtype=float)
	fi_thr = float(np.percentile(all_fi_values, fi_threshold_percentile)) if all_fi_values.size > 0 else 0.0
	print(f"  全局FI阈值（{fi_threshold_percentile}%）：{fi_thr:.8f}")

	# 3) 权重键空间（由已计算的样本FI推导），并可选裁剪Top-K
	import os as _os_sbfl_conf
	if len(all_indices) > 0:
		any_idx = all_indices[0]
		all_weight_keys_full = list(all_fi_per_sample[any_idx].keys())
	else:
		all_weight_keys_full = []

	topk = int(_os_sbfl_conf.environ.get('SBFL_PRUNE_TOPK_WEIGHTS', '0') or '0')
	if topk > 0 and topk < len(all_weight_keys_full):
		# 以所有样本的 max FI 作为权重重要性进行候选裁剪
		weight_max_fi = {}
		for sidx in all_indices:
			for wk, val in all_fi_per_sample[sidx].items():
				if wk not in weight_max_fi or val > weight_max_fi[wk]:
					weight_max_fi[wk] = val
		sorted_wks = sorted(weight_max_fi.items(), key=lambda kv: kv[1], reverse=True)
		selected_keys = [wk for wk, _ in sorted_wks[:topk]]
		print(f"  裁剪候选权重: 选择Top-{topk}/{len(all_weight_keys_full)}")
	else:
		selected_keys = list(all_weight_keys_full)

	# 准备计数数组（仅对被选中的权重做计数）
	num_sel = len(selected_keys)
	ef = np.zeros(num_sel, dtype=float)
	ep = np.zeros(num_sel, dtype=float)
	nf = np.zeros(num_sel, dtype=float)
	np_val = np.zeros(num_sel, dtype=float)

	# 4) 逐权重、逐样本统计（仅对选中键），其余权重在结果中补0分
	print("计算加权SBFL计数（balanced）...")
	key_index = {wk: i for i, wk in enumerate(selected_keys)}
	for sample_idx in all_indices:
		fi_map = all_fi_per_sample[sample_idx]
		ok = sample_correct[sample_idx]
		wf = sample_weights_fail[sample_idx]
		wp = sample_weights_pass[sample_idx]
		wne = sample_weights_notexec[sample_idx]
		for wk, i in key_index.items():
			fi_val = fi_map.get(wk, 0.0)
			executed = fi_val > fi_thr
			if executed:
				if ok:
					ep[i] += wp
				else:
					ef[i] += wf
			else:
				if ok:
					np_val[i] += wne
				else:
					nf[i] += wne

	# 5) 四种SBFL公式，并为未选中键补0分以保持完整长度
def _compose_results(scores_arr):
		# 选中键的得分
		sel_scores = {wk: float(scores_arr[i]) for wk, i in key_index.items()}
		# 其余键补0
		rest_keys = [wk for wk in all_weight_keys_full if wk not in key_index]
		full_list = list(sel_scores.items()) + [(wk, 0.0) for wk in rest_keys]
		# 排序（降序）
		full_list.sort(key=lambda kv: kv[1], reverse=True)
		return full_list

	results = {}
	denom_ochiai = np.sqrt((ef + ep) * (ef + nf))
	score_ochiai = np.divide(ef, denom_ochiai, out=np.zeros_like(ef), where=denom_ochiai > 0)
	ef_rate = np.divide(ef, ef + nf, out=np.zeros_like(ef), where=(ef + nf) > 0)
	ep_rate = np.divide(ep, ep + np_val, out=np.zeros_like(ep), where=(ep + np_val) > 0)
	denom_tarantula = ef_rate + ep_rate
	score_tarantula = np.divide(ef_rate, denom_tarantula, out=np.zeros_like(ef_rate), where=denom_tarantula > 0)
	denom_dstar = ep + nf
	score_dstar = np.divide(ef * ef, denom_dstar, out=np.zeros_like(ef), where=denom_dstar > 0)
	denom_jaccard = ef + ep + nf
	score_jaccard = np.divide(ef, denom_jaccard, out=np.zeros_like(ef), where=denom_jaccard > 0)

	for name, scores in [
		('ochiai', score_ochiai),
		('tarantula', score_tarantula),
		('dstar', score_dstar),
		('jaccard', score_jaccard),
	]:
		results[name] = _compose_results(scores)
		nonzero = int(sum(1 for _, sc in results[name] if sc > 0))
		print(f"  {name.upper()}: 非零={nonzero}/{len(all_weight_keys_full)}")

	print(f"  最终结果: 生成{len(all_weight_keys_full)}个权重排名（四公式）")
	return results


def localise_by_FI_only_confidence(
	X, y,
	indices_to_chgd,
	indices_to_unchgd,
	target_weights,
	path_to_keras_model = None,
	is_multi_label = True):
	"""
	置信度加权的FI-only故障定位
	对每个样本计算 fi_score * (1 + confidence)，然后聚合
	"""
	from collections.abc import Iterable
	import tensorflow as tf
	import numpy as np
	
	total_cands = {}
	# set loss func (虽然FI不直接用loss，但compute_FI_and_GL需要)
	loss_func = model_util.get_loss_func(is_multi_label = is_multi_label)
	
	# 加载模型计算置信度
	model = tf.keras.models.load_model(path_to_keras_model, compile=False)
	
	## slice inputs
	for idx_to_tl, vs in target_weights.items():
		t_w, lname = vs
		print(f"targeting layer {idx_to_tl} ({lname})")
		
		if model_util.is_C2D(lname) or model_util.is_FC(lname): # either FC or C2D
			# 计算changed样本的置信度加权FI
			weighted_fi_chgd = np.zeros(t_w.size)
			print(f"Processing {len(indices_to_chgd)} changed samples...")
			
			for sample_idx in indices_to_chgd:
				# 计算单个样本的FI
				fi_and_gl_single = compute_FI_and_GL(
					X, y, [sample_idx], {idx_to_tl: vs},
					is_multi_label = is_multi_label,
					path_to_keras_model = path_to_keras_model)
				
				# 计算这个样本的置信度
				pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
				# 使用 margin (top1 - top2) 代替 max probability
				if pred.size >= 2:
					# 计算 margin: top1 - top2
					top2 = np.partition(pred, -2)[-2:]
					confidence = top2.max() - top2.min()
				else:
					# 如果只有一个输出，使用原来的最大概率
					# 使用 margin (top1 - top2) 代替 max probability
					if pred.size >= 2:
						# 计算 margin: top1 - top2
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						# 如果只有一个输出，使用原来的最大概率
						confidence = np.max(pred)
				
				# FI * (1 + confidence)
				fi_single = fi_and_gl_single[idx_to_tl]['costs'][:, 1]
				weighted_fi_chgd += fi_single * (1.0 + confidence)
			
			# 计算unchanged样本的置信度加权FI
			weighted_fi_unchgd = np.zeros(t_w.size)
			print(f"Processing {len(indices_to_unchgd)} unchanged samples...")
			
			for sample_idx in indices_to_unchgd:
				# 计算单个样本的FI
				fi_and_gl_single = compute_FI_and_GL(
					X, y, [sample_idx], {idx_to_tl: vs},
					is_multi_label = is_multi_label,
					path_to_keras_model = path_to_keras_model)
				
				# 计算这个样本的置信度
				pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
				# 使用 margin (top1 - top2) 代替 max probability
				if pred.size >= 2:
					# 计算 margin: top1 - top2
					top2 = np.partition(pred, -2)[-2:]
					confidence = top2.max() - top2.min()
				else:
					# 如果只有一个输出，使用原来的最大概率
					# 使用 margin (top1 - top2) 代替 max probability
					if pred.size >= 2:
						# 计算 margin: top1 - top2
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						# 如果只有一个输出，使用原来的最大概率
						confidence = np.max(pred)
				
				# FI * (1 + confidence)
				fi_single = fi_and_gl_single[idx_to_tl]['costs'][:, 1]
				weighted_fi_unchgd += fi_single * (1.0 + confidence)
			
			# 计算置信度加权的FI比值：weighted_FI_changed / (1 + weighted_FI_unchanged)
			final_scores = weighted_fi_chgd / (1.0 + weighted_fi_unchgd)
			
			total_cands[idx_to_tl] = {
				'shape': t_w.shape, 
				'costs': final_scores}
				
		elif model_util.is_LSTM(lname):
			# LSTM层的处理
			num_matrices = len(t_w)  # LSTM通常有多个权重矩阵
			total_cands[idx_to_tl] = {'shape': [], 'costs': []}
			
			for matrix_idx in range(num_matrices):
				matrix_shape = t_w[matrix_idx].shape
				matrix_size = t_w[matrix_idx].size
				
				# 计算changed样本的置信度加权FI
				weighted_fi_chgd = np.zeros(matrix_size)
				
				for sample_idx in indices_to_chgd:
					# 计算单个样本的FI
					fi_and_gl_single = compute_FI_and_GL(
						X, y, [sample_idx], {idx_to_tl: vs},
						is_multi_label = is_multi_label,
						path_to_keras_model = path_to_keras_model)
					
					# 计算这个样本的置信度
					pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
					# 使用 margin (top1 - top2) 代替 max probability
					if pred.size >= 2:
						# 计算 margin: top1 - top2
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						# 如果只有一个输出，使用原来的最大概率
						confidence = np.max(pred)
					
					# FI * (1 + confidence)
					fi_single = fi_and_gl_single[idx_to_tl]['costs'][matrix_idx][:, 1]
					weighted_fi_chgd += fi_single * (1.0 + confidence)
				
				# 计算unchanged样本的置信度加权FI
				weighted_fi_unchgd = np.zeros(matrix_size)
				
				for sample_idx in indices_to_unchgd:
					# 计算单个样本的FI
					fi_and_gl_single = compute_FI_and_GL(
						X, y, [sample_idx], {idx_to_tl: vs},
						is_multi_label = is_multi_label,
						path_to_keras_model = path_to_keras_model)
					
					# 计算这个样本的置信度
					pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
					# 使用 margin (top1 - top2) 代替 max probability
					if pred.size >= 2:
						# 计算 margin: top1 - top2
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						# 如果只有一个输出，使用原来的最大概率
						confidence = np.max(pred)
					
					# FI * (1 + confidence)
					fi_single = fi_and_gl_single[idx_to_tl]['costs'][matrix_idx][:, 1]
					weighted_fi_unchgd += fi_single * (1.0 + confidence)
				
				# 计算这个矩阵的最终分数
				final_scores = weighted_fi_chgd / (1.0 + weighted_fi_unchgd)
				
				total_cands[idx_to_tl]['shape'].append(matrix_shape)
				total_cands[idx_to_tl]['costs'].append(final_scores)
		else:
			print ("{} not supported yet".format(lname))
			assert False

	# 生成排序结果（完全类似GL算法的结构）
	indices_to_tl = list(total_cands.keys())
	costs_and_keys = []
	for idx_to_tl in indices_to_tl:
		if not model_util.is_LSTM(target_weights[idx_to_tl][1]):
			for local_i,c in enumerate(total_cands[idx_to_tl]['costs']):
				cost_and_key = ([idx_to_tl, np.unravel_index(local_i, total_cands[idx_to_tl]['shape'])], c) 
				costs_and_keys.append(cost_and_key)
		else:
			num = len(total_cands[idx_to_tl]['shape'])
			for idx_to_w in range(num):
				for local_i, c in enumerate(total_cands[idx_to_tl]['costs'][idx_to_w]):
					cost_and_key = (
						[(idx_to_tl, idx_to_w), 
						np.unravel_index(local_i, total_cands[idx_to_tl]['shape'][idx_to_w])], c) 
					costs_and_keys.append(cost_and_key)

	sorted_costs_and_keys = sorted(costs_and_keys, key = lambda vs:vs[1], reverse = True)
	print(f"✅ FI-only-confidence完成，共{len(sorted_costs_and_keys)}个权重")
	return sorted_costs_and_keys

def localise_by_FI_SBFL_smooth(
	X, y, indices_to_chgd, indices_to_unchgd, target_weights,
	path_to_keras_model=None, is_multi_label=True, scale_percentile=95):
	"""
	FI-SBFL 平滑分配版本 - 使用FI值的归一化概率进行平滑分配
	
	核心思想:
	1. 按层计算FI值的分布阈值（如95%分位数）
	2. 将FI值归一化为参与概率：a = min(|FI|/threshold, 1.0)
	3. 平滑分配：执行类累加a，未执行类累加(1-a)
	4. 保持度量一致性和总量守恒
	
	Args:
		scale_percentile: 用于层级标准化的百分位数，默认95%
	"""
	
	import numpy as np
	import tensorflow as tf
	from tqdm import tqdm
	
	print(f"🚀 FI-SBFL平滑分配版本: scale_percentile={scale_percentile}%")
	
	# 合并所有样本索引
	all_indices = list(indices_to_chgd) + list(indices_to_unchgd)
	print(f"  使用样本数: {len(all_indices)} (changed: {len(indices_to_chgd)}, unchanged: {len(indices_to_unchgd)})")
	
	# 载入模型
	model = tf.keras.models.load_model(path_to_keras_model, compile=False)
	
	# 预计算所有样本的FI值和分类正确性
	print("逐样本计算FI值和分类正确性...")
	sample_data = {}
	
	for sample_idx in tqdm(all_indices, desc="处理样本"):
		sample_x = X[sample_idx:sample_idx+1]
		sample_y = y[sample_idx:sample_idx+1]
		
		# 1. 计算该样本的FI值
		single_sample_indices = np.array([sample_idx])
		fi_and_gl_results = compute_FI_and_GL(X, y, single_sample_indices, target_weights,
											is_multi_label=is_multi_label, 
											path_to_keras_model=path_to_keras_model)
		
		# 提取FI值（按层组织）
		sample_fi_dict = {}
		for layer_idx, layer_results in fi_and_gl_results.items():
			costs = layer_results['costs']  # shape = (N, 2)，第0列是GL，第1列是FI
			shape = layer_results['shape']
			
			for weight_flat_idx, (gl_score, fi_score) in enumerate(costs):
				weight_multidim_idx = np.unravel_index(weight_flat_idx, shape)
				weight_key = (layer_idx, weight_multidim_idx)
				sample_fi_dict[weight_key] = fi_score
		
		# 2. 判断该样本分类是否正确
		pred = model.predict(sample_x, verbose=0)
		sample_is_correct = is_correct(pred, sample_y, is_multi_label)
		
		sample_data[sample_idx] = {
			'fi_values': sample_fi_dict,
			'is_correct': sample_is_correct
		}
	
	# 统计分类正确性
	correct_count = sum(1 for data in sample_data.values() if data['is_correct'])
	wrong_count = len(all_indices) - correct_count
	print(f"  分类统计: 正确{correct_count}个, 错误{wrong_count}个")
	
	# 计算每层的FI分布阈值（用于标准化）
	print(f"计算每层FI值的{scale_percentile}%分位数阈值...")
	layer_scales = {}  # {layer_idx: scale_threshold}
	
	# 收集每层的所有FI值
	layer_fi_values = {}
	for sample_data_item in sample_data.values():
		for weight_key, fi_val in sample_data_item['fi_values'].items():
			layer_idx = weight_key[0]
			if layer_idx not in layer_fi_values:
				layer_fi_values[layer_idx] = []
			layer_fi_values[layer_idx].append(abs(fi_val))  # 使用绝对值
	
	# 计算每层的分位数阈值
	for layer_idx, fi_vals in layer_fi_values.items():
		if fi_vals:
			scale_threshold = np.percentile(fi_vals, scale_percentile)
			layer_scales[layer_idx] = max(scale_threshold, 1e-8)  # 避免除零
			print(f"    层{layer_idx}: {scale_percentile}%分位数 = {scale_threshold:.8f}")
		else:
			layer_scales[layer_idx] = 1e-8
			print(f"    层{layer_idx}: 无有效FI值，使用默认阈值1e-8")
	
	# 收集所有权重键
	all_weight_keys = set()
	for data in sample_data.values():
		all_weight_keys.update(data['fi_values'].keys())
	all_weight_keys = list(all_weight_keys)
	
	# 初始化平滑SBFL计数器
	print("计算平滑分配SBFL计数...")
	num_weights = len(all_weight_keys)
	weight_index = {wk: i for i, wk in enumerate(all_weight_keys)}
	
	ef = np.zeros(num_weights, dtype=float)
	ep = np.zeros(num_weights, dtype=float)
	nf = np.zeros(num_weights, dtype=float)
	np_val = np.zeros(num_weights, dtype=float)
	
	# 平滑分配累加
	for sample_idx in all_indices:
		sample_info = sample_data[sample_idx]
		fi_map = sample_info['fi_values']
		is_ok = sample_info['is_correct']
		
		for i, wk in enumerate(all_weight_keys):
			fi_val = fi_map.get(wk, 0.0)
			layer_idx = wk[0]
			s = layer_scales[layer_idx]  # 该层的分位数阈值
			
			# 计算参与概率：a = min(|FI| / threshold, 1.0)
			a = min(abs(fi_val) / max(s, 1e-8), 1.0)
			
			# 平滑分配
			if is_ok:
				ep[i] += a          # 执行且成功
				np_val[i] += (1-a)  # 未执行且成功
			else:
				ef[i] += a          # 执行且失败
				nf[i] += (1-a)      # 未执行且失败
	
	# 计算四种SBFL公式
	print("计算SBFL可疑度分数...")
	
	# Ochiai: ef / sqrt((ef + ep) * (ef + nf))
	denominator_ochiai = np.sqrt((ef + ep) * (ef + nf))
	score_ochiai = np.divide(ef, denominator_ochiai, out=np.zeros_like(ef), where=denominator_ochiai>0)
	
	# Tarantula: ef/(ef+nf) / (ef/(ef+nf) + ep/(ep+np))
	ef_rate = np.divide(ef, ef + nf, out=np.zeros_like(ef), where=(ef+nf)>0)
	ep_rate = np.divide(ep, ep + np_val, out=np.zeros_like(ep), where=(ep+np_val)>0)
	denom_tarantula = ef_rate + ep_rate
	score_tarantula = np.divide(ef_rate, denom_tarantula, out=np.zeros_like(ef_rate), where=denom_tarantula>0)
	
	# DStar: ef^2 / (ep + nf)
	denom_dstar = ep + nf
	score_dstar = np.divide(ef*ef, denom_dstar, out=np.zeros_like(ef), where=denom_dstar>0)
	
	# Jaccard: ef / (ef + ep + nf)
	denom_jaccard = ef + ep + nf
	score_jaccard = np.divide(ef, denom_jaccard, out=np.zeros_like(ef), where=denom_jaccard>0)
	
	# 创建排序结果
	results = {}
	for formula_name, scores in [
		('ochiai', score_ochiai),
		('tarantula', score_tarantula),
		('dstar', score_dstar),
		('jaccard', score_jaccard),
	]:
		sorted_indices = np.argsort(scores)[::-1]
		sorted_results = [(all_weight_keys[i], float(scores[i])) for i in sorted_indices]
		results[formula_name] = sorted_results
		# 统计非零分数数量
		nonzero_count = int(np.sum(scores > 0))
		print(f"  {formula_name.capitalize()}: {nonzero_count}/{len(scores)}个权重有非零分数")
	
	# 统计信息
	total_samples = len(all_indices)
	avg_ef = np.mean(ef)
	avg_ep = np.mean(ep)
	avg_nf = np.mean(nf)
	avg_np = np.mean(np_val)
	
	print(f"  📊 平滑分配统计 (共{total_samples}个样本):")
	print(f"    平均ef={avg_ef:.2f}, ep={avg_ep:.2f}, nf={avg_nf:.2f}, np={avg_np:.2f}")
	print(f"    总量验证: ef+ep+nf+np = {np.mean(ef+ep+nf+np_val):.2f} (应约等于{total_samples})")
	
	return results


def localise_by_FI_only_confidence(
	X, y,
	indices_to_chgd,
	indices_to_unchgd,
	target_weights,
	path_to_keras_model = None,
	is_multi_label = True):
	"""
	置信度加权的FI-only故障定位
	对每个样本计算 fi_score * (1 + confidence)，然后聚合
	"""
	from collections.abc import Iterable
	import tensorflow as tf
	import numpy as np
	
	total_cands = {}
	# set loss func (虽然FI不直接用loss，但compute_FI_and_GL需要)
	loss_func = model_util.get_loss_func(is_multi_label = is_multi_label)
	
	# 加载模型计算置信度
	model = tf.keras.models.load_model(path_to_keras_model, compile=False)
	
	## slice inputs
	for idx_to_tl, vs in target_weights.items():
		t_w, lname = vs
		print(f"targeting layer {idx_to_tl} ({lname})")
		
		if model_util.is_C2D(lname) or model_util.is_FC(lname): # either FC or C2D
			# 计算changed样本的置信度加权FI
			weighted_fi_chgd = np.zeros(t_w.size)
			print(f"Processing {len(indices_to_chgd)} changed samples...")
			
			for sample_idx in indices_to_chgd:
				# 计算单个样本的FI
				fi_and_gl_single = compute_FI_and_GL(
					X, y, [sample_idx], {idx_to_tl: vs},
					is_multi_label = is_multi_label,
					path_to_keras_model = path_to_keras_model)
				
				# 计算这个样本的置信度
				pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
				# 使用 margin (top1 - top2) 代替 max probability
				if pred.size >= 2:
					# 计算 margin: top1 - top2
					top2 = np.partition(pred, -2)[-2:]
					confidence = top2.max() - top2.min()
				else:
					# 如果只有一个输出，使用原来的最大概率
					# 使用 margin (top1 - top2) 代替 max probability
					if pred.size >= 2:
						# 计算 margin: top1 - top2
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						# 如果只有一个输出，使用原来的最大概率
						confidence = np.max(pred)
				
				# FI * (1 + confidence)
				fi_single = fi_and_gl_single[idx_to_tl]['costs'][:, 1]
				weighted_fi_chgd += fi_single * (1.0 + confidence)
			
			# 计算unchanged样本的置信度加权FI
			weighted_fi_unchgd = np.zeros(t_w.size)
			print(f"Processing {len(indices_to_unchgd)} unchanged samples...")
			
			for sample_idx in indices_to_unchgd:
				# 计算单个样本的FI
				fi_and_gl_single = compute_FI_and_GL(
					X, y, [sample_idx], {idx_to_tl: vs},
					is_multi_label = is_multi_label,
					path_to_keras_model = path_to_keras_model)
				
				# 计算这个样本的置信度
				pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
				# 使用 margin (top1 - top2) 代替 max probability
				if pred.size >= 2:
					# 计算 margin: top1 - top2
					top2 = np.partition(pred, -2)[-2:]
					confidence = top2.max() - top2.min()
				else:
					# 如果只有一个输出，使用原来的最大概率
					# 使用 margin (top1 - top2) 代替 max probability
					if pred.size >= 2:
						# 计算 margin: top1 - top2
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						# 如果只有一个输出，使用原来的最大概率
						confidence = np.max(pred)
				
				# FI * (1 + confidence)
				fi_single = fi_and_gl_single[idx_to_tl]['costs'][:, 1]
				weighted_fi_unchgd += fi_single * (1.0 + confidence)
			
			# 计算置信度加权的FI比值：weighted_FI_changed / (1 + weighted_FI_unchanged)
			final_scores = weighted_fi_chgd / (1.0 + weighted_fi_unchgd)
			
			total_cands[idx_to_tl] = {
				'shape': t_w.shape, 
				'costs': final_scores}
				
		elif model_util.is_LSTM(lname):
			# LSTM层的处理
			num_matrices = len(t_w)  # LSTM通常有多个权重矩阵
			total_cands[idx_to_tl] = {'shape': [], 'costs': []}
			
			for matrix_idx in range(num_matrices):
				matrix_shape = t_w[matrix_idx].shape
				matrix_size = t_w[matrix_idx].size
				
				# 计算changed样本的置信度加权FI
				weighted_fi_chgd = np.zeros(matrix_size)
				
				for sample_idx in indices_to_chgd:
					# 计算单个样本的FI
					fi_and_gl_single = compute_FI_and_GL(
						X, y, [sample_idx], {idx_to_tl: vs},
						is_multi_label = is_multi_label,
						path_to_keras_model = path_to_keras_model)
					
					# 计算这个样本的置信度
					pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
					# 使用 margin (top1 - top2) 代替 max probability
					if pred.size >= 2:
						# 计算 margin: top1 - top2
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						# 如果只有一个输出，使用原来的最大概率
						confidence = np.max(pred)
					
					# FI * (1 + confidence)
					fi_single = fi_and_gl_single[idx_to_tl]['costs'][matrix_idx][:, 1]
					weighted_fi_chgd += fi_single * (1.0 + confidence)
				
				# 计算unchanged样本的置信度加权FI
				weighted_fi_unchgd = np.zeros(matrix_size)
				
				for sample_idx in indices_to_unchgd:
					# 计算单个样本的FI
					fi_and_gl_single = compute_FI_and_GL(
						X, y, [sample_idx], {idx_to_tl: vs},
						is_multi_label = is_multi_label,
						path_to_keras_model = path_to_keras_model)
					
					# 计算这个样本的置信度
					pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
					# 使用 margin (top1 - top2) 代替 max probability
					if pred.size >= 2:
						# 计算 margin: top1 - top2
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						# 如果只有一个输出，使用原来的最大概率
						confidence = np.max(pred)
					
					# FI * (1 + confidence)
					fi_single = fi_and_gl_single[idx_to_tl]['costs'][matrix_idx][:, 1]
					weighted_fi_unchgd += fi_single * (1.0 + confidence)
				
				# 计算这个矩阵的最终分数
				final_scores = weighted_fi_chgd / (1.0 + weighted_fi_unchgd)
				
				total_cands[idx_to_tl]['shape'].append(matrix_shape)
				total_cands[idx_to_tl]['costs'].append(final_scores)
		else:
			print ("{} not supported yet".format(lname))
			assert False

	# 生成排序结果（完全类似GL算法的结构）
	indices_to_tl = list(total_cands.keys())
	costs_and_keys = []
	for idx_to_tl in indices_to_tl:
		if not model_util.is_LSTM(target_weights[idx_to_tl][1]):
			for local_i,c in enumerate(total_cands[idx_to_tl]['costs']):
				cost_and_key = ([idx_to_tl, np.unravel_index(local_i, total_cands[idx_to_tl]['shape'])], c) 
				costs_and_keys.append(cost_and_key)
		else:
			num = len(total_cands[idx_to_tl]['shape'])
			for idx_to_w in range(num):
				for local_i, c in enumerate(total_cands[idx_to_tl]['costs'][idx_to_w]):
					cost_and_key = (
						[(idx_to_tl, idx_to_w), 
						np.unravel_index(local_i, total_cands[idx_to_tl]['shape'][idx_to_w])], c) 
					costs_and_keys.append(cost_and_key)

	sorted_costs_and_keys = sorted(costs_and_keys, key = lambda vs:vs[1], reverse = True)
	print(f"✅ FI-only-confidence完成，共{len(sorted_costs_and_keys)}个权重")
	return sorted_costs_and_keys


import numpy as np
import tensorflow as tf

def localise_by_FI_SBFL_enhanced(
	X, y,
	indices_to_chgd, indices_to_unchgd,
	target_weights,
	path_to_keras_model=None,
	is_multi_label=True,
	fi_threshold_percentile=99  # 全局阈值百分位（原始 |FI|）
):
	"""
	FI+SBFL（增强版，兼容老接口）
	- 执行判定：基于全局百分位阈值（原始 |FI|，不做 log），二值 exec∈{0,1}
	- 失败/成功：用 is_correct(预测, 真实) 判定
	- 公式：Ochiai / Tarantula / DStar / Jaccard
	- 返回：dict[{formula} -> sorted list of ((layer_key, weight_pos), score)]
	"""

	import numpy as np
	import tensorflow as tf
	from tqdm import tqdm

	EPS = 1e-12

	# 0) 汇总样本索引
	all_indices = list(indices_to_chgd) + list(indices_to_unchgd)
	if len(all_indices) == 0:
		print("⚠️ 没有样本可用。")
		return {'ochiai': [], 'tarantula': [], 'dstar': [], 'jaccard': []}

	# 1) 逐样本预计算 FI（一次性缓存）
	print("🧮 预计算每个样本的 FI ...")
	all_sample_fi = {}  # {sample_idx: {layer_idx: {'shape':..., 'costs':...} 或 LSTM 的 list}}
	for sample_idx in tqdm(all_indices):
		all_sample_fi[sample_idx] = compute_FI_and_GL(
			X, y, [sample_idx], target_weights,
			is_multi_label=is_multi_label,
			path_to_keras_model=path_to_keras_model
		)

	# 2) 计算每个样本是否预测正确（成功/失败）
	print("🔎 判定样本 成功/失败 ...")
	model = tf.keras.models.load_model(path_to_keras_model, compile=False)
	X_batch = X[all_indices]
	preds = model.predict(X_batch, verbose=0)

	passed_samples = set()
	failed_samples = set()
	for k, sample_idx in enumerate(all_indices):
		pred = preds[k]
		if is_correct(pred, y[sample_idx], is_multi_label):
			passed_samples.add(sample_idx)
		else:
			failed_samples.add(sample_idx)

	print(f"  成功样本数: {len(passed_samples)} | 失败样本数: {len(failed_samples)}")

	# 3) 计算全局阈值：对所有层、所有样本的 |FI| 取第 fi_threshold_percentile 百分位
	print("📏 计算全局 |FI| 百分位阈值 ...")
	all_fi_abs = []
	for sample_idx in all_indices:
		fi_dict = all_sample_fi[sample_idx]
		for layer_idx, vs in target_weights.items():
			lname = vs[1]
			entry = fi_dict[layer_idx]
			if not model_util.is_LSTM(lname):
				# entry['costs'] shape: (num_weights, 2), 第1列是 FI
				all_fi_abs.extend(np.abs(entry['costs'][:, 1]).tolist())
			else:
				# LSTM: entry['costs'] 是 list，每个元素 shape: (num_weights_in_matrix, 2)
				for mat_cost in entry['costs']:
					all_fi_abs.extend(np.abs(mat_cost[:, 1]).tolist())

	all_fi_abs = np.array(all_fi_abs, dtype=float)
	if all_fi_abs.size == 0:
		print("⚠️ 未收集到 FI 值。")
		return {'ochiai': [], 'tarantula': [], 'dstar': [], 'jaccard': []}

	fi_threshold = np.percentile(all_fi_abs, fi_threshold_percentile)
	print(f"  全局阈值 = 第 {fi_threshold_percentile}% 百分位 = {fi_threshold:.8e}")
	print(f"  FI 分布: min={np.min(all_fi_abs):.3e}, max={np.max(all_fi_abs):.3e}, "
		f"p75={np.percentile(all_fi_abs,75):.3e}, p90={np.percentile(all_fi_abs,90):.3e}, "
		f"p98={np.percentile(all_fi_abs,98):.3e}, p99={np.percentile(all_fi_abs,99):.3e}")

	# 4) 逐层逐权重，统计 ef/ep/nf/np 并计算四个可疑度
	total_cands = {}  # layer_idx -> dict or lstm dict
	activation_report = []  # 打印各层激活占比（虽然是全局阈值）

	for layer_idx, vs in target_weights.items():
		t_w, lname = vs
		first_entry = all_sample_fi[all_indices[0]][layer_idx]

		if model_util.is_C2D(lname) or model_util.is_FC(lname):
			layer_shape = tuple(first_entry['shape'])
			num_weights = int(np.prod(layer_shape))

			ef = np.zeros(num_weights, dtype=float)
			ep = np.zeros(num_weights, dtype=float)
			nf = np.zeros(num_weights, dtype=float)
			npv = np.zeros(num_weights, dtype=float)

			# 遍历样本，累加计数
			for sample_idx in all_indices:
				costs = all_sample_fi[sample_idx][layer_idx]['costs']  # (num_weights, 2)
				fi_vec = np.abs(costs[:, 1])
				exec_mask = fi_vec > fi_threshold
				is_ok = sample_idx in passed_samples

				if is_ok:
					ep += exec_mask.astype(float)
					npv += (~exec_mask).astype(float)
				else:
					ef += exec_mask.astype(float)
					nf += (~exec_mask).astype(float)

			# 计算四个分数
			suspiciousness_ochiai = np.zeros(num_weights, dtype=float)
			suspiciousness_tarantula = np.zeros(num_weights, dtype=float)
			suspiciousness_dstar = np.zeros(num_weights, dtype=float)
			suspiciousness_jaccard = np.zeros(num_weights, dtype=float)

			# Ochiai
			denom_och = np.sqrt((ef + ep) * (ef + nf) + EPS)
			suspiciousness_ochiai = ef / denom_och

			# Tarantula
			ef_rate = ef / (ef + nf + EPS)
			ep_rate = ep / (ep + npv + EPS)
			suspiciousness_tarantula = ef_rate / (ef_rate + ep_rate + EPS)

			# DStar
			suspiciousness_dstar = (ef * ef) / (ep + nf + EPS)

			# Jaccard
			suspiciousness_jaccard = ef / (ef + ep + nf + EPS)

			total_cands[layer_idx] = dict(
				shape=layer_shape,
				costs_ochiai=suspiciousness_ochiai,
				costs_tarantula=suspiciousness_tarantula,
				costs_dstar=suspiciousness_dstar,
				costs_jaccard=suspiciousness_jaccard
			)

			act_ratio = float(np.sum((ef + ep) > 0)) / num_weights * 100.0
			activation_report.append((layer_idx, act_ratio))

		elif model_util.is_LSTM(lname):
			# LSTM: 多个矩阵
			layer_shapes = first_entry['shape']            # list of shapes
			num_mats = len(layer_shapes)

			total_cands[layer_idx] = dict(
				shape=[],
				costs_ochiai=[],
				costs_tarantula=[],
				costs_dstar=[],
				costs_jaccard=[]
			)

			for m in range(num_mats):
				matrix_shape = tuple(layer_shapes[m])
				num_weights = int(np.prod(matrix_shape))

				ef = np.zeros(num_weights, dtype=float)
				ep = np.zeros(num_weights, dtype=float)
				nf = np.zeros(num_weights, dtype=float)
				npv = np.zeros(num_weights, dtype=float)

				# 累加
				for sample_idx in all_indices:
					mat_costs = all_sample_fi[sample_idx][layer_idx]['costs'][m]  # (num_weights, 2)
					fi_vec = np.abs(mat_costs[:, 1])
					exec_mask = fi_vec > fi_threshold
					is_ok = sample_idx in passed_samples

					if is_ok:
						ep += exec_mask.astype(float)
						npv += (~exec_mask).astype(float)
					else:
						ef += exec_mask.astype(float)
						nf += (~exec_mask).astype(float)

				suspiciousness_ochiai = np.zeros(num_weights, dtype=float)
				suspiciousness_tarantula = np.zeros(num_weights, dtype=float)
				suspiciousness_dstar = np.zeros(num_weights, dtype=float)
				suspiciousness_jaccard = np.zeros(num_weights, dtype=float)

				denom_och = np.sqrt((ef + ep) * (ef + nf) + EPS)
				suspiciousness_ochiai = ef / denom_och

				ef_rate = ef / (ef + nf + EPS)
				ep_rate = ep / (ep + npv + EPS)
				suspiciousness_tarantula = ef_rate / (ef_rate + ep_rate + EPS)

				suspiciousness_dstar = (ef * ef) / (ep + nf + EPS)
				suspiciousness_jaccard = ef / (ef + ep + nf + EPS)

				total_cands[layer_idx]['shape'].append(matrix_shape)
				total_cands[layer_idx]['costs_ochiai'].append(suspiciousness_ochiai)
				total_cands[layer_idx]['costs_tarantula'].append(suspiciousness_tarantula)
				total_cands[layer_idx]['costs_dstar'].append(suspiciousness_dstar)
				total_cands[layer_idx]['costs_jaccard'].append(suspiciousness_jaccard)

				act_ratio = float(np.sum((ef + ep) > 0)) / num_weights * 100.0
				activation_report.append((f"{layer_idx}-mat{m}", act_ratio))

		else:
			print(f"{lname} not supported yet")
			raise AssertionError

	# 打印各层激活占比（基于全局阈值）
	for lid, ratio in activation_report:
		print(f"  层{lid}: 激活权重占比≈ {ratio:.2f}%")

	# 5) 展平为排序列表
	costs_and_keys_ochiai = []
	costs_and_keys_tarantula = []
	costs_and_keys_dstar = []
	costs_and_keys_jaccard = []

	for layer_idx, vs in total_cands.items():
		lname = target_weights[layer_idx][1]
		if not model_util.is_LSTM(lname):
			shape = vs['shape']
			size = int(np.prod(shape))
			for flat_i in range(size):
				pos = np.unravel_index(flat_i, shape)
				key = [layer_idx, pos]
				costs_and_keys_ochiai.append((key, float(vs['costs_ochiai'][flat_i])))
				costs_and_keys_tarantula.append((key, float(vs['costs_tarantula'][flat_i])))
				costs_and_keys_dstar.append((key, float(vs['costs_dstar'][flat_i])))
				costs_and_keys_jaccard.append((key, float(vs['costs_jaccard'][flat_i])))
		else:
			num_mats = len(vs['shape'])
			for m in range(num_mats):
				shape = vs['shape'][m]
				size = int(np.prod(shape))
				for flat_i in range(size):
					pos = np.unravel_index(flat_i, shape)
					key = [(layer_idx, m), pos]
					costs_and_keys_ochiai.append((key, float(vs['costs_ochiai'][m][flat_i])))
					costs_and_keys_tarantula.append((key, float(vs['costs_tarantula'][m][flat_i])))
					costs_and_keys_dstar.append((key, float(vs['costs_dstar'][m][flat_i])))
					costs_and_keys_jaccard.append((key, float(vs['costs_jaccard'][m][flat_i])))

	sorted_ochiai = sorted(costs_and_keys_ochiai, key=lambda x: x[1], reverse=True)
	sorted_tarantula = sorted(costs_and_keys_tarantula, key=lambda x: x[1], reverse=True)
	sorted_dstar = sorted(costs_and_keys_dstar, key=lambda x: x[1], reverse=True)
	sorted_jaccard = sorted(costs_and_keys_jaccard, key=lambda x: x[1], reverse=True)

	print(f"✅ 生成排名数：{len(sorted_ochiai)}")
	# 可选：估算平均激活率（多少权重至少在一个样本中“执行过”）
	# 这里用 Ochiai>0 近似
	act_rate = 100.0 * np.mean([sc > 0 for _, sc in sorted_ochiai])
	print(f"平均激活率 ≈ {act_rate:.2f}% （建议 1–3% 左右，过高/过低都可调 fi_threshold_percentile）")

	return {
		'ochiai': sorted_ochiai,
		'tarantula': sorted_tarantula,
		'dstar': sorted_dstar,
		'jaccard': sorted_jaccard
	}


def localise_by_fi_continuous_sbfl(
	X, y,
	indices_to_chgd, indices_to_unchgd,
	target_weights,
	path_to_keras_model=None,
	is_multi_label=True,
	fi_threshold_percentile=99  # 用于确定log归一化的上界
):
	"""
	连续版本的FI-SBFL故障定位：
	- 不使用阈值判断"执行"，而是使用连续的执行强度
	- 对FI值进行log归一化处理：log1p(fi) / log1p(threshold)
	- 其他逻辑与enhanced版本相同：样本权重、加权计数、防除零公式
	"""
	import numpy as np
	import tensorflow as tf
	from tqdm import tqdm

	EPS = 1e-12

	# 0) 汇总样本索引
	all_indices = list(indices_to_chgd) + list(indices_to_unchgd)
	if len(all_indices) == 0:
		print("⚠️ 没有样本可用。")
		return {'ochiai': [], 'tarantula': [], 'dstar': [], 'jaccard': []}

	# 1) 逐样本预计算 FI（一次性缓存）
	print("🧮 预计算每个样本的 FI ...")
	all_sample_fi = {}  # {sample_idx: {layer_idx: {'shape':..., 'costs':...} 或 LSTM 的 list}}
	for sample_idx in tqdm(all_indices):
		all_sample_fi[sample_idx] = compute_FI_and_GL(
			X, y, [sample_idx], target_weights,
			is_multi_label=is_multi_label,
			path_to_keras_model=path_to_keras_model
		)

	# 2) 计算每个样本是否预测正确（成功/失败）
	print("🔎 判定样本 成功/失败 ...")
	model = tf.keras.models.load_model(path_to_keras_model, compile=False)
	X_batch = X[all_indices]
	preds = model.predict(X_batch, verbose=0)

	passed_samples = set()
	failed_samples = set()
	confidences = []
	
	for k, sample_idx in enumerate(all_indices):
		pred = preds[k]
		true_label = y[sample_idx]
		
		# 计算置信度 - 修复索引问题
		if is_multi_label:
			# 打印调试信息（仅前几个样本）
			if k < 3:
				print(f"  DEBUG: sample {k}, pred.shape={pred.shape}, true_label.shape={true_label.shape if hasattr(true_label, 'shape') else 'scalar'}")
			
			if len(true_label.shape) == 1 and len(true_label) > 1 and len(pred.shape) > 0 and len(pred) >= len(true_label):
				positive_labels = np.where(true_label > 0.5)[0]
				confidence = np.mean(pred[positive_labels]) if len(positive_labels) > 0 else np.max(pred)
			else:
				confidence = np.max(pred)
		else:
			confidence = np.max(pred)
		confidences.append(confidence)
		
		# 判断预测正确性 - 修复形状匹配问题
		if is_multi_label:
			if len(true_label.shape) == 1 and len(true_label) > 1 and len(pred.shape) > 0 and len(pred) >= len(true_label):
				pred_binary = (pred > 0.5).astype(int)
				true_binary = (true_label > 0.5).astype(int)
				is_correct = np.array_equal(pred_binary, true_binary)
			else:
				# 单标签情况或形状不匹配时
				pred_label = np.argmax(pred)
				true_label_idx = int(true_label) if np.isscalar(true_label) else np.argmax(true_label)
				is_correct = (pred_label == true_label_idx)
		else:
			pred_label = np.argmax(pred)
			true_label_idx = int(true_label) if np.isscalar(true_label) else np.argmax(true_label)
			is_correct = (pred_label == true_label_idx)
		
		if is_correct:
			passed_samples.add(sample_idx)
		else:
			failed_samples.add(sample_idx)

	# 3) 置信度min-max归一化 -> 样本权重
	confidences = np.array(confidences)
	min_conf, max_conf = np.min(confidences), np.max(confidences)
	if max_conf > min_conf:
		normalized_weights = (confidences - min_conf) / (max_conf - min_conf)
	else:
		normalized_weights = np.ones_like(confidences)

	print(f"  分类统计: 成功{len(passed_samples)}个, 失败{len(failed_samples)}个")
	print(f"  置信度范围: [{min_conf:.4f}, {max_conf:.4f}]")

	# 4) 计算全局FI归一化阈值（用于log归一化的上界）
	print("🎯 计算全局FI归一化阈值...")
	all_fi_values = []
	for idx_to_tl, vs in target_weights.items():
		for sample_idx in all_indices:
			fi_costs = all_sample_fi[sample_idx][idx_to_tl]['costs']
			if not model_util.is_LSTM(vs[1]):
				all_fi_values.extend(np.abs(fi_costs[:, 1]))  # FI值在索引1，取绝对值
			else:
				for matrix_costs in fi_costs:
					all_fi_values.extend(np.abs(matrix_costs[:, 1]))
	
	fi_threshold = np.percentile(all_fi_values, fi_threshold_percentile)
	log_threshold = np.log1p(fi_threshold)
	print(f"  FI归一化阈值({fi_threshold_percentile}%): {fi_threshold:.8f}")
	print(f"  log1p(阈值): {log_threshold:.8f}")

	# 5) 计算连续执行强度的加权SBFL计数
	print("📊 计算连续执行强度的SBFL计数...")
	
	# 收集所有权重位置
	all_weight_keys = set()
	for idx_to_tl, vs in target_weights.items():
		if not model_util.is_LSTM(vs[1]):
			layer_shape = all_sample_fi[all_indices[0]][idx_to_tl]['shape']
			for flat_idx in range(np.prod(layer_shape)):
				weight_pos = np.unravel_index(flat_idx, layer_shape)
				all_weight_keys.add((idx_to_tl, weight_pos))
		else:
			# LSTM处理
			for matrix_idx, matrix_shape in enumerate(all_sample_fi[all_indices[0]][idx_to_tl]['shape']):
				for flat_idx in range(np.prod(matrix_shape)):
					weight_pos = np.unravel_index(flat_idx, matrix_shape)
					all_weight_keys.add(((idx_to_tl, matrix_idx), weight_pos))
	
	all_weight_keys = sorted(list(all_weight_keys))
	num_weights = len(all_weight_keys)
	weight_to_idx = {key: i for i, key in enumerate(all_weight_keys)}

	# 初始化连续SBFL计数
	ef = np.zeros(num_weights)  # Executed & Failed (连续强度)
	ep = np.zeros(num_weights)  # Executed & Passed (连续强度)
	nf = np.zeros(num_weights)  # Not Executed & Failed (连续强度)
	np_val = np.zeros(num_weights)  # Not Executed & Passed (连续强度)

	# 逐样本累加连续执行强度
	for sample_i, sample_idx in enumerate(all_indices):
		sample_weight = normalized_weights[sample_i]
		weighted_contribution = 1.0 + sample_weight  # (1 + W_i)
		is_correct = sample_idx in passed_samples

		# 遍历该样本的所有权重
		for idx_to_tl, vs in target_weights.items():
			if not model_util.is_LSTM(vs[1]):
				# 非LSTM层
				fi_costs = all_sample_fi[sample_idx][idx_to_tl]['costs']
				layer_shape = all_sample_fi[sample_idx][idx_to_tl]['shape']
				
				for flat_idx in range(len(fi_costs)):
					weight_pos = np.unravel_index(flat_idx, layer_shape)
					weight_key = (idx_to_tl, weight_pos)
					
					if weight_key in weight_to_idx:
						weight_idx = weight_to_idx[weight_key]
						fi_val = abs(fi_costs[flat_idx, 1])  # FI值取绝对值
						
						# 连续执行强度：log1p(fi) / log1p(threshold)
						# 如果fi_val很小，log1p(fi_val)接近0，执行强度接近0
						# 如果fi_val接近threshold，执行强度接近1
						log_fi = np.log1p(fi_val)
						execution_strength = min(log_fi / log_threshold, 1.0) if log_threshold > 0 else 0.0
						non_execution_strength = 1.0 - execution_strength
						
						# 累加连续SBFL计数
						if is_correct:
							# Passed
							ep[weight_idx] += execution_strength * weighted_contribution
							np_val[weight_idx] += non_execution_strength * weighted_contribution
						else:
							# Failed
							ef[weight_idx] += execution_strength * weighted_contribution
							nf[weight_idx] += non_execution_strength * weighted_contribution
			else:
				# LSTM层处理
				for matrix_idx, matrix_costs in enumerate(all_sample_fi[sample_idx][idx_to_tl]['costs']):
					matrix_shape = all_sample_fi[sample_idx][idx_to_tl]['shape'][matrix_idx]
					
					for flat_idx in range(len(matrix_costs)):
						weight_pos = np.unravel_index(flat_idx, matrix_shape)
						weight_key = ((idx_to_tl, matrix_idx), weight_pos)
						
						if weight_key in weight_to_idx:
							weight_idx = weight_to_idx[weight_key]
							fi_val = abs(matrix_costs[flat_idx, 1])
							
							log_fi = np.log1p(fi_val)
							execution_strength = min(log_fi / log_threshold, 1.0) if log_threshold > 0 else 0.0
							non_execution_strength = 1.0 - execution_strength
							
							if is_correct:
								ep[weight_idx] += execution_strength * weighted_contribution
								np_val[weight_idx] += non_execution_strength * weighted_contribution
							else:
								ef[weight_idx] += execution_strength * weighted_contribution
								nf[weight_idx] += non_execution_strength * weighted_contribution

	# 6) 计算SBFL可疑度分数（加权版本，带ε防除零）
	print("🎯 计算SBFL可疑度分数...")
	
	# Ochiai: ef / sqrt((ef+ep)(ef+nf)+ε)
	denom_ochiai = np.sqrt((ef + ep) * (ef + nf) + EPS)
	score_ochiai = ef / denom_ochiai
	
	# Tarantula: (ef/(ef+nf+ε)) / ((ef/(ef+nf+ε)) + (ep/(ep+np+ε)) + ε)
	ef_rate = ef / (ef + nf + EPS)
	ep_rate = ep / (ep + np_val + EPS)
	score_tarantula = ef_rate / (ef_rate + ep_rate + EPS)
	
	# DStar: ef^2 / (ep+nf+ε)
	score_dstar = (ef * ef) / (ep + nf + EPS)
	
	# Jaccard: ef / (ef + ep + nf + ε)
	score_jaccard = ef / (ef + ep + nf + EPS)

	# 7) 创建排序结果
	results = {}
	for formula_name, scores in [
		('ochiai', score_ochiai),
		('tarantula', score_tarantula),
		('dstar', score_dstar),
		('jaccard', score_jaccard),
	]:
		sorted_indices = np.argsort(scores)[::-1]
		sorted_results = [(all_weight_keys[i], float(scores[i])) for i in sorted_indices]
		results[formula_name] = sorted_results
		
		# 统计非零分数数量
		nonzero_count = int(np.sum(scores > EPS))
		print(f"  {formula_name.capitalize()}: {nonzero_count}/{len(scores)}个权重有非零分数")

	# 8) 统计信息
	total_samples = len(all_indices)
	avg_ef = np.mean(ef)
	avg_ep = np.mean(ep)
	avg_nf = np.mean(nf)
	avg_np = np.mean(np_val)
	
	print(f"  📊 连续版SBFL统计 (共{total_samples}个样本):")
	print(f"    平均ef={avg_ef:.2f}, ep={avg_ep:.2f}, nf={avg_nf:.2f}, np={avg_np:.2f}")
	print(f"    连续强度总和验证: ef+ep+nf+np = {np.mean(ef+ep+nf+np_val):.2f}")
	print(f"    执行强度分布: ef+ep = {np.mean(ef+ep):.2f}, nf+np = {np.mean(nf+np_val):.2f}")
	
	return results


def localise_by_FI_only_confidence(
	X, y,
	indices_to_chgd,
	indices_to_unchgd,
	target_weights,
	path_to_keras_model = None,
	is_multi_label = True):
	"""
	置信度加权的FI-only故障定位
	对每个样本计算 fi_score * (1 + confidence)，然后聚合
	"""
	from collections.abc import Iterable
	import tensorflow as tf
	import numpy as np
	
	total_cands = {}
	# set loss func (虽然FI不直接用loss，但compute_FI_and_GL需要)
	loss_func = model_util.get_loss_func(is_multi_label = is_multi_label)
	
	# 加载模型计算置信度
	model = tf.keras.models.load_model(path_to_keras_model, compile=False)
	
	## slice inputs
	for idx_to_tl, vs in target_weights.items():
		t_w, lname = vs
		print(f"targeting layer {idx_to_tl} ({lname})")
		
		if model_util.is_C2D(lname) or model_util.is_FC(lname): # either FC or C2D
			# 计算changed样本的置信度加权FI
			weighted_fi_chgd = np.zeros(t_w.size)
			print(f"Processing {len(indices_to_chgd)} changed samples...")
			
			for sample_idx in indices_to_chgd:
				# 计算单个样本的FI
				fi_and_gl_single = compute_FI_and_GL(
					X, y, [sample_idx], {idx_to_tl: vs},
					is_multi_label = is_multi_label,
					path_to_keras_model = path_to_keras_model)
				
				# 计算这个样本的置信度
				pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
				# 使用 margin (top1 - top2) 代替 max probability
				if pred.size >= 2:
					# 计算 margin: top1 - top2
					top2 = np.partition(pred, -2)[-2:]
					confidence = top2.max() - top2.min()
				else:
					# 如果只有一个输出，使用原来的最大概率
					# 使用 margin (top1 - top2) 代替 max probability
					if pred.size >= 2:
						# 计算 margin: top1 - top2
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						# 如果只有一个输出，使用原来的最大概率
						confidence = np.max(pred)
				
				# FI * (1 + confidence)
				fi_single = fi_and_gl_single[idx_to_tl]['costs'][:, 1]
				weighted_fi_chgd += fi_single * (1.0 + confidence)
			
			# 计算unchanged样本的置信度加权FI
			weighted_fi_unchgd = np.zeros(t_w.size)
			print(f"Processing {len(indices_to_unchgd)} unchanged samples...")
			
			for sample_idx in indices_to_unchgd:
				# 计算单个样本的FI
				fi_and_gl_single = compute_FI_and_GL(
					X, y, [sample_idx], {idx_to_tl: vs},
					is_multi_label = is_multi_label,
					path_to_keras_model = path_to_keras_model)
				
				# 计算这个样本的置信度
				pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
				# 使用 margin (top1 - top2) 代替 max probability
				if pred.size >= 2:
					# 计算 margin: top1 - top2
					top2 = np.partition(pred, -2)[-2:]
					confidence = top2.max() - top2.min()
				else:
					# 如果只有一个输出，使用原来的最大概率
					# 使用 margin (top1 - top2) 代替 max probability
					if pred.size >= 2:
						# 计算 margin: top1 - top2
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						# 如果只有一个输出，使用原来的最大概率
						confidence = np.max(pred)
				
				# FI * (1 + confidence)
				fi_single = fi_and_gl_single[idx_to_tl]['costs'][:, 1]
				weighted_fi_unchgd += fi_single * (1.0 + confidence)
			
			# 计算置信度加权的FI比值：weighted_FI_changed / (1 + weighted_FI_unchanged)
			final_scores = weighted_fi_chgd / (1.0 + weighted_fi_unchgd)
			
			total_cands[idx_to_tl] = {
				'shape': t_w.shape, 
				'costs': final_scores}
				
		elif model_util.is_LSTM(lname):
			# LSTM层的处理
			num_matrices = len(t_w)  # LSTM通常有多个权重矩阵
			total_cands[idx_to_tl] = {'shape': [], 'costs': []}
			
			for matrix_idx in range(num_matrices):
				matrix_shape = t_w[matrix_idx].shape
				matrix_size = t_w[matrix_idx].size
				
				# 计算changed样本的置信度加权FI
				weighted_fi_chgd = np.zeros(matrix_size)
				
				for sample_idx in indices_to_chgd:
					# 计算单个样本的FI
					fi_and_gl_single = compute_FI_and_GL(
						X, y, [sample_idx], {idx_to_tl: vs},
						is_multi_label = is_multi_label,
						path_to_keras_model = path_to_keras_model)
					
					# 计算这个样本的置信度
					pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
					# 使用 margin (top1 - top2) 代替 max probability
					if pred.size >= 2:
						# 计算 margin: top1 - top2
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						# 如果只有一个输出，使用原来的最大概率
						confidence = np.max(pred)
					
					# FI * (1 + confidence)
					fi_single = fi_and_gl_single[idx_to_tl]['costs'][matrix_idx][:, 1]
					weighted_fi_chgd += fi_single * (1.0 + confidence)
				
				# 计算unchanged样本的置信度加权FI
				weighted_fi_unchgd = np.zeros(matrix_size)
				
				for sample_idx in indices_to_unchgd:
					# 计算单个样本的FI
					fi_and_gl_single = compute_FI_and_GL(
						X, y, [sample_idx], {idx_to_tl: vs},
						is_multi_label = is_multi_label,
						path_to_keras_model = path_to_keras_model)
					
					# 计算这个样本的置信度
					pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
					# 使用 margin (top1 - top2) 代替 max probability
					if pred.size >= 2:
						# 计算 margin: top1 - top2
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						# 如果只有一个输出，使用原来的最大概率
						confidence = np.max(pred)
					
					# FI * (1 + confidence)
					fi_single = fi_and_gl_single[idx_to_tl]['costs'][matrix_idx][:, 1]
					weighted_fi_unchgd += fi_single * (1.0 + confidence)
				
				# 计算这个矩阵的最终分数
				final_scores = weighted_fi_chgd / (1.0 + weighted_fi_unchgd)
				
				total_cands[idx_to_tl]['shape'].append(matrix_shape)
				total_cands[idx_to_tl]['costs'].append(final_scores)
		else:
			print ("{} not supported yet".format(lname))
			assert False

	# 生成排序结果（完全类似GL算法的结构）
	indices_to_tl = list(total_cands.keys())
	costs_and_keys = []
	for idx_to_tl in indices_to_tl:
		if not model_util.is_LSTM(target_weights[idx_to_tl][1]):
			for local_i,c in enumerate(total_cands[idx_to_tl]['costs']):
				cost_and_key = ([idx_to_tl, np.unravel_index(local_i, total_cands[idx_to_tl]['shape'])], c) 
				costs_and_keys.append(cost_and_key)
		else:
			num = len(total_cands[idx_to_tl]['shape'])
			for idx_to_w in range(num):
				for local_i, c in enumerate(total_cands[idx_to_tl]['costs'][idx_to_w]):
					cost_and_key = (
						[(idx_to_tl, idx_to_w), 
						np.unravel_index(local_i, total_cands[idx_to_tl]['shape'][idx_to_w])], c) 
					costs_and_keys.append(cost_and_key)

	sorted_costs_and_keys = sorted(costs_and_keys, key = lambda vs:vs[1], reverse = True)
	print(f"✅ FI-only-confidence完成，共{len(sorted_costs_and_keys)}个权重")
	return sorted_costs_and_keys


def localise_by_fi_linear_sbfl(
	X, y,
	indices_to_chgd, indices_to_unchgd,
	target_weights,
	path_to_keras_model=None,
	is_multi_label=True,
	fi_threshold_percentile=99  # 用于确定线性缩放的上界
):
	"""
	线性缩放版本的FI-SBFL故障定位：
	- 不使用阈值判断"执行"，而是使用连续的执行强度
	- 对FI值进行线性缩放处理：fi / threshold
	- 其他逻辑与enhanced版本相同：样本权重、加权计数、防除零公式
	"""
	import numpy as np
	import tensorflow as tf
	from tqdm import tqdm

	EPS = 1e-12

	# 0) 汇总样本索引
	all_indices = list(indices_to_chgd) + list(indices_to_unchgd)
	if len(all_indices) == 0:
		print("⚠️ 没有样本可用。")
		return {'ochiai': [], 'tarantula': [], 'dstar': [], 'jaccard': []}

	# 1) 逐样本预计算 FI（一次性缓存）
	print("🧮 预计算每个样本的 FI ...")
	all_sample_fi = {}  # {sample_idx: {layer_idx: {'shape':..., 'costs':...} 或 LSTM 的 list}}
	for sample_idx in tqdm(all_indices):
		all_sample_fi[sample_idx] = compute_FI_and_GL(
			X, y, [sample_idx], target_weights,
			is_multi_label=is_multi_label,
			path_to_keras_model=path_to_keras_model
		)

	# 2) 计算每个样本是否预测正确（成功/失败）
	print("🔎 判定样本 成功/失败 ...")
	model = tf.keras.models.load_model(path_to_keras_model, compile=False)
	X_batch = X[all_indices]
	preds = model.predict(X_batch, verbose=0)

	passed_samples = set()
	failed_samples = set()
	confidences = []
	
	for k, sample_idx in enumerate(all_indices):
		pred = preds[k]
		true_label = y[sample_idx]
		
		# 计算置信度 - 修复索引问题
		if is_multi_label:
			# 打印调试信息（仅前几个样本）
			if k < 3:
				print(f"  DEBUG: sample {k}, pred.shape={pred.shape}, true_label.shape={true_label.shape if hasattr(true_label, 'shape') else 'scalar'}")
			
			if len(true_label.shape) == 1 and len(true_label) > 1 and len(pred.shape) > 0 and len(pred) >= len(true_label):
				positive_labels = np.where(true_label > 0.5)[0]
				confidence = np.mean(pred[positive_labels]) if len(positive_labels) > 0 else np.max(pred)
			else:
				confidence = np.max(pred)
		else:
			confidence = np.max(pred)
		confidences.append(confidence)
		
		# 判断预测正确性 - 修复形状匹配问题
		if is_multi_label:
			if len(true_label.shape) == 1 and len(true_label) > 1 and len(pred.shape) > 0 and len(pred) >= len(true_label):
				pred_binary = (pred > 0.5).astype(int)
				true_binary = (true_label > 0.5).astype(int)
				is_correct = np.array_equal(pred_binary, true_binary)
			else:
				# 单标签情况或形状不匹配时
				pred_label = np.argmax(pred)
				true_label_idx = int(true_label) if np.isscalar(true_label) else np.argmax(true_label)
				is_correct = (pred_label == true_label_idx)
		else:
			pred_label = np.argmax(pred)
			true_label_idx = int(true_label) if np.isscalar(true_label) else np.argmax(true_label)
			is_correct = (pred_label == true_label_idx)
		
		if is_correct:
			passed_samples.add(sample_idx)
		else:
			failed_samples.add(sample_idx)

	# 3) 置信度min-max归一化 -> 样本权重
	confidences = np.array(confidences)
	min_conf, max_conf = np.min(confidences), np.max(confidences)
	if max_conf > min_conf:
		normalized_weights = (confidences - min_conf) / (max_conf - min_conf)
	else:
		normalized_weights = np.ones_like(confidences)

	print(f"  分类统计: 成功{len(passed_samples)}个, 失败{len(failed_samples)}个")
	print(f"  置信度范围: [{min_conf:.4f}, {max_conf:.4f}]")

	# 4) 计算全局FI线性缩放阈值（用于线性缩放的上界）
	print("🎯 计算全局FI线性缩放阈值...")
	all_fi_values = []
	for idx_to_tl, vs in target_weights.items():
		for sample_idx in all_indices:
			fi_costs = all_sample_fi[sample_idx][idx_to_tl]['costs']
			if not model_util.is_LSTM(vs[1]):
				all_fi_values.extend(np.abs(fi_costs[:, 1]))  # FI值在索引1，取绝对值
			else:
				for matrix_costs in fi_costs:
					all_fi_values.extend(np.abs(matrix_costs[:, 1]))
	
	fi_threshold = np.percentile(all_fi_values, fi_threshold_percentile)
	print(f"  FI线性缩放阈值({fi_threshold_percentile}%): {fi_threshold:.8f}")

	# 5) 计算连续执行强度的加权SBFL计数
	print("📊 计算线性缩放连续执行强度的SBFL计数...")
	
	# 收集所有权重位置
	all_weight_keys = set()
	for idx_to_tl, vs in target_weights.items():
		if not model_util.is_LSTM(vs[1]):
			layer_shape = all_sample_fi[all_indices[0]][idx_to_tl]['shape']
			for flat_idx in range(np.prod(layer_shape)):
				weight_pos = np.unravel_index(flat_idx, layer_shape)
				all_weight_keys.add((idx_to_tl, weight_pos))
		else:
			# LSTM处理
			for matrix_idx, matrix_shape in enumerate(all_sample_fi[all_indices[0]][idx_to_tl]['shape']):
				for flat_idx in range(np.prod(matrix_shape)):
					weight_pos = np.unravel_index(flat_idx, matrix_shape)
					all_weight_keys.add(((idx_to_tl, matrix_idx), weight_pos))
	
	all_weight_keys = sorted(list(all_weight_keys))
	num_weights = len(all_weight_keys)
	weight_to_idx = {key: i for i, key in enumerate(all_weight_keys)}

	# 初始化连续SBFL计数
	ef = np.zeros(num_weights)  # Executed & Failed (连续强度)
	ep = np.zeros(num_weights)  # Executed & Passed (连续强度)
	nf = np.zeros(num_weights)  # Not Executed & Failed (连续强度)
	np_val = np.zeros(num_weights)  # Not Executed & Passed (连续强度)

	# 逐样本累加连续执行强度
	for sample_i, sample_idx in enumerate(all_indices):
		sample_weight = normalized_weights[sample_i]
		# weighted_contribution = 1.0 + sample_weight  # (1 + W_i)
		weighted_contribution = sample_weight  # (1 + W_i)
		is_correct = sample_idx in passed_samples

		# 遍历该样本的所有权重
		for idx_to_tl, vs in target_weights.items():
			if not model_util.is_LSTM(vs[1]):
				# 非LSTM层
				fi_costs = all_sample_fi[sample_idx][idx_to_tl]['costs']
				layer_shape = all_sample_fi[sample_idx][idx_to_tl]['shape']
				
				for flat_idx in range(len(fi_costs)):
					weight_pos = np.unravel_index(flat_idx, layer_shape)
					weight_key = (idx_to_tl, weight_pos)
					
					if weight_key in weight_to_idx:
						weight_idx = weight_to_idx[weight_key]
						fi_val = abs(fi_costs[flat_idx, 1])  # FI值取绝对值
						
						# 线性缩放执行强度：fi / threshold
						# 如果fi_val很小，执行强度接近0
						# 如果fi_val接近threshold，执行强度接近1
						execution_strength = min(fi_val / fi_threshold, 1.0) if fi_threshold > 0 else 0.0
						non_execution_strength = 1.0 - execution_strength
						
						# 累加连续SBFL计数
						if is_correct:
							# Passed
							ep[weight_idx] += execution_strength * weighted_contribution
							# np_val[weight_idx] += non_execution_strength * weighted_contribution
						else:
							# Failed
							ef[weight_idx] += execution_strength * weighted_contribution
							# nf[weight_idx] += non_execution_strength * weighted_contribution
			else:
				# LSTM层处理
				for matrix_idx, matrix_costs in enumerate(all_sample_fi[sample_idx][idx_to_tl]['costs']):
					matrix_shape = all_sample_fi[sample_idx][idx_to_tl]['shape'][matrix_idx]
					
					for flat_idx in range(len(matrix_costs)):
						weight_pos = np.unravel_index(flat_idx, matrix_shape)
						weight_key = ((idx_to_tl, matrix_idx), weight_pos)
						
						if weight_key in weight_to_idx:
							weight_idx = weight_to_idx[weight_key]
							fi_val = abs(matrix_costs[flat_idx, 1])
							
							execution_strength = min(fi_val / fi_threshold, 1.0) if fi_threshold > 0 else 0.0
							non_execution_strength = 1.0 - execution_strength
							
							if is_correct:
								ep[weight_idx] += execution_strength * weighted_contribution
								# np_val[weight_idx] += non_execution_strength * weighted_contribution
							else:
								ef[weight_idx] += execution_strength * weighted_contribution
								# nf[weight_idx] += non_execution_strength * weighted_contribution

	# 6) 计算SBFL可疑度分数（加权版本，带ε防除零）
	print("🎯 计算SBFL可疑度分数...")
	
	# Ochiai: ef / sqrt((ef+ep)(ef+nf)+ε)
	denom_ochiai = np.sqrt((ef + ep) * (ef + nf) + EPS)
	score_ochiai = ef / denom_ochiai
	
	# Tarantula: (ef/(ef+nf+ε)) / ((ef/(ef+nf+ε)) + (ep/(ep+np+ε)) + ε)
	ef_rate = ef / (ef + nf + EPS)
	ep_rate = ep / (ep + np_val + EPS)
	score_tarantula = ef_rate / (ef_rate + ep_rate + EPS)
	
	# DStar: ef^2 / (ep+nf+ε)
	score_dstar = (ef * ef) / (ep + nf + EPS)
	
	# Jaccard: ef / (ef + ep + nf + ε)
	score_jaccard = ef / (ef + ep + nf + EPS)

	# 7) 创建排序结果
	results = {}
	for formula_name, scores in [
		('ochiai', score_ochiai),
		('tarantula', score_tarantula),
		('dstar', score_dstar),
		('jaccard', score_jaccard),
	]:
		sorted_indices = np.argsort(scores)[::-1]
		sorted_results = [(all_weight_keys[i], float(scores[i])) for i in sorted_indices]
		results[formula_name] = sorted_results
		
		# 统计非零分数数量
		nonzero_count = int(np.sum(scores > EPS))
		print(f"  {formula_name.capitalize()}: {nonzero_count}/{len(scores)}个权重有非零分数")

	# 8) 统计信息
	total_samples = len(all_indices)
	avg_ef = np.mean(ef)
	avg_ep = np.mean(ep)
	avg_nf = np.mean(nf)
	avg_np = np.mean(np_val)
	
	print(f"  📊 线性缩放版SBFL统计 (共{total_samples}个样本):")
	print(f"    平均ef={avg_ef:.2f}, ep={avg_ep:.2f}, nf={avg_nf:.2f}, np={avg_np:.2f}")
	print(f"    连续强度总和验证: ef+ep+nf+np = {np.mean(ef+ep+nf+np_val):.2f}")
	print(f"    执行强度分布: ef+ep = {np.mean(ef+ep):.2f}, nf+np = {np.mean(nf+np_val):.2f}")
	
	return results


def localise_by_FI_only_confidence(
	X, y,
	indices_to_chgd,
	indices_to_unchgd,
	target_weights,
	path_to_keras_model = None,
	is_multi_label = True):
	"""
	基于置信度加权的FI-only故障定位（简化版本）
	使用最直接的方法：FI * (1 + confidence)
	其中confidence使用margin方式计算（top1 - top2）
	"""
	from collections.abc import Iterable
	import tensorflow as tf
	import numpy as np
	
	total_cands = {}
	# set loss func (虽然FI不直接用loss，但compute_FI_and_GL需要)
	loss_func = model_util.get_loss_func(is_multi_label = is_multi_label)
	
	# 在循环外加载模型，避免重复加载
	model = tf.keras.models.load_model(path_to_keras_model, compile=False)
	
	## slice inputs
	for idx_to_tl, vs in target_weights.items():
		t_w, lname = vs
		print(f"targeting layer {idx_to_tl} ({lname})")
		
		if model_util.is_C2D(lname) or model_util.is_FC(lname): # either FC or C2D
			
			# 累加所有样本的加权FI值
			total_weighted_fi = np.zeros(t_w.size)
			
			print(f"Processing {len(indices_to_chgd)} changed samples...")
			for sample_idx in indices_to_chgd:
				fi_and_gl_single = compute_FI_and_GL(
					X, y, [sample_idx], {idx_to_tl: vs},
					is_multi_label = is_multi_label,
					path_to_keras_model = path_to_keras_model)
				
				# 计算这个样本的置信度 - 使用 margin (top1 - top2)
				pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
				if pred.size >= 2:
					# 计算 margin: top1 - top2
					top2 = np.partition(pred, -2)[-2:]
					confidence = top2.max() - top2.min()
				else:
					# 如果只有一个输出，使用原来的最大概率
					confidence = np.max(pred)
				
				# FI * (1 + confidence)
				fi_single = fi_and_gl_single[idx_to_tl]['costs'][:, 1]
				weighted_fi = fi_single * (1.0 + confidence)
				total_weighted_fi += weighted_fi
			
			print(f"Processing {len(indices_to_unchgd)} unchanged samples...")
			for sample_idx in indices_to_unchgd:
				fi_and_gl_single = compute_FI_and_GL(
					X, y, [sample_idx], {idx_to_tl: vs},
					is_multi_label = is_multi_label,
					path_to_keras_model = path_to_keras_model)
				
				# 计算这个样本的置信度 - 使用 margin (top1 - top2)
				pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
				if pred.size >= 2:
					# 计算 margin: top1 - top2
					top2 = np.partition(pred, -2)[-2:]
					confidence = top2.max() - top2.min()
				else:
					# 如果只有一个输出，使用原来的最大概率
					confidence = np.max(pred)
				
				# FI * (1 + confidence)
				fi_single = fi_and_gl_single[idx_to_tl]['costs'][:, 1]
				weighted_fi = fi_single * (1.0 + confidence)
				total_weighted_fi += weighted_fi
			
			print(f"  加权FI分数范围: [{np.min(total_weighted_fi):.4f}, {np.max(total_weighted_fi):.4f}]")
			
			total_cands[idx_to_tl] = {
				'shape': t_w.shape, 
				'costs': total_weighted_fi}
				
		elif model_util.is_LSTM(lname):
			# LSTM层的处理
			num_matrices = len(t_w)  # LSTM通常有多个权重矩阵
			total_cands[idx_to_tl] = {'shape': [], 'costs': []}
			
			for matrix_idx in range(num_matrices):
				matrix_shape = t_w[matrix_idx].shape
				matrix_size = t_w[matrix_idx].size
				
				# 累加所有样本的加权FI值
				total_weighted_fi = np.zeros(matrix_size)
				
				for sample_idx in indices_to_chgd:
					fi_and_gl_single = compute_FI_and_GL(
						X, y, [sample_idx], {idx_to_tl: vs},
						is_multi_label = is_multi_label,
						path_to_keras_model = path_to_keras_model)
					
					# 计算这个样本的置信度 - 使用 margin (top1 - top2)
					pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
					if pred.size >= 2:
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						confidence = np.max(pred)
					
					# FI * (1 + confidence)
					fi_single = fi_and_gl_single[idx_to_tl]['costs'][matrix_idx][:, 1]
					weighted_fi = fi_single * (1.0 + confidence)
					total_weighted_fi += weighted_fi
				
				for sample_idx in indices_to_unchgd:
					fi_and_gl_single = compute_FI_and_GL(
						X, y, [sample_idx], {idx_to_tl: vs},
						is_multi_label = is_multi_label,
						path_to_keras_model = path_to_keras_model)
					
					# 计算这个样本的置信度 - 使用 margin (top1 - top2)
					pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
					if pred.size >= 2:
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						confidence = np.max(pred)
					
					# FI * (1 + confidence)
					fi_single = fi_and_gl_single[idx_to_tl]['costs'][matrix_idx][:, 1]
					weighted_fi = fi_single * (1.0 + confidence)
					total_weighted_fi += weighted_fi
				
				total_cands[idx_to_tl]['shape'].append(matrix_shape)
				total_cands[idx_to_tl]['costs'].append(total_weighted_fi)
		else:
			print ("{} not supported yet".format(lname))
			assert False

	# 生成排序结果（完全类似GL算法的结构）
	indices_to_tl = list(total_cands.keys())
	costs_and_keys = []
	for idx_to_tl in indices_to_tl:
		if not model_util.is_LSTM(target_weights[idx_to_tl][1]):
			for local_i,c in enumerate(total_cands[idx_to_tl]['costs']):
				cost_and_key = ([idx_to_tl, np.unravel_index(local_i, total_cands[idx_to_tl]['shape'])], c) 
				costs_and_keys.append(cost_and_key)
		else:
			num = len(total_cands[idx_to_tl]['shape'])
			for idx_to_w in range(num):
				for local_i, c in enumerate(total_cands[idx_to_tl]['costs'][idx_to_w]):
					cost_and_key = (
						[(idx_to_tl, idx_to_w), 
						np.unravel_index(local_i, total_cands[idx_to_tl]['shape'][idx_to_w])], c) 
					costs_and_keys.append(cost_and_key)

	sorted_costs_and_keys = sorted(costs_and_keys, key = lambda vs:vs[1], reverse = True)
	print(f"✅ FI-only-confidence (简化版)完成，共{len(sorted_costs_and_keys)}个权重")
	return sorted_costs_and_keys





def localise_by_continuous_sbfl_success_failure_balanced(
	X, y,
	indices_to_chgd,
	indices_to_unchgd,
	target_weights,
	path_to_keras_model = None,
	is_multi_label = True,
	badness_method = "one_minus_confidence"):
	"""
	连续版本的SBFL故障定位
	
	基本思路：
	1. 参与度 p_w(x)：使用FI值作为权重w在样本x上的参与度
	2. 坏程度 g(x)：连续化的"失败程度"
	3. 可疑度公式：F_w = Σ[p_w(x) * g(x)] / (Σ[p_w(x)] + ε)
	
	新逻辑：
	- 不再区分changed/unchanged，而是区分成功/失败样本
	- 根据模型预测正确性分类样本
	- 确保成功和失败样本数量相等（平衡采样）
	
	Args:
		badness_method: 坏程度计算方法
			- "one_minus_confidence": g(x) = 1 - confidence(x)
			- "loss_based": g(x) = loss(x) 
			- "neg_margin": g(x) = 1 - margin(x)
	"""
	from collections.abc import Iterable
	import tensorflow as tf
	import numpy as np
	
	print(f"🚀 连续SBFL故障定位 (坏程度方法: {badness_method})")
	print("   新逻辑: 基于预测成功/失败分类，样本数量平衡")
	
	total_cands = {}
	# set loss func
	loss_func = model_util.get_loss_func(is_multi_label = is_multi_label)
	
	# 在循环外加载模型，避免重复加载
	model = tf.keras.models.load_model(path_to_keras_model, compile=False)
	
	# 合并所有样本并根据预测正确性分类
	all_indices = list(indices_to_chgd) + list(indices_to_unchgd)
	print(f"原始样本总数: {len(all_indices)} (changed: {len(indices_to_chgd)}, unchanged: {len(indices_to_unchgd)})")
	
	# 第一步：分类所有样本为成功/失败
	success_indices = []
	failure_indices = []
	
	print("🔍 分析样本预测正确性...")
	for sample_idx in all_indices:
		# 动态预测 - 按照FI only的模式
		pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
		true_label = y[sample_idx]
		
		# 使用统一的is_correct函数判断预测是否正确
		is_prediction_correct = is_correct(pred, true_label, is_multi_label)
		
		if is_prediction_correct:
			success_indices.append(sample_idx)
		else:
			failure_indices.append(sample_idx)
	
	print(f"  成功样本: {len(success_indices)}")
	print(f"  失败样本: {len(failure_indices)}")
	
	# 第二步：平衡采样，确保成功和失败样本数量相等
	min_count = min(len(success_indices), len(failure_indices))
	if min_count == 0:
		print("❌ 没有足够的成功或失败样本进行平衡采样")
		return []
	
	# 随机采样相等数量
	np.random.seed(42)  # 确保可重现
	balanced_success = np.random.choice(success_indices, min_count, replace=False)
	balanced_failure = np.random.choice(failure_indices, min_count, replace=False)
	
	print(f"✅ 平衡采样后: 成功 {len(balanced_success)}, 失败 {len(balanced_failure)}")
	
	# 使用平衡后的样本
	all_balanced_indices = list(balanced_success) + list(balanced_failure)
	
	## slice inputs
	for idx_to_tl, vs in target_weights.items():
		t_w, lname = vs
		print(f"targeting layer {idx_to_tl} ({lname})")
		
		if model_util.is_C2D(lname) or model_util.is_FC(lname): # either FC or C2D
			
			# 初始化累积器：分子(参与度*坏程度)和分母(参与度)
			numerator = np.zeros(t_w.size)    # Σ[p_w(x) * g(x)]
			denominator = np.zeros(t_w.size)  # Σ[p_w(x)]
			
			print(f"Processing {len(all_balanced_indices)} balanced samples...")
			for sample_idx in all_balanced_indices:
				# 计算FI值作为参与度
				fi_and_gl_single = compute_FI_and_GL(
					X, y, [sample_idx], {idx_to_tl: vs},
					is_multi_label = is_multi_label,
					path_to_keras_model = path_to_keras_model)
				
				# 获取FI值作为参与度 p_w(x)
				fi_single = fi_and_gl_single[idx_to_tl]['costs'][:, 1]  # 取FI列
				participation = fi_single  # p_w(x)
				
				# 动态预测并计算坏程度 g(x) - 按照FI only模式
				pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
				true_label = y[sample_idx]
				
				# 使用统一的is_correct函数判断样本是成功还是失败
				is_success = is_correct(pred, true_label, is_multi_label)
				
				# 根据成功/失败状态和坏程度方法计算g(x)
				if badness_method == "one_minus_confidence":
					# 计算置信度 - 使用与FI only相同的方法
					if pred.size >= 2:
						# 使用 margin (top1 - top2) 作为置信度
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						# 如果只有一个输出，使用最大概率
						confidence = np.max(pred)
					
					if is_success:
						badness = confidence  # 成功样本的"好程度"
					else:
						badness = 1.0 - confidence  # 失败样本的"坏程度"
						
				elif badness_method == "loss_based":
					# 直接使用损失值作为坏程度
					true_label_batch = y[sample_idx:sample_idx+1]
					pred_batch = pred.reshape(1, -1)
					loss_value = float(loss_func(true_label_batch, pred_batch).numpy())
					badness = loss_value
					
				elif badness_method == "neg_margin":
					# 基于margin计算坏程度 - 与FI only的置信度计算一致
					if pred.size >= 2:
						top2 = np.partition(pred, -2)[-2:]
						margin = top2.max() - top2.min()
					else:
						margin = np.max(pred)
					
					if is_success:
						badness = margin  # 成功样本的margin作为"好程度"
					else:
						badness = 1.0 - margin  # 失败样本的反margin作为"坏程度"
				else:
					raise ValueError(f"Unknown badness_method: {badness_method}")
				
				# 累积到分子和分母
				numerator += participation * badness      # Σ[p_w(x) * g(x)]
				denominator += participation              # Σ[p_w(x)]
			
			# 计算最终的可疑度分数 F_w
			eps = 1e-6
			suspiciousness = numerator / (denominator + eps)
			
			# 存储结果
			total_cands[idx_to_tl] = {
				'costs': np.column_stack([np.zeros_like(suspiciousness), suspiciousness]),  # [GL, FI]格式
				'shape': t_w.shape
			}
			
			print(f"Layer {idx_to_tl}: 计算了{len(suspiciousness)}个权重的可疑度")
			print(f"  可疑度范围: [{np.min(suspiciousness):.6f}, {np.max(suspiciousness):.6f}]")
			print(f"  非零可疑度: {np.sum(suspiciousness > 0)}/{len(suspiciousness)}")
			
		elif model_util.is_LSTM(lname): # LSTM layer
			print(f"Processing LSTM layer with {len(t_w)} matrices...")
			
			lstm_results = []
			# LSTM有多个权重矩阵
			for idx_to_w in range(len(t_w)):
				matrix_shape = t_w[idx_to_w].shape
				matrix_size = t_w[idx_to_w].size
				
				# 初始化累积器
				numerator = np.zeros(matrix_size)
				denominator = np.zeros(matrix_size)
				
				# 对平衡后的样本进行处理
				for sample_idx in all_balanced_indices:
					# 计算单个样本的FI
					fi_and_gl_single = compute_FI_and_GL(
						X, y, [sample_idx], {idx_to_tl: vs},
						is_multi_label = is_multi_label,
						path_to_keras_model = path_to_keras_model)
					
					# 获取LSTM矩阵的FI值
					fi_matrix = fi_and_gl_single[idx_to_tl]['costs'][idx_to_w][:, 1]  # 取FI列
					participation = fi_matrix  # p_w(x)
					
					# 动态预测并计算坏程度 g(x)
					pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
					true_label = y[sample_idx]
					is_success = is_correct(pred, true_label, is_multi_label)
					
					# 计算坏程度
					if badness_method == "one_minus_confidence":
						if pred.size >= 2:
							top2 = np.partition(pred, -2)[-2:]
							confidence = top2.max() - top2.min()
						else:
							confidence = np.max(pred)
						badness = confidence if is_success else (1.0 - confidence)
					elif badness_method == "loss_based":
						true_label_batch = y[sample_idx:sample_idx+1]
						pred_batch = pred.reshape(1, -1)
						loss_value = float(loss_func(true_label_batch, pred_batch).numpy())
						badness = loss_value
					elif badness_method == "neg_margin":
						if pred.size >= 2:
							top2 = np.partition(pred, -2)[-2:]
							margin = top2.max() - top2.min()
						else:
							margin = np.max(pred)
						badness = margin if is_success else (1.0 - margin)
					
					# 累积
					numerator += participation * badness
					denominator += participation
				
				# 计算可疑度
				eps = 1e-6
				suspiciousness = numerator / (denominator + eps)
				lstm_results.append(np.column_stack([np.zeros_like(suspiciousness), suspiciousness]))
			
			total_cands[idx_to_tl] = {
				'costs': lstm_results,
				'shape': [matrix.shape for matrix in t_w]
			}
			
			print(f"LSTM Layer {idx_to_tl}: 处理了{len(t_w)}个矩阵")
		
		else:
			print(f"⚠️ 不支持的层类型: {lname}")
			continue
	
	# 整理结果并排序
	indices_to_tl = list(total_cands.keys())
	costs_and_keys = []
	
	for idx_to_tl in indices_to_tl:
		if not model_util.is_LSTM(target_weights[idx_to_tl][1]):
			# 普通层：直接处理costs数组
			for local_i, c in enumerate(total_cands[idx_to_tl]['costs']):
				cost_and_key = ([idx_to_tl, np.unravel_index(local_i, total_cands[idx_to_tl]['shape'])], c[1])  # 取FI值
				costs_and_keys.append(cost_and_key)
		else:
			# LSTM层：处理多个矩阵
			num_matrices = len(total_cands[idx_to_tl]['shape'])
			for idx_to_w in range(num_matrices):
				for local_i, c in enumerate(total_cands[idx_to_tl]['costs'][idx_to_w]):
					cost_and_key = (
						[(idx_to_tl, idx_to_w), 
						np.unravel_index(local_i, total_cands[idx_to_tl]['shape'][idx_to_w])], c[1])  # 取FI值
					costs_and_keys.append(cost_and_key)
	
	# 按可疑度降序排序
	sorted_costs_and_keys = sorted(costs_and_keys, key=lambda vs: vs[1], reverse=True)
	
	print(f"✅ 连续SBFL故障定位完成，共{len(sorted_costs_and_keys)}个权重")
	print(f"   坏程度方法: {badness_method}")
	print(f"   最高可疑度: {sorted_costs_and_keys[0][1]:.6f}")
	print(f"   最低可疑度: {sorted_costs_and_keys[-1][1]:.6f}")
	
	return sorted_costs_and_keys


def localise_by_FI_only_confidence_true(
	X, y,
	indices_to_chgd,
	indices_to_unchgd,
	target_weights,
	path_to_keras_model = None,
	is_multi_label = True):
	"""
	真正的FI-only confidence方法
	严格按照changed vs unchanged的逻辑：计算changed和unchanged样本FI的加权比值
	使用 changed_weighted_FI / unchanged_weighted_FI 的方式
	"""
	from collections.abc import Iterable
	import tensorflow as tf
	import numpy as np
	
	total_cands = {}
	# set loss func
	loss_func = model_util.get_loss_func(is_multi_label = is_multi_label)
	
	# 在循环外加载模型，避免重复加载
	model = tf.keras.models.load_model(path_to_keras_model, compile=False)
	
	## slice inputs
	for idx_to_tl, vs in target_weights.items():
		t_w, lname = vs
		print(f"targeting layer {idx_to_tl} ({lname})")
		
		if model_util.is_C2D(lname) or model_util.is_FC(lname): # either FC or C2D
			
			# 分别计算changed和unchanged样本的加权FI值
			changed_weighted_fi = np.zeros(t_w.size)
			unchanged_weighted_fi = np.zeros(t_w.size)
			
			print(f"Processing {len(indices_to_chgd)} changed samples...")
			for sample_idx in indices_to_chgd:
				fi_and_gl_single = compute_FI_and_GL(
					X, y, [sample_idx], {idx_to_tl: vs},
					is_multi_label = is_multi_label,
					path_to_keras_model = path_to_keras_model)
				
				# 计算这个样本的置信度 - 使用 margin (top1 - top2)
				pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
				if pred.size >= 2:
					# 计算 margin: top1 - top2
					top2 = np.partition(pred, -2)[-2:]
					confidence = top2.max() - top2.min()
				else:
					# 如果只有一个输出，使用原来的最大概率
					confidence = np.max(pred)
				
				# FI * (1 + confidence) - changed样本
				fi_single = fi_and_gl_single[idx_to_tl]['costs'][:, 1]
				weighted_fi = fi_single * (1.0 + confidence)
				changed_weighted_fi += weighted_fi
			
			print(f"Processing {len(indices_to_unchgd)} unchanged samples...")
			for sample_idx in indices_to_unchgd:
				fi_and_gl_single = compute_FI_and_GL(
					X, y, [sample_idx], {idx_to_tl: vs},
					is_multi_label = is_multi_label,
					path_to_keras_model = path_to_keras_model)
				
				# 计算这个样本的置信度 - 使用 margin (top1 - top2)
				pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
				if pred.size >= 2:
					# 计算 margin: top1 - top2
					top2 = np.partition(pred, -2)[-2:]
					confidence = top2.max() - top2.min()
				else:
					# 如果只有一个输出，使用原来的最大概率
					confidence = np.max(pred)
				
				# FI * (1 + confidence) - unchanged样本
				fi_single = fi_and_gl_single[idx_to_tl]['costs'][:, 1]
				weighted_fi = fi_single * (1.0 + confidence)
				unchanged_weighted_fi += weighted_fi
			
			# 真正的FI-only逻辑：changed / unchanged (加上小的eps避免除零)
			eps = 1e-8
			total_weighted_fi = changed_weighted_fi / (unchanged_weighted_fi + eps)
			print(f"  FI比值分数范围: [{np.min(total_weighted_fi):.4f}, {np.max(total_weighted_fi):.4f}]")
			print(f"  Changed FI sum: {np.sum(changed_weighted_fi):.2f}, Unchanged FI sum: {np.sum(unchanged_weighted_fi):.2f}")
			print(f"  比值>1的权重数: {np.sum(total_weighted_fi > 1.0)}/{len(total_weighted_fi)}")
			
			total_cands[idx_to_tl] = {
				'shape': t_w.shape, 
				'costs': total_weighted_fi}
				
		elif model_util.is_LSTM(lname):
			# LSTM层的处理
			num_matrices = len(t_w)  # LSTM通常有多个权重矩阵
			total_cands[idx_to_tl] = {'shape': [], 'costs': []}
			
			for matrix_idx in range(num_matrices):
				matrix_shape = t_w[matrix_idx].shape
				matrix_size = t_w[matrix_idx].size
				
				# 分别计算changed和unchanged样本的加权FI值
				changed_weighted_fi = np.zeros(matrix_size)
				unchanged_weighted_fi = np.zeros(matrix_size)
				
				for sample_idx in indices_to_chgd:
					fi_and_gl_single = compute_FI_and_GL(
						X, y, [sample_idx], {idx_to_tl: vs},
						is_multi_label = is_multi_label,
						path_to_keras_model = path_to_keras_model)
					
					# 计算这个样本的置信度 - 使用 margin (top1 - top2)
					pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
					if pred.size >= 2:
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						confidence = np.max(pred)
					
					# FI * (1 + confidence) - changed样本
					fi_single = fi_and_gl_single[idx_to_tl]['costs'][matrix_idx][:, 1]
					weighted_fi = fi_single * (1.0 + confidence)
					changed_weighted_fi += weighted_fi
				
				for sample_idx in indices_to_unchgd:
					fi_and_gl_single = compute_FI_and_GL(
						X, y, [sample_idx], {idx_to_tl: vs},
						is_multi_label = is_multi_label,
						path_to_keras_model = path_to_keras_model)
					
					# 计算这个样本的置信度 - 使用 margin (top1 - top2)
					pred = model.predict(X[sample_idx:sample_idx+1], verbose=0)[0]
					if pred.size >= 2:
						top2 = np.partition(pred, -2)[-2:]
						confidence = top2.max() - top2.min()
					else:
						confidence = np.max(pred)
					
					# FI * (1 + confidence) - unchanged样本
					fi_single = fi_and_gl_single[idx_to_tl]['costs'][matrix_idx][:, 1]
					weighted_fi = fi_single * (1.0 + confidence)
					unchanged_weighted_fi += weighted_fi
				
				# 真正的FI-only逻辑：changed / unchanged (加上小的eps避免除零)
				eps = 1e-8
				total_weighted_fi = changed_weighted_fi / (unchanged_weighted_fi + eps)
				total_cands[idx_to_tl]['shape'].append(matrix_shape)
				total_cands[idx_to_tl]['costs'].append(total_weighted_fi)
		else:
			print ("{} not supported yet".format(lname))
			assert False

	# 生成排序结果
	indices_to_tl = list(total_cands.keys())
	costs_and_keys = []
	for idx_to_tl in indices_to_tl:
		if not model_util.is_LSTM(target_weights[idx_to_tl][1]):
			for local_i,c in enumerate(total_cands[idx_to_tl]['costs']):
				cost_and_key = ([idx_to_tl, np.unravel_index(local_i, total_cands[idx_to_tl]['shape'])], c) 
				costs_and_keys.append(cost_and_key)
		else:
			num = len(total_cands[idx_to_tl]['shape'])
			for idx_to_w in range(num):
				for local_i, c in enumerate(total_cands[idx_to_tl]['costs'][idx_to_w]):
					cost_and_key = (
						[(idx_to_tl, idx_to_w), 
						np.unravel_index(local_i, total_cands[idx_to_tl]['shape'][idx_to_w])], c) 
					costs_and_keys.append(cost_and_key)

	sorted_costs_and_keys = sorted(costs_and_keys, key = lambda vs:vs[1], reverse = True)
	print(f"✅ FI-only-confidence (真正版本)完成，共{len(sorted_costs_and_keys)}个权重")
	return sorted_costs_and_keys



