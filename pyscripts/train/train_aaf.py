from __future__ import print_function

import argparse
import math
import os
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import network.aaf.losses as lossx
import utils.general
from seg_models.image_reader import ImageReader
from seg_models.models.pspnet import pspnet_resnet101 as model

IMG_MEAN = np.array((122.675, 116.669, 104.008), dtype=np.float32)


def get_arguments():
	"""Parse all the arguments provided from the CLI.
	
	Returns:
	  A list of parsed arguments.
	"""
	parser = argparse.ArgumentParser(description='Semantic Segmentation')
	# Data parameters
	parser.add_argument('--batch-size', type=int, default=1,
						help='Number of images in one step.')
	parser.add_argument('--data-dir', type=str, default='',
						help='/path/to/dataset/.')
	parser.add_argument('--data-list', type=str, default='',
						help='/path/to/datalist/file.')
	parser.add_argument('--ignore-label', type=int, default=255,
						help='The index of the label to ignore.')
	parser.add_argument('--input-size', type=str, default='336,336',
						help='Comma-separated string with H and W of image.')
	parser.add_argument('--random-seed', type=int, default=1234,
						help='Random seed to have reproducible results.')
	# Training paramters
	parser.add_argument('--is-training', action='store_true',
						help='Whether to updates weights.')
	parser.add_argument('--use-global-status', action='store_true',
						help='Whether to updates moving mean and variance.')
	parser.add_argument('--learning-rate', type=float, default=2.5e-4,
						help='Base learning rate.')
	parser.add_argument('--power', type=float, default=0.9,
						help='Decay for poly learing rate policy.')
	parser.add_argument('--momentum', type=float, default=0.9,
						help='Momentum component of the optimiser.')
	parser.add_argument('--weight-decay', type=float, default=5e-4,
						help='Regularisation parameter for L2-loss.')
	parser.add_argument('--num-classes', type=int, default=21,
						help='Number of classes to predict.')
	parser.add_argument('--num-steps', type=int, default=20000,
						help='Number of training steps.')
	parser.add_argument('--iter-size', type=int, default=10,
						help='Number of iteration to update weights')
	parser.add_argument('--random-mirror', action='store_true',
						help='Whether to randomly mirror the inputs.')
	parser.add_argument('--random-crop', action='store_true',
						help='Whether to randomly crop the inputs.')
	parser.add_argument('--random-scale', action='store_true',
						help='Whether to randomly scale the inputs.')
	parser.add_argument('--kld-margin', type=float,
						help='margin for affinity loss')
	parser.add_argument('--kld-lambda-1', type=float,
						help='Lambda for affinity loss at edge.')
	parser.add_argument('--kld-lambda-2', type=float,
						help='Lambda for affinity loss at non-edge.')
	# Misc paramters
	parser.add_argument('--restore-from', type=str, default='',
						help='Where restore model parameters from.')
	parser.add_argument('--save-pred-every', type=int, default=10000,
						help='Save summaries and checkpoint every often.')
	parser.add_argument('--update-tb-every', type=int, default=20,
						help='Update summaries every often.')
	parser.add_argument('--snapshot-dir', type=str, default='',
						help='Where to save snapshots of the model.')
	parser.add_argument('--not-restore-classifier', action='store_true',
						help='Whether to not restore classifier layers.')
	
	return parser.parse_args()


def save(saver, sess, logdir, step):
	"""Saves the trained weights.
	
	Args:
	  saver: TensorFlow Saver object.
	  sess: TensorFlow session.
	  logdir: path to the snapshots directory.
	  step: current training step.
	"""
	model_name = 'model.ckpt'
	checkpoint_path = os.path.join(logdir, model_name)
	
	if not os.path.exists(logdir):
		os.makedirs(logdir)
	saver.save(sess, checkpoint_path, global_step=step)
	print('The checkpoint has been created.')


def load(saver, sess, ckpt_path):
	"""Loads the trained weights.
	
	Args:
	  saver: TensorFlow Saver object.
	  sess: TensorFlow session.
	  ckpt_path: path to checkpoint file with parameters.
	"""
	saver.restore(sess, ckpt_path)
	print('Restored model parameters from {}'.format(ckpt_path))


def main():
	"""Creates the model and start training.
	"""
	# Read CL arguments and snapshot the arguments into text file.
	args = get_arguments()
	utils.general.snapshot_arg(args)
	
	# The segmentation network is stride 8 by default.
	h, w = map(int, args.input_size.split(','))
	input_size = (h, w)
	innet_size = (int(math.ceil(h / 8)), int(math.ceil(w / 8)))
	
	# Initialize the random seed.
	tf.set_random_seed(args.random_seed)
	
	# Create queue coordinator.
	coord = tf.train.Coordinator()
	
	# current step
	step_ph = tf.placeholder(dtype=tf.float32, shape=())
	
	# Load the data reader.
	with tf.device('/cpu:0'):
		with tf.name_scope('create_inputs'):
			reader = ImageReader(
				args.data_dir,
				args.data_list,
				input_size,
				args.random_scale,
				args.random_mirror,
				args.random_crop,
				args.ignore_label,
				IMG_MEAN)
			
			image_batch, label_batch = reader.dequeue(args.batch_size)
	
	# Shrink labels to the size of the network output.
	labels = tf.image.resize_nearest_neighbor(
		label_batch, innet_size, name='label_shrink')
	labels_flat = tf.reshape(labels, [-1, ])
	
	# Ignore the location where the label value is larger than args.num_classes.
	not_ignore_pixel = tf.less_equal(labels_flat, args.num_classes - 1)
	
	# Extract the indices of pixel where the gradients are propogated.
	pixel_inds = tf.squeeze(tf.where(not_ignore_pixel), 1)
	
	# Create network and output predictions.
	outputs = model(image_batch,
					args.num_classes,
					args.is_training,
					args.use_global_status)
	
	# Grab variable names which should be restored from checkpoints.
	restore_var = [
		v for v in tf.global_variables()
		if 'block5' not in v.name or not args.not_restore_classifier
	]
	
	# Sum the losses from output branches.
	labels_gather = tf.to_int32(tf.gather(labels_flat, pixel_inds))
	one_hot_lab = tf.one_hot(tf.squeeze(labels, axis=-1), depth=args.num_classes)
	
	seg_losses = []
	aaf_losses = []
	for i, output in enumerate(outputs):
		# Define softmax loss.
		output_2d = tf.reshape(output, [-1, args.num_classes])
		output_gather = tf.gather(output_2d, pixel_inds)
		seg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=output_gather, labels=labels_gather)
		seg_loss = tf.reduce_mean(seg_loss)
		seg_losses.append(seg_loss)
		
		# Define AAF loss.
		prob = tf.nn.softmax(output, axis=-1)
		w_edge = tf.get_variable(
			name='edge_w',
			shape=(1, 1, 1, args.num_classes, 1, 3),
			dtype=tf.float32,
			initializer=tf.constant_initializer(0))
		
		w_edge = tf.nn.softmax(w_edge, axis=-1)
		
		w_not_edge = tf.get_variable(
			name='nonedge_w',
			shape=(1, 1, 1, args.num_classes, 1, 3),
			dtype=tf.float32,
			initializer=tf.constant_initializer(0))
		w_not_edge = tf.nn.softmax(w_not_edge, axis=-1)
		
		n, h, w = w_edge.get_shape().as_list()[:3]
		
		# Apply AAF on 3x3 patch.
		eloss_1, neloss_1 = lossx.adaptive_affinity_loss(labels,
														 one_hot_lab,
														 prob,
														 1,
														 args.num_classes,
														 args.kld_margin,
														 w_edge[..., 0],
														 w_not_edge[..., 0])
		# Apply AAF on 5x5 patch.
		eloss_2, neloss_2 = lossx.adaptive_affinity_loss(labels,
														 one_hot_lab,
														 prob,
														 2,
														 args.num_classes,
														 args.kld_margin,
														 w_edge[..., 1],
														 w_not_edge[..., 1])
		# Apply AAF on 7x7 patch.
		eloss_3, neloss_3 = lossx.adaptive_affinity_loss(labels,
														 one_hot_lab,
														 prob,
														 3,
														 args.num_classes,
														 args.kld_margin,
														 w_edge[..., 2],
														 w_not_edge[..., 2])
		# Apply exponential decay to the AAF loss.
		dec = tf.pow(10.0, -step_ph / args.num_steps)
		aaf_loss = tf.reduce_mean(eloss_1) * args.kld_lambda_1 * dec
		aaf_loss += tf.reduce_mean(eloss_2) * args.kld_lambda_1 * dec
		aaf_loss += tf.reduce_mean(eloss_3) * args.kld_lambda_1 * dec
		aaf_loss += tf.reduce_mean(neloss_1) * args.kld_lambda_2 * dec
		aaf_loss += tf.reduce_mean(neloss_2) * args.kld_lambda_2 * dec
		aaf_loss += tf.reduce_mean(neloss_3) * args.kld_lambda_2 * dec
		aaf_losses.append(aaf_loss)
	
	# Define weight regularization loss.
	w = args.weight_decay
	l2_losses = [w * tf.nn.l2_loss(v) for v in tf.trainable_variables()
				 if 'weights' in v.name]
	
	# Sum all loss terms.
	mean_seg_loss = tf.add_n(seg_losses)
	mean_aaf_loss = tf.add_n(aaf_losses)
	mean_l2_loss = tf.add_n(l2_losses)
	reduced_loss = mean_seg_loss + mean_l2_loss + mean_aaf_loss
	
	# Grab variable names which are used for training.
	all_trainable = tf.trainable_variables()
	fc_trainable = [v for v in all_trainable if 'block5' in v.name]  # lr*10
	base_trainable = [v for v in all_trainable
					  if 'block5' not in v.name and 'edge' not in v.name]  # lr*1
	aaf_trainable = [v for v in all_trainable if 'edge' in v.name]
	
	# Computes gradients per iteration.
	grads = tf.gradients(reduced_loss, base_trainable + fc_trainable + aaf_trainable)
	grads_base = grads[:len(base_trainable)]
	grads_fc = grads[len(base_trainable):len(base_trainable) + len(fc_trainable)]
	grads_aaf = grads[len(base_trainable) + len(fc_trainable):]
	grads_aaf = [-g for g in grads_aaf]  # gradient ascent
	
	# Define optimisation parameters.
	base_lr = tf.constant(args.learning_rate)
	learning_rate = tf.scalar_mul(
		base_lr,
		tf.pow((1 - step_ph / args.num_steps), args.power))
	
	opt_base = tf.train.MomentumOptimizer(learning_rate * 1.0, args.momentum)
	opt_fc = tf.train.MomentumOptimizer(learning_rate * 10.0, args.momentum)
	opt_aaf = tf.train.MomentumOptimizer(learning_rate * 1.0, args.momentum)
	
	# Define tensorflow operations which apply gradients to update variables.
	train_op_base = opt_base.apply_gradients(zip(grads_base, base_trainable))
	train_op_fc = opt_fc.apply_gradients(zip(grads_fc, fc_trainable))
	train_op_aaf = opt_aaf.apply_gradients(zip(grads_aaf, aaf_trainable))
	train_op = tf.group(train_op_base, train_op_fc, train_op_aaf)
	
	# Process for visualisation.
	with tf.device('/cpu:0'):
		# Image summary for input image, ground-truth label and prediction.
		output_vis = tf.image.resize_nearest_neighbor(
			outputs[-1], tf.shape(image_batch)[1:3, ])
		output_vis = tf.argmax(output_vis, axis=3)
		output_vis = tf.expand_dims(output_vis, dim=3)
		output_vis = tf.cast(output_vis, dtype=tf.uint8)
		
		labels_vis = tf.cast(label_batch, dtype=tf.uint8)
		
		in_summary = tf.py_func(
			utils.general.inv_preprocess,
			[image_batch, IMG_MEAN],
			tf.uint8)
		gt_summary = tf.py_func(
			utils.general.decode_labels,
			[labels_vis, args.num_classes],
			tf.uint8)
		out_summary = tf.py_func(
			utils.general.decode_labels,
			[output_vis, args.num_classes],
			tf.uint8)
		# Concatenate image summaries in a row.
		total_summary = tf.summary.image(
			'images',
			tf.concat(axis=2, values=[in_summary, gt_summary, out_summary]),
			max_outputs=args.batch_size)
		
		# Scalar summary for different loss terms.
		seg_loss_summary = tf.summary.scalar('seg_loss', mean_seg_loss)
		aaf_loss_summary = tf.summary.scalar('aaf_loss', mean_aaf_loss)
		total_summary = tf.summary.merge_all()
		
		summary_writer = tf.summary.FileWriter(args.snapshot_dir,
											   graph=tf.get_default_graph())
	
	# Set up tf session and initialize variables.
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	init = tf.global_variables_initializer()
	
	sess.run(init)
	
	# Saver for storing checkpoints of the model.
	saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)
	
	# Load variables if the checkpoint is provided.
	if args.restore_from is not None:
		loader = tf.train.Saver(var_list=restore_var)
		load(loader, sess, args.restore_from)
	
	# Start queue threads.
	threads = tf.train.start_queue_runners(coord=coord, sess=sess)
	
	# Iterate over training steps.
	pbar = tqdm(range(args.num_steps))
	for step in pbar:
		start_time = time.time()
		feed_dict = {step_ph: step}
		
		step_loss = 0
		for it in range(args.iter_size):
			# Update summary periodically.
			if it == args.iter_size - 1 and step % args.update_tb_every == 0:
				sess_outs = [reduced_loss, total_summary, train_op]
				loss_value, summary, _ = sess.run(sess_outs,
												  feed_dict=feed_dict)
				summary_writer.add_summary(summary, step)
			else:
				sess_outs = [reduced_loss, train_op]
				loss_value, _ = sess.run(sess_outs, feed_dict=feed_dict)
			
			step_loss += loss_value
		
		step_loss /= args.iter_size
		
		lr = sess.run(learning_rate, feed_dict=feed_dict)
		
		# Save trained model periodically.
		if step % args.save_pred_every == 0 and step > 0:
			save(saver, sess, args.snapshot_dir, step)
		
		duration = time.time() - start_time
		desc = 'loss = {:.3f}, lr = {:.6f}'.format(step_loss, lr)
		pbar.set_description(desc)
	
	coord.request_stop()
	coord.join(threads)


if __name__ == '__main__':
	main()
