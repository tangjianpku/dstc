"""Example running MemN2N on a single bAbI task.
Download tasks from facebook.ai/babi """
from __future__ import absolute_import
from __future__ import print_function

from sklearn import cross_validation, metrics
from memn2n import MemN2N
from itertools import chain
from six.moves import range, reduce
from reader import TextReader

import tensorflow as tf
import numpy as np

tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 1, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 50, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 20, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
tf.flags.DEFINE_integer("class_size", 6, "Number of classes.")
tf.flags.DEFINE_integer("random_state", 20, "Random state.")
tf.flags.DEFINE_string("dataset", "dblp", "Directory containing bAbI tasks")
tf.flags.DEFINE_integer("sentence_size", "15", "maximum sentence size")
tf.flags.DEFINE_string("nonlin", "None", "nonlinear function ")
tf.flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoints]")
tf.flags.DEFINE_string("encoder", "avg", " the approach of sentence encoder")
tf.flags.DEFINE_string("filter_sizes", "3", "the filter sizes in convolutional neural networks")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability")
tf.flags.DEFINE_integer("share_memory_size", 20, "Share memory size")
tf.flags.DEFINE_string("embedding_file", "doc.txt", "document embedding file")
FLAGS = tf.flags.FLAGS

print("class_size:", FLAGS.class_size)

with tf.Session() as sess:
	reader =TextReader(FLAGS)
	model = MemN2N( session=sess, reader=reader, config= FLAGS)
	model.train()
