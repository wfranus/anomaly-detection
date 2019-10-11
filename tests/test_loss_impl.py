"""Compare two implementation of the same loss function

Before run make sure that:
- these version are installed:
    pip install tensorflow==1.9.0 Theano==1.0.2 numpy==1.5.1
- Tensorflow backend for Keras is set:
    KERAS_BACKEND=tensorflow python test.py
- the same value of n_bag is set here and in paper_loss
- do not change n_seg because value 32 is hardcoded into papr_loss (wrrr..)
"""
import numpy as np
np.random.seed(42)
from tensorflow import set_random_seed
set_random_seed(42)

import tensorflow as tf
import theano.tensor as T
import math

from src.loss import custom_loss
from tests.ref_loss import paper_loss

n_seg = 32
n_bag = 60
n_exp = (n_bag * n_seg) // 2

rand_pred = np.random.uniform(0.0, 1.0, n_bag * n_seg)
targets = np.array(([0.] * n_exp) + ([1.] * n_exp), dtype='float32')

## THEANO
y_pred = T.fvector('y_pred')
y_true = T.fvector('y_true')
loss_ref = paper_loss(y_true, y_pred)
loss_ref_val = loss_ref.eval({y_pred: np.array(rand_pred, dtype='float32'),
                              y_true: targets})
print(f'Theano loss: {loss_ref_val}')


## TENSORFLOW
y_pred = tf.constant(rand_pred, dtype='float32')
y_true = tf.constant(targets)
loss_new = custom_loss(n_bags=n_bag, n_seg=n_seg, coef1=1.0, coef2=1.0)(y_true, y_pred)

with tf.Session() as sess:
    loss_new_val = sess.run([loss_new])[0]
    print(f'Tensorflow loss: {loss_new_val}')

print(f'Difference between implementations: {math.fabs(loss_ref_val-loss_new_val)}')
assert math.isclose(loss_ref_val, loss_new_val, rel_tol=1e-09)
