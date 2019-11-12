import keras.backend as K
import numpy as np
import tensorflow as tf

from robust_loss.adaptive import lossfun


class RobustAdaptativeLoss(object):
  def __init__(self):
    z = np.array([[0]])
    self.v_alpha = K.variable(z)

  def loss(self, y_true, y_pred):
    x = y_true - y_pred
    # x = K.reshape(x, shape=(-1, -1))
    with tf.variable_scope("lossfun"): #, reuse=True):
      loss, alpha, scale = lossfun(x)
    op = K.update(self.v_alpha, alpha)
    # The alpha update must be part of the graph but it should
    # not influence the result.
    return loss + 0 * op

  def alpha(self, y_true, y_pred):
    return self.v_alpha