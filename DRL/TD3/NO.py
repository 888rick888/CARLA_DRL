import tensorflow as tf
import numpy as np

discounted_r = []
reward_buffer = np.ones(10)
v_s_ = 1
for r in reward_buffer[::-1]:
    v_s_ = r + 0.9 * v_s_
    discounted_r.append(v_s_)
discounted_r.reverse()
discounted_r = np.array(discounted_r)[:, np.newaxis]
a = tf.reduce_mean([[1, 1], [2, 2]])
print(discounted_r, a)