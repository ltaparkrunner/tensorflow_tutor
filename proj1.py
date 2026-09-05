import os
import sys

# 2. Глушим стандартный вывод TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 3. Дополнительно перенаправляем stderr в пустоту на момент импорта, 
# если что-то всё ещё пытается пробиться
sys.stderr = open(os.devnull, 'w')

import tensorflow as tf
print(tf.version.VERSION)  # make sure the version is 2.x
print(f'tensorflow version = {tf.__version__}')
# Возвращаем стандартный вывод ошибок обратно после импорта, 
# чтобы не пропустить реальные ошибки в вашем коде
sys.stderr = sys.__stderr__

rtensor = tf.Variable([['test', 'ok'], ['test', 'yes']], tf.string)
print(f'rtensor rank = {tf.rank(rtensor)}')
print(tf.rank(rtensor))
print(rtensor.shape)
print(f'tensor shape = {rtensor.shape}')

tens1 = tf.ones([1,2,3])
tens2 = tf.reshape(tens1, [2,3,1])
tens3 = tf.reshape(tens1, [3, -1])

print(tens1, tens2, tens3)

t = tf.zeros([5,5,5,5])
t = tf.reshape(t, [125, -1])
print(t)