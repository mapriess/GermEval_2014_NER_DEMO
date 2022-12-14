# scipy
import scipy
print('scipy: %s' % scipy.__version__)

# numpy
import numpy
print('numpy: %s' % numpy.__version__)

# matplotlib
import matplotlib
print('matplotlib: %s' % matplotlib.__version__)

# pandas
import pandas
print('pandas: %s' % pandas.__version__)

# scikit-learn
import sklearn
print('sklearn: %s' % sklearn.__version__)

# theano
import theano
print('theano: %s' % theano.__version__)

# tensorflow
import tensorflow
print('tensorflow: %s' % tensorflow.__version__)

# keras
import keras
print('keras: %s' % keras.__version__)
print('keras backend: ' + keras.backend.backend())

# tf keras
from tensorflow import keras as tf_keras
print('tf keras: %s' % tf_keras.__version__)
print('tf keras backend: ' + tf_keras.backend.backend())
