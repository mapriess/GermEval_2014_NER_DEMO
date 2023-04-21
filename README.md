## [ Python environment ]

Python requirements are all listet in YML file `ml_environment_mpriess.yml`.
To install the environment in Anaconda use:

```python
conda-env create -f ml_environment_mpriess.yml
```

It comprises (among others) of the following modules:
* scipy
* numpy
* matplotlib
* pandas
* sklearn
* theano
* tensorflow
* keras
* tf keras

## [ Google Colab ]
When running the jupyter notebook files on [*Google Colab*](https://colab.research.google.com), upload the input data to the session memory (file menu on the lhs on Google Colab). Furthermore, installation of packages might be necessary.

## [ Microsoft Azure ]
The BERT model has been trained within [*Microsoft Azure Machine Learning Studio*](https://learn.microsoft.com/en-us/azure/machine-learning/overview-what-is-machine-learning-studio) as part of [Microsoft Azure](https://azure.microsoft.com/).

## [ Tensorflow ]
TensorFlow is an open source library for fast numerical computing. It was created and is maintained by Google and released under the Apache 2.0 open source license. The API is nominally for the Python programming language, although there is access to the underlying C++ API. Unlike other numerical libraries intended for use in Deep Learning like Theano, TensorFlow was designed for use both in research and development and in production systems,
not least RankBrain in Google search and the fun DeepDream project . It can run on single CPU systems, GPUs as well as mobile devices and large scale distributed systems of hundreds of machines.

## [ Theano ]
Theano is an open source project released under the BSD license and was developed by the LISA (now MILA1) group at the University of Montreal.
It uses a host of clever code optimizations to squeeze as much performance as possible from your hardware. At it’s heart Theano is a compiler for mathematical expressions in Python. It knows how to take your structures and turn them into very efficient code that uses NumPy, efficient native libraries like BLAS and native code to run as fast as possible on CPUs or GPUs.

## [ Keras Backend: Tensorflow, Theano ]
Keras is a lightweight API and rather than providing an implementation of the required mathematical operations needed for deep learning it provides a consistent interface to efficient numerical libraries. Assuming you have both Theano and TensorFlow installed, you can configure the backend used by Keras. The easiest way is by adding or editing the Keras configuration file in your home directory (`~/.keras/keras.json`) which has the following format.

Theano `keras.json`:
```json
{
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "theano",
    "image_data_format": "channels_last",
    "image_dim_ordering": "th"
}
```

Tensorflow `keras.json`:
```json
{
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "tensorflow",
    "image_data_format": "channels_last",
    "image_dim_ordering": "tf"
}
```

You can confirm the backend used by Keras using the following script on the command line:

```python
python -c "from keras import backend; print(backend.backend())"
```
