# Keras Autoencoder

A collection of different autoencoder types in Keras. It is inspired by [this blog post](http://blog.keras.io/building-autoencoders-in-keras.html).

## Installation

Python is easiest to use with a virtual environment. All packages are sandboxed in a local folder so that they do not interfere nor pollute the global installation:

    virtualenv --system-site-packages venv
    
Whenever you now want to use this package, type
    
    source venv/bin/activate
    
in every terminal that wants to make use of it. To install the dependencies, use **pip**:
    
    pip install -r requirements.txt
    
If you want to use **tensorflow** as the backend, you have to install it as described [in the tensorflow install guide][1]. Then, change the backend for Keras like described [here][2]. Now everything is ready for use!

## Usage

One can change the type of autoencoder in **main.py**. 

    python main.py

I currently use it for an university project relating robots, that is why this dataset is in there. Feel free to use your own!

[1]: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md#virtualenv-installation
[2]: http://keras.io/backend/

