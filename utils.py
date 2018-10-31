from tensorflow.keras import initializers

def init_normal(shape=[0,0.05], seed=None):
    """ A easy initializer"""
    mean, stddev = shape
    return initializers.RandomNormal(mean=mean, stddev=stddev, seed=seed)