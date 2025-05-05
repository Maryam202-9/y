import tensorflow as tf

def load_model(path='blood_cnn_model.h5'):
    model = tf.keras.models.load_model(path)
    return model
