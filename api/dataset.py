import tensorflow as tf
from tensorflow.python import keras

# Full length of the dataset.
BUFFER_SIZE = 60000

# Training batch size.
BATCH_SIZE = 256


def load_normalized_dataset():
    """
    Loads MNIST dataset,  normalizes it to [-1, +1] values,
    and shuffles.

    :return: Normalized dataset.
    """
    (train_images, train_labels), (_, _) = keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    return train_dataset
