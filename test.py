import matplotlib as plt
import tensorflow as tf
import imageio

image_path = '/Users/aniruddhapandit/Desktop/Testing/GITHUB/ML-player-recognition-/Training/sample.jpeg'
input_image = tf.io.read_file(image_path)
plt.imshow(image_path)
_ = plt.axis('off')
