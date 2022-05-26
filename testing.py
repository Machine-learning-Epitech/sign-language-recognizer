import os
import model_lib
import numpy as np
from PIL import Image
import tensorflow as tf
from skimage import transform

img_height = 200
img_width = 200

model = model_lib.load_model()

class_names = model_lib.get_class_names()

# Test
test_image_path = os.path.abspath('asl_alphabet_test/asl_alphabet_test/L_test.jpg')

img = Image.open(test_image_path)
img = np.array(img).astype('float32') / 255
img = transform.resize(img, (img_height, img_width, 3))
img = np.expand_dims(img, axis=0)

predictions = model.predict(img)
score = tf.nn.softmax(predictions[0])

print("Expected output: L")
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
