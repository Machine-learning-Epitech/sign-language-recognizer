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
result = 0

classes = {
    "0": ["A", "a"],
    "1": ["B", "b"],
    "2": ["C", "c"],
    "3": ["D", "d"],
    "4": ["DEL", "del"],
    "5": ["E", "e"],
    "6": ["F", "f"],
    "7": ["G", "g"],
    "7": ["H", "h"],
    "8": ["I", "i"],
    "9": ["J", "j"],
    "10": ["K", "k"],
    "11": ["L", "l"],
    "12": ["M", "m"],
    "13": ["N", "n"],
    "14": ["Nothing", "nothing"],
    "15": ["O", "o"],
    "16": ["P", "p"],
    "17": ["Q", "q"],
    "18": ["R", "r"],
    "19": ["S", "s"],
    "20": ["Space", "space"],
    "21": ["T", "t"],
    "22": ["U", "u"],
    "23": ["V", "v"],
    "24": ["W", "w"],
    "25": ["X", "x"],
    "26": ["Y", "y"],
    "27": ["Z", "z"],
    "28": ["_", " "],
    "29": [".", ","],
}

def decode_predictions(preds, top=28):
    if len(preds.shape) != 2 or preds.shape[1] != 30:
        raise ValueError('`decode_predictions` expects '
                        'a batch of predictions '
                        '(i.e. a 2D array of shape (samples, 1000)). '
                     'Found array with shape: ' + str(preds.shape))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(classes[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results

def predict(image_path, IMG_SIZE=200):
    image = Image.open(image_path)
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    image = tf.expand_dims(image, axis =-1)
    resp = model.predict(image)
    return resp;

for class_name in class_names:
    test_image_path = os.path.abspath('asl_alphabet_test/asl_alphabet_test/' + class_name + '_test.jpg')

    predictions = predict(test_image_path)
    score = tf.nn.softmax(predictions[0])

    print("Expected output: " + class_name)
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

    if class_names[np.argmax(score)] == class_name:
      result += 1

print("Found " + str(result) + "/" + str(len(class_names)) + " letters")
print("{:.2f}".format(result / len(class_names)) + "%")
