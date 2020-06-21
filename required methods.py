from PIL import Image
import numpy as np
from keras.models import load_model
from tensorflow.keras.models import model_from_json
import os


def create_image_for_prediction(image_pixels):

    """
    :param image_pixels: numpy array of image pixels: numpy.array(*PIL-object*)
    :return: PIL object.
    """

    min_x = 100000
    min_y = 100000
    max_x = 0
    max_y = 0
    for p in range(len(image_pixels)):
        image_pixels[p] = (int(image_pixels[p][0]), int(image_pixels[p][1]))
        p = image_pixels[p]
        if p[1] > max_x:
            max_x = p[1]
        if p[1] < min_x:
            min_x = p[1]
        if p[0] > max_y:
            max_y = p[0]
        if p[0] < min_y:
            min_y = p[0]

    if min_x == 100000:
        min_x = 0
    if min_y == 100000:
        min_y = 0

    value = np.empty((), dtype=object)
    value[()] = (0, 0, 0, 255)
    new_pic = np.full((PADDING * 2 + max_y + 1 - min_y, PADDING * 2 + max_x + 1 - min_x), value, dtype=object)
    for pixel in image_pixels:
        x = PADDING + pixel[1] - min_x
        y = PADDING + pixel[0] - min_y
        new_pic[y][x] = (255, 255, 255, 255)

    im = Image.new('RGBA', (PADDING * 2 + max_y + 1 - min_y, PADDING * 2 + max_x + 1 - min_x))
    row_index = 0
    column_index = 0
    for row in new_pic:
        for pixel in row:
            im.putpixel((row_index, column_index), pixel)
            column_index += 1
        column_index = 0
        row_index += 1

    im = im.resize((28, 28))
    im = im.convert('L')

    return im


def get_model():

    """
    load model from file
    :return: model (keras model object)
    """

    json_file = open('classifier.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('weights.h5')
    return loaded_model


def predict_character(model, im):
    to_predict = np.array(im)
    arr = to_predict.reshape(1, 28, 28, 1)
    arr = arr.astype('float32')
    arr /= 255
    return label_meaning[int(model.predict_classes(arr)[0])]

