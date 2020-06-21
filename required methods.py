from PIL import Image
import numpy as np
from tensorflow.keras.models import model_from_json

PADDING = 10

label_meaning = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B',
                 12: 'C', 13: 'D',
                 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O',
                 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
                 36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j', 46: 'k',
                 47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't', 56: 'u', 57: 'v',
                 58: 'w', 59: 'x', 60: 'y', 61: 'z'}



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
    """

    :param model: model loaded using get_model()
    :param im: PIL object returned from create_image_for_prediction
    :return: predicted character.
    """
    to_predict = np.array(im)
    arr = to_predict.reshape(1, 28, 28, 1)
    arr = arr.astype('float32')
    arr /= 255
    return label_meaning[int(model.predict_classes(arr)[0])]

