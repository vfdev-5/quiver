import base64
import PIL.Image
from cStringIO import StringIO

import numpy as np

from flask import send_file

def send_input_image(inputs, image_id):
    print("!!!! TEST !!!!")
    image_id = int(image_id)
    image = inputs[image_id, :, :, :]
    if not (image.shape[2] == 1 or image.shape[2] == 3) or \
        image.dtype != np.uint8:
            pass

    #return send_file(image_to_png(image), mimetype='application/octets')
    data = image_to_base64(image) 
    data = "\"data:image/png;base64,%s\"" % data.decode('utf8')
    return data


def image_to_png(img):
    f = StringIO()
    PIL.Image.fromarray(img).save(f, 'png')
    return f


def image_to_base64(img):
    f = image_to_png(img)
    image_base64 = base64.b64encode(f.getvalue())
    return image_base64


def list_inputs(inputs):
    return ['%i' % i for i in range(inputs.shape[0])]