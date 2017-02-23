
import os
import shutil
from os import listdir
from os.path import abspath, relpath, dirname, join, exists
import json

import webbrowser

from flask import Flask, send_from_directory
from flask.json import jsonify
from flask_cors import CORS

from gevent.wsgi import WSGIServer

import numpy as np

# Project 
from new_utils import send_input_image, list_inputs


# Global Flask definition
Application = Flask(__name__)
Application.threaded = True
CORS(Application)


@Application.route('/')
def home():
    return send_from_directory(
        join(Application.html_base_dir, 'quiverboard/dist'),
        'index.html'
    )
    
    
@Application.route('/<path>')
def get_board_files(path):
    return send_from_directory(
        join(Application.html_base_dir, 'quiverboard/dist'),
        path
    )  
    

@Application.route('/model')
def get_config():
    return jsonify(json.loads(Application.model.to_json()))

# @Application.route('/layer/<layer_name>/<input_path>')
# def get_layer_outputs(layer_name, input_path):
#     return jsonify(
#         save_layer_outputs(
#             load_img(
#                 join(abspath(Application.input_folder), input_path),
#                 Application.single_input_shape,
#                 grayscale=(Application.input_channels == 1)
#             ),
#             Application.model,
#             layer_name,
#             Application.temp_folder,
#             input_path
#         )
#     )

# @Application.route('/predict/<input_path>')
# def get_prediction(input_path):
#     with get_evaluation_context():
#         return safe_jsonnify(
#             decode_predictions(
#                 Application.model.predict(
#                     load_img(
#                         join(abspath(Application.input_folder), input_path),
#                         Application.single_input_shape,
#                         grayscale=(Application.input_channels == 1)
#                     )
#                 ),
#                 Application.classes,
#                 Application.top
#             )
#         )    

@Application.route('/inputs')
def get_inputs():
     return jsonify(list_inputs(Application.inputs))    


@Application.route('/input-file/<image_id>')
def get_input_file(image_id):
    print("-- get_input_file : ", image_id)
    return send_input_image(Application.inputs, image_id)



def _run_app(app, port=5000):
    print ("--- App: ", app, id(app))
    http_server = WSGIServer(('', port), app)
    try:
        webbrowser.open_new('http://localhost:' + str(port))
        http_server.serve_forever()
    except KeyboardInterrupt:
        http_server.stop()
        http_server = None


def _configure(app, model, inputs, classes, top, html_base_dir):
    '''
    The base of the Flask application to be run
    :param model: the model to show
    :param classes: list of names of output classes to show in the GUI.
        if None passed - ImageNet classes will be used
    :param top: number of top predictions to show in the GUI
    :param html_base_dir: the directory for the HTML (usually inside the
        packages, quiverboard/dist must be a subdirectory)
    :param temp_folder: where the temporary image data should be saved
    :param input_folder: the image directory for the raw data
    :return:
    '''
    app.inputs = inputs
    app.model = model
    app.classes = classes
    app.top = top
    app.html_base_dir = html_base_dir


def launch(model, inputs, classes=None, top=5, port=5000, html_base_dir=None):
    """
    Method to launch server
    :param model: Keras model
    :param classes: target class labels
    :param top: number of top predictions
    
    :param temp_folder: temporary folder to store images
    :param input_folder: folder with input images .png, .jpg, .gif to run predictions on
    
    :param inputs: ndarray of shape `(n_images, height, width, n_channels)` of images to run predictions on. 
        The array should be preprocessed as when it is given to `model.predict` function.
                        
    :param port: quiver server port
    :param html_base_dir:
    """
    print("-- launch : ", inputs is not None)
    assert inputs is not None, "You should specify either input_data"
        
    if inputs is not None:
        assert isinstance(inputs, np.ndarray) and len(inputs.shape) == 4, \
            "Parameter input_data should be an ndarray of shape (n_images, n_channels, height, width)"
        assert inputs.shape[3] == 1 or inputs.shape[3] == 3, "Input data n_channels should be 1 or 3"
        
    html_base_dir = html_base_dir if html_base_dir is not None else dirname(abspath(__file__))
    print('Starting webserver from:', html_base_dir)
    assert exists(join(html_base_dir, "quiverboard")), "Quiverboard must be a " \
                                                                       "subdirectory of {}".format(html_base_dir)
    assert exists(join(html_base_dir, "quiverboard", "dist")), "Dist must be a " \
                                                                               "subdirectory of quiverboard"
    assert exists(join(html_base_dir, "quiverboard", "dist", "index.html")), "Index.html missing"
        
    _configure(Application, model, inputs, classes, top, html_base_dir)
    return _run_app(Application, port)


