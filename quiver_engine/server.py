from __future__ import print_function

import json

import os
import shutil
from os import listdir
from os.path import abspath, relpath, dirname, join, exists, splitext
import webbrowser

from flask import Flask, send_from_directory
from flask.json import jsonify
from flask_cors import CORS

from gevent.wsgi import WSGIServer

from quiver_engine.util import (
    load_img, safe_jsonnify, decode_predictions,
    get_input_config, get_evaluation_context,
    validate_launch
)

from quiver_engine.file_utils import list_img_files, save_input_data
from quiver_engine.vis_utils import save_layer_outputs

import numpy as np


# Global Flask definition
Application = Flask(__name__)
Application.threaded = True
CORS(Application)

'''
    Static Routes
'''
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

    
@Application.route('/temp-file/<path>')
def get_temp_file(path):
    print("-- get_input_file : ", Application.temp_folder)
    return send_from_directory(abspath(Application.temp_folder), path)

    
@Application.route('/input-file/<path>')
def get_input_file(path):
    if splitext(path) == '.npz':
        path += '.png'
    print("-- get_input_file : ", Application.input_folder, path)
    return send_from_directory(abspath(Application.input_folder), path)    
    
'''
    Computations
'''
@Application.route('/model')
def get_config():
    return jsonify(json.loads(Application.model.to_json()))


@Application.route('/inputs')
def get_inputs():
    print("-- get_inputs : input_folder=", Application.input_folder)
    print("-- get_inputs : temp_folder=", Application.temp_folder)
    return jsonify(list_img_files(Application.input_folder))


@Application.route('/layer/<layer_name>/<input_path>')
def get_layer_outputs(layer_name, input_path):
    return jsonify(
        save_layer_outputs(
            load_img(
                join(abspath(Application.input_folder), input_path),
                Application.single_input_shape,
                grayscale=(Application.input_channels == 1)
            ),
            Application.model,
            layer_name,
            Application.temp_folder,
            input_path
        )
    )

@Application.route('/predict/<input_path>')
def get_prediction(input_path):
    with get_evaluation_context():
        return safe_jsonnify(
            decode_predictions(
                Application.model.predict(
                    load_img(
                        join(abspath(Application.input_folder), input_path),
                        Application.single_input_shape,
                        grayscale=(Application.input_channels == 1)
                    )
                ),
                Application.classes,
                Application.top
            )
        )    
    
    
def configure(app, model, classes, top, html_base_dir, temp_folder='./tmp', input_folder='', input_data=None):
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
    single_input_shape, input_channels = get_input_config(model)
    
    app.single_input_shape = single_input_shape
    app.input_channels = input_channels
    app.input_data = input_data
    app.input_folder = input_folder
    app.temp_folder = temp_folder
    app.model = model
    app.classes = classes
    app.top = top
    app.html_base_dir = html_base_dir
   

def run_app(app, port=5000):
    print ("--- App: ", app, id(app))
    http_server = WSGIServer(('', port), app)
    try:
        webbrowser.open_new('http://localhost:' + str(port))
        http_server.serve_forever()
    except KeyboardInterrupt:
        http_server.stop()
        http_server = None
        if app.input_data is not None:
            # remove 'input_data_temp'
            shutil.rmtree(app.input_folder)

    
def launch(model, classes=None, top=5, temp_folder='./tmp', input_folder='', input_data=None, port=5000, html_base_dir=None):
    """
    Method to launch server
    :param model: Keras model
    :param classes: predictions classes
    :param temp_folder: temporary folder to store images
    :param input_folder: folder with input images .png, .jpg, .gif to run predictions on
    :param input_data: ndarray of shape `(n_images, height, width, n_channels)` of images to run predictions on. 
                        `n_channels` should be 1 or 3. Input data is stored in `temp_folder/input_data_temp`.
    :param port: quiver server port
    :param html_base_dir:
    """
    print("-- launch : ", input_data is not None, "input_folder=", input_folder)
    
    assert (len(input_folder) > 0 and input_data is None) or \
        (len(input_folder) == 0 and input_data is not None), "You should specify either input_folder or input_data"

    if not exists(temp_folder):
        os.makedirs(temp_folder)
        
    if input_data is not None:
        assert isinstance(input_data, np.ndarray) and len(input_data.shape) == 4, \
            "Parameter input_data should be an ndarray of shape (n_images, n_channels, height, width)"
        assert input_data.shape[3] == 1 or input_data.shape[3] == 3, "Input data n_channels should be 1 or 3"
    
        input_folder = join(temp_folder, 'input_data_temp')
        os.makedirs(input_folder)
        save_input_data(input_data, input_folder)
    
    html_base_dir = html_base_dir if html_base_dir is not None else dirname(abspath(__file__))
    print('Starting webserver from:', html_base_dir)
    assert exists(join(html_base_dir, "quiverboard")), "Quiverboard must be a " \
                                                                       "subdirectory of {}".format(html_base_dir)
    assert exists(join(html_base_dir, "quiverboard", "dist")), "Dist must be a " \
                                                                               "subdirectory of quiverboard"
    assert exists(join(html_base_dir, "quiverboard", "dist", "index.html")), "Index.html missing"
        
    configure(
            Application,
            model, classes, top, 
            html_base_dir=html_base_dir,
            temp_folder=temp_folder, 
            input_folder=input_folder, 
            input_data=input_data)
    return run_app(Application, port)
