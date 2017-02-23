import re
from os.path import relpath, abspath, join
from os import listdir

import numpy as np
from scipy.misc import imsave
from quiver_engine.util import deprocess_image


def save_layer_img(layer_outputs, layer_name, idx, temp_folder, input_path):
    filename = get_output_filename(layer_name, idx, temp_folder, input_path)
    imsave(filename, deprocess_image(layer_outputs))
    return relpath(filename, abspath(temp_folder))


def get_output_filename(layer_name, z_idx, temp_folder, input_path):
    return '{}/{}_{}_{}.png'.format(temp_folder, layer_name, str(z_idx), input_path)

    
def list_img_files(input_folder):
    image_regex = re.compile(r'.*\.(jpg|png|gif|npz)$')
    return [
        filename
        for filename in listdir(
            abspath(input_folder)
        )
        if image_regex.match(filename) is not None
    ]
    

def save_data_iconview(img, filename):
    """
    Method to save image `img` as png for browser icons
    :param img: ndarray of shape (height, width, n_channels)
    """
    if (img.shape[2] == 3 or img.shape[2] == 1) and img.dtype == np.uint8:
        imsave(filename, img)
    else:
        img = img[:, :, 0]
        imsave(filename, img)
        
def save_input_data(input_data, input_folder):
    """
    Method to save input_data as npz compressed files and create its png icons for browser
    """
    for i in range(input_data.shape[0]):
        filename = join(input_folder, "image_data_%i.npz" % i)
        np.savez_compressed(filename, input_data[i,:,:,:])
        # create png icons
        #filename += '.png'        
        #save_data_iconview(input_data[i,:,:,:], filename) 
        