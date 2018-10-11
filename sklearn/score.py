import json
import numpy as np
from tensorflow.keras.models import load_model
from azureml.core.model import Model
import skimage.io
from skimage.transform import resize
import base64
from PIL import Image
import configparser

# Read config file
config = configparser.ConfigParser()
config.read('ml-config.ini')

def init():
    global model
    model_path = Model.get_model_path(config['train']['model_name'])
    model = load_model(model_path + '/model.h5')

# Here raw_data is a json with 'data' key as a base64 encoded string
def run(raw_data):
    class_dict = {'0': 'electro', '1': 'smart'}

    try:
        data = json.loads(raw_data)["data"]
        data_bytes = bytes(data,encoding='utf-8')

        with open("imageToSave.png", "wb") as fh:
            fh.write(base64.decodestring(data_bytes))

        # Convert image back to numpy array
        img = np.asarray(Image.open('imageToSave.png'))

        # Resize image to required size
        image_resized = resize(img, (225,225))

        # Normalize image
        img = image_resized/255.0

        # Expand dims
        img = np.expand_dims(img,axis=0)

        # Predict
        result = class_dict[str(model.predict_classes(img)[0])]
        
    except Exception as e:
        result = str(e)
    return json.dumps({"result": result})