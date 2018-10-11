# TODO: configure paths correctly
import azure 
import azureml.core
from azureml.core import Workspace
from azureml.core import ScriptRunConfig
import configparser
import ast

# Read config file
config = configparser.ConfigParser()
config.read('config.ini')

# Initialize workspace from config
ws = Workspace.from_config()

# Retrive registered model by version and name
from azureml.core.model import Model

regression_models = Model.list(workspace=ws,name=config['train']['model_name']) # vgg
for m in regression_models:
    if m.version == int(config['docker']['model_version']):
        model = m

# Create conda environment for docker image
from azureml.core.conda_dependencies import CondaDependencies 

# DEFINE CONDA DEPENDENCIES 
myenv = CondaDependencies.create(pip_packages=ast.literal_eval(config['docker']['pip_packages']),conda_packages=ast.literal_eval(config['train']['conda_packages']))
myenv.add_pip_package("pynacl==1.2.1")

# CREATE CONDA ENVIRONMENT FILE
with open(config['docker']['conda_env_file'],"w") as f:
    f.write(myenv.serialize_to_string())

# Create docker image
from azureml.core.image import Image, ContainerImage

image_config = ContainerImage.image_configuration(runtime= "python",
                                 execution_script=config['docker']['path_scoring_script'],
                                 conda_file=config['docker']['conda_env_file'],
                                 tags = {'area': "meter_classification", 'type': "meter_classification"},
                                 description = "Image with re-trained vgg model")

image = Image.create(name = config['docker']['docker_image_name'],
                     # this is the model object 
                     models = [model],
                     image_config = image_config, 
                     workspace = ws)

image.wait_for_creation(show_output = True)

