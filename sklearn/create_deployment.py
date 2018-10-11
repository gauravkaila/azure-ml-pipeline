from azureml.core import Image
import azure 
import azureml.core
from azureml.core import Workspace
from azureml.core import ScriptRunConfig
import configparser

# Read config file
config = configparser.ConfigParser()
config.read('config.ini')

# Initialize workspace from config
ws = Workspace.from_config()

for i in Image.list(workspace = ws,image_name=config['docker']['docker_image_name']):
    if i.version == int(config['deploy']['docker_image_version']):
        image = i

from azureml.core.webservice import AciWebservice

aciconfig = AciWebservice.deploy_configuration(cpu_cores = int(config['deploy']['cpu_cores']), 
                                               memory_gb = int(config['deploy']['memory']), 
                                               tags = {'area': "meter_classification", 'type': "meter_classification"}, 
                                               description = "Image with re-trained vgg model")

from azureml.core.webservice import Webservice

aci_service_name = config['deploy']['service_name']
print(aci_service_name)
aci_service = Webservice.deploy_from_image(deployment_config = aciconfig,
                                           image = image,
                                           name = aci_service_name,
                                           workspace = ws)
aci_service.wait_for_deployment(True)
print(aci_service.state)                                              

