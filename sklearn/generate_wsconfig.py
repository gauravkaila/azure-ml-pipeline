# Create workspace configuration file

from azureml.core import Workspace
import configparser

# Read config file
config = configparser.ConfigParser()
config.read('ml-config.ini')

subscription_id = config['ws_config']['subscription_id']
resource_group = config['ws_config']['resource_group']
workspace_name = config['ws_config']['workspace_name']

try:
   ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
   ws.write_config()
   print('Library configuration succeeded')
except:
   print('Workspace not found')