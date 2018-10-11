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

# Create an experiment
from azureml.core import Experiment
# NOTE: New experiment is created if one with the following name is not found
experiment_name = config['train']['experiment_name']
exp = Experiment(workspace=ws, name=experiment_name)

# Create a target compute - VM
from azureml.core.compute import DsvmCompute
from azureml.core.compute_target import ComputeTargetException

# NOTE: New compute target is created is one with the following name is not found
compute_target_name = config['train']['compute_target_name']

try:
    dsvm_compute = DsvmCompute(workspace=ws, name=compute_target_name)
    print('found existing:', dsvm_compute.name)
except ComputeTargetException:
    print('creating new.')
    dsvm_config = DsvmCompute.provisioning_configuration(vm_size= config['train']['vm_size'])
    dsvm_compute = DsvmCompute.create(ws, name=compute_target_name, provisioning_configuration=dsvm_config)
    dsvm_compute.wait_for_completion(show_output=True)

# get the default datastore and upload data from local folder to VM
ds = ws.get_default_datastore()
print(ds.name, ds.datastore_type, ds.account_name, ds.container_name)

# Upload data to default data storage
data_folder = config['train']['data_folder']
ds.upload(config['train']['local_directory'],target_path=data_folder,overwrite=True)
print ('Finished Uploading Data.')

# Run Configuration
from azureml.core.runconfig import DataReferenceConfiguration
dr = DataReferenceConfiguration(datastore_name=ds.name, 
                   path_on_datastore=data_folder, 
                   mode='download', # download files from datastore to compute target
                   overwrite=True)

from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies

# create a new RunConfig object
conda_run_config = RunConfiguration(framework="python")

# Set compute target to the Linux DSVM
conda_run_config.target = dsvm_compute.name

# set the data reference of the run coonfiguration
conda_run_config.data_references = {ds.name: dr}

# specify conda packages to install on the VM
conda_run_config.environment.python.conda_dependencies = CondaDependencies.create(conda_packages=ast.literal_eval(config['train']['conda_packages']))

from azureml.core import Run
from azureml.core import ScriptRunConfig

src = ScriptRunConfig(source_directory='./', 
                      script= config['train']['script'],
                      run_config=conda_run_config, 
                      # pass the datastore reference as a parameter to the training script
                      arguments=['--data-folder', str(ds.as_download())] 
                     ) 
run = exp.submit(config=src)
run.wait_for_completion(show_output=True)

# Register the model
print('Registering model...')
model = run.register_model(model_name=config['train']['model_name'], model_path='./outputs/model.h5')
print('Done registering model.')




