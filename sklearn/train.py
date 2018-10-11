
from azure.storage.blob import BlockBlobService
from azureml.core.run import Run

run = Run.get_submitted_run()

##############Start of Train Script##############

##############End of Train Script##############

# Save the model to ./outputs
# model.save('./outputs/model.h5')