
from azure.storage.blob import BlockBlobService
from azureml.core.run import Run

run = Run.get_submitted_run()

##############Start of Train Script##############
import os
import argparse

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import numpy as np

os.makedirs('./outputs', exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str,
                    dest='data_folder', help='data folder')
args = parser.parse_args()

print('Data folder is at:', args.data_folder)
print('List all files: ', os.listdir(args.data_folder))

X = np.load(os.path.join(args.data_folder, 'features.npy'))
y = np.load(os.path.join(args.data_folder, 'labels.npy'))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
data = {"train": {"X": X_train, "y": y_train},
        "test": {"X": X_test, "y": y_test}}

alpha = 0.95

reg = Ridge(alpha=alpha)
reg.fit(data["train"]["X"], data["train"]["y"])

preds = reg.predict(data["test"]["X"])
mse = mean_squared_error(preds, data["test"]["y"])
run.log('alpha', alpha)
run.log('mse', mse)

model_file_name = 'ridge_1.pkl'
with open(model_file_name, "wb") as file:
    print('saving model: {0}'.format(model_file_name))
    joblib.dump(value=reg, filename='outputs/' + model_file_name)
    print('path of file: {0}'.format(os.path.abspath('outputs/'+model_file_name)))


##############End of Train Script##############

