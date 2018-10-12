import json
import numpy as np
from sklearn.externals import joblib
from azureml.core.model import Model

def init():
    global model
    # note here "sklearn_regression_model.pkl" is the name of the model registered under
    # this is a different behavior than before when the code is run locally, even though the code is the same.
    model_path = Model.get_model_path('sklearn-test')
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path + '/ridge_1.pkl')

# note you can pass in multiple rows for scoring
def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        data = np.array(data)
        result = model.predict(data)
    except Exception as e:
        result = str(e)
    return json.dumps({"result": result.tolist()})
