import pickle
import json
import numpy
from sklearn.externals import joblib
from sklearn.linear_model import Ridge

# Load model
model = joblib.load('./examples/sklearn/ridge_0.40.pkl')

# Test model locally
test_sample = json.dumps({'data': [
    [1,2,3,4,5,6,7,8,9,10], 
    [10,9,8,7,6,5,4,3,2,1]
]})

data = json.loads(test_sample)['data']
data = numpy.array(data)
result = model.predict(data)

print(result)

