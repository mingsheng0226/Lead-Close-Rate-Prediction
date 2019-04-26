from flask import Flask, jsonify, request
import pickle
import json
import jsonpickle as jp
import numpy as np
import pandas as pd
import xgboost as xgb

app = Flask(__name__)

# Choose the objective
label = 'close_3d'
# Set the parameters
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'auc'
params['eta'] = 0.1
params['max_depth'] = 5
params['num_rounds'] = 200
params['alpha'] = 1
params['lambda'] = 3
params['seed'] = 31
params['silent'] = 1

model = None
result = {}

def load_training(data, label='close_3d'):
    """Load all historical json records for training"""
    df = pd.read_json(data, orient='records')
    ids = df['Quote.No']
    features = np.array(df[df.columns.difference(['Quote.No','close_3d','close_2w'])]).astype('float32')
    labels = np.array(df['close_3d'])
    return ids, features, labels

def train_xgb(features, labels, params):
    """Train and save model"""
    xgb_train = xgb.DMatrix(features, label=labels)
    model = xgb.train(params, xgb_train, params['num_rounds'])
    pk = pickle.dumps(model)
    return pk

def load_testing(data):
    """Load sample for prediction"""
    test = pd.read_json(data, orient='records')
    id_test, X_test = test['Quote.No'], test[test.columns.difference(['Quote.No'])]
    return id_test, X_test

def predict_xgb(id_test, X_test, pk):
    """Predict the lead score and return the JSON result"""
    clt = pickle.loads(jp.decode(pk))
    X_test.columns = ['f0','f1','f2','f3','f4','f5','f6','f7','f8','f9',
                      'f10','f11','f12','f13','f14','f15','f16','f17','f18','f19',
                      'f20','f21','f22','f23','f24','f25','f26','f27','f28','f29']
    X = xgb.DMatrix(X_test)
    pred = clt.predict(X)
    result[id_test.values[0]] = str(pred[0])
    return result

@app.route('/model', methods = ['GET', 'POST', 'PUT'])
def api_model():
    global model
    if request.method == 'GET':
        return model

    elif request.method == 'POST':
        if request.headers['Content-Type'] == 'application/json':
            ids, features, labels = load_training(json.dumps(request.get_json()))
            clt = train_xgb(features, labels, params)
            model = jp.encode(clt)
            return model
        
        else:
            return "415 Unsupported Media Type!"

    elif request.method == 'PUT':
        if request.headers['Content-Type'] == 'application/json':
            ids, features, labels = load_training(json.dumps(request.get_json()))
            clt = train_xgb(features, labels, params)
            model = jp.encode(clt)
            return model
        
        else:
            return "415 Unsupported Media Type!"
        
@app.route('/result', methods = ['GET'])
def api_result():
    if request.method == 'GET':
        if request.headers['Content-Type'] == 'application/json':
            id_test, X_test = load_testing(json.dumps(request.get_json()))
            result = predict_xgb(id_test, X_test, model)
            return jsonify(result)
        else:
            return "415 Unsupported Media Type!"

if __name__ == '__main__':
    app.run(port=7000, debug=True)
    print model
