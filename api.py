from flask import Flask,request,jsonify
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
model=joblib.load('model2');

app=Flask(__name__)

@app.route('/')
def index():
  return "Hello"

@app.route('/predict',methods=['POST'])

def predict():
    age=request.form.get('age')
    g=request.form.get('gender')
    if g=='Male':
        g=1
    else:
        g=0
    Bp=request.form.get('bp')
    Hr=request.form.get('hr')
    os=request.form.get('os')
    input1 = np.array([[age, g, Bp, Hr, os]])
    sc_X=StandardScaler()
    scaler = sc_X.fit(input1)
    x = scaler.transform(input1)
    x = x.reshape(1, 5).astype('float32')
    result = model.predict(x)
    if result[0][1]>0.8:
        return jsonify(status="High Risk")
    else:
        return jsonify(status="Safe")
if __name__=='__main__':
    app.run(debug=True)