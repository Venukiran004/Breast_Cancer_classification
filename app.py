# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

#loading svm 
filename = 'breast_cancer_model.pkl'
with open(filename,'rb') as f:
     model=pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
        rad = float(request.form['Radius'])
        text = float(request.form['Texture'])
        area= float(request.form['Area'])
        smt = float(request.form['Smoothness'])
        compact = float(request.form['Compactness'])
        conca = float(request.form['Concavity'])
        concave = float(request.form['Concave'])
        symm= float(request.form['symmetry'])
        frac = float(request.form['fractal_dimension'])
        data = np.array([[rad,text,area,smt,compact,conca,concave,symm,frac]])
        my_prediction = model.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)