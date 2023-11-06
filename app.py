import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    # Add your code to render the home page here
    return render_template('home.html')

@app.route('/detect_cancer')
def detect_cancer():
    # Render the index.html page for the "Detect Breast Cancer" link
    return render_template('index.html')

@app.route('/info')
def info():
    # Render the index.html page for the "Detect Breast Cancer" link
    return render_template('info.html')

@app.route('/predict',methods=['POST'])
def predict():
  input_features = [int(x) for x in request.form.values()]
  features_value = [np.array(input_features)]

  features_name = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
       'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
       'bland_chromatin', 'normal_nucleoli', 'mitoses']

  df = pd.DataFrame(features_value, columns=features_name)
  output = model.predict(df)

  if output == 4:
      res_val = "Breast cancer"
  else:
      res_val = "no Breast cancer"


  return render_template('index.html', prediction_text='Patient has {}'.format(res_val))

if __name__ == "__main__":
  app.run()
