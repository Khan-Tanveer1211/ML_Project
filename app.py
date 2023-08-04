from flask import Flask,request,render_template
# request is used to capture the POST request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler   # use it beacuse the use of pickle file

app = Flask(__name__)  # __name__ gives the enterpoint in Flask

# Create Route for home page

@app.route('/')     # creating the homepage

def index():
    return render_template('index.html')

@app.route('/predictdata',method=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        pass