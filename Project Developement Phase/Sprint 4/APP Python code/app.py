from keras.models import load_model
from flask import Flask,render_template,request
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
from keras.utils import load_img,img_to_array
import pandas as pd
import numpy as np
import base64

app = Flask(__name__)

m = load_model('Fruits.h5')
model = load_model('veg_model.h5')

def show(img):
    img = img
    data = base64.b64encode(img.getbuffer()).decode()
    return data

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/prediction')
def prediction():
    h = 'hidden'
    return render_template('predict.html',h=h)

@app.route('/predict',methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        pic = show(f)
        basepath=os.path.dirname(__file__)     
        file_path=os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)
        print(f)
        img=load_img(file_path,target_size=(128,128))
        x=img_to_array(img)
        x=np.expand_dims(x,axis=0)
        plant=request.form['select']
        print(plant)
        if(plant=="veg"):
            preds=(model.predict(x) > 0.5).astype("int32")
            print(preds)
            preds = preds[0]
            preds = np.where(preds == 1)
            df=pd.read_excel('precautions-veg.xlsx')
            data = df.iloc[preds]
            print(data)
            return render_template('predict.html',data = data,show = pic)
        else:
            preds=(m.predict(x) > 0.5).astype("int32")
            preds = preds[0]
            preds = np.where(preds == 1)
            df=pd.read_excel('precautions-fruits.xlsx')
            data = df.iloc[preds]
            print(data)
            return render_template('predict.html',data = data,show = pic)
    




if __name__=='__main__':
    app.run()