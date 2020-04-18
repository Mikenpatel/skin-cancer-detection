import os

from flask import * 
from flask import jsonify 

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,load_model,save_model
from keras.layers import Activation,Dropout,Flatten,Conv2D,MaxPooling2D,Dense
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
import tensorflow
import os
from tensorflow import lite

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/',methods=['POST'])  
def success():  
    app.config['UPLOAD_FOLDER']='uploads/'
    if request.method == 'POST':  
        f =  request.files['file']  
        
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],f.filename)  ) 
        my_image_path=os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
        
        model1=load_model('malenoma.h5')
        def input_images(image):

            bkl_img=load_img(image,target_size=(150,150))
            bkl_img=img_to_array(bkl_img)
            bkl_img=np.expand_dims(bkl_img,axis=0)
            bkl_img=bkl_img/255
            bkl_img=model1.predict_classes(bkl_img)
            if bkl_img[0][0]==0:
                return 'Melanoma!!!(You need to see doctor)'
            else:
                return 'You are safe(feel free)'
            
        result=input_images(my_image_path)
        
        # return result
        # return make_response(result)
        return render_template("index.html", imagename = f.filename,path= my_image_path,result=result)  

app.run(debug=True)