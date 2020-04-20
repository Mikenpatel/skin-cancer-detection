
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,load_model,save_model
from keras.layers import Activation,Dropout,Flatten,Conv2D,MaxPooling2D,Dense
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
import tensorflow
import os
from tensorflow import lite

image_gen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

image_gen.flow_from_directory('DermMel/train_sep/')

model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),
        input_shape=(150,150,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),
        input_shape=(150,150,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),
        input_shape=(150,150,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])



model.summary()

batch_size=15

train_image_gen=image_gen.flow_from_directory('DermMel/train_sep/',
                                              target_size=(150,150),
                                             batch_size=batch_size,
                                             class_mode='binary')




validation_image_gen=image_gen.flow_from_directory('DermMel/valid/',
                                                  target_size=(150,150),
                                                  batch_size=batch_size,
                                                  class_mode='binary')


train_image_gen.class_indices




results=model.fit_generator(train_image_gen,
                            epochs=5,
                            validation_data=validation_image_gen,
                            validation_steps=12)





history=model.fit_generator(train_image_gen,
                            steps_per_epoch=10682/40,
                            epochs=50,
                            validation_data=validation_image_gen,
                            validation_steps=3562/40)



model.save('malenoma.h5')

model.save_weights('malenoma_weights.h5')




model2=load_model('malenoma.h5')

def input_images(image):
    
        bkl_img=load_img(image,target_size=(150,150))
        bkl_img=img_to_array(bkl_img)
        bkl_img=np.expand_dims(bkl_img,axis=0)
        bkl_img=bkl_img/255
        bkl_img=model2.predict_classes(bkl_img)
        if bkl_img[0][0]==0:
            return 'Melanoma!!!(You need to see doctor)'
        else:
            return 'You are safe(feel free)'


result=input_images('Normal_mole.jpg')

print(result)


test_generator=ImageDataGenerator(rescale=1./255)

validation_image_gen=test_generator.flow_from_directory('DermMel/test/',
                                                  target_size=(150,150),
                                                  batch_size=1,
                                                  class_mode='binary')



converter=lite.TocoConverter.from_keras_model_file('malenoma.h5')


tflite_model=converter.convert()



open('melanoma.tflite','wb').write(tflite_model)







