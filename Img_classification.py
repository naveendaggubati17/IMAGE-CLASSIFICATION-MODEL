import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Dense
(tr_img,tr_labels),(testing_img,testing_labels)=datasets.cifar10.load_data()
tr_img.shape
tr_labels.shape
tr_labels[0]
testing_img.shape
testing_labels.shape
tr_img_scaled,testing_img_scaled=tr_img/255,testing_img/255
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
tr_labels[0][0]
plt.figure(figsize=(10,10))
for i in range(100):
    plt.subplot(10,10,i + 1)
    plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.6,hspace=0.9)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(tr_img[i])
    plt.xlabel(class_names[tr_labels[i][0]])
plt.show()

model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(layers.Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(tr_img_scaled,tr_labels,epochs=10, validation_data=(testing_img_scaled,testing_labels))
loss,accuracy=model.evaluate(testing_img_scaled,testing_labels)
print('loss=',loss)
print('accuracy=',accuracy)
model.save("Image_classification")
model=models.load_model('Image_classification')
import pickle
pickle.dump(model,open('img_class.pkl','wb'))
pickled_model=pickle.load(open('img_class.pkl','rb'))
img_test1=cv.imread("C:/Users/Shruti/Downloads/Bird.jpg")
img_test1
plt.imshow(img_test1)

img_test1=cv.cvtColor(img_test1,cv.COLOR_BGR2RGB)
img_test1_resized = cv.resize(img_test1, (32, 32))
img_test1_scaled=img_test1_resized/255
img_test1_final=np.reshape(img_test1_scaled,(1,32,32,3))
prediction=model.predict(img_test1_final)
prediction
index=np.argmax(prediction)
class_names[index]


OUTPUT :

Bird
