import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
import cv2
import numpy as np

model2 = Sequential()
model2.add(Conv2D(filters = 16,kernel_size=2,padding='same',activation='relu',input_shape=(100,100,3)))
model2.add(MaxPooling2D(pool_size=2))
model2.add(Conv2D(filters=32,kernel_size=2,padding='same',activation='relu'))
model2.add(MaxPooling2D(pool_size=2))
model2.add(Conv2D(filters=64,kernel_size=2,padding='same',activation='relu'))
model2.add(MaxPooling2D(pool_size=2))
model2.add(Conv2D(filters=128,kernel_size=2,padding='same',activation='relu'))
model2.add(MaxPooling2D(pool_size=2))

model2.add(Dropout(0.2))
model2.add(Flatten())
model2.add(Dense(1000,activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(1200,activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(1000,activation='relu'))
model2.add(Dropout(0.2))


model2.add(Dense(3,activation='softmax'))
# model2.summary()

#compile the model
# model2.compile(loss='mean_squared_error', optimizer='rmsprop',metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint
# checkpointer = ModelCheckpoint(filepath='modelFruit.weights.best.hdf5', verbose=1,
#                                save_best_only=True)
# hist = model2.fit(x_train,y_train,batch_size=32,epochs=100,validation_data=(x_valid,y_valid),callbacks=[checkpointer],verbose=2,shuffle=True)
# load weight with best validation score
model2.load_weights('modelFruit.weights.best.hdf5')

Labels = ['Potato', 'Orange', 'Peanuts']
def classify(img):
    global Labels
    img = np.array(cv2.resize(img,(100,100)))
    img = np.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2]))
    label = model2.predict(img)
    id = np.argmax(label)
    return Labels[id]

# img = cv2.imread("Pictures/Orange/0_100.jpg")
# print(classify(img))
