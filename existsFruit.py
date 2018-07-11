import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os

os.chdir('Pictures_exist')
fruitTypes = os.listdir(".")
# print(fruitTypes)
os.chdir('..')


def prepare_X_Y(path):
    lbl = 0
    fruitList = glob.glob(path+fruitTypes[0]+'/*')
    # print(fruitList)
    # print('\n\n')
    fruits = np.array([np.array(cv2.resize(cv2.imread(fruit),(100,100))) for fruit in fruitList])
    x_y = np.array([[img,lbl] for img in fruits])
    # print(x_y.shape)
    lbl += 1
    rest = fruitTypes[1:]
    # print(rest)
    for f in rest:
        # print(f)
        fruitList = glob.glob(path + f + '/*')
        # print(fruitList)
        fruits = np.array([np.array(cv2.resize(cv2.imread(fruit),(100,100))) for fruit in fruitList])
        x_y = np.vstack((x_y,np.array([[img,lbl] for img in fruits])))
        # print(x_y.shape)
        lbl += 1
    # print(x_y.shape)
    # print x_y[:,1]
    np.random.shuffle(x_y)
    return x_y
x_y_train = prepare_X_Y('Pictures_exist/')
x_y_test = prepare_X_Y('Validation_exist/')
for i in range(36):
    plt.subplot(6,6,i+1,xticks=[],yticks=[])
    plt.imshow(x_y_train[i,0][:,:,[2,1,0]],interpolation='nearest',aspect='auto')
plt.show()


x_y_train = x_y_train
x_y_test = x_y_test
#
# print(x_y_train.shape)
# print(x_y_test.shape)

def divide_img_lbl(data):
    """ split data into image and label"""
    x = []
    y = []
    for [item,lbl] in data:
        x.append(item)
        y.append([lbl])
    x = np.array(x)
    y = np.array(y)
    return x,y

x_train,y_train = divide_img_lbl(x_y_train)

# print(x_train.shape)
# print(y_train.shape)

x_test,y_test = divide_img_lbl(x_y_test)

print(x_test.shape)
print(y_test.shape)

# rescale [0,255]  --> [0,1]
x_train = x_train[0:10000].astype('float32')/255
x_test = x_test[0:10000].astype('float32')/255
y_train = y_train[0:10000]
y_test = y_test [0:10000]
# print x_train[0]
bins = np.arange(0,36)
lbl = fruitTypes
# import seaborn as sns
# sns.set()
# plt.hist(y_train,bins,ec='black')
# plt.xlabel('Labels')
# plt.ylabel('Frequency')
# plt.xticks(bins,lbl)
# plt.show()
import keras

# one Hot_Encoding
# print y_train
num_classes = len(fruitTypes)
# print len(y_train)
# print y_train.max(),y_train.min()
# print y_test.max(),y_test.min()
# print num_classes
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# break trainset into trainset and validationset
#take first 80% as train and 20% as validation
uptill = int(len(x_train)*0.8)
(x_train,x_valid) = x_train[:uptill],x_train[uptill:]
(y_train,y_valid) = y_train[:uptill],y_train[uptill:]

print(x_train.shape)
print(x_valid.shape)
print(y_train.shape)
print(y_valid.shape)


# create the model
train = False
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout

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


model2.add(Dense(2,activation='softmax'))
# model2.summary()

#compile the model
model2.compile(loss='mean_squared_error', optimizer='rmsprop',metrics=['accuracy'])
if train:
    from keras.callbacks import ModelCheckpoint
    checkpointer = ModelCheckpoint(filepath='modelExistFruit.weights.best.hdf5', verbose=1,
                                   save_best_only=True)
    hist = model2.fit(x_train,y_train,batch_size=32,epochs=100,validation_data=(x_valid,y_valid),callbacks=[checkpointer],verbose=2,shuffle=True)

# load weight with best validation score
model2.load_weights('modelExistFruit.weights.best.hdf5')

score = model2.evaluate(x_test,y_test,verbose=0)
print('test accuracy',score[1])

Labels = fruitTypes
print(Labels)
print(x_test.shape)
y_hat = (model2.predict(x_test))
for i in range(25):
    plt.subplot(5,5,i+1,xticks=[],yticks=[])
    plt.imshow(np.squeeze(x_test[i][:,:,[2,1,0]]))
    pred_idx = np.argmax(y_hat[i])
    true_idx = np.argmax(y_test[i])
    plt.title("{} ({})".format(Labels[pred_idx],Labels[true_idx] ),color=("green" if pred_idx == true_idx else "red"))
# plt.tight_layout()
plt.show()
