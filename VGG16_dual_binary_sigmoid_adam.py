## go into three sets of data (train,, dev, test) with images, the first latter of file name is the category.
# convert the label into a dictionary: train_label, dev_label, test-label. The list is ordered amse as image order
# convert image into numpy array, and store in dictionary:  train_data, dev_data, test_data

def imageload(mypath_train, mypath_dev, mypath_test, filename_label_position, dimheight,dimwidth):
# resize to (dimheight,dimwidth) using cv
# path to training, development, test=mypath_train, mypath_dev, mypath_test
    import os
    import cv2
    import matplotlib
    import numpy as np


# training set
    train_label={}          # creat train_label and train_data dictionary
    train_data={}
    set_of_file=set()
    i=0
    for root, dir, files in os.walk(mypath_train):   # walk through all files in the folder
        for file in files:
            if file[0]==".":
                continue
            if file in set_of_file:
                continue
            set_of_file.add(file)      # make sure file not duplicate
            mypath_file=os.path.join(mypath_train,file)   # define the path with directory
            img = cv2.imread(mypath_file)   # read with OPenCV first
            print(mypath_file)
            
    # read with Open CV. note: expect color images
            resized_img = cv2.resize(img, (dimheight,dimwidth),interpolation = cv2.INTER_AREA)  # resize with OpenCV
            train_label[i]= file[filename_label_position]   # obtain label of data (the character position is used to indicate categories)
            
            img1=np.array(resized_img)  # numparize the resized image into a numpay array 
            train_data[i]=img1 #store the numpay array into a dictionary (here this is train_data)
            i+=1
    
#same idea repeated below for development set and test set .........
            
# development set
    dev_label={}
    dev_data={}
    set_of_file=set()
    i=0

    for root, dir, files in os.walk(mypath_dev):
        for file in files:
            
            if file[0]==".":
                continue
            if file in set_of_file:
                continue
            set_of_file.add(file)
            mypath_file=os.path.join(mypath_dev,file)
            print(mypath_file)
            img = cv2.imread(mypath_file, cv2.IMREAD_COLOR)
        
            resized_img = cv2.resize(img, (dimheight,dimwidth),interpolation = cv2.INTER_AREA)
            dev_label[i]= file[filename_label_position]
            img1=np.array(resized_img)
            dev_data[i]=img1
            i+=1

# test set
    test_label={}
    test_data={}
    set_of_file=set()
    i=0

    for root, dir, files in os.walk(mypath_test):
        for file in files:
            if file[0]==".":
                continue
            if file in set_of_file:
                continue
            set_of_file.add(file)
            mypath_file=os.path.join(mypath_test,file)
            print(mypath_file)
            img = cv2.imread(mypath_file, cv2.IMREAD_COLOR)
            resized_img = cv2.resize(img, (dimheight,dimwidth),interpolation = cv2.INTER_AREA)
            test_label[i]= file[filename_label_position]
            
            img1=np.array(resized_img)
            test_data[i]=img1
            i+=1

    return train_label,train_data,dev_label,dev_data,test_label,test_data


## convert dictionaries train_label, dev_label, test-label,  (train_data, dev_data, test_data into numpy array
## assume that all images are same shape

def numparize(train_label,train_data,dev_label,dev_data,test_label,test_data):

    #1. check if label and data set have same number
    import numpy as np

    if (len(train_label)!=len(train_data)) or (len(dev_label)!=len(dev_data)) or (len(test_label)!=len(test_data)):
        print("error in data and label size")
        exit()
    

    #2 obtaining size of the dictionaries in numpy arrays
    len_train=len(train_label)   ##all these sets were dictionary
    len_dev=len(dev_label)
    len_test=len(test_label)
    
    imageshape=np.shape(train_data[0])   ## train_data is supposed to be a dictionary of (index, numpy array of resized image). 
                                        ## we only look at the first one
    height, width, channel=imageshape      ## the height and width were obtained from the shaoe abive
                                                  #expect color image
   

    #3 Initializing new numpy array
    train_label_np=np.zeros((len_train,1))
    dev_label_np=np.zeros((len_dev,1))
    test_label_np=np.zeros((len_test,1))

    train_data_np=np.zeros((len_train,height, width,channel))
    dev_data_np=np.zeros((len_dev,height, width,channel))
    test_data_np=np.zeros((len_test,height, width,channel))
    
    # 4  converting data to numpy
    
    for i in range(len_train):
        train_data_np[i,:,:,:]=train_data[i]
        train_label_np[i]=train_label[i]
    for i in range(len_dev):
        dev_data_np[i,:,:,:]=dev_data[i]
        dev_label_np[i]=dev_label[i]
    for i in range(len_test):
        test_data_np[i,:,:,:]=test_data[i]
        test_label_np[i]=test_label[i]
    
    return train_label_np,train_data_np,dev_label_np,dev_data_np,test_label_np,test_data_np

def describe(train_label_np,train_data_np,dev_label_np,dev_data_np,test_label_np,test_data_np):
    #this takes numpy array training data and return: height, width, train_size, dev_size, test_size,categories, category_num
    import numpy as np
    train_size,height,width,channel=np.shape(train_data_np)

    dev_size,_,_,_=np.shape(dev_data_np)
    test_size,_,_,_=np.shape(test_data_np)
    categories=np.unique(train_label_np)
    category_num=len(categories)
    return height, width, channel, train_size, dev_size, test_size,categories, category_num


## Main Program go here

### Part 1: extract first digit data (Normal or abnormal)

# define the paths to training, development, and test sets
mypath_train="./data/train"
mypath_dev="./data/dev"
mypath_test="./data/test"

# call to load images into initial arrays, 
#the data file digits preceding the "-" are the label, for instance filename_label_position=0, 
# means the position 0 (first digit) is the label of interest
#in our dataset, first digit is normal (=0), vs abnormal (=1)
# second digit is which artery (0=left main, 1=LAD, 2= LCx, 3= RCA)
#the picture is converted to dimenston height/difeth set below, using CV2)
train_labels,train_data,dev_labels,dev_data,test_labels,test_data= imageload(mypath_train, mypath_dev, mypath_test, filename_label_position=0, dimheight=128,dimwidth=128 )

#numparize the array
train_label_np,train_data_np,dev_label_np,dev_data_np,test_label_np,test_data_np= numparize(train_labels,train_data,dev_labels,dev_data,test_labels,test_data)

# Describe the data
height, width, channel, train_size, dev_size, test_size,categories, category_num=describe(train_label_np,train_data_np,dev_label_np,dev_data_np,test_label_np,test_data_np)

print("Sanity Check: for first set of input")
print("height=",height, " width=", width, " channel=", channel)
print("training set size=", train_size, "   development set size=", dev_size, "   test set size=", test_size)
print("categories=", categories)
print("number of category=", category_num)

print("------------")


## connecting to Keras below


### Data extraction Part 2: extract first digit data (Left main or LAD or LCx or RCA)

train_labels1,train_data1,dev_labels1,dev_data1,test_labels1,test_data1= imageload(mypath_train, mypath_dev, mypath_test, filename_label_position=1, dimheight=64,dimwidth=64 )

#numparize the array
train_label_np1,train_data_np1,dev_label_np1,dev_data_np1,test_label_np1,test_data_np1= numparize(train_labels1,train_data1,dev_labels1,dev_data1,test_labels1,test_data1)

# Describe the data
height1, width1, channel1, train_size1, dev_size1, test_size1,categories1, category_num1=describe(train_label_np1,train_data_np1,dev_label_np1,dev_data_np1,test_label_np1,test_data_np1)
print("Sanity Check: for second set of input")
print("height=",height1, " width=", width1, " channel=", channel1)
print("training set size=", train_size1, "   development set size=", dev_size1, "   test set size=", test_size1)
print("categories=", categories1)
print("number of category=", category_num1)


from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False, 
                 input_shape=(height, width, channel))


# flatten the image and ensure it can go into Keras properly
# notice the name is different from train_data.
train_images = train_data_np.reshape((train_size, height,width,channel))
train_images = train_images.astype('float32') / 255
dev_images = dev_data_np.reshape((dev_size, height,width,channel))
dev_images = dev_images.astype('float32') / 255
test_images = test_data_np.reshape((test_size, height, width,channel))
test_images = test_images.astype('float32') / 255

train_images1 = train_data_np1.reshape((train_size1, height1, width1, channel1))
train_images1 = train_images1.astype('float32') / 255
dev_images1 = dev_data_np1.reshape((dev_size1, height1,width1,channel1))
dev_images1 = dev_images1.astype('float32') / 255
test_images1 = test_data_np1.reshape((test_size1, height1, width1,channel1))
test_images1 = test_images1.astype('float32') / 255


from keras.utils import to_categorical
import numpy as np

# general categoricla variable
# notice I am add "s" at train_label, to distinguish from earlier numpy.
train_labels = to_categorical(train_label_np)
dev_labels = to_categorical(dev_label_np)
test_labels = to_categorical(test_label_np)

train_labels1 = to_categorical(train_label_np1)
dev_labels1 = to_categorical(dev_label_np1)
test_labels1 = to_categorical(test_label_np1)

# joining the two numpy arrray of one-shot into one
train_label_shape=np.shape(train_labels)    # describe the train-label and label-1
train_label1_shape=np.shape(train_labels1)
oneshot_column_merged=(train_label_shape[1]-1)+train_label1_shape[1]  #so how many columns do we need in tatpe
sample_size=train_label_shape[0]    # sample size
train_labels_merged=np.zeros((sample_size,oneshot_column_merged))  #create new numpy array of 0

train_labels_merged[0:sample_size,0:(train_label_shape[1]-1)]=train_labels[0:sample_size,1].reshape((sample_size,1))    # put the smaller array into the bigger array
train_labels_merged[0:sample_size,(train_label_shape[1]-1):oneshot_column_merged]=train_labels1

# joining the two numpy arrray of one-shot into one
dev_label_shape=np.shape(dev_labels)    # describe the train-label and label-1
dev_label1_shape=np.shape(dev_labels1)
oneshot_column_merged=(dev_label_shape[1]-1)+dev_label1_shape[1]  #so how many columns do we need in tatpe
sample_size=dev_label_shape[0]    # sample size
dev_labels_merged=np.zeros((sample_size,oneshot_column_merged))  #create new numpy array of 0

dev_labels_merged[0:sample_size,0:(dev_label_shape[1]-1)]=dev_labels[0:sample_size,1].reshape((sample_size,1))   # put the smaller array into the bigger array
dev_labels_merged[0:sample_size,(dev_label_shape[1]-1):oneshot_column_merged]=dev_labels1

# joining the two numpy arrray of one-shot into one
test_label_shape=np.shape(test_labels)    # describe the train-label and label-1
test_label1_shape=np.shape(test_labels1)
oneshot_column_merged=(test_label_shape[1]-1)+test_label1_shape[1]  #so how many columns do we need in tatpe
sample_size=test_label_shape[0]    # sample size
test_labels_merged=np.zeros((sample_size,oneshot_column_merged))  #create new numpy array of 0

test_labels_merged[0:sample_size,0:(test_label_shape[1]-1)]=test_labels[0:sample_size,1].reshape((sample_size,1))    # put the smaller array into the bigger array
test_labels_merged[0:sample_size,(test_label_shape[1]-1):oneshot_column_merged]=test_labels1
for i in range(sample_size):
    print(test_labels_merged[i,:],",",test_label_np[i],",",test_label_np1[i])


print(conv_base.summary())


from keras import models
from keras import layers
from keras import optimizers

# this code takes VGG16, and then add on a lauer of softmax to classify stuff.

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(5, activation='sigmoid'))

# Below freeze all the the VGG16 as untrainable except the last few layers. Look at structure of the VGG16 listed above

conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
# notice the very low learning rate set below.  lr=0.001 does not work

INIT_LR=0.0001
adam = optimizers.Adam(lr=INIT_LR)
model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['accuracy'])
print("dimension of train_images, train_labels,dev_images, dev_labels")
print(train_images.shape, train_labels_merged.shape, dev_images.shape,  dev_labels_merged.shape)
history=model.fit(train_images, train_labels_merged, epochs=30, batch_size=16,validation_data=(dev_images, dev_labels_merged))


# save the pareameter of the model
model.save('PL_CV_engine2.h5')


# print the graph of learning history for diagnostic purpose.

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Disease Training and validation accuracy (Dev set)')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Disease Training and validation loss (Dev Set)')
plt.legend()

plt.show()


#validation of small network
test_loss, test_acc = model.evaluate(x= test_images,y= test_labels_merged,  batch_size=None, verbose=1)
print('test_accuracy for test set:', test_acc)

#print test images and results using matplotlib

import numpy as np
import cv2
from matplotlib import pyplot as plt

CAD={0:"Normal",1:"Stenotic"}
Anatomy={0:"Left Main or LAD/LCx",1:"LAD",2:"LCx",3:"RCA"}
prediction = model.predict(test_images)

counter_stenosis=0
counter_anatomy=0

for i in range(test_data_np.shape[0]):
    print("test images #",i)
    plt.imshow(test_data[i], cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    
    prediction_row=prediction[i]
    print(prediction_row)
    
    cad=int(prediction_row[0]>0.5)     # this is specific for one shot
    anatomy=np.argmax(prediction_row[1:5])  #this is specific for one shot
    
    if (cad==int(test_label_np[i])):  
        counter_stenosis=counter_stenosis+1
    if (anatomy==int(test_label_np1[i])):
        counter_anatomy=counter_anatomy+1
    
    print("deciphered:",cad,anatomy)
    print("Predicted",CAD[cad],Anatomy[anatomy])
    print("Actual",CAD[int(test_label_np[i])],Anatomy[int(test_label_np1[i])])
    print("--------")
correct_stenosis_ratio=counter_stenosis/test_data_np.shape[0]
correct_anatomy_ratio=counter_anatomy/test_data_np.shape[0]
print("stenosis correct ratio:",correct_stenosis_ratio,"; anatomy correct ratio:",correct_anatomy_ratio)
