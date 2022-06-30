
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialize CNN
classifier = Sequential()

#Step1 : Convolution
classifier.add(Convolution2D(128, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

#Step 2 : Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#ADD ANOTHER CONVOLUTIONAL LAYER
classifier.add(Convolution2D(64, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Step 3 : Flattening
classifier.add(Flatten())

#Step 4 : Full Connection
classifier.add(Dense(output_dim = 64, activation = 'relu'))		#Hidden Layer
classifier.add(Dense(output_dim = 16, activation = 'relu'))		#2nd Hidden Layer
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))	#Output Layer

#Compile CNN, Define which Algorithms will be used
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#DATA AUGMENTATION
#TAKEN SAMPLE CODE FROM KERAS DOCUMENTATION
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'mini_dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'mini_dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         epochs = 50,
                         validation_data = test_set,
                         nb_val_samples = 1000)

#MAKE A PREDICTION
import numpy as np
from keras.preprocessing import image
test_img = image.load_img('prediction\cat_or_dog_1.jpg', target_size = (64,64))
test_img = image.img_to_array(test_img)
test_img = np.expand_dims(test_img, axis=0)
pred = classifier.predict(test_img)
training_set.class_indices['cats']
if pred[0][0] == training_set.class_indices['cats']:
    print('It is a CAT')
else:
    print('It is DOG')







