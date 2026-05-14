import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout

#generators - batches

train_ds = keras.utils.image_dataset_from_directory(
    directory = BASE_DIR
    labels = 'inferred',
    label_mode = 'int',
    batch_size = 32,
    image_size = (256,256)
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory = BASE_DIR,
    labels = 'inferred',
    label_mode = 'int',
    batch_size = 32,
    image_size = (256,256)
)

#Normalize

def normalize(image, label):
    image = tf.cast(image/255. ,tf.float32)
    return image, label

train_ds = train_ds.map(normalize)
validation_ds = validation_ds.map(normalize)

#create CNN Model

model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), padding = 'valid', activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))

model.add(Conv2D(64, kernel_size=(3,3), padding = 'valid', activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))

model.add(Conv2D(128, kernel_size=(3,3), padding = 'valid', activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))

model.add(Flatten())

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, epochs=10, validation_data=validation_ds)




# Validation

import matplotlib.pyplot as plt

plt.plot (history.history['accuracy'], color='red', label = 'train')
plt.plot (history.history['val_accuracy'], color='blue', label = 'validation')
plt.legend()
plt.show()

# Loss

plt.plot(history.history['loss'], color='red', label='train')
plt.plot(history.history['val_loss'], color = 'blue', label ='validation')
plt.legend()
plt.show()

# Prediction

import cv2

test_img = cv2.imread('/content/dog.jpg')
plt.imshow (test_img)
test_img.shape
test_img = cv2.resize(test_img, (256,256))
test_input = test_img.reshape((1,256,256,3))

model.predict(test_input)
