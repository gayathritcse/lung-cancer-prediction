import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
# TensorFlow
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models, layers
# Transfer Learning
from tensorflow.keras.applications import VGG16
from keras.applications.efficientnet import preprocess_input

print("Importing libraries completed.")

path = 'Data/'

train_folder = path + "Train/"


# image size
img_width = 100
img_height = 100

# model
batch_size = 32
epochs =4


train_class_names = os.listdir(train_folder)
print("Train class names: %s" % (train_class_names))


# Declaring variables
x = []  # to store array value of the images
y = []  # to store the labels of the images

for folder in os.listdir(train_folder):
    image_list = os.listdir(train_folder + "/" + folder)
    for img_name in image_list:
        # Loading images
        img = image.load_img(train_folder + "/" + folder + "/" + img_name, target_size=(img_width, img_height))
        img = image.img_to_array(img)
        img = preprocess_input(img)

        x.append(img)
        y.append(train_class_names.index(folder))  # appending class index to the array

print("Training Dataset")

x = np.array(x)
print(x.shape)

y = to_categorical(y)
print(y.shape)


from tensorflow.keras.applications import EfficientNetB3

model_vgg16 = EfficientNetB3(weights='imagenet')
model_vgg16.summary()

print("Summary of Custom VGG16 model.\n")
input_layer = layers.Input(shape=(img_width, img_height, 3))
model_vgg16 = VGG16(weights='imagenet', input_tensor=input_layer, include_top=False)
model_vgg16.summary()
last_layer = model_vgg16.output
flatten = layers.Flatten()(last_layer)
output_layer = layers.Dense(3, activation='softmax')(flatten)
model = models.Model(inputs=input_layer, outputs=output_layer)

model.summary()
for layer in model.layers[:-1]:
    layer.trainable = False
model.summary()

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)
print("Splitting data for train and test completed.")

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Model compilation completed.")

history2 = model.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size, verbose=True, validation_data=(xtest, ytest))

print("Fitting the model completed.")

model.save("model.h5")

acc = history2.history['accuracy']
val_acc = history2.history['val_accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot Model Loss
loss_train = history2.history['loss']
loss_val = history2.history['val_loss']
plt.plot(epochs, loss_train, label='Training Loss')
plt.plot(epochs, loss_val, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


y_pred = model.predict(xtest)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)
y_test=np.argmax(ytest, axis=1)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix of EfficientNetB0')
plt.show()
