from imutils import paths
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Model.Simplenet import SimpleNet
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report


args = {"train": "train_image",
        "test": "test_image",
        "epochs": 50,
        "plots": "plots"}

TrainimagePaths = list(paths.list_images(args["train"]))
TestimagePaths = list(paths.list_images(args["test"]))
print("total train set:", len(TrainimagePaths))
print("total test set:", len(TestimagePaths))


data = []
labels = []

for imagePath in TrainimagePaths:

    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    # print(f"image={imagePath.split(os.path.sep)[-1]}")
    image = cv2.resize(image, (64, 64))

    data.append(image)
    labels.append(label)
    

testX = []
testY = []

for imagePath in TestimagePaths:

    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    print(f"image={imagePath.split(os.path.sep)[-1]}")
    image = cv2.resize(image, (64, 64))
    testX.append(image)
    testY.append(label)
    
(trainX, ValX, trainY, ValY) = train_test_split(data, labels,
                                                  test_size=0.30,
                                                  stratify=labels,
                                                  random_state=42)
print(f"train:{len(trainX)}\nval:{len(ValX)}\ntest:{len(testX)}")

for i in range(10):
  plt.subplot(1, 10, i+1)
  plt.imshow(trainX[i], 'gray')
plt.show()
print(trainY[0:10])

trainX = np.array(trainX, dtype="float") / 255.0
ValX = np.array(ValX, dtype="float") / 255.0
testX = np.array(testX, dtype="float") / 255.0

le = LabelEncoder()
trainY = le.fit_transform(trainY)
trainY = to_categorical(trainY, 10)
ValY = le.fit_transform(ValY)
ValY = to_categorical(ValY, 10)
testY = le.fit_transform(testY)
testY = to_categorical(testY, 10)
print(f'Trainset={type(trainX)},Trainset_label={type(trainY)}')
print(f'Valset={type(ValX)},Valset_lable={type(ValY)}')
print(f'Testset={type(testX)}, Testset_lable={type(testY)}')

# random crop
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.5,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.5,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest")

model = SimpleNet.build(width=64, height=64, depth=3,
                        classes=len(le.classes_), reg=l2(0.0002))
opt = Adam(lr=1e-4, decay=1e-4 / args["epochs"])
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

print("[INFO] training network for {} epochs...".format(
    args["epochs"]))
H = model.fit(x=aug.flow(trainX, trainY, batch_size=32),  
              validation_data=(ValX, ValY),
              batch_size=32, 
              epochs=args["epochs"], 
              verbose=1)

print("[INFO] predicting network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=le.classes_))

print("[INFO] evaluating network...")
evaluate = model.evaluate(testX, testY, verbose = 0)
print("Test loss:", evaluate[0])
print("Test Accuracy:", evaluate[1])


# plot the training accuracy
N = args["epochs"]
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plots"])

# plot the training loss 
N = args["epochs"]
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(args["plots"])


