import os
import shutil
import random

source_dir = "."      # original dataset
target_dir = "dataset_split"  # new split dataset

classes = ["dry", "normal", "oily"]
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

random.seed(42)

for cls in classes:
    cls_path = os.path.join(source_dir, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    for split, split_imgs in zip(
        ["train", "val", "test"],
        [train_imgs, val_imgs, test_imgs]
    ):
        split_path = os.path.join(target_dir, split, cls)
        os.makedirs(split_path, exist_ok=True)

        for img in split_imgs:
            src = os.path.join(cls_path, img)
            dst = os.path.join(split_path, img)
            shutil.copy(src, dst)

    print(f"{cls}: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

print("Dataset split completed successfully.")

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1
)

val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    "dataset_split/train",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_data = val_gen.flow_from_directory(
    "dataset_split/val",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_data = test_gen.flow_from_directory(
    "dataset_split/test",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1
)

val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    "dataset_split/train",
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical'
)

val_data = val_gen.flow_from_directory(
    "dataset_split/val",
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical'
)

test_data = test_gen.flow_from_directory(
    "dataset_split/test",
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical',
    shuffle=False
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    train_data,
    epochs=25,
    validation_data=val_data
)
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.show()

test_loss, test_acc = model.evaluate(test_data)
print("Test Accuracy:", test_acc)

model.save("skin_type_mobilenetv2.h5")

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

pred = model.predict(test_data)
y_pred = np.argmax(pred, axis=1)
y_true = test_data.classes

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['dry','normal','oily']))

import cv2
import numpy as np

img_path = "C:\\Users\\swati\\OneDrive\\Desktop\\dataset\\dry\\dry13.jpeg"  # change image
img = cv2.imread(img_path)
img = cv2.resize(img, (224,224))
img = img.astype('float32') / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
class_names = ['dry','normal','oily']

print("Predicted Skin Type:", class_names[np.argmax(prediction)])

