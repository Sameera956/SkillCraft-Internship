import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Dataset path
dataset_path = "leapGestRecog"

images = []
labels = []
label = 0

# Load dataset
for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    for gesture in os.listdir(person_path):
        gesture_path = os.path.join(person_path, gesture)

        if not os.path.isdir(gesture_path):
            continue

        for file in os.listdir(gesture_path):
            img_path = os.path.join(gesture_path, file)

            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (64, 64))
            images.append(img)
            labels.append(label)

        label += 1

images = np.array(images, dtype="float32") / 255.0
labels = np.array(labels)

print("Total Images Loaded:", len(images))
print("Total Classes:", len(np.unique(labels)))

X_train, X_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42
)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),

    Dense(128, activation='relu'),
    Dense(len(np.unique(labels)), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

loss, accuracy = model.evaluate(X_test, y_test)

print(f"Accuracy: {accuracy*100:.2f}%")

model.save("hand_gesture_model.h5")

print("Model Saved Successfully!")