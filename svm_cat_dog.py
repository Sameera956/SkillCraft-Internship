import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

data = []
labels = []

dataset = "train"

categories = ["cats", "dogs"]

for label, category in enumerate(categories):
    path = os.path.join(dataset, category)

    for img in os.listdir(path)[:500]:   # Load first 500 images
        img_path = os.path.join(path, img)

        image = cv2.imread(img_path)

        if image is None:
            continue

        image = cv2.resize(image, (64, 64))
        image = image.flatten()

        data.append(image)
        labels.append(label)

X = np.array(data)
y = np.array(labels)

X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel='linear')

print("Training Model...")
model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, prediction,
      target_names=["Cat", "Dog"]))