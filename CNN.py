import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Path to dataset
dataset_path = 'path_of_dataset'  # Change this to your dataset path

# Parameters
img_size = 128
num_classes = 5

# Load dataset
data = []
labels = []
classes = os.listdir(dataset_path)

# Check each subdirectory for images
for split in ['train', 'test', 'val']:  # Assuming your dataset is split into these folders
    split_path = os.path.join(dataset_path, split)
    for class_id, class_name in enumerate(os.listdir(split_path)):
        class_path = os.path.join(split_path, class_name)
        if os.path.isdir(class_path):  # Check if the path is a directory
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):  # Ensure it's an image file
                    img = cv2.imread(img_path)
                    if img is not None:  # Check if image is loaded successfully
                        img = cv2.resize(img, (img_size, img_size))
                        data.append(img)
                        labels.append(class_id)
                    else:
                        print(f"Failed to load image: {img_path}")
                else:
                    print(f"Skipping non-image file: {img_path}")

data = np.array(data)
labels = np.array(labels)

# Normalize pixel values
data = data / 255.0

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")

#MOdel Building
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

#evaluating the model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# Evaluate model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Accuracy
accuracy = accuracy_score(y_test, y_pred_classes)
print(f'Accuracy: {accuracy}')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


#exporting model into .h5 extention
from tensorflow.keras.models import load_model

# Save the model
model.save('/content/drive/MyDrive/H5model/my_model2.h5')

