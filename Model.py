import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16  # VGG16 modelini transfer öğrenme için dahil ediyoruz
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings('ignore')

img_path = 'archive/'


df_train = pd.read_csv('archive/Train.csv')
df_test = pd.read_csv('archive/Test.csv')

print(df_train.head(10))


train_ratio = 0.8
random_state = 42  

# Her sınıf için veriyi %80 eğitim ve %20 doğrulama olacak şekilde ayırma
train_data = pd.DataFrame()
val_data = pd.DataFrame()

for label, group in df_train.groupby('ClassId'):
    train_samples = group.sample(frac=train_ratio, random_state=random_state)
    val_samples = group.drop(train_samples.index)
    
    train_data = pd.concat([train_data, train_samples])
    val_data = pd.concat([val_data, val_samples])

print("Eğitim verisi boyutu:", train_data.shape)
print("Doğrulama verisi boyutu:", val_data.shape)

# Görüntü işleme ve boyutlandırma fonksiyonları
def resize_images_from_df(df, img_path, target_width, target_height, maintain_aspect_ratio=True):
    resized_images = []
    
    for img_name in df['Path']:
        img = cv2.imread(os.path.join(img_path, img_name))
        
        if maintain_aspect_ratio:
            (h, w) = img.shape[:2]
            if h > w:
                new_h = target_height
                new_w = int(w * (target_height / h))
            else:
                new_w = target_width
                new_h = int(h * (target_width / w))
            
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            top_padding = (target_height - img_resized.shape[0]) // 2
            bottom_padding = target_height - img_resized.shape[0] - top_padding
            left_padding = (target_width - img_resized.shape[1]) // 2
            right_padding = target_width - img_resized.shape[1] - left_padding
            img_padded = cv2.copyMakeBorder(img_resized, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        else:
            img_padded = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        
        resized_images.append(img_padded)
    
    return np.array(resized_images)


train_images = resize_images_from_df(train_data, img_path, 32, 32)
val_images = resize_images_from_df(val_data, img_path, 32, 32)


def normalize_resized_images(images):
    return images.astype('float32') / 255.0

train_images_normalized = normalize_resized_images(train_images)
val_images_normalized = normalize_resized_images(val_images)

# Veri artırma işlemi
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=False,
    fill_mode='nearest'
)

# CNN modelini oluşturma
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(43, activation='softmax')  # 43 sınıfı için çıkış katmanı
])

# Alternatif: Transfer öğrenme için VGG16 kullanma
# model = Sequential([
#     VGG16(include_top=False, weights='imagenet', input_shape=(32, 32, 3)),
#     GlobalAveragePooling2D(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(43, activation='softmax')
# ])

# Modeli derleme
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modelin yapısını gözden geçirme
model.summary()

# Modeli eğitme
history = model.fit(datagen.flow(train_images_normalized, train_data['ClassId'].values, batch_size=32),
                    epochs=5,
                    validation_data=(val_images_normalized, val_data['ClassId'].values))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Make predictions on the validation set
val_predictions = np.argmax(model.predict(val_images_normalized), axis=1)
val_true_labels = val_data['ClassId'].values

# Calculate the metrics
accuracy = accuracy_score(val_true_labels, val_predictions)
precision = precision_score(val_true_labels, val_predictions, average='weighted')
recall = recall_score(val_true_labels, val_predictions, average='weighted')
f1 = f1_score(val_true_labels, val_predictions, average='weighted')

# Print the classification report for detailed metrics per class
print("Classification Report:")
print(classification_report(val_true_labels, val_predictions, target_names=[str(i) for i in range(43)]))

# Print the overall metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
