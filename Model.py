import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

img_path = 'archive/'

# Veri kümesini yükle
df_train = pd.read_csv('archive/Train.csv')
df_test = pd.read_csv('archive/Test.csv')

print(df_train.head(10))

# Eğitim ve doğrulama veri oranlarını belirleme
train_ratio = 0.8
random_state = 42  # Rastgelelik için sabit bir değer

# Her sınıf için veriyi %80 eğitim ve %20 doğrulama olacak şekilde ayırma
train_data = pd.DataFrame()  # Eğitim verisi için boş bir DataFrame
val_data = pd.DataFrame()  # Doğrulama verisi için boş bir DataFrame

# Sınıf etiketine göre gruplandırma ve her gruptan %80/%20 bölme
for label, group in df_train.groupby('ClassId'):  # 'ClassId' sınıf etiketinin olduğu sütun adı
    train_samples = group.sample(frac=train_ratio, random_state=random_state)  # %80 eğitim
    val_samples = group.drop(train_samples.index)  # Geriye kalan %20 doğrulama
    
    train_data = pd.concat([train_data, train_samples])  # Eğitim verisine ekleme
    val_data = pd.concat([val_data, val_samples])  # Doğrulama verisine ekleme

# Sonuçların kontrolü
print("Eğitim verisi boyutu:", train_data.shape)
print("Doğrulama verisi boyutu:", val_data.shape)

# Her bir sınıftaki dağılımın doğrulanması
print("Eğitim seti sınıf dağılımı:\n", train_data['ClassId'].value_counts())
print("Doğrulama seti sınıf dağılımı:\n", val_data['ClassId'].value_counts())

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

# Eğitim ve doğrulama verilerinin boyutlandırılması
train_images = resize_images_from_df(train_data, img_path, 32, 32)
val_images = resize_images_from_df(val_data, img_path, 32, 32)

# Görüntüleri normalize etme
def normalize_resized_images(images):
    return images.astype('float32') / 255.0

train_images_normalized = normalize_resized_images(train_images)
val_images_normalized = normalize_resized_images(val_images)

# Veri artırma işlemi
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=False,
    fill_mode='nearest'
)

# Augment edilmiş görüntüleri görselleştirme
augmented_images = datagen.flow(train_images_normalized, batch_size=1)

for i in range(5):
    batch = next(augmented_images)
    image = batch[0]
    plt.imshow(image)
    plt.axis('off')
    plt.show()
