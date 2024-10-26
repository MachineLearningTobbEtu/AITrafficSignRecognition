import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import warnings

warnings.filterwarnings('ignore') 


from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D,Dense,Flatten,Input,MaxPooling2D,Dropout,BatchNormalization,Reshape
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from sklearn.metrics import confusion_matrix,classification_report
from keras.utils import to_categorical


img_path='archive/Train/'

df_train = pd.read_csv('archive/Train.csv')

df_test = pd.read_csv('archive/Test.csv')

df_meta  = pd.read_csv('archive/Meta.csv')

print(df_train.head(10))



# labels=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42']

#img_path='C:/Users/ASUS/Desktop/archive/'

#print(df3.head(10))
"""
# Örnek bir veri kümesi yükleme ve toplu normalizasyon
def load_and_normalize_images(df, img_path):
    image_list = []
    
    for img_name in df['Path']:  # DataFrame'deki 'Path' sütunu
        # Görüntüyü yükle
        img = cv2.imread(os.path.join(img_path, img_name))
        
        # Görüntüyü ekle (0-255 aralığında ham görüntü)
        image_list.append(img)
    
    # Listeyi NumPy dizisine dönüştür
    images = np.array(image_list)
    
    # Tüm görüntüleri vektörize olarak normalleştir (0-255 -> 0-1)
    images_normalized = images.astype('float32') / 255.0
    
    return images_normalized
"""
# Tüm görüntüleri yükle ve normalize et
#normalized_images = load_and_normalize_images(df_train, img_path)
#   normalized_images = load_and_normalize_images(df_test, img_path+'Test/')

def resize_images_from_df(df, img_path, target_width, target_height, maintain_aspect_ratio=True):
    resized_images = []
    
    for img_name in df['Path']:  # DataFrame'deki 'Path' sütunu ile dosya yollarını alıyoruz
        # Görüntü dosyasını yükle
        img = cv2.imread(os.path.join(img_path, img_name))
        
        # Eğer aspect ratio korunacaksa
        if maintain_aspect_ratio:
            # Oranı koruyarak yeniden boyutlandırma
            (h, w) = img.shape[:2]
            if h > w:
                new_h = target_height
                new_w = int(w * (target_height / h))
            else:
                new_w = target_width
                new_h = int(h * (target_width / w))
            
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Padding ekleyerek görüntüyü tam boyut yapma (64x64 gibi)
            top_padding = (target_height - img_resized.shape[0]) // 2
            bottom_padding = target_height - img_resized.shape[0] - top_padding
            left_padding = (target_width - img_resized.shape[1]) // 2
            right_padding = target_width - img_resized.shape[1] - left_padding
            img_padded = cv2.copyMakeBorder(img_resized, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        else:
            # Aspect ratio'yu korumadan direkt yeniden boyutlandırma
            img_padded = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        
        # Yeniden boyutlandırılmış görüntüyü listeye ekle
        resized_images.append(img_padded)
    
    # Tüm görüntüler NumPy dizisine dönüştürülür
    return np.array(resized_images)


def normalize_resized_images(resized_images):
    # Görüntüleri normalize et (0-255 -> 0-1)
    normalized_images = resized_images.astype('float32') / 255.0
    return normalized_images