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
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator


warnings.filterwarnings("ignore")

# ---------------------------Veriyi Oku---------------------------
img_path = "archive/"


df_train = pd.read_csv("archive/Train.csv")
df_test = pd.read_csv("archive/Test.csv")

print(df_train.head(10))

train_ratio = 0.8
random_state = 42

#Comparison Parameters
plotNoComprision=False
plotDenseUnitComparison=True
plotConvFiltersComparison=False
plotAugmentationComparison=False
plotNormalizationComparison=False
plotEpochComparison=False

# ---------------------------Veri Okuma Bitti---------------------------
# ---------------------------------------------------------------------------------------------------------
# ---------------------------Veriyi Ayır---------------------------
# Her sınıf için veriyi %80 eğitim ve %20 doğrulama olacak şekilde ayırma
train_data = pd.DataFrame()
val_data = pd.DataFrame()

for label, group in df_train.groupby("ClassId"):
    train_samples = group.sample(frac=train_ratio, random_state=random_state)
    val_samples = group.drop(train_samples.index)

    train_data = pd.concat([train_data, train_samples])
    val_data = pd.concat([val_data, val_samples])

print("Eğitim verisi boyutu:", train_data.shape)
print("Doğrulama verisi boyutu:", val_data.shape)

# ---------------------------Veriyi Ayırma Bitti---------------------------
# ---------------------------------------------------------------------------------------------------------
# ---------------------------Görüntü resize ve normalize et---------------------------


# Görüntü işleme ve boyutlandırma fonksiyonları
def resize_images_from_df(
    df, img_path, target_width, target_height, maintain_aspect_ratio=True
):
    resized_images = []

    for img_name in df["Path"]:
        img = cv2.imread(os.path.join(img_path, img_name))

        if maintain_aspect_ratio:
            (h, w) = img.shape[:2]
            if h > w:
                new_h = target_height
                new_w = int(w * (target_height / h))
            else:
                new_w = target_width
                new_h = int(h * (target_width / w))

            img_resized = cv2.resize(
                img, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )

            top_padding = (target_height - img_resized.shape[0]) // 2
            bottom_padding = target_height - img_resized.shape[0] - top_padding
            left_padding = (target_width - img_resized.shape[1]) // 2
            right_padding = target_width - img_resized.shape[1] - left_padding
            img_padded = cv2.copyMakeBorder(
                img_resized,
                top_padding,
                bottom_padding,
                left_padding,
                right_padding,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )

        else:
            img_padded = cv2.resize(
                img, (target_width, target_height), interpolation=cv2.INTER_LINEAR
            )

        resized_images.append(img_padded)

    return np.array(resized_images)


train_images = resize_images_from_df(train_data, img_path, 32, 32)
val_images = resize_images_from_df(val_data, img_path, 32, 32)


def normalize_resized_images(images):
    return images.astype("float32") / 255.0


train_images_normalized = normalize_resized_images(train_images)
val_images_normalized = normalize_resized_images(val_images)

# ---------------------------Görüntü resize ve normalize et Bitti---------------------------
# ---------------------------------------------------------------------------------------------------------
# ---------------------------Veri Genelleme (Augmentation)---------------------------

# Veri genelleme işlemi
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=False,
    fill_mode="nearest",
)
# ---------------------------Veri genelleme Bitti (Augmentation)---------------------------
# ---------------------------------------------------------------------------------------------------------

if plotNoComprision:
    # ---------------------------Model ve Eğitim---------------------------

    # CNN modelini oluşturma
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(43, activation="softmax"),
        ]
    )


    # Modeli derleme
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Modelin yapısını gözden geçirme
    model.summary()

    # Modeli eğitme
    history = model.fit(
        datagen.flow(train_images_normalized, train_data["ClassId"].values, batch_size=32),
        epochs=10,
        validation_data=(val_images_normalized, val_data["ClassId"].values),
    )

    # ---------------------------Model ve Eğitim Bitti---------------------------
    # ---------------------------------------------------------------------------------------------------------
    # ---------------------------Model Basarimi---------------------------

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report
    )

    test_images = resize_images_from_df(df_test, img_path, 32, 32)
    test_images_normalized = normalize_resized_images(test_images)
    test_true_labels = df_test["ClassId"].values


    # Test seti üzerinde tahmin yapma
    test_predictions = np.argmax(model.predict(test_images_normalized), axis=1)

    print("Classification Report:")
    print(
        classification_report(
            test_true_labels, test_predictions, target_names=[str(i) for i in range(43)]
        )
    )

    # Test başarımını hesaplama
    accuracy_test = accuracy_score(test_true_labels, test_predictions)
    precision_test = precision_score(test_true_labels, test_predictions, average="weighted")
    recall_test = recall_score(test_true_labels, test_predictions, average="weighted")
    f1_test = f1_score(test_true_labels, test_predictions, average="weighted")

    # Test sonuçlarını yazdırma
    print("Test Sonuçları:")
    print(f"Accuracy: {accuracy_test:.4f}")
    print(f"Precision: {precision_test:.4f}")
    print(f"Recall: {recall_test:.4f}")
    print(f"F1-score: {f1_test:.4f}")


    # ---------------------------Model Basarimi Bitti---------------------------

if plotDenseUnitComparison:
# ---------------------------Model ve Eğitim---------------------------
    # CNN modelini oluşturma

    units = [32, 64, 128, 256]
    results = {
        "units": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "val_accuracy": []
    }

    for unit in units:
        print(unit)
        model = Sequential(
            [
                Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation="relu"),
                MaxPooling2D((2, 2)),
                Conv2D(128, (3, 3), activation="relu"),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(unit, activation="relu"),
                Dropout(0.5),
                Dense(43, activation="softmax"),
            ]
        )


        # Modeli derleme
        model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

        # Modelin yapısını gözden geçirme
        model.summary()

        # Modeli eğitme
        history = model.fit(
            datagen.flow(train_images_normalized, train_data["ClassId"].values, batch_size=32),
            epochs=5,
            validation_data=(val_images_normalized, val_data["ClassId"].values),
        )

        # ---------------------------Model ve Eğitim Bitti---------------------------
        # ---------------------------------------------------------------------------------------------------------
        # ---------------------------Model Basarimi---------------------------

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            classification_report
        )

        test_images = resize_images_from_df(df_test, img_path, 32, 32)
        test_images_normalized = normalize_resized_images(test_images)
        test_true_labels = df_test["ClassId"].values


        # Test seti üzerinde tahmin yapma
        test_predictions = np.argmax(model.predict(test_images_normalized), axis=1)
        final_val_accuracy = history.history['val_accuracy'][-1]
        
        print("Classification Report:")
        print(
            classification_report(
                test_true_labels, test_predictions, target_names=[str(i) for i in range(43)]
            )
        )

        # Test başarımını hesaplama
        accuracy_test = accuracy_score(test_true_labels, test_predictions)
        precision_test = precision_score(test_true_labels, test_predictions, average="weighted")
        recall_test = recall_score(test_true_labels, test_predictions, average="weighted")
        f1_test = f1_score(test_true_labels, test_predictions, average="weighted")

        # Test sonuçlarını yazdırma
        """print("Test Sonuçları:")
        print(f"Accuracy: {accuracy_test:.4f}")
        print(f"Precision: {precision_test:.4f}")
        print(f"Recall: {recall_test:.4f}")
        print(f"F1-score: {f1_test:.4f}")"""
        results["units"].append(unit)
        results["accuracy"].append(accuracy_test)
        results["precision"].append(precision_test)
        results["recall"].append(recall_test)
        results["f1_score"].append(f1_test)
        results["val_accuracy"].append(final_val_accuracy)
        
        print(f"Units: {unit}, Accuracy: {accuracy_test:.4f}, Precision: {precision_test:.4f}, Recall: {recall_test:.4f}, F1: {f1_test:.4f}, Val_Accuracy: {final_val_accuracy:.4f}")


    # ---------------------------Model Basarimi Bitti---------------------------

    # ---------------------------Grafiklestirme---------------------------
    plt.figure(figsize=(12, 8))
    plt.plot(results["units"], results["accuracy"], marker="o", label="Accuracy")
    plt.plot(results["units"], results["precision"], marker="s", label="Precision")
    plt.plot(results["units"], results["recall"], marker="^", label="Recall")
    plt.plot(results["units"], results["f1_score"], marker="d", label="F1-Score")
    plt.title("Performance Metrics vs Dense Units")
    plt.xlabel("Dense Units")
    plt.ylabel("Performance Metrics")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Performance_Metrics_vs_Dense_Units.png", dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(results["units"], results["accuracy"], marker="s", color="orange", label="Training Accuracy")
    plt.plot(results["units"], results["val_accuracy"], marker="*", label="Validation Accuracy") 
    plt.title("Performance Metrics vs Dense Units")
    plt.xlabel("Dense Units")
    plt.ylabel("Performance Metrics")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Training_Accuracy_Validation_Accuracy_vs_Dense_Units.png", dpi=300, bbox_inches='tight')
    plt.show()

    # ---------------------------Grafiklestirme Bitti---------------------------

if plotConvFiltersComparison:
# ---------------------------Model ve Eğitim---------------------------
    # CNN modelini oluşturma

    filter_values = [16, 32, 64, 128]
    results = {
        "filters": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "val_accuracy": []
    }

    for filters in filter_values:
        model = Sequential(
            [
                Conv2D(filters, (3, 3), activation="relu", input_shape=(32, 32, 3)),
                MaxPooling2D((2, 2)),
                Conv2D(2*filters, (3, 3), activation="relu"),
                MaxPooling2D((2, 2)),
                Conv2D(4*filters, (3, 3), activation="relu"),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(128, activation="relu"),
                Dropout(0.5),
                Dense(43, activation="softmax"),
            ]
        )


        # Modeli derleme
        model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

        # Modelin yapısını gözden geçirme
        model.summary()

        # Modeli eğitme
        history = model.fit(
            datagen.flow(train_images_normalized, train_data["ClassId"].values, batch_size=32),
            epochs=5,
            validation_data=(val_images_normalized, val_data["ClassId"].values),
        )

        # ---------------------------Model ve Eğitim Bitti---------------------------
        # ---------------------------------------------------------------------------------------------------------
        # ---------------------------Model Basarimi---------------------------

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            classification_report
        )

        test_images = resize_images_from_df(df_test, img_path, 32, 32)
        test_images_normalized = normalize_resized_images(test_images)
        test_true_labels = df_test["ClassId"].values


        # Test seti üzerinde tahmin yapma
        test_predictions = np.argmax(model.predict(test_images_normalized), axis=1)
        final_val_accuracy = history.history['val_accuracy'][-1]

        print("Classification Report:")
        print(
            classification_report(
                test_true_labels, test_predictions, target_names=[str(i) for i in range(43)]
            )
        )

        # Test başarımını hesaplama
        accuracy_test = accuracy_score(test_true_labels, test_predictions)
        precision_test = precision_score(test_true_labels, test_predictions, average="weighted")
        recall_test = recall_score(test_true_labels, test_predictions, average="weighted")
        f1_test = f1_score(test_true_labels, test_predictions, average="weighted")
    
        # Test sonuçlarını yazdırma
        """print("Test Sonuçları:")
        print(f"Accuracy: {accuracy_test:.4f}")
        print(f"Precision: {precision_test:.4f}")
        print(f"Recall: {recall_test:.4f}")
        print(f"F1-score: {f1_test:.4f}")"""
        results["filters"].append(filters)
        results["accuracy"].append(accuracy_test)
        results["precision"].append(precision_test)
        results["recall"].append(recall_test)
        results["f1_score"].append(f1_test)
        results["val_accuracy"].append(final_val_accuracy)
        
        print(f"Filters: {filters}, Accuracy: {accuracy_test:.4f}, Precision: {precision_test:.4f}, Recall: {recall_test:.4f}, F1: {f1_test:.4f}, Val_Accuracy: {final_val_accuracy:.4f}")


    # ---------------------------Model Basarimi Bitti---------------------------

    # ---------------------------Grafiklestirme---------------------------
    plt.figure(figsize=(12, 8))
    plt.plot(results["filters"], results["accuracy"], marker="o", label="Accuracy")
    plt.plot(results["filters"], results["precision"], marker="s", label="Precision")
    plt.plot(results["filters"], results["recall"], marker="^", label="Recall")
    plt.plot(results["filters"], results["f1_score"], marker="d", label="F1-Score")

    plt.title("Performance Metrics vs Conv2D Filters")
    plt.xlabel("Conv2D Filters")
    plt.ylabel("Performance Metrics")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Performance_Metrics_vs_Conv2D_Filters.png", dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(results["filters"], results["accuracy"], marker="s", color="orange", label="Training Accuracy")
    plt.plot(results["filters"], results["val_accuracy"], marker="*", label="Validation Accuracy") 
    plt.title("Performance Metrics vs Conv2D Filters")
    plt.xlabel("Conv2D Filters")
    plt.ylabel("Performance Metrics")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Val_acc_training_acc_vs_Conv2D_Filters.png", dpi=300, bbox_inches='tight')
    plt.show()

    # ---------------------------Grafiklestirme Bitti---------------------------

# ---------------------------Augmentation'siz Model ve Eğitim---------------------------
# Modeli yeniden oluşturma
if(plotAugmentationComparison):
    model_no_aug = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(43, activation="softmax"),
        ]
    )

    # Modeli derleme
    model_no_aug.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Modelin yapısını kontrol etme
    model_no_aug.summary()

    # ---------------------------Augmentation'siz Model ve Eğitim---------------------------

    # Modeli eğitme
    history_no_aug = model_no_aug.fit(
        train_images_normalized,
        train_data["ClassId"].values,
        epochs=10,
        batch_size=32,
        validation_data=(val_images_normalized, val_data["ClassId"].values),
    )

    # ---------------------------Model Basarimi Augmentation'siz---------------------------

    # Tahminler
    test_predictions_no_aug = np.argmax(model_no_aug.predict(test_images_normalized), axis=1)

    # Başarı metriklerini hesaplama
    accuracy_test_no_aug = accuracy_score(test_true_labels, test_predictions_no_aug)
    precision_test_no_aug = precision_score(
        test_true_labels, test_predictions_no_aug, average="weighted"
    )
    recall_test_no_aug = recall_score(
        test_true_labels, test_predictions_no_aug, average="weighted"
    )
    f1_test_no_aug = f1_score(test_true_labels, test_predictions_no_aug, average="weighted")

    # Augmentation olmadan başarımlar
    print("Augmentation'siz Model Sonuçları:")
    print(f"Accuracy: {accuracy_test_no_aug:.4f}")
    print(f"Precision: {precision_test_no_aug:.4f}")
    print(f"Recall: {recall_test_no_aug:.4f}")
    print(f"F1-score: {f1_test_no_aug:.4f}")


    # Augmentation'lı ve Augmentation'siz kıyaslama
    print("\nKıyaslama:")
    print(f"Accuracy Farkı: {accuracy_test - accuracy_test_no_aug:.4f}")
    print(f"Precision Farkı: {precision_test - precision_test_no_aug:.4f}")
    print(f"Recall Farkı: {recall_test - recall_test_no_aug:.4f}")
    print(f"F1-score Farkı: {f1_test - f1_test_no_aug:.4f}")

    # ---------------------------Model Basarimi Augmentation'siz Bitti---------------------------

    # ---------------------------Augmentation Yes vs No---------------------------
    # Verileri oluştur
    data = {
        "Model": ["Basic CNN", "Enhanced CNN"],
        "Data Augmentation": ["No", "Yes"],
        "Accuracy (%)": [accuracy_test_no_aug, accuracy_test],
        "F1-Score": [f1_test_no_aug,f1_test]
    }

    # DataFrame oluştur
    df = pd.DataFrame(data)

    # Tabloyu çizmek için bir figür oluştur
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis("tight")
    ax.axis("off")

    # Tabloyu oluştur
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )

    # Stil düzenlemeleri
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    # Tabloyu göster
    plt.show()
    # ---------------------------Augmentation Yes vs No---------------------------
    # Verileri oluştur
    data = {
        "Model": ["Basic CNN", "Enhanced CNN"],
        "Data Augmentation": ["No", "Yes"],
        "Accuracy (%)": [accuracy_test_no_aug, accuracy_test],
        "F1-Score": [f1_test_no_aug,f1_test]
    }

    # DataFrame oluştur
    df = pd.DataFrame(data)

    # Tabloyu çizmek için bir figür oluştur
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis("tight")
    ax.axis("off")

    # Tabloyu oluştur
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )

    # Stil düzenlemeleri
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    # Tabloyu göster
    plt.show()
# ---------------------------Augmentation Yes vs No Bitti---------------------------

# ---------------------------Normalization'siz Model ve Eğitim---------------------------
# Modeli yeniden oluşturma

if(plotNormalizationComparison):
    model_no_normalize = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(43, activation="softmax"),
        ]
    )

    # Modeli derleme
    model_no_normalize.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Modelin yapısını kontrol etme
    model_no_normalize.summary()
    # Modeli eğitme
    history_no_aug = model_no_normalize.fit(
        train_images,
        train_data["ClassId"].values,
        epochs=10,
        batch_size=32,
        validation_data=(val_images, val_data["ClassId"].values),
    )

    # ---------------------------Model Basarimi Normalization'siz---------------------------

    # Tahminler
    test_predictions_no_normalize = np.argmax(model_no_normalize.predict(test_images), axis=1)

    # Başarı metriklerini hesaplama
    accuracy_test_no_normalize = accuracy_score(test_true_labels, test_predictions_no_normalize)
    precision_test_no_normalize = precision_score(
        test_true_labels, test_predictions_no_normalize, average="weighted"
    )
    recall_test_no_normalize = recall_score(
        test_true_labels, test_predictions_no_normalize, average="weighted"
    )
    f1_test_no_normalize = f1_score(test_true_labels, test_predictions_no_normalize, average="weighted")

    # Augmentation olmadan başarımlar
    print("Augmentation'siz Model Sonuçları:")
    print(f"Accuracy: {accuracy_test_no_normalize:.4f}")
    print(f"Precision: {precision_test_no_normalize:.4f}")
    print(f"Recall: {recall_test_no_normalize:.4f}")
    print(f"F1-score: {f1_test_no_normalize:.4f}")


    # Augmentation'lı ve Augmentation'siz kıyaslama
    print("\nKıyaslama:")
    print(f"Accuracy Farkı: {accuracy_test - accuracy_test_no_normalize:.4f}")
    print(f"Precision Farkı: {precision_test - precision_test_no_normalize:.4f}")
    print(f"Recall Farkı: {recall_test - recall_test_no_normalize:.4f}")
    print(f"F1-score Farkı: {f1_test - f1_test_no_normalize:.4f}")

    # ---------------------------Normalize Yes vs No---------------------------

    # Verileri oluştur
    data = {
        "Model": ["Basic CNN", "Enhanced CNN"],
        "Image Normalization": ["No", "Yes"],
        "Accuracy (%)": [accuracy_test_no_normalize, accuracy_test],
        "F1-Score": [f1_test_no_normalize,f1_test]
    }

    # DataFrame oluştur
    df = pd.DataFrame(data)

    # Tabloyu çizmek için bir figür oluştur
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis("tight")
    ax.axis("off")

    # Tabloyu oluştur
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )

    # Stil düzenlemeleri
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    # Tabloyu göster
    plt.show()

# ---------------------------Normalize Yes vs No Bitti---------------------------

# ---------------------------------------------------------------------------------------------------------
# ---------------------------Epoch degisimi gozlem---------------------------

if(plotEpochComparison):
    # Epoch sayısını 50'ye çıkararak modeli eğitme
    history = model.fit(
        datagen.flow(train_images_normalized, train_data["ClassId"].values, batch_size=32),
        epochs=50,
        validation_data=(val_images_normalized, val_data["ClassId"].values),
    )

    # Eğitim ve doğrulama başarımlarını grafik üzerinde gösterme
    plt.figure(figsize=(12, 6))

    # Eğitim ve doğrulama doğruluğu (accuracy) grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Eğitim Doğruluğu")
    plt.plot(history.history["val_accuracy"], label="Doğrulama Doğruluğu")
    plt.title("Model Doğruluğu")
    plt.xlabel("Epoch")
    plt.ylabel("Doğruluk")
    plt.legend()

    # Eğitim ve doğrulama kaybı (loss) grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Eğitim Kaybı")
    plt.plot(history.history["val_loss"], label="Doğrulama Kaybı")
    plt.title("Model Kaybı")
    plt.xlabel("Epoch")
    plt.ylabel("Kayıp")
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("Test sonuçları başarıyla tahmin edildi.")


# ---------------------------Epoch degisimi gozlem bitti---------------------------
# ---------------------------------------------------------------------------------------------------------
# Print the classification report for detailed metrics per class
