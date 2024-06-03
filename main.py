import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data import load_iiit5k_data, preprocess_images, recognize_text_with_tesseract
from model import build_model
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
import tensorflow as tf

def encode_labels(labels, num_classes, max_len):
    all_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    label_encoder = LabelEncoder()
    label_encoder.fit(list(all_characters))

    encoded_labels = []
    for label in labels:
        try:
            encoded_label = label_encoder.transform(list(label))
            padded_label = pad_sequences([encoded_label], maxlen=max_len, padding='post')
            one_hot_label = to_categorical(padded_label, num_classes=num_classes)
            encoded_labels.append(one_hot_label[0])
        except Exception as e:
            print(f"Error encoding label {label}: {e}")
    
    return np.array(encoded_labels)

def decode_predictions(preds, label_encoder):
    decoded_preds = []
    for pred in preds:
        decoded_pred = label_encoder.inverse_transform(np.argmax(pred, axis=1))
        decoded_preds.append("".join(decoded_pred).strip())
    return decoded_preds

def display_predictions(images, true_labels, predictions, num_samples=10):
    plt.figure(figsize=(20, 10))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"True: {true_labels[i]}\nPred: {predictions[i]}")
        plt.axis('off')
    plt.show()

def save_predictions(images, true_labels, predictions, save_dir='predictions'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i, (image, true_label, pred_label) in enumerate(zip(images, true_labels, predictions)):
        image_pil = Image.fromarray((image.squeeze() * 255).astype(np.uint8)).convert('RGB')
        draw = ImageDraw.Draw(image_pil)
        font = ImageFont.load_default()
        text = f"True: {true_label}, Pred: {pred_label}"
        draw.text((10, 10), text, font=font, fill=(255, 0, 0))
        image_pil.save(os.path.join(save_dir, f"prediction_{i}.png"))

def predict_new_images(model, new_images_dir, input_shape, label_encoder, max_len):
    image_paths = [os.path.join(new_images_dir, img) for img in os.listdir(new_images_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
    new_images = preprocess_images(image_paths, input_shape)
    predictions = model.predict(new_images)
    decoded_predictions = decode_predictions(predictions, label_encoder)

    for img_path, pred in zip(image_paths, decoded_predictions):
        print(f"Image: {img_path}, Predicted Text: {pred}")

def main():
    data_dir = 'D:/WORK/IME_DIGITAL/English_OCR_Model/IIIT5K'
    input_shape = (32, 128, 1)
    num_classes = 62
    max_len = 10

    # Load and preprocess data
    train_images, train_labels = load_iiit5k_data(data_dir, 'trainCharBound.mat')
    val_images, val_labels = load_iiit5k_data(data_dir, 'testCharBound.mat')
    train_images = preprocess_images(train_images, input_shape)
    val_images = preprocess_images(val_images, input_shape)

    # Encode labels
    all_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    label_encoder = LabelEncoder()
    label_encoder.fit(list(all_characters))
    train_encoded_labels = encode_labels(train_labels, num_classes, max_len)
    val_encoded_labels = encode_labels(val_labels, num_classes, max_len)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1
    )

    # Load the existing model or build a new one
    model_path = 'D:/WORK/IME_DIGITAL/English_OCR_Model/saved_model/my_ocr_model.h5'
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = build_model(input_shape, num_classes, max_len)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    checkpoint = ModelCheckpoint('D:/WORK/IME_DIGITAL/English_OCR_Model/saved_model/my_ocr_model_retrained.keras', save_best_only=True)
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

    # Train the model with data augmentation
    model.fit(
        datagen.flow(train_images, train_encoded_labels, batch_size=32),
        validation_data=(val_images, val_encoded_labels),
        epochs=20,
        callbacks=[checkpoint, early_stopping]
    )

    # Enable unsafe deserialization
    tf.keras.config.enable_unsafe_deserialization()

    # Load the best model from retraining
    model = load_model('D:/WORK/IME_DIGITAL/English_OCR_Model/saved_model/my_ocr_model_retrained.keras')

    # Make predictions on the validation set
    predictions = model.predict(val_images)

    # Decode the predictions
    decoded_predictions = decode_predictions(predictions, label_encoder)

    # Display some sample predictions
    display_predictions(val_images, val_labels, decoded_predictions, num_samples=10)

    # Save the predictions
    save_predictions(val_images, val_labels, decoded_predictions, save_dir='D:/WORK/IME_DIGITAL/English_OCR_Model/predictions')

    # Test on new images
    new_images_dir = 'D:/WORK/IME_DIGITAL/English_OCR_Model/new_images'
    predict_new_images(model, new_images_dir, input_shape, label_encoder, max_len)

if __name__ == '__main__':
    main()
