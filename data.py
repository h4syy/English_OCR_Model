import os
import numpy as np
from PIL import Image
import scipy.io
import pytesseract

def load_iiit5k_data(data_dir, mat_file):
    mat = scipy.io.loadmat(os.path.join(data_dir, mat_file))
    images = []
    labels = []

    if 'train' in mat_file:
        data = mat['trainCharBound'][0]
    else:
        data = mat['testCharBound'][0]

    for item in data:
        img_name = item[0][0]
        label = item[1][0]
        img_path = os.path.join(data_dir, img_name)
        if not os.path.isfile(img_path):
            print(f"File not found: {img_path}")
            continue

        images.append(img_path)
        labels.append(label)
    
    print(f"Loaded {len(images)} images and {len(labels)} labels from {mat_file}")

    return images, labels

def preprocess_images(image_paths, input_shape):
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert('L')
            img = Image.eval(img, lambda x: 255 - x)  # Invert the image: black background, white text
            img = img.resize((input_shape[1], input_shape[0]))
            img = np.array(img)
            img = img / 255.0
            img = np.expand_dims(img, axis=-1)  # Add channel dimension
            images.append(img)
        except Exception as e:
            print(f"Error processing image {path}: {e}")
    
    return np.array(images)

def recognize_text_with_tesseract(image):
    """
    Recognize text in the input image using Tesseract OCR.

    Parameters:
        image (PIL.Image): Input image as a PIL Image.

    Returns:
        str: Recognized text extracted from the image.
    """
    # Use pytesseract to extract text from the image
    recognized_text = pytesseract.image_to_string(image)
    return recognized_text
