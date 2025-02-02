import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.xception import Xception, preprocess_input
from keras.models import Model
import pickle

# مسیر تصاویر دیتاست
IMAGE_DIR = "C:\\Users\\ariyan\\AI\\Flicker8k_Dataset"

# بارگذاری مدل Xception برای استخراج ویژگی‌ها
base_model = Xception(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

def extract_features(image_path):
    """Load image, preprocess, and extract features"""
    image = load_img(image_path, target_size=(299, 299))  
    image = img_to_array(image)  
    image = preprocess_input(image)  
    image = np.expand_dims(image, axis=0)  
    feature = model.predict(image)  
    return feature

# استخراج ویژگی‌ها از تمام تصاویر
features = {}
image_files = os.listdir(IMAGE_DIR)

for img_file in image_files:
    img_path = os.path.join(IMAGE_DIR, img_file)
    feature = extract_features(img_path)
    features[img_file] = feature
    print(f"Extracting features for image: {img_file}")

# ذخیره ویژگی‌ها
with open("C:\\Users\\ariyan\\AI\\image_features.pkl", "wb") as f:
    pickle.dump(features, f)

print("✅ Image features extracted and saved successfully!")
