import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import heapq

# بارگذاری مدل ذخیره‌شده
model = load_model("C:\\Users\\ariyan\\AI\\image_captioning_model.keras")

# بارگذاری ویژگی‌های تصویری مدل Xception
xception_model = Xception(weights="E:/Anaconda/Lib/site-packages/keras/src/applications/xception_weights_tf_dim_ordering_tf_kernels_notop.h5", include_top=False, pooling="avg")


# بارگذاری توکنایزر
with open(r"C:\Users\ariyan\AI\tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# تعیین حداکثر طول جملات (مطابق با مرحله آموزش)
max_length = 15 

def extract_features(image_path):
    """استخراج ویژگی‌های تصویر با Xception"""
    image = load_img(image_path, target_size=(299, 299))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return xception_model.predict(image)

def generate_caption(image_path):
    """تولید کپشن برای یک تصویر"""
    features = extract_features(image_path)

    caption = ["startseq"]  # شروع جمله
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding="post")

        y_pred = model.predict([features, sequence], verbose=0)
        next_word_index = np.argmax(y_pred)
        next_word = {index: word for word, index in tokenizer.word_index.items()}.get(next_word_index, None)

        if next_word is None or next_word == "endseq":
            break

        caption.append(next_word)

    return " ".join(caption[1:])  # حذف "startseq"

# تست روی یک تصویر جدید
test_image = r"C:\Users\ariyan\AI\test\test.jpg"  # مسیر تصویر تستی
generated_caption = generate_caption(test_image)
print(f"🖼️ Generated Caption: {generated_caption}")