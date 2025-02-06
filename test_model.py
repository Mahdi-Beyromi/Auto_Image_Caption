import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import os

# 🚀 بارگذاری مدل آموزش‌دیده
model = load_model(r"C:\Users\ariyan\AI\image_captioning_model.keras")

# 📂 بارگذاری توکنایزر
with open(r"C:\Users\ariyan\AI\tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# 📂 بارگذاری ویژگی‌های تصویری از `image_features.pkl`
with open(r"C:\Users\ariyan\AI\image_features.pkl", "rb") as f:
    image_features = pickle.load(f)

# محاسبه `max_length` از توکنایزر
max_length = max(len(seq) for seq in tokenizer.texts_to_sequences(tokenizer.word_index.keys()))
print(f"📏 Computed max_length: {max_length}")

def generate_caption(image_path):
    """تولید کپشن برای یک تصویر با استفاده از ویژگی‌های از قبل استخراج شده"""
    image_name = os.path.basename(image_path)  # فقط نام فایل بدون مسیر
    
    if image_name not in image_features:
        print(f"❌ Error: Image {image_name} not found in image_features.pkl")
        return ""

    features = image_features[image_name]  # دریافت ویژگی‌های تصویر
    caption = ["startseq"]

    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding="post")

        y_pred = model.predict([features, sequence], verbose=0)
        next_word_index = np.argmax(y_pred)
        next_word = tokenizer.index_word.get(next_word_index, "UNK")

        if next_word == "endseq" or next_word == "UNK":
            break

        caption.append(next_word)

    return " ".join(caption[1:])  # حذف "startseq"

# 🖼️ تست مدل روی یک تصویر جدید
test_image_name = r"C:\Users\ariyan\AI\test\667626_18933d713e.jpg"  # نام تصویر تستی
generated_caption = generate_caption(test_image_name)
print(f"🖼️ Generated Caption for {test_image_name}: {generated_caption}")
