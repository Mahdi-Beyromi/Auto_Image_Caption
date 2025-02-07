import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Dropout, Concatenate

# ---------------------------
# بارگذاری ویژگی‌های تصویری
# ---------------------------
with open(r"C:\Users\ariyan\AI\image_features.pkl", "rb") as f:
    image_features = pickle.load(f)

# ---------------------------
# بارگذاری کپشن‌ها
# ---------------------------
with open(r"C:\Users\ariyan\AI\captions.pkl", "rb") as f:
    captions = pickle.load(f)

print(f"✅ Loaded {len(image_features)} image features and {len(captions)} captions.")

# ---------------------------
# تبدیل کپشن‌ها از دیکشنری به لیست (و flatten کردن)
# ---------------------------
all_captions = list(captions.values())
flat_captions = [caption for sublist in all_captions for caption in sublist]

print(f"✅ Total captions: {len(flat_captions)}")
print("🔍 Sample captions:", flat_captions[:5])  # نمایش ۵ کپشن اول

# ---------------------------
# ساخت توکنایزر و تعیین ۱۰,۰۰۰ کلمه پرتکرار
# ---------------------------
tokenizer = Tokenizer(num_words=10000, oov_token="UNK")
tokenizer.fit_on_texts(flat_captions)

# ذخیره توکنایزر
with open(r"C:\Users\ariyan\AI\tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print(f"✅ Tokenizer created with {len(tokenizer.word_index)} words.")

# ---------------------------
# تعیین طول حداکثری کپشن‌ها
# ---------------------------
max_length = max(len(caption.split()) for caption in flat_captions)
print(f"📏 Maximum caption length: {max_length}")

# ذخیره max_length برای استفاده در فایل test_model.py
with open(r"C:\Users\ariyan\AI\max_length.pkl", "wb") as f:
    pickle.dump(max_length, f)
print(f"📏 max_length saved: {max_length}")

# ---------------------------
# آماده‌سازی داده‌های آموزشی: تبدیل کپشن‌ها به توالی‌های عددی
# ---------------------------
X_train, y_train, image_input = [], [], []

for key, caps in captions.items():
    image_id = key.split(".")[0] + ".jpg"  # حذف .1, .2 و ...
    if image_id not in image_features:
        print(f"⚠ Skipping {image_id} (not found in features)")
        continue

    for caption in caps:  # استفاده از تمام 5 کپشن برای هر تصویر
        seq = tokenizer.texts_to_sequences([caption])[0]
        if len(seq) < 2:
            continue

        for i in range(1, len(seq)):
            X_train.append(seq[:i])
            y_train.append(seq[i])
            image_input.append(image_features[image_id])

X_train = pad_sequences(X_train, maxlen=max_length, padding='post')
y_train = np.array(y_train)
image_input = np.array(image_input).squeeze()

print(f"✅ Training data prepared: {X_train.shape}, {y_train.shape}, {image_input.shape}")

# ---------------------------
# ساخت مدل LSTM
# ---------------------------
# ورودی ویژگی‌های تصویر (با ابعاد 2048)
image_input_layer = Input(shape=(2048,))
img_dense = Dense(256, activation="relu")(image_input_layer)

# ورودی کپشن (با طول max_length)
caption_input = Input(shape=(max_length,))
caption_embedding = Embedding(input_dim=len(tokenizer.word_index)+1,
                              output_dim=256, mask_zero=True)(caption_input)
caption_lstm = LSTM(256)(caption_embedding)

# ادغام ویژگی‌های تصویری و متنی با استفاده از Concatenate
merged = Concatenate()([img_dense, caption_lstm])
merged = Dense(256, activation="relu")(merged)
merged = Dropout(0.3)(merged)
output = Dense(len(tokenizer.word_index)+1, activation="softmax")(merged)

model = Model(inputs=[image_input_layer, caption_input], outputs=output)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
print("✅ Model created successfully!")

# ---------------------------
# آموزش مدل
# ---------------------------
model.fit([image_input, X_train], y_train, epochs=10, batch_size=64, verbose=1)

# ---------------------------
# ذخیره مدل با فرمت جدید Keras
# ---------------------------
model.save(r"C:\Users\ariyan\AI\image_captioning_model.keras")
print("✅ Model trained and saved successfully!")
