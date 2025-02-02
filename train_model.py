import pickle

# بارگذاری ویژگی‌های تصویری
with open("C:\\Users\\ariyan\\AI\\image_features.pkl", "rb") as f:
    image_features = pickle.load(f)

# بارگذاری کپشن‌ها
with open("C:\\Users\\ariyan\\AI\\captions.pkl", "rb") as f:
    captions = pickle.load(f)

print(f"✅ Loaded {len(image_features)} image features and {len(captions)} captions.")

# TODO: ادامه کد برای پردازش داده‌ها و آموزش مدل LSTM را اضافه کنید...

from tensorflow.keras.preprocessing.text import Tokenizer


# بررسی مقدار `all_captions`
all_captions = list(captions.values())
print(f"✅ Total captions: {len(all_captions)}")
print("🔍 Sample captions:", all_captions[:5])  # نمایش ۵ کپشن اول

# ساخت توکنایزر
tokenizer = Tokenizer(num_words=10000, oov_token="UNK")
tokenizer.fit_on_texts(all_captions)

# بررسی تعداد کلمات در توکنایزر
print(f"✅ Total words in tokenizer: {len(tokenizer.word_index)}")

# ذخیره توکنایزر برای استفاده بعدی
with open("C:\\Users\\ariyan\\AI\\tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print(f"✅ Tokenizer created with {len(tokenizer.word_index)} words.")



from keras.preprocessing.sequence import pad_sequences
import numpy as np

# تعیین طول حداکثری جملات
max_length = max(len(caption.split()) for caption in all_captions)

# تبدیل کپشن‌ها به توالی‌های عددی
X_train, y_train, image_input = [], [], []

for key, caption in captions.items():  # حالا فقط یک کپشن برای هر تصویر داریم
    image_id = key.split(".")[0] + ".jpg"  # حذف پسوند اضافی

    if image_id not in image_features:
        print(f"⚠ Skipping {image_id} (not found in features)")
        continue  # این تصویر رو نادیده بگیر

    seq = tokenizer.texts_to_sequences([caption])[0]
    if len(seq) < 2:  # اگر کپشن خیلی کوتاه بود، ردش کن
        print(f"⚠ Skipping empty sequence for caption: {caption}")
        continue

    for i in range(1, len(seq)):
        X_train.append(seq[:i])
        y_train.append(seq[i])
        image_input.append(image_features[image_id])

print(f"✅ Final dataset size: X_train={len(X_train)}, y_train={len(y_train)}, image_input={len(image_input)}")


# اعمال پدینگ برای یکسان‌سازی طول ورودی‌ها
X_train = pad_sequences(X_train, maxlen=max_length, padding='post')
y_train = np.array(y_train)
image_input = np.array(image_input).squeeze()

print(f"✅ Training data prepared: {X_train.shape}, {y_train.shape}, {image_input.shape}")




from keras.layers import Input, Embedding, LSTM, Dense, Dropout, Add
from keras.models import Model

# ورودی ویژگی‌های تصویری
image_input_layer = Input(shape=(2048,))
image_features = Dense(256, activation="relu")(image_input_layer)

# ورودی کپشن‌ها
caption_input = Input(shape=(max_length,))
caption_embedding = Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=256, mask_zero=True)(caption_input)
caption_lstm = LSTM(256)(caption_embedding)

# ترکیب اطلاعات تصویری و متنی
merged = Add()([image_features, caption_lstm])
merged = Dense(256, activation="relu")(merged)
merged = Dropout(0.3)(merged)
output = Dense(len(tokenizer.word_index)+1, activation="softmax")(merged)

# ساخت مدل
model = Model(inputs=[image_input_layer, caption_input], outputs=output)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

print("✅ Model created successfully!")



model.fit([image_input, X_train], y_train, epochs=10, batch_size=64, verbose=1)

# ذخیره مدل
model.save("C:\\Users\\ariyan\\AI\\image_captioning_model.keras")
print("✅ Model trained and saved successfully!")
