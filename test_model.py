import os
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import heapq  # برای پیاده‌سازی Beam Search

# ---------------------------
# بارگذاری مدل آموزش‌دیده
# ---------------------------
model_path = r"C:\Users\ariyan\AI\image_captioning_model.keras"
model = load_model(model_path)
print(f"✅ Loaded model from {model_path}")

# ---------------------------
# بارگذاری توکنایزر
# ---------------------------
tokenizer_path = r"C:\Users\ariyan\AI\tokenizer.pkl"
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)
print(f"✅ Loaded tokenizer from {tokenizer_path}")

# ---------------------------
# بارگذاری ویژگی‌های تصویری
# ---------------------------
features_path = r"C:\Users\ariyan\AI\image_features.pkl"
with open(features_path, "rb") as f:
    image_features = pickle.load(f)
print(f"✅ Loaded image features from {features_path}")

# ---------------------------
# بارگذاری مقدار max_length
# ---------------------------
max_length_path = r"C:\Users\ariyan\AI\max_length.pkl"
if os.path.exists(max_length_path):
    with open(max_length_path, "rb") as f:
        max_length = pickle.load(f)
    print(f"📏 Loaded max_length: {max_length}")
else:
    max_length = 33  # مقدار پیش‌فرض در صورت عدم وجود فایل
    print(f"📏 max_length file not found. Using default max_length: {max_length}")

# ---------------------------
# تنظیم اندیس توکن UNK (برای جلوگیری از انتخاب آن در Beam Search)
# ---------------------------
unk_index = tokenizer.word_index.get("UNK", None)
if unk_index is None:
    # اگر به هر دلیلی توکن "UNK" در tokenizer وجود ندارد، مقدار پیش‌فرض تعیین می‌شود.
    unk_index = -1

# ---------------------------
# تابع تولید کپشن با Beam Search (بهبود یافته)
# ---------------------------
def beam_search_caption(features, beam_width=5):
    """تولید کپشن با استفاده از Beam Search با length normalization"""
    start_seq = [tokenizer.word_index.get("startseq", 1)]
    sequences = [(start_seq, 0.0)]
    
    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            sequence_input = pad_sequences([seq], maxlen=max_length, padding="post")
            y_pred = model.predict([features, sequence_input], verbose=0)[0]
            top_indices = np.argsort(y_pred)[-beam_width:]
            for idx in top_indices:
                if idx == tokenizer.word_index.get("UNK", 0):
                    continue
                candidate = seq + [idx]
                # استفاده از length normalization: تقسیم نمره بر طول توالی
                candidate_score = (score - np.log(y_pred[idx] + 1e-10)) / float(len(candidate))
                all_candidates.append((candidate, candidate_score))
                
        if not all_candidates:
            break
        sequences = heapq.nsmallest(beam_width, all_candidates, key=lambda tup: tup[1])
        if all(seq[-1] == tokenizer.word_index.get("endseq", -1) for seq, _ in sequences):
            break
            
    best_seq = sequences[0][0]
    caption = [tokenizer.index_word.get(idx, "") for idx in best_seq]
    caption = [word for word in caption if word not in ["startseq", "endseq", "UNK", ""]]
    return " ".join(caption)


# ---------------------------
# تابع تولید کپشن برای یک تصویر
# ---------------------------
def generate_caption(image_path, beam_width=3):
    """
    تولید کپشن برای یک تصویر با استفاده از ویژگی‌های از قبل استخراج‌شده.
    توجه: نام فایل (بدون مسیر) باید دقیقا با کلیدهای موجود در image_features مطابقت داشته باشد.
    """
    image_name = os.path.basename(image_path)
    if image_name not in image_features:
        print(f"❌ Error: Image {image_name} not found in image_features.pkl")
        return ""
    
    features = image_features[image_name]  # دریافت ویژگی‌های تصویر
    return beam_search_caption(features, beam_width=beam_width)

# ---------------------------
# تست مدل روی یک تصویر جدید
# ---------------------------
test_image_path = r"C:\Users\ariyan\AI\test\667626_18933d713e.jpg"  # مسیر تصویر تستی؛ مطمئن شوید که نام آن با کلیدهای image_features همخوانی دارد.
generated_caption = generate_caption(test_image_path, beam_width=5)
print(f"🖼️ Generated Caption for {test_image_path}: {generated_caption}")
