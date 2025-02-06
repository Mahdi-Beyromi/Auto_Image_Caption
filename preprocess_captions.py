# import string
# import pickle

# # مسیر فایل کپشن‌ها
# CAPTION_FILE = "C:\\Users\\ariyan\\AI\\Flickr8k.token.txt"

# def load_captions(filename):
#     """Load captions from file and store only one caption per image"""
#     captions_dict = {}
#     with open(filename, 'r') as file:
#         for line in file:
#             parts = line.strip().split("\t")
#             if len(parts) < 2:
#                 continue
#             image_id, caption = parts[0].split("#")[0], parts[1]  # استخراج نام تصویر و کپشن

#             # اگر این تصویر قبلاً کپشن گرفته، از پردازش رد می‌شیم
#             if image_id in captions_dict:
#                 continue

#             captions_dict[image_id] = preprocess_caption(caption)  # ذخیره فقط یک کپشن

#     return captions_dict

# def preprocess_caption(caption):
#     """Convert captions to lowercase, remove punctuation, and clean text"""
#     caption = caption.lower()
#     caption = caption.translate(str.maketrans('', '', string.punctuation))  # حذف نقطه‌گذاری
#     caption = " ".join(caption.split())  # حذف فاصله‌های اضافی
#     return caption

# # پردازش و ذخیره کپشن‌ها
# captions = load_captions(CAPTION_FILE)

# # ذخیره کپشن‌های اصلاح‌شده
# with open("C:\\Users\\ariyan\\AI\\captions.pkl", "wb") as f:
#     pickle.dump(captions, f)

# print(f"✅ Captions processed and saved successfully! Total unique images: {len(captions)}")

import string
import pickle

# مسیر فایل کپشن‌ها
CAPTION_FILE = "C:\\Users\\ariyan\\AI\\Flickr8k.token.txt"

def load_captions(filename):
    """Load captions from file and store all 5 captions per image"""
    captions_dict = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            image_id, caption = parts[0].split("#")[0], parts[1]

            # اگر این تصویر قبلاً اضافه نشده، یک لیست جدید برایش بساز
            if image_id not in captions_dict:
                captions_dict[image_id] = []

            captions_dict[image_id].append(preprocess_caption(caption))  # اضافه کردن همه‌ی کپشن‌ها

    return captions_dict


def preprocess_caption(caption):
    """Convert captions to lowercase, remove punctuation, and clean text"""
    caption = caption.lower().strip()
    caption = caption.translate(str.maketrans('', '', string.punctuation))  # حذف نقطه‌گذاری
    caption = " ".join(caption.split())  # حذف فاصله‌های اضافی
    return caption

# پردازش و ذخیره کپشن‌ها
captions = load_captions(CAPTION_FILE)

# ذخیره کپشن‌های اصلاح‌شده
with open("C:\\Users\\ariyan\\AI\\captions.pkl", "wb") as f:
    pickle.dump(captions, f)

print(f"✅ Captions processed successfully! Total unique images: {len(captions)}")
