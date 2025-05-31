import os
import cv2
import albumentations as A
from glob import glob

input_root = "Dataset/raw"
N_AUG = 10  # Số ảnh augment tạo thêm từ mỗi ảnh gốc

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.2, rotate_limit=35, p=0.8),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    A.MotionBlur(blur_limit=5, p=0.2),
])

# Lặp qua từng thư mục của mỗi người
for person_name in os.listdir(input_root):
    person_dir = os.path.join(input_root, person_name)
    if not os.path.isdir(person_dir):
        continue

    image_paths = glob(os.path.join(person_dir, "*.png")) + \
                  glob(os.path.join(person_dir, "*.jpg")) + \
                  glob(os.path.join(person_dir, "*.jpeg"))

    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        base_name = os.path.splitext(os.path.basename(img_path))[0]

        for i in range(N_AUG):
            augmented = transform(image=img)["image"]
            save_path = os.path.join(person_dir, f"{base_name}_aug_{i}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

    print(f"{person_name}: Augmented {len(image_paths)} images × {N_AUG} each.")
