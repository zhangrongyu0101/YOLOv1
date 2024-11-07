import pandas as pd
import os


img_dir = "./data/images"
label_dir = "./data/labels"

csv_file = "./train.csv"
df = pd.read_csv(csv_file, header=None, names=["image", "label"])


def file_exists(row, img_dir, label_dir):
    img_path = os.path.join(img_dir, row['image'])
    label_path = os.path.join(label_dir, row['label'])
    img_exists = os.path.exists(img_path)
    label_exists = os.path.exists(label_path)
    if not img_exists or not label_exists:
        # Print missing paths
        if not img_exists:
            print(f"Missing image file: {img_path}")
        if not label_exists:
            print(f"Missing label file: {label_path}")
    return img_exists and label_exists


df['exists'] = df.apply(file_exists, axis=1, args=(img_dir, label_dir))
df_cleaned = df[df['exists']]


df_cleaned.drop(columns=['exists'], inplace=True)


updated_csv_file = "./updated_train.csv"
df_cleaned.to_csv(updated_csv_file, index=False, header=False)