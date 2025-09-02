import os
import pandas as pd
import requests
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import torch

# --- CONFIG ---
csv_path = 'tasks_by_majority.csv'      # Your filtered CSV
output_root = 'images_by_label_yolo'    # Output directory
valid_labels = ['Aggression', 'Anxiety', 'Contentment', 'Fear']

# --- LOAD DATA ---
df = pd.read_csv(csv_path)

filtered_df = df[
    (df['final_label'].isin(valid_labels)) &
    (df['num_annotators'] >= 2)
].copy()

print(filtered_df.head())
print(filtered_df.columns)
print(f"About to process {len(filtered_df)} images...")

# --- INIT YOLO ---
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # or yolov8n for YOLOv8

# --- COUNTERS FOR ORDER NUMBER ---
label_counters = {label: 1 for label in valid_labels}

# --- MAKE FOLDERS ---
for label in valid_labels:
    os.makedirs(os.path.join(output_root, label), exist_ok=True)

# --- MAIN LOOP ---
for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
    url = row['url']
    label = row['final_label']

    try:
        # 1. Download image
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"Failed to download {url}")
            continue
        img = Image.open(BytesIO(response.content)).convert("RGB")

        # 2. YOLO detection
        results = model(img, size=640)
        detections = results.pandas().xyxy[0]
        dog_detections = detections[detections['name'] == 'dog']

        if len(dog_detections) == 0:
            # No dog detected: skip/delete
            continue

        # 3. Pick largest dog bounding box (prefer full body)
        dog_detections['area'] = (dog_detections['xmax'] - dog_detections['xmin']) * \
                                 (dog_detections['ymax'] - dog_detections['ymin'])
        dog = dog_detections.iloc[dog_detections['area'].argmax()]
        xmin, ymin, xmax, ymax = map(int, [dog['xmin'], dog['ymin'], dog['xmax'], dog['ymax']])

        # 4. Crop to dog
        cropped_img = img.crop((xmin, ymin, xmax, ymax))

        # 5. Save with incremental order number per label
        order_number = label_counters[label]
        filename = f"{label}_{order_number}.jpg"
        save_path = os.path.join(output_root, label, filename)
        cropped_img.save(save_path)
        label_counters[label] += 1

    except Exception as e:
        print(f"Error with {url}: {e}")

print("Done!")
