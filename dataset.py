import cv2
import os
import pandas as pd
import numpy as np


dataset_path = "dataset"

data = []

# Loop through each folder (light, medium, dark)
for label in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, label)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue
        # Directly use image (already face)
        face = cv2.resize(img, (100, 100))

        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)

        h_val = np.mean(hsv[:, :, 0])
        s_val = np.mean(hsv[:, :, 1])
        v_val = np.mean(hsv[:, :, 2])

        data.append([h_val, s_val, v_val, label])
       

# Create DataFrame
df = pd.DataFrame(data, columns=['hue', 'saturation', 'value', 'skin_tone'])

# Save CSV
df.to_csv("skin_dataset.csv", index=False)

print("CSV file created successfully ✅")