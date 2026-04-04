import cv2
import numpy as np

img = cv2.imread("test.jpg")

if img is None:
    print("Image not found!")
else:
    print("Image loaded successfully!")
    print("Original Shape:", img.shape)

    # Resize image
    height, width = img.shape[:2]
    max_width = 400
    scale_ratio = max_width / width
    new_height = int(height * scale_ratio)

    resized_img = cv2.resize(img, (max_width, new_height))

    # Convert to Grayscale (for face detection)
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    print("Number of faces detected:", len(faces))

    # 🔥 Recommendation Dictionary (Define Once)
    recommendations = {
        "Fair": {
            "Outfit": ["Pastel Blue", "Lavender", "Soft Pink",
                       "Mint Green", "Powder Blue", "Light Peach"],
            "Lipstick": ["Peach", "Soft Pink", "Nude", "Rose"],
            "Foundation": "Cool / Neutral"
        },
        "Medium": {
            "Outfit": ["Mustard", "Olive Green", "Teal",
                       "Rust", "Coral", "Deep Blue"],
            "Lipstick": ["Coral", "Rose", "Warm Red"],
            "Foundation": "Warm / Neutral"
        },
        "Olive": {
            "Outfit": ["Emerald", "Burgundy", "Terracotta",
                       "Forest Green", "Navy Blue"],
            "Lipstick": ["Brick Red", "Burnt Orange"],
            "Foundation": "Warm"
        },
        "Dark": {
            "Outfit": ["Royal Blue", "Fuchsia", "Bright Yellow",
                       "Crimson", "Turquoise"],
            "Lipstick": ["Berry", "Deep Plum", "Wine Red"],
            "Foundation": "Warm / Cool"
        }
    }

    # Loop through all detected faces
    for i, (x, y, w, h) in enumerate(faces):

        # Draw rectangle
        cv2.rectangle(resized_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Add face number label
        cv2.putText(resized_img, f"Face {i+1}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Crop face
        face_region = resized_img[y:y+h, x:x+w]

        # Convert cropped face to HSV
        face_hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)

        # Calculate average HSV values
        average_hsv = np.mean(face_hsv, axis=(0, 1))

        h_val, s_val, v_val = average_hsv

        # Skin Tone Classification
        if v_val > 200:
            skin_tone = "Fair"
        elif v_val > 160:
            skin_tone = "Medium"
        elif v_val > 120:
            skin_tone = "Olive"
        else:
            skin_tone = "Dark"

        rec = recommendations[skin_tone]

        # 🔥 Clean Structured Output
        print("\n" + "="*50)
        print(f"RESULT FOR FACE {i+1}")
        print("="*50)
        print(f"Average HSV: {average_hsv}")
        print(f"Detected Skin Tone: {skin_tone}")

        print("\nRecommended Outfit Colors:")
        for color in rec["Outfit"]:
            print("  •", color)

        print("\nRecommended Lipstick Shades:")
        for shade in rec["Lipstick"]:
            print("  •", shade)

        print("\nRecommended Foundation Type:")
        print("  •", rec["Foundation"])
        print("="*50)

    # Show full image with rectangles
    cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)
    cv2.imshow("Face Detection", resized_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()