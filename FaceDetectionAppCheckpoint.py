import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# Title
st.title("🖼️ Viola-Jones Face Detection App")

# Instructions
st.markdown("""
### 📌 Instructions:
1. **Upload** an image in JPG or PNG format.  
2. **Pick a rectangle color** for detected faces.  
3. **Adjust parameters** for `scaleFactor` and `minNeighbors` to fine-tune detection.  
4. Click **'Detect Faces'** to run the algorithm.  
5. **Download** the processed image with detected faces.
---
""")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Color picker for rectangle
rect_color = st.color_picker("Choose rectangle color", "#FF0000")  # default red
# Convert hex to BGR for OpenCV
rect_color_bgr = tuple(int(rect_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

# Sliders for parameters
scale_factor = st.slider("Scale Factor", min_value=1.05, max_value=2.0, value=1.1, step=0.05)
min_neighbors = st.slider("Min Neighbors", min_value=1, max_value=10, value=5, step=1)

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    if st.button("🔍 Detect Faces"):
        # Detect faces
        faces = face_cascade.detectMultiScale(
            image_gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors
        )

        # Draw rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(image_np, (x, y), (x + w, y + h), rect_color_bgr, 2)

        # Show result
        st.image(image_np, caption=f"Detected {len(faces)} face(s)", use_column_width=True)

        # Save to file
        save_path = "detected_faces.jpg"
        cv2.imwrite(save_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        # Download button
        with open(save_path, "rb") as file:
            btn = st.download_button(
                label="💾 Download Image",
                data=file,
                file_name="faces_detected.jpg",
                mime="image/jpeg"
            )
