import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/detect_nano', methods=['POST'])
def detect_nano():
    # Load the SEM image from the request
    img = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to smooth the image and reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image to create a binary mask of the nanoparticles
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours of the nanoparticles in the mask
    cnts, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define the thresholds for circularity and aspect ratio to classify nanoparticles
    circularity_thresh = 0.9
    aspect_ratio_thresh = 1.3

    # Initialize counters for each nanoparticle class
    num_spheres = 0
    num_rods = 0
    num_cubes = 0

    # Loop over the contours and classify each nanoparticle
    for c in cnts:
        # Calculate the area and perimeter of the contour
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)

        # Calculate the circularity and aspect ratio of the contour
        circularity = 4 * np.pi * area / (perimeter ** 2)
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = float(w)/h if w > h else float(h)/w

        # Classify the nanoparticle based on its circularity and aspect ratio
        if circularity > circularity_thresh:
            if aspect_ratio > aspect_ratio_thresh:
                num_rods += 1
            else:
                num_spheres += 1
        else:
            num_cubes += 1

    # Return the number of nanoparticles in each class as JSON
    result = {
        'spheres': num_spheres,
        'rods': num_rods,
        'cubes': num_cubes
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
