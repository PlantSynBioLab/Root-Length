#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

def find_horizontal_line_based_on_color_dip(image):
    y_coordinate = np.diff((image[:,:,0].mean(1))).argmax() + 3
    return y_coordinate

def annotate_image(image, text, position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 0, 0)  # Blue color
    line_type = 2
    cv2.putText(image, text, position, font, font_scale, font_color, line_type)

def process_image(image_path, csv_writer):
    image = cv2.imread(image_path)
    horizontal_line_y = find_horizontal_line_based_on_color_dip(image)
    cropped_image = image[horizontal_line_y:, :, :]
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 35, 50)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    longest_contour = None
    max_length = 0
    for contour in contours:
        length = cv2.arcLength(contour, True)
        if length > max_length:
            max_length = length
            longest_contour = contour

    if longest_contour is not None:
        cv2.drawContours(cropped_image, [longest_contour], -1, (0, 255, 0), 3)
        length_cm = max_length * 0.00183364  # Convert to cm
        annotate_image(cropped_image, f"Length: {length_cm:.2f} cm", (10, 50))
        csv_writer.writerow([os.path.basename(image_path), length_cm])

    save_path = os.path.splitext(image_path)[0] + "_processed.png"
    cv2.imwrite(save_path, cropped_image)

def process_folder(folder_path):
    csv_path = os.path.join(folder_path, 'image_lengths.csv')
    try:
        with open(csv_path, 'w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['Image Name', 'Length (cm)'])
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(folder_path, filename)
                    process_image(image_path, csv_writer)
        print(f"CSV file created at: {csv_path}")
    except Exception as e:
        print(f"Error: {e}")
# Example usage
process_folder("C:\\My data\\Virginia tech\\Wright Labs\\Epithitis model\\Roots\\Batch 1")


# In[ ]:




