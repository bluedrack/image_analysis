# import os
import cv2

# import xml.etree.ElementTree as ET
# arr = os.listdir("./imgs")
# print(arr)
# for i in range(80):
#     # load color image
#     img = cv2.imread(f"./imgs/{i}.jpg", cv2.IMREAD_COLOR)

#     # load xml
#     tree 

import os
import xml.etree.ElementTree as ET
import numpy as np

# Function to extract bounding box coordinates from a single XML file
def extract_bounding_boxes_from_xml(xml_file):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Initialize an empty list to store bounding box coordinates
    bounding_boxes = []

    # Iterate over each object element
    for obj in root.findall('object'):
        # Extract the bounding box coordinates
        xmin = int(obj.find('bndbox').find('xmin').text)
        xmax = int(obj.find('bndbox').find('xmax').text)
        ymin = int(obj.find('bndbox').find('ymin').text)
        ymax = int(obj.find('bndbox').find('ymax').text)

        # Append the bounding box coordinates to the list
        bounding_boxes.append([xmin, ymin, xmax, ymax])

    # Convert the list of bounding box coordinates to a NumPy array
    bounding_boxes_array = np.array(bounding_boxes)
    
    return bounding_boxes_array

# Path to the folder containing XML files
folder_path = r"./imgs/"

# Initialize an empty list to store bounding box arrays for all XML files
bounding_boxes_all_files = []

def sort_by_numeric_value(filename):
    return int(os.path.splitext(filename)[0])

for filename in sorted(os.listdir(folder_path), key=sort_by_numeric_value):
    if filename.endswith('.xml'):
        # print(filename)
        xml_file_path = os.path.join(folder_path, filename)
        bounding_boxes_array = extract_bounding_boxes_from_xml(xml_file_path)
        for box in bounding_boxes_array:

            bounding_boxes_all_files.append({"img" : filename.removesuffix(".xml"), "bounding": box})

#np.save(r"C:\Users\USER\Documents\Training_coins\bounding_boxes.npy", bounding_boxes_all_files)

# Concatenate bounding box arrays from all XML files into a single NumPy array
#bounding_boxes_combined = np.concatenate(bounding_boxes_all_files, axis=0)

# Print the combined NumPy array containing bounding box coordinates
# print(bounding_boxes_all_files[0].shape)
#print(bounding_boxes_all_files)

coin_images =  []
print(bounding_boxes_all_files)
id = 0
for i in bounding_boxes_all_files:
    img = i["img"]
    bounding = i["bounding"]
    # Load the image
    img_path = os.path.join(folder_path, img + ".jpg")
    image = cv2.imread(img_path)
    # Bounding box
    # print(bounding)
    # for box in bounding:
    print("box:",bounding)
    xmin, ymin, xmax, ymax = bounding 
    # Extract img from bounding box
    roi = image[ymin:ymax, xmin:xmax]
    coin_images.append(roi)
    # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # Display the image with bounding boxes
    # cv2.imshow('image', image)
    # Save image
    cv2.imwrite(f"./output/{img}-{id}.jpg",roi)
    id += 1

# for id, img in enumerate(coin_images):
#     print("Save ", id)
#     cv2.imwrite(f"./output/{id}.jpg", img)
