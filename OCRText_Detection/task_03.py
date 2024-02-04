import os
import cv2
from easyocr import Reader

def read_images_from_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    return [os.path.join(folder_path, file) for file in image_files]

def run_easyocr(input_image_path):
    reader = Reader(['en'])
    result = reader.readtext(input_image_path, detail=1)
    return result

def rewrite_ocr_values(image_path, output_folder):
    result = run_easyocr(image_path)
    img = cv2.imread(image_path)

    for detection in result:
        text = detection[1]
        coordinates = detection[0]

        x_min, y_min = map(int, min(coordinates, key=lambda x: x[1]))
        x_max, y_max = map(int, max(coordinates, key=lambda x: x[1]))

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red rectangle outline

        x, y = x_min + 5, y_min + 50  

        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        color = (0, 255, 0)  # Green color for text
        thickness = 2

        cv2.putText(img, text, (x, y), font, 1, color, thickness, cv2.LINE_AA)

    output_path = os.path.join(output_folder, f"output_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, img)

if __name__ == "__main__":
    input_folder = "License_Images"
    output_folder = "License_Outputs"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = read_images_from_folder(input_folder)

    for image_path in image_paths:
        rewrite_ocr_values(image_path, output_folder)
