import cv2
import numpy as np
from rembg import remove

def adjust_contrast(image, alpha=1.5, beta=2):
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image

def remove_background_and_add_contours(roi, original_image, roi_coordinates,input_image_path):
    if input_image_path == "2.jpg" :
        roi_adjusted = adjust_contrast(roi)
    else:
        roi_adjusted = roi

    _, temp_input_path = cv2.imencode('.png', roi_adjusted)
    roi_bytes = temp_input_path.tobytes()

    output_bytes = remove(roi_bytes)

    output_np = np.frombuffer(output_bytes, dtype=np.uint8)
    processed_roi = cv2.imdecode(output_np, cv2.IMREAD_UNCHANGED)

    gray_processed_roi = cv2.cvtColor(processed_roi, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(gray_processed_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_shifted = [contour + roi_coordinates[:2] for contour in contours]
    cv2.drawContours(original_image, contours_shifted, -1, (0, 255, 0), 3)

    return original_image

def main():

    input_image_path = "3.jpg"

    while True:

        original_image = cv2.imread(input_image_path)

        image_height, image_width = original_image.shape[:2]

        window_width = min(1200, image_width) 
        window_height = min(800, image_height)  

        if image_width > window_width or image_height > window_height:
            original_image = cv2.resize(original_image, (window_width, window_height))

        display_image = original_image.copy()

        cv2.imshow("Original Image", display_image)

        roi_coordinates = cv2.selectROI("Select ROI", display_image, fromCenter=False, showCrosshair=False)

        x, y, w, h = roi_coordinates

        selected_roi = original_image[y:y + h, x:x + w]

        result_image = remove_background_and_add_contours(selected_roi, original_image, roi_coordinates,input_image_path)
        
        cv2.imshow("Result", result_image)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            cv2.destroyAllWindows()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
