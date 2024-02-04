import cv2
import numpy as np

def detect_green_polka_dots(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 50, 25])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    green_dot_centers = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                green_dot_centers.append((cx, cy))

    return green_dot_centers

def draw_red_dots(frame, green_dot_centers):
    for center in green_dot_centers:
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

    return frame

def main(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(5)

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        green_dot_centers = detect_green_polka_dots(frame)

        frame_with_red_dots = draw_red_dots(frame.copy(), green_dot_centers)

        out.write(frame_with_red_dots)

        cv2.imshow("Output", frame_with_red_dots)
        if cv2.waitKey(1) & 0xFF == 27: # Press 'Esc' to exit 
            break

    cap.release()
    out.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video_path = "task_2_video.mp4"
    output_video_path = "output_video.mp4"
    main(input_video_path, output_video_path)
