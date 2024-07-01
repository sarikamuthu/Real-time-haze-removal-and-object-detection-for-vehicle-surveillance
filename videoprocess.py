import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import threading

from ultralytics import YOLO


model = YOLO("lightWeight.pt")


def segment_sky(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, sky_mask = cv2.threshold(hsv[:, :, 2], 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel)
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)
    non_sky_mask = cv2.bitwise_not(sky_mask)
    return non_sky_mask


def dehaze(frame, non_sky_mask):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    min_channel = np.min(frame, axis=2)
    dark_channel = cv2.erode(min_channel, np.ones((15, 15), np.uint8))
    atmospheric_light = np.max(dark_channel)
    omega = 0.85
    transmission = 1 - omega * (dark_channel / atmospheric_light)
    transmission = np.clip(transmission, 0.1, 1)
    epsilon = 0.001
    recovered_scene_radiance = np.zeros_like(frame, dtype=np.float32)
    for i in range(3):
        recovered_scene_radiance[:, :, i] = (frame[:, :, i].astype(np.float32) - atmospheric_light) / np.maximum(
            transmission, epsilon) + atmospheric_light
    for i in range(3):
        recovered_scene_radiance[:, :, i] = np.where(non_sky_mask == 255, recovered_scene_radiance[:, :, i],
                                                     frame[:, :, i])
    recovered_scene_radiance = np.clip(recovered_scene_radiance, 0, 255).astype(np.uint8)
    return recovered_scene_radiance

def dehaze_indoor(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    min_channel = np.min(frame, axis=2)
    dark_channel = cv2.erode(min_channel, np.ones((15, 15), np.uint8))
    atmospheric_light = np.max(dark_channel)
    omega = 0.85
    transmission = 1 - omega * (dark_channel / atmospheric_light)
    transmission = np.clip(transmission, 0.1, 1)
    epsilon = 0.001
    recovered_scene_radiance = np.zeros_like(frame, dtype=np.float32)
    for i in range(3):
        recovered_scene_radiance[:, :, i] = (frame[:, :, i].astype(np.float32) - atmospheric_light) / np.maximum(
            transmission, epsilon) + atmospheric_light

    recovered_scene_radiance = np.clip(recovered_scene_radiance, 0, 255).astype(np.uint8)
    return recovered_scene_radiance

def enhance_contrast(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y = clahe.apply(y)
    merged = cv2.merge((y, cr, cb))
    enhanced_image = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
    return enhanced_image


def enhance_saturation(image, saturation_scale=1.3):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    s = np.clip(s * saturation_scale, 0, 255).astype(np.uint8)
    enhanced_hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)


def apply_gamma_correction(image, gamma=1.1):
    inv_gamma = 1.0 / gamma
    table = (np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)])).astype("uint8")
    return cv2.LUT(image, table)


def correct_white_balance(frame):
    result = cv2.xphoto.createSimpleWB().balanceWhite(frame)
    return result


def enhance_image(image):
    image = enhance_saturation(image)
    image = apply_gamma_correction(image)
    image = correct_white_balance(image)
    return image


def perform_object_detection(cv_img):
    results = model.predict(cv_img)
    result = results[0]
    output = []
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([x1, y1, x2, y2, result.names[class_id], prob])
        detections.append((x1, y1, x2, y2, result.names[class_id], prob))
    return output


def draw_boxes(cv_img, boxes):
    for box in boxes:
        x1, y1, x2, y2, label, prob = box
        cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        


def process_video(video_path):
    print("Processing video:", video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return

    
    input_frame_rate = cap.get(cv2.CAP_PROP_FPS)
    print(f"Input Frame Rate: {input_frame_rate} FPS")

    
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        
        #non_sky_mask = segment_sky(frame)
        dehazed_frame = dehaze_indoor(frame)

        # Enhance dehazed frame
        dehazed_frame = enhance_image(dehazed_frame)


        
        side_by_side = np.hstack((frame, dehazed_frame))
        cv2.imshow('Original vs Dehazed', side_by_side)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    elapsed_time = time.time() - start_time
    output_frame_rate = frame_count / elapsed_time if elapsed_time > 0 else 0
    print(f"Output Frame Rate: {output_frame_rate} FPS")

    cap.release()
    cv2.destroyAllWindows()


def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
    if file_path:
        process_video(file_path)

def perform_detection(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return

    
    input_frame_rate = cap.get(cv2.CAP_PROP_FPS)
    print(f"Input Frame Rate: {input_frame_rate} FPS")

    
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

       
        non_sky_mask = segment_sky(frame)
        dehazed_frame = dehaze(frame, non_sky_mask)

        # Enhance dehazed frame
        dehazed_frame = enhance_image(dehazed_frame)

        
        boxes = perform_object_detection(dehazed_frame)
        draw_boxes(dehazed_frame, boxes)

        
        side_by_side = np.hstack((frame, dehazed_frame))
        cv2.imshow('Original vs Dehazed', side_by_side)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    elapsed_time = time.time() - start_time
    output_frame_rate = frame_count / elapsed_time if elapsed_time > 0 else 0
    print(f"Output Frame Rate: {output_frame_rate} FPS")

    cap.release()
    cv2.destroyAllWindows()

def obj_det():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
    if file_path:
        perform_detection(file_path)

def indoor_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return

    
    input_frame_rate = cap.get(cv2.CAP_PROP_FPS)
    print(f"Input Frame Rate: {input_frame_rate} FPS")

    
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        
        dehazed_frame = dehaze_indoor(frame)
        side_by_side = np.hstack((frame, dehazed_frame))
        cv2.imshow('Original vs Dehazed', side_by_side)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    elapsed_time = time.time() - start_time
    output_frame_rate = frame_count / elapsed_time if elapsed_time > 0 else 0
    print(f"Output Frame Rate: {output_frame_rate} FPS")

    cap.release()
    cv2.destroyAllWindows()

def indoor():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
    if file_path:
        indoor_video(file_path)

def quit_app():
    root = tk.Tk()
    root.quit()


def main():
    root = tk.Tk()
    root.title("Dehazing Application")
    button_bg_color = "#06038D"  
    button_fg_color = "white"     
    root.geometry("800x600")

    heading_label = tk.Label(root, text="Real-time haze removal and object detection for vehicle surveillance", font=('Helvetica', 18, 'bold'))
    heading_label.pack(pady=20)

    select_button = tk.Button(root, text="Outdoor Dehazing", command=select_file,bg=button_bg_color,fg=button_fg_color, font=('Helvetica', 12, 'bold'),width=20, height=2)
    select_button.pack(pady=20)

    detect_button = tk.Button(root, text="Perform Object Detection", command=obj_det,bg=button_bg_color, fg=button_fg_color, font=('Helvetica', 12, 'bold'),width=20, height=2)
    detect_button.pack(pady=10)

    indoor_button = tk.Button(root, text="Indoor Dehazing", command=indoor,bg=button_bg_color, fg=button_fg_color, font=('Helvetica', 12, 'bold'),width=20, height=2)
    indoor_button.pack(pady=10)

    quit_button = tk.Button(root, text="Quit", command=quit_app,bg=button_bg_color, fg=button_fg_color, font=('Helvetica', 12, 'bold'),width=20, height=2)
    quit_button.pack(pady=20)

    
    root.mainloop()

if __name__ == '__main__':
    main()