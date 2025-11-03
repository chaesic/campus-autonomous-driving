import cv2
from ultralytics import YOLO

POSE_MODEL  = "/home/jaemin/yolopose/model/yolo11s-pose.pt"
VIDEO_PATH  = "/home/jaemin/JAAD/JAAD_clips/video_0006.mp4"

def main():
    model = YOLO(POSE_MODEL)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

    window_name = "test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)  

    while True: 
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, verbose=False)
        plotted = results[0].plot()

        cv2.imshow(window_name, plotted)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
