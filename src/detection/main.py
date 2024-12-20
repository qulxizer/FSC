import cv2
from ultralytics import YOLO
from listener import listener






if __name__ == "__main__":
    try:
        shm_name = "shared_image"
        print(f"Listening for new images at {shm_name} ")
        listener(shm_name)
    finally:
        cv2.destroyAllWindows()  # Ensure OpenCV windows are closed on exit
