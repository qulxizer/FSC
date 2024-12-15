import time
import cv2
import numpy as np
from multiprocessing import shared_memory
from ultralytics import YOLO
from ultralytics.engine.results import Results


def read_shared_cvmat(name, shape, dtype):
    # Attach to the shared memory segment
    shm = shared_memory.SharedMemory(name=name)
    # Create a NumPy array from shared memory
    array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return array, shm


def listener():
    shm_name = "shared_image"  # Shared memory name
    model_path = "models/best.pt"
    model = YOLO(model_path)

    # Assuming the image is in BGR format (OpenCV default)
    width, height, channels = 512, 424, 4
    image_shape = (height, width, channels)
    image_dtype = np.uint8

    while True:
        try:
            image, shm = read_shared_cvmat(shm_name, image_shape, image_dtype)
            try:
                # Perform model inference

                cv2.imshow("Shared Memory Image", image[:, :, :3])
                cv2.waitKey(1)  # Display the image (non-blocking wait)
                results_list = model.predict(source=image[:, :, :3])

                # Process each `Results` object in the `results_list`
                for results in results_list:
                    if (
                        results.boxes is not None and results.boxes.xyxy.shape[0] > 0
                    ):  # Check if there are any detections
                        # Extract bounding boxes
                        x1, y1, x2, y2 = results.boxes.xyxy
                        confidences = (
                            results.boxes.conf
                        )  # Confidence scores as a numpy array
                        class_ids = results.boxes.cls  # Class IDs as a numpy array

                        print("Bounding Boxes:", x1, y2, x2, y2)
                        print("Confidences:", confidences)
                        print("Class IDs:", class_ids)
                    else:
                        print("No bounding boxes detected in this result.")
            finally:
                shm.unlink()  # Free the shared memory
                shm.close()  # Release the shared memory handle

            # Optional sleep for controlling CPU usage
            time.sleep(0.1)

        except FileNotFoundError:
            print(f"Shared memory '{shm_name}' not found, retrying in 1 second...")
            time.sleep(1)  # Retry after 1 second
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(1)  # Retry after handling the error


if __name__ == "__main__":
    try:
        listener()
    finally:
        cv2.destroyAllWindows()  # Ensure OpenCV windows are closed on exit
