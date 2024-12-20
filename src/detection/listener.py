from ultralytics import YOLO
import numpy as np
from utils import readSharedCvmat, parseResultsIntoJson, writeJsonToSharedMemory
import cv2

def listener(shm_name: str):
    modelPath = "models/best.pt"
    model = YOLO(modelPath)

    # Assuming the image is in BGR format (OpenCV default)
    width, height, channels = 512, 424, 4
    imageShape = (height, width, channels)
    imageDtype = np.uint8

    while True:
        try:
            image, shm = readSharedCvmat(shm_name, imageShape, imageDtype)

            try:
                # Perform model inference
                cv2.waitKey(1)  # Display the image (non-blocking wait)
                results_list = model.predict(source=image[:, :, :3], show=True)
                # Process each `Results` object in the `results_list`
                detectionsJson = parseResultsIntoJson(results_list)
                writeJsonToSharedMemory("2d_coordinates", detectionsJson)  # type: ignore

            finally:
                shm.unlink()  # Free the shared memory
                shm.close()  # Release the shared memory handle

        # This execption if the shared_memory is not created yet
        except FileNotFoundError:
            pass

        # This expection if the frame in the shared_memory
        except ValueError:
            pass
        except Exception as e:
            print(f"{type(e)}: {e}")
            continue