import cv2
import numpy as np
import json
from multiprocessing import shared_memory
from ultralytics import YOLO
from enum import Enum


class ClassNames(Enum):
    unripe = 2
    healthy_ripe = 1
    bad = 0


def readSharedCvmat(name, shape, dtype):
    # Attach to the shared memory segment
    shm = shared_memory.SharedMemory(name=name)
    # Create a NumPy array from shared memory
    array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return array, shm


def writeDetection(name, detection):
    array = np.array(detection, dtype=np.float32)
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=array.size)
        print("Shared memory created.")
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=name)
        print("Shared memory already exists.")

    buffer = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    buffer[:] = array[:]

    print(f"Data written to {name}:", buffer[:])


def writeJsonToSharedMemory(name, data: list):
    json_data = json.dumps(data)
    print(json_data)
    json_bytes = json_data.encode("utf-8")

    try:
        # Try to open shared memory, it will raise FileExistsError if it already exists
        shm = shared_memory.SharedMemory(name=name)
        print(f"Shared memory with name {name} already exists. Overwriting...")
        # Unlink the existing shared memory before overwriting
        shm.unlink()
    except FileNotFoundError:
        # Shared memory does not exist, no need to unlink
        pass
    
    # Create shared memory with the correct size
    shm = shared_memory.SharedMemory(name=name, create=True, size=len(json_bytes))
    # Write the JSON bytes to the shared memory buffer
    np.copyto(np.ndarray(len(json_bytes), dtype=np.uint8, buffer=shm.buf), np.frombuffer(json_bytes, dtype=np.uint8))
    print(f"JSON data written to shared memory {name}.")


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
            detectionsJson = []

            try:
                # Perform model inference
                cv2.waitKey(1)  # Display the image (non-blocking wait)
                results_list = model.predict(source=image[:, :, :3], show=True)

                # Process each `Results` object in the `results_list`
                for results in results_list:
                    if (
                        results.boxes is not None and results.boxes.xyxy.shape[0] > 0
                    ):  # Check if there are any detections
                        for box in results.boxes:
                            b = box.xyxy  # get box coordinates in (left, top, right, bottom) format
                            c = box.cls
                            bboxs = b.tolist()
                            detections = []
                            for bbox, className in zip(bboxs, c):
                                xCenter = (bbox[0] + bbox[2]) / 2 # type: ignore
                                yCenter = (bbox[1] + bbox[3]) / 2 # type: ignore
                                detections.append([xCenter, yCenter, className.item()])

                            for detection in detections:
                                xCenter = detection[0]
                                yCenter = detection[1]
                                detectionClass = ClassNames(detection[2])
                                detectionJson = {
                                    "Class": detectionClass.name,
                                    "xCenter": xCenter,
                                    "yCenter": yCenter,
                                }
                                detectionsJson.append(detectionJson)
                                print(
                                    f"xCenter: {xCenter}, yCenter: {yCenter}, , Class: {detectionClass.name}"
                                )
                            writeJsonToSharedMemory("2d_coordinates", detectionsJson)
                
                # if no detection were found
                
                if len(detectionsJson) < 1:
                    detectionsJson = {
                        "message": "no detections were found"
                    }
                    writeJsonToSharedMemory("2d_coordinates", detectionsJson) # type: ignore


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


if __name__ == "__main__":
    try:
        shm_name = "shared_image"
        print(f"Listening for new images at {shm_name} ")
        listener(shm_name)
    finally:
        cv2.destroyAllWindows()  # Ensure OpenCV windows are closed on exit
