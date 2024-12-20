import numpy as np
import json
from multiprocessing import shared_memory
from ultralytics import YOLO
from enums import ClassNames

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
    # Dumping the list into json object and encoding it to utf-8
    json_data = json.dumps(data)
    json_bytes = json_data.encode("utf-8")

    shm = None
    try:
        # Creating shared_memory with buffer size of 1 megabyte
        shm = shared_memory.SharedMemory(name=name, create=True, size=1000)
        print("Shared memory created.")

    # If there is already shared_memory instance with the same name we use it
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=name)
        print("Shared memory already exists.")

    # Write the JSON bytes to the shared memory buffer
    if shm is not None:
        np.copyto(
            np.ndarray(len(json_bytes), dtype=np.uint8, buffer=shm.buf),
            np.frombuffer(json_bytes, dtype=np.uint8),
        )
        print(f"JSON data written to shared memory {name}.")


def parseResultsIntoJson(results_list):
    # For extra information about what is happenning here i recommend reading the docs for ultralytics
    detectionsJson = []
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
                    xCenter = (bbox[0] + bbox[2]) / 2  # type: ignore
                    yCenter = (bbox[1] + bbox[3]) / 2  # type: ignore
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

    if len(detectionsJson) < 1:
        detectionsJson = {"message": "no detections were found"}

    return detectionsJson



