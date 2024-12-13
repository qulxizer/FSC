from multiprocessing.shared_memory import SharedMemory
import numpy as np
import cv2


def read_shared_cvmat(name, shape, dtype):
    # Attach to the shared memory segment
    shm = SharedMemory(name=name)

    # Create numpy array from shared memory
    array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    return array, shm


if __name__ == "__main__":
    # Define the shape and type of the image
    img_shape = (640, 640, 3)  # Example: Height, Width, Channels
    img_dtype = np.uint8  # Example: CV_8UC3 -> np.uint8

    # Read the image from shared memory
    image, shm = read_shared_cvmat("shared_image", img_shape, img_dtype)

    # Display the image
    cv2.imshow("Shared Image", image)
    cv2.waitKey(0)

    # Cleanup
    shm.close()
    shm.unlink()  # Remove shared memory when done
