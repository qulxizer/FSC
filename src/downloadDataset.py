from detector.Detector import Detector
import sys
detector = Detector()

# print(os.getcwd())
detector.download_dataset(
    api_key=sys.argv[1],
    path_to_dataset="dataset/"
)