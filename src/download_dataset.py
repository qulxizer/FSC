from detector.Detector import Detector
import sys
detector = Detector()


detector.download_dataset(
    api_key=sys.argv[1],
    path_to_dataset=sys.argv[2]
)