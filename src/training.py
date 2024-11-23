from detector.Detector import Detector

detector = Detector()

detector.train(
    path_dataset_yaml="dataset/tomato_checker/data.yaml",
    path_training_runs="dataset/tomato_checker/"
)