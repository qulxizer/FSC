PYTHONPATH := $(shell pwd)/src

run:
	python3 ./src/main.py

getImages:
	sh -c "PYTHONPATH=$(PYTHONPATH) python3 ./src/getImages.py"

listPorts:
	sh -c "PYTHONPATH=$(PYTHONPATH) python3 ./src/listports.py"

calibrate:
	sh -c "PYTHONPATH=$(PYTHONPATH) python3 ./src/calibrate.py $(ARGS)"

# left first
# usage: make calibrate "dataset/our_dataset/calibration/left_camera/" "dataset/our_dataset/calibration/right_camera/"

download_dataset:
	sh -c "PYTHONPATH=$(PYTHONPATH) python3 ./src/download_dataset.py $(ARGS)"

train:
	python3 ./src/training.py 