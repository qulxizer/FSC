PYTHONPATH := $(shell pwd)/src

run:
	python3 ./src/main.py

getImages:
	sh -c "PYTHONPATH=$(PYTHONPATH) python3 ./src/getImages.py"

listPorts:
	sh -c "PYTHONPATH=$(PYTHONPATH) python3 ./src/listPorts.py"

# left first + use full path please
# usage: make calibrate "/home/$USER/repo-location/dataset/our_dataset/calibration/left_camera/" "/home/$USER/repo-location/dataset/our_dataset/calibration/right_camera/"
calibrate:
	sh -c "PYTHONPATH=$(PYTHONPATH) python3 ./src/calibrate.py $(ARGS)"


downloadDataset:
	-mkdir $(DOWNLOAD_DIRECTORY)
	curl -L -o $(DOWNLOAD_DIRECTORY)/tomato_checker.zip https://universe.roboflow.com/ds/Z3DdGaSUAo?key=rR58Mle25h
	unzip $(DOWNLOAD_DIRECTORY)/tomato_checker.zip -d $(DOWNLOAD_DIRECTORY)
	rm $(DOWNLOAD_DIRECTORY)/tomato_checker.zip

train:
	python3 ./src/training.py  