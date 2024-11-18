PYTHONPATH := $(shell pwd)/src

run:
	python3 ./src/main.py

.PHONY: test

getImages:
	sh -c "PYTHONPATH=$(PYTHONPATH) python3 ./src/getImages.py"

listPorts:
	sh -c "PYTHONPATH=$(PYTHONPATH) python3 ./src/listports.py"

calibrate:
	sh -c "PYTHONPATH=$(PYTHONPATH) python3 ./src/calibrate.py"


download_dataset:
	sh -c "PYTHONPATH=$(PYTHONPATH) python3 ./src/download_dataset.py $(ARGS)"


train:
	python3 ./src/training.py 