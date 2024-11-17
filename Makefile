

PYTHONPATH := $(shell pwd)/src

run:
	python3 ./src/main.py


.PHONY: test
test:
	@# using sh explicitly to run test to avoid shell-specific syntax 
	sh -c "PYTHONPATH=$(PYTHONPATH) pytest -s"

getImages:
	sh -c "PYTHONPATH=$(PYTHONPATH) python3 ./src/getImages.py"

listPorts:
	sh -c "PYTHONPATH=$(PYTHONPATH) python3 ./src/listports.py"

calibrate:
	sh -c "PYTHONPATH=$(PYTHONPATH) python3 ./src/calibrate.py"