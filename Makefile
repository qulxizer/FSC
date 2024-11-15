run:
	python3 ./src/main.py


.PHONY: test
test:
	# using sh explicitly to run test to avoid shell-specific syntax 
	sh -c "PYTHONPATH=$$(pwd) pytest -s"