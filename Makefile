init:
	mkdir build

run:
	g++ -o build/build src/main.cpp src/frameProcessing.cpp src/calculation.cpp $$(pkg-config --cflags --libs opencv4 freenect2)
	build/build