init:
	mkdir build

run:
	g++ -o build/build src/main.cpp $$(pkg-config --cflags --libs opencv4 freenect2)
	build/build