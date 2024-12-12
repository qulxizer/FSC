init:
	mkdir build

run:
	g++ -o build/build $$(ls src/*.cpp | grep -v 'main.cpp') $$(pkg-config --cflags --libs opencv4 freenect2) -Isrc/
	build/build
