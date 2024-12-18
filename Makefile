run:
	@-mkdir build
	(cd build && cmake ..)
	make -C build/
	build/main
