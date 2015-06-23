corner_harris: corner_harris.cpp
	g++ `pkg-config --cflags opencv` -c $< `pkg-config --libs opencv` -o $@

test: test.cpp corner_harris
	g++ `pkg-config --cflags opencv` $<  corner_harris `pkg-config --libs opencv` -o $@