main.out:  main.o perceptron.o
	g++ -Wall -g  -larmadillo  main.o perceptron.o -o main

main.o: main.cpp
	g++ -Wall -g -larmadillo  -c main.cpp
perceptron.o: perceptron.cpp perceptron.h
	g++ -Wall -g -larmadillo -c perceptron.cpp
clean: 
	rm *.o 
