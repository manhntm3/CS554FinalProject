# CS554 Final Project
This is a repo for final project of class CS554 taught by professor Mansky at UIC. The syllabus of the course could be found [here](https://www.cs.uic.edu/~mansky/teaching/cs554/fa25/syllabus.html).

The goal of this project is to investigate weak memory behaviour on Nvidia GPU. 
The final report of this project could be found [here](CS554FinalProject.pdf).


Build and run: 
```
nvcc -std=c++17 -arch=sm_75 ./litmus_tests.cu -o litmus_test
./litmus_test
```
