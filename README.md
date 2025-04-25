# CUDA_Assignment1_KNS

## Choosing Grid and Block Dimensions
Because the matrix is 10x10 and we are performing the same modification on each entry, I chose the block dimension to be 10x10 for a thread to perform each modification concurrently. 
The block dimension is only 1x1 as only one block is necessary.

## Thread Mapping within Kernel
The threads are mapped to individual matrix elements by using the thread index to represent the row and column, then computing the correct index in the flattened 2-D array

## Expected Output
The expected result matrix should contain the same row 10 times, with those rows expected to be the squares of the numbers 1-10

## Compilation and execution
For the purposes of this project, one may simply paste the CUDA file into LeetGPU and execute on their free cloud service.
