/*
* Kirin Sharma
* CS-425 Advanced Architecture
* CUDA Assignment 1
*
* This program creates and runs a CUDA kernel to modify a matrix based on a specific criteria
* in parallel using 1 block with 10x10 threads in the block.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cuda.h>
#include <stdlib.h>

using namespace std;

// CUDA kernel function to modify the matrix such that each element Mij = Mij * Vj
__global__ void modify(int *matrix, int *vector, int size)
{
    int row = threadIdx.y;
    int col = threadIdx.x;	
    int index = row * size + col;
	matrix[index] = matrix[index] * vector[col];
}

// Helper function to print a matrix
void printMatrix(int* matrix, int size) {
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            cout << matrix[size * i + j] << "   ";
        }
        cout << "\n";
    }
}

// Helper function to print a vector
void printVector(int* vector, int size) {
    for(int i = 0; i < size; i++) {
        cout << vector[i] << "   ";
    }
}

int main()
{
    // Allocate unified memory for the (flattened) 2-D matrix and vector
	int size = 10;
	int *matrix;
	int *vector;
    cudaMallocManaged(&matrix, size * size * sizeof(int));
    cudaMallocManaged(&vector, size * sizeof(int));

    // Initialize the matrix and vector
    for(int i = 0; i < size; i++) {
        vector[i] = i + 1;
        for(int j = 0; j < size; j++) {
            matrix[i * size + j] = j + 1;
        }
    }

    // Print the original vector
    cout << "Original Vector:\n";
    printVector(vector, size);
    cout << "\n\n";

    // Print the original matrix
    cout << "Original Matrix:\n";
    printMatrix(matrix, size);
    cout << "\n\n";

    // Specify block dimension to be 10x10 threads and grid dimension to be 1x1
    dim3 gridDimension(1, 1);
    dim3 blockDimension(10, 10);

    // Launch the cuda kernel
    modify<<<gridDimension, blockDimension>>>(matrix, vector, size);
    cudaDeviceSynchronize();

    // Verify matrix is modified correctly by printing to the screen
    // Expected to be the squares of the numbers 1-10 in each row
    cout << "Modified Matrix:\n";
    printMatrix(matrix, size);

    // Free cuda memory
    cudaFree(matrix);
    cudaFree(vector);

    return 0;

} // end main
