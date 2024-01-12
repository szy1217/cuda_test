#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdlib>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include <vector>

#define MAX_N_NAME 64U

// CUDA kernel for vector addition
__global__ void vectorAdd(int* a, int* b, int* c, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < n) {
		c[tid] = a[tid] + b[tid];
	}
}

enum Operation { Add, Multiply };

class VectorAdd {
public:
	VectorAdd(size_t n);
	std::vector<int*> hostVarList_, deviceVarList_;
	size_t createInputVariable(int* vec);
	int* applyOperation(Operation op, int* a, int* b);
	size_t applyOperation(Operation op, size_t id_a, size_t id_b);
	void declareOutputVariable(int* o);
	void declareOutputVariable(size_t id);
	void finalizeCalculation(std::vector<std::vector<int>>& out);
	void freeVariable(size_t id);
private:
	int NUM_THREADS;
	int NUM_BLOCKS;
	std::vector<int*> outputPtr_;
	size_t n_;
};

VectorAdd::VectorAdd(size_t n): n_(n) {
	NUM_THREADS = 256;
	hostVarList_.clear();
	deviceVarList_.clear();
}

size_t VectorAdd::createInputVariable(int* vec) {
	// Check if the vector is in device memory
	auto it = std::find(deviceVarList_.begin(), deviceVarList_.end(), vec);
	if (it == deviceVarList_.end()) {
		int* dMem;
		deviceVarList_.push_back(dMem);
		// Allocate memory in device
		cudaMalloc(&deviceVarList_.back(), sizeof(int) * n_);
		// Copy the vector from host to device
		cudaMemcpy(deviceVarList_.back(), vec, sizeof(int) * n_, cudaMemcpyHostToDevice);
		std::cout << " not found" << std::endl;
		return deviceVarList_.size() - 1;
	}
	else {
		std::cout << " found" << std::endl;
		return std::distance(deviceVarList_.begin(), it);
	}
}

int* VectorAdd::applyOperation(Operation op, int* a, int* b) {
	int* d_a = deviceVarList_[createInputVariable(a)];
	int* d_b = deviceVarList_[createInputVariable(b)];
	int* d_res;
	cudaMalloc(&d_res, sizeof(int) * n_);
	NUM_BLOCKS = (n_ + NUM_THREADS - 1) / NUM_THREADS;
	if (op == Add)
		vectorAdd << <NUM_BLOCKS, NUM_THREADS >> > (d_a, d_b, d_res, n_);
	deviceVarList_.push_back(d_res);
	return d_res;
}

size_t VectorAdd::applyOperation(Operation op, size_t id_a, size_t id_b) {
	int* d_a = deviceVarList_[id_a];
	int* d_b = deviceVarList_[id_b];
	int* d_res;
	// Allocate memory for result array
	cudaMalloc(&d_res, sizeof(int) * n_);
	NUM_BLOCKS = (n_ + NUM_THREADS - 1) / NUM_THREADS;
	// Compute vector addition in the device
	if (op == Add)
		vectorAdd << <NUM_BLOCKS, NUM_THREADS >> > (d_a, d_b, d_res, n_);
	deviceVarList_.push_back(d_res);
	return deviceVarList_.size() - 1;
}

void VectorAdd::declareOutputVariable(int* o) {
	auto it = std::find(deviceVarList_.begin(), deviceVarList_.end(), o);
	if (it != deviceVarList_.end()) {
		outputPtr_.push_back(o);
	}
	else {
		std::cout << "Output variable not found on device" << std::endl;
	}
}

void VectorAdd::declareOutputVariable(size_t id) {
	if (id < deviceVarList_.size()) {
		outputPtr_.push_back(deviceVarList_[id]);
	}
	else {
		std::cout << "Output variable not found on device" << std::endl;
	}
}

void VectorAdd::freeVariable(size_t id) {
	cudaFree(deviceVarList_[id]);
}

void VectorAdd::finalizeCalculation(std::vector<std::vector<int>>& out) {
	size_t i = 0;
	for (auto const& oPtr : outputPtr_) {
		int* h_res;
		h_res = (int*)malloc(sizeof(int) * n_);
		// copy the output from device to host
		cudaMemcpy(h_res, oPtr, sizeof(int) * n_, cudaMemcpyDeviceToHost);
		std::copy(h_res, h_res + n_, out[i].begin());
		++i;
	}
}

// Initialize vector of size n
void vector_init(int* a, int n) {
	for (int i = 0; i < n; i++) {
		a[i] = rand() % 100;
	}
}

// Check vector add result
void check_answer(int* a, int* b, std::vector<int> c, int n) {
	for (int i = 0; i < n; i++) {
		if (i == 0) {
			std::cout << a[0] << " " << b[0] << " " << c[0] << std::endl;
		}
		assert(c[i] == 2*(a[i] + b[i]));
	}
}

int main() {
	//test error
	int device_count;
	cudaError_t cudaRes = cudaGetDeviceCount(&device_count);
	std::cout << "device_count = " << device_count << std::endl;
	std::cout << "cudaRes = " << (cudaRes == cudaSuccess) << std::endl;

	cudaDeviceProp device_prop;
	cudaRes = cudaGetDeviceProperties(&device_prop, 0);
	char device_name[MAX_N_NAME];
	std::cout << "device_name = " << device_prop.name << std::endl;
	std::cout << "cudaRes = " << (cudaRes == cudaSuccess) << std::endl;

	// Vector size
	int n = 1 << 16;

	// Host vector pointers
	int* h_a;
	int* h_b;
	h_a = (int*)malloc(sizeof(int) * n);
	h_b = (int*)malloc(sizeof(int) * n);

	vector_init(h_a, n);
	vector_init(h_b, n);

	VectorAdd c(n);
	/*
	int* d_c;
	d_c = c.applyOperation(Operation::Add, h_a, h_b);
	d_c = c.applyOperation(Operation::Add, d_c, d_c);
	c.declareOutputVariable(d_c);
	std::vector<std::vector<int>> output(1, std::vector<int>(n));
	c.finalizeCalculation(output);
	*/
	size_t id_a = c.createInputVariable(h_a);
	size_t id_b = c.createInputVariable(h_b);
	size_t id_c = c.applyOperation(Operation::Add, id_a, id_b);
	id_c = c.applyOperation(Operation::Add, id_c, id_c);
	c.freeVariable(id_a);
	c.freeVariable(id_b);
	c.declareOutputVariable(id_c);
	std::vector<std::vector<int>> output(1, std::vector<int>(n));
	c.finalizeCalculation(output);

	// Check result for errors
	check_answer(h_a, h_b, output[0], n);

	std::cout << "DONE" << std::endl;

	return 0;
}




