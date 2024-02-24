#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <nvrtc.h>
#include <cstdlib>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <string>

#include <math_functions.h>

#define MAX_N_NAME 64U

// CUDA kernel for vector addition

enum Operation { Add, Multiply };
enum ArrayType { Scalar, Vector };

class VectorAdd {
public:
	VectorAdd(size_t n);
	size_t createInputVariable(int* vec);
	size_t applyOperation(Operation op, size_t id_a, size_t id_b);
	void declareOutputVariable(size_t id);
	void finalizeCalculation(std::vector<int*>& out);
	void freeVariable(size_t id);
	void releaseProgram(nvrtcProgram& p);
private:
	int NUM_THREADS;
	int NUM_BLOCKS;
	std::vector<int*> hostVarList_, deviceVarList_;
	std::vector<size_t> outputId_;
	std::vector<int> arrayTypes_;
	size_t n_;
	std::string source_;
};

VectorAdd::VectorAdd(size_t n): n_(n) {
	NUM_THREADS = 256;
	hostVarList_.clear();
	deviceVarList_.clear();
	arrayTypes_.clear();
	source_ = "extern \"C\" __global__ void operationKernel(int** input, int* arrayTypes, int n) {\n"
		      "    int tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
}

size_t VectorAdd::createInputVariable(int* vec) {
	int* dMem;
	deviceVarList_.push_back(dMem);
	arrayTypes_.push_back(Vector);
	hostVarList_.push_back(vec);
	//cudaMalloc(&deviceVarList_.back(), sizeof(int) * n_);
	//cudaMemcpy(deviceVarList_.back(), vec, sizeof(int) * n_, cudaMemcpyHostToDevice);
	return deviceVarList_.size() - 1;
}

size_t VectorAdd::applyOperation(Operation op, size_t id_a, size_t id_b) {
	int* d_res;
	hostVarList_.push_back(nullptr);
	//cudaMalloc(&d_res, sizeof(int) * n_);
	size_t id_res = deviceVarList_.size();
	NUM_BLOCKS = (n_ + NUM_THREADS - 1) / NUM_THREADS;
	// Compute vector addition in the device, do not consider scalar here
	if (op == Add)
		source_ += "if (tid < n) input[" + std::to_string(id_res) + "][tid] = input["
		+ std::to_string(id_a) + "][tid] + input[" + std::to_string(id_b) + "][tid];\n";
	deviceVarList_.push_back(d_res);
	arrayTypes_.push_back(Vector);
	return id_res;
}

void VectorAdd::declareOutputVariable(size_t id) {
	if (id < deviceVarList_.size()) {
		outputId_.push_back(id);
	}
	else {
		std::cerr << "Output variable not found on device" << std::endl;
	}
}

void VectorAdd::freeVariable(size_t id) {
	cudaFree(deviceVarList_[id]);
}

void VectorAdd::releaseProgram(nvrtcProgram& p) {
	nvrtcResult err = nvrtcDestroyProgram(&p);
	std::cout << "NVRTC Error: " << nvrtcGetErrorString(err) << std::endl;
}

void VectorAdd::finalizeCalculation(std::vector<int*>& out) {
	NUM_BLOCKS = (n_ + NUM_THREADS - 1) / NUM_THREADS;

	// generate int* arrayType
	int* h_arrayTypes = new int[arrayTypes_.size()];
	for (size_t i = 0; i < arrayTypes_.size(); i++) {
		h_arrayTypes[i] = arrayTypes_[i];
	}
	
	// Allocate memory and copy to device
	int** input;
	int* arrayTypes;
	cudaMalloc(&input, deviceVarList_.size() * sizeof(int*));
	cudaMalloc(&arrayTypes, arrayTypes_.size() * sizeof(int));
	for (size_t i = 0; i < deviceVarList_.size(); i++) {
		// Allocate memory in device
		cudaMalloc(&deviceVarList_[i], sizeof(int) * n_);
		// Copy the vector from host to device
		if (hostVarList_[i] != nullptr) {
			auto err = cudaMemcpy(deviceVarList_[i], hostVarList_[i], sizeof(int) * n_, cudaMemcpyHostToDevice);
		}
		auto err = cudaMemcpy(&input[i], &deviceVarList_[i], sizeof(int*), cudaMemcpyHostToDevice);
	}
	cudaMemcpy(arrayTypes, h_arrayTypes, arrayTypes_.size() * sizeof(int), cudaMemcpyHostToDevice);

	// finish source code
	source_ += "}";
	std::cout << source_ << std::endl;

	// compile source code
	nvrtcProgram program;
	nvrtcCreateProgram(&program, source_.c_str(), "kernel.cu", 0, nullptr, nullptr);

	const char* compileOptions[] = { "--gpu-architecture=compute_52", "-std=c++17", nullptr };
	nvrtcResult compileResult = nvrtcCompileProgram(program, 2, compileOptions);

	// Check compilation result
	if (compileResult != NVRTC_SUCCESS) {
		std::cerr << nvrtcGetErrorString(compileResult) << std::endl;
	}

	// Retrieve the compiled PTX code
	size_t ptxSize;
	nvrtcGetPTXSize(program, &ptxSize);
	char* ptx = new char[ptxSize];
	nvrtcGetPTX(program, ptx);
	nvrtcResult nvrtc_err = nvrtcDestroyProgram(&program);
	std::cout << "NVRTC Error: " << nvrtcGetErrorString(nvrtc_err) << std::endl;

	// Initialize CUDA context and module
	CUcontext cuContext;
	cuInit(0);
	CUresult cu_err = cuCtxCreate(&cuContext, 0, 0);
	if (cu_err != CUDA_SUCCESS) {
		const char* errStr;
		cuGetErrorString(cu_err, &errStr);
		std::cout << "CUDA Error: " << errStr << std::endl;
	}

	CUmodule cuModule;
	cuModuleLoadData(&cuModule, ptx);

	// Launch the kernel
	CUfunction cuFunction;
	cuModuleGetFunction(&cuFunction, cuModule, "operationKernel");

	void* args[] = { &input, &arrayTypes, &n_ };
	CUresult result = cuLaunchKernel(cuFunction, NUM_THREADS, 1, 1, NUM_BLOCKS, 1, 1, 0, 0, args, nullptr);
	if (result == CUDA_SUCCESS)
		std::cout << "Kernel launched successfully." << std::endl;
	else {
		const char* errorStr;
		cuGetErrorString(result, &errorStr);
		std::cerr << "CUDA error: " << errorStr<< std::endl;
	}

	// Allocate memory for output
	size_t i = 0;
	for (auto const& oId : outputId_) {
		out[i] = (int*)malloc(sizeof(int) * n_);
		// copy the output from device to host
		cudaMemcpy(out[i], deviceVarList_[oId], sizeof(int) * n_, cudaMemcpyDeviceToHost);
		//std::copy(h_res, h_res + n_, out[i].begin());
		++i;
	}
	std::cout << "cuCtxDestroy" << cuCtxDestroy(cuContext) << std::endl;
}

// Initialize vector of size n
void vector_init(int* a, int n) {
	for (int i = 0; i < n; i++) {
		a[i] = rand() % 100;
	}
}

// Check vector add result
void check_answer(int* a, int* b, int* c, int n) {
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
	size_t id_a = c.createInputVariable(h_a);
	size_t id_b = c.createInputVariable(h_b);
	size_t id_c = c.applyOperation(Operation::Add, id_a, id_b);
	id_c = c.applyOperation(Operation::Add, id_c, id_c);
	c.declareOutputVariable(id_c);
	std::vector<int*> output(1);
	c.finalizeCalculation(output);
	c.freeVariable(id_a);
	c.freeVariable(id_b);

	// Check result for errors
	check_answer(h_a, h_b, output[0], n);

	std::cout << "DONE" << std::endl;

	return 0;
}




