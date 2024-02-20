//
// Created by leo on 2/17/24.
//

#include <iostream>
#include <algorithm>
#include <cstdlib>
#include "VectorOps.cuh"
#include <curand.h>
#include <curand_kernel.h>

#include "boost/program_options/variables_map.hpp"
#include "boost/program_options/options_description.hpp"

namespace po = boost::program_options;

__global__ void add_vecs(float *vec1, float *vec2, float *outVec) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    outVec[i] = vec1[i] + vec2[i];
}

__global__ void sub_vecs(float *vec1, float *vec2, float *outVec) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    outVec[i] = vec1[i] - vec2[i];
}

__global__ void dot_product_vecs(float *vec1, float *vec2, float *outVec) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    outVec[i] = vec1[i] * vec2[i];
}

void print_vector(float* vec, int size) {
    std::cout << "[ ";
    if (size <= 15) {
        for(int i=0;i<size;++i)
            std::cout << vec[i] << ' ';
    } else {
        for(int i=0;i<15;++i)
            std::cout << vec[i] << ' ';
        std::cout << "... " << vec[size-1];
    }
    std::cout << " ]" << std::endl;
}

int VectorOps::main(const po::variables_map &vm) {

    uint amountElements = 15;
    if (vm.count("numElements")) {
        amountElements = vm["numElements"].as<int>();
        std::cout << "Randomizing " << amountElements << " elements and performing operations" << std::endl;
    }

    float *vec1;
    float *vec2;
    cudaMalloc(&vec1, amountElements * sizeof(float));
    cudaMalloc(&vec2, amountElements * sizeof(float));
    float *add_result;
    float *sub_result;
    float *dot_prod_result;
    cudaMalloc(&add_result, amountElements * sizeof(float));
    cudaMalloc(&sub_result, amountElements * sizeof(float));
    cudaMalloc(&dot_prod_result, amountElements * sizeof(float));

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    curandGenerateUniform(gen, vec1, amountElements);
    curandGenerateUniform(gen, vec2, amountElements);

    curandDestroyGenerator(gen);

    int threadsPerBlock = 256;
    int blocksPerGrid = (amountElements + threadsPerBlock - 1) / threadsPerBlock;

    float* host_vec1 = new float[amountElements];
    cudaMemcpy(host_vec1, vec1, amountElements * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Vec1: ";
    print_vector(host_vec1,amountElements);

    float* host_vec2 = new float[amountElements];
    cudaMemcpy(host_vec2, vec2, amountElements * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Vec2: ";
    print_vector(host_vec2,amountElements);

    delete[] host_vec1;
    delete[] host_vec2;

    // Addition
    add_vecs<<<blocksPerGrid, threadsPerBlock>>>(vec1, vec2, add_result);

    // Subtraction
    sub_vecs<<<blocksPerGrid, threadsPerBlock>>>(vec1, vec2, sub_result);

    // Dot product
    dot_product_vecs<<<blocksPerGrid, threadsPerBlock>>>(vec1, vec2, dot_prod_result);

    // Synchronize Device
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    float* host_add_result = new float[amountElements];
    cudaMemcpy(host_add_result, add_result, amountElements * sizeof(float), cudaMemcpyDeviceToHost);
    float* host_sub_result = new float[amountElements];
    cudaMemcpy(host_sub_result, sub_result, amountElements * sizeof(float), cudaMemcpyDeviceToHost);
    float* host_dot_prod_result = new float[amountElements];
    cudaMemcpy(host_dot_prod_result, dot_prod_result, amountElements * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Addition output vector: ";
    print_vector(host_add_result, amountElements);

    std::cout << "Subtraction output vector: ";
    print_vector(host_sub_result, amountElements);

    float dotProduct = 0.0f;
    for (int i = 0; i < amountElements; i++) {
        dotProduct += host_dot_prod_result[i];
    }

    printf("Dot product: %.5f\n", dotProduct);

    cudaFree(vec1);
    cudaFree(vec2);
    cudaFree(add_result);
    cudaFree(sub_result);
    cudaFree(dot_prod_result);

    delete[] host_add_result;
    delete[] host_sub_result;
    delete[] host_dot_prod_result;

    return 0;
}

VectorOps::VectorOps() {

}

void VectorOps::addParams(po::options_description *desc) {
    desc->add_options() ("numElements,n", po::value<int>()->implicit_value(15), "Number of elements to generate");
}