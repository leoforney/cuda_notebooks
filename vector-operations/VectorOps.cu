//
// Created by leo on 2/17/24.
//

#include <iostream>
#include <algorithm>
#include <cstdlib>
#include "VectorOps.cuh"

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

void print_vector(std::vector<float> vec) {
    std::cout << "[ ";
    std::for_each(vec.begin(), vec.end(), [](float i) { std::cout << i << ' '; });
    std::cout << " ]" << std::endl;
}

int VectorOps::main(const po::variables_map &vm) {

    uint amountElements = 15;
    if (vm.count("numElements")) {
        amountElements = vm["numElements"].as<int>();
        std::cout << "Randomizing " << amountElements << " elements and performing operations" << std::endl;
    }

    std::vector<float> vec1(amountElements);
    std::generate(vec1.begin(), vec1.end(), []() { return static_cast <float> (rand()) / static_cast <float> (RAND_MAX); });
    std::cout << "Vec1: ";
    print_vector(vec1);

    std::vector<float> vec2(amountElements);
    std::generate(vec2.begin(), vec2.end(), []() { return static_cast <float> (rand()) / static_cast <float> (RAND_MAX); });
    std::cout << "Vec2: ";
    print_vector(vec2);

    float *vec_result;
    float *vec1_data;
    float *vec2_data;
    cudaMalloc(&vec1_data, amountElements * sizeof(float));
    cudaMalloc(&vec2_data, amountElements * sizeof(float));
    cudaMalloc(&vec_result, amountElements * sizeof(float));

    cudaMemcpy(vec1_data, vec1.data(), amountElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vec2_data, vec2.data(), amountElements * sizeof(float), cudaMemcpyHostToDevice);

    add_vecs<<<1, amountElements>>>(vec1_data, vec2_data, vec_result);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    float* host_result = new float[amountElements];
    cudaMemcpy(host_result, vec_result, amountElements * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> output_vector(host_result, host_result + amountElements);
    std::cout << "Addition output vector: ";
    print_vector(output_vector);

    sub_vecs<<<1, amountElements>>>(vec1_data, vec2_data, vec_result);

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    cudaMemcpy(host_result, vec_result, amountElements * sizeof(float), cudaMemcpyDeviceToHost);

    output_vector = std::vector<float>(host_result, host_result + amountElements);
    std::cout << "Subtraction output vector: ";
    print_vector(output_vector);

    delete[] host_result;

    cudaFree(vec1_data);
    cudaFree(vec2_data);
    cudaFree(vec_result);

    return 0;
}

VectorOps::VectorOps() {

}

void VectorOps::addParams(po::options_description *desc) {
    desc->add_options() ("numElements,n", po::value<int>()->implicit_value(15), "Number of elements to generate");
}
