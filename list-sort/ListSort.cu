//
// Created by leo on 2/21/24.
//

#include "ListSort.cuh"
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <random>
#include <set>
#include <algorithm>
#include <memory>

#define MAX_THREADS 512
#define MAX_BLOCKS 32768

__global__ void bitonic_sort_step(float *dev_values, int j, int k) {
    unsigned int i, ixj;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    ixj = i^j;

    if ((ixj) > i) {
        if ((i&k) == 0) {
            if (dev_values[i] > dev_values[ixj]) {
                float temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
        if ((i&k) != 0) {
            if (dev_values[i] < dev_values[ixj]) {
                float temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
    }
}

// Done with assist from https://gist.github.com/mre/1392067
void bitonic_sort(float *values, int n_vals) {
    float *dev_values;
    size_t size = n_vals * sizeof(float);

    cudaMalloc((void**) &dev_values, size);
    cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

    dim3 blocks(min(n_vals/MAX_THREADS, MAX_BLOCKS),1);
    dim3 threads(min(n_vals, MAX_THREADS),1);

    for (int k = 2; k <= n_vals; k <<= 1) {
        for (int j = k >> 1; j > 0; j = j >> 1)
            bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
    }
    cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
    cudaFree(dev_values);
}

std::vector<float> ListSort::sortElements(std::vector<float> elements) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 100.0);

    const size_t initialCount = elements.size();
    const size_t minSize = MAX_THREADS;

    size_t finalCount = std::max(minSize, initialCount);

    int nearestPowerOf2 = std::pow(2, std::ceil(std::log2(finalCount)));

    float placeholderValue;
    do {
        placeholderValue = dis(gen);
    } while (std::find(elements.begin(), elements.end(), placeholderValue) != elements.end());

    unsigned long amountOfItemsToInsert = nearestPowerOf2 - elements.size();
    std::cout << "Inserting " << amountOfItemsToInsert << " placeholder elements in order to reach a multiple of 2" << std::endl;
    for (int i = 0; i < amountOfItemsToInsert; i++) {
        elements.push_back(placeholderValue);
    }

    std::unique_ptr<float[]> cArrayData(new float[nearestPowerOf2]);
    std::copy(elements.begin(), elements.end(), cArrayData.get());

    bitonic_sort(cArrayData.get(), nearestPowerOf2);

    std::vector<float> sortedData;

    for (int i = 0; i < nearestPowerOf2; i++) {
        if (cArrayData[i] != 0.0f && cArrayData[i] != placeholderValue)
            sortedData.push_back(cArrayData[i]);
    }
    return sortedData;
}

ListSort::ListSort() {

}

int ListSort::main(const po::variables_map &vm) {
    uint amountElements = 15;
    if (vm.count("numElements")) {
        amountElements = vm["numElements"].as<int>();
        std::cout << "Randomizing " << amountElements << " elements and performing operations" << std::endl;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 100.0);

    const size_t initialCount = 510;
    const size_t minSize = MAX_THREADS;

    std::vector<float> data;
    for (int i = 0; i < initialCount; i++) {
        data.push_back(dis(gen));
    }

    std::cout << "Unsorted data: ";
    for (float i : data) {
        std::cout << i << ", ";
    }
    std::cout << std::endl;

    std::vector<float> sortedElements = sortElements(data);

    std::cout << "Sorted data: ";
    for (float i : sortedElements) {
        std::cout << i << ", ";
    }

    return 0;
}

void ListSort::addParams(po::options_description *desc) {
    //desc->add_options() ("numElements,n", po::value<int>()->implicit_value(15), "Number of elements to generate");
}
