//
// Created by leo on 3/15/24.
//

#ifndef CUDA_NOTEBOOKS_AUDIOPROCESSOR_CUH
#define CUDA_NOTEBOOKS_AUDIOPROCESSOR_CUH

#include "VisualizerConstant.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cufft.h>

ChunkVector transformAudio(std::vector<cufftReal> &readData) {
    const int chunkSize = 2048;
    const int overlap = chunkSize / 2;
    const int stepSize = chunkSize - overlap;

    cufftHandle plan;
    cufftPlan1d(&plan, chunkSize, CUFFT_R2C, 1);

    ChunkVector cached_chunks;

    for (int start = 0; start + chunkSize <= readData.size(); start += stepSize) {
        thrust::device_vector<cufftReal> d_buffer(readData.begin() + start, readData.begin() + start + chunkSize);
        thrust::device_vector<cufftComplex> d_output(chunkSize);

        cufftExecR2C(plan, thrust::raw_pointer_cast(d_buffer.data()), thrust::raw_pointer_cast(d_output.data()));
        cudaDeviceSynchronize();

        ComplexVector copied_buffer(chunkSize);
        thrust::copy(d_output.begin(), d_output.end(), copied_buffer.begin());

        cached_chunks.push_back(copied_buffer);
    }

    cufftDestroy(plan);

    return cached_chunks;
}

#endif //CUDA_NOTEBOOKS_AUDIOPROCESSOR_CUH
