//
// Created by leo on 3/12/24.
//

#include <iostream>
#include <cufft.h>
#include "AudioVisualizer.cuh"
#include "sndfile.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

AudioVisualizer::AudioVisualizer() {

}

int AudioVisualizer::main(const po::variables_map &vm) {

    const char* file_path = "gettysburg.wav";
    SF_INFO sf_info;
    SNDFILE* file = sf_open(file_path, SFM_READ, &sf_info);
    if (file == nullptr) {
        std::cerr << "Failed to open file." << std::endl;
        return -1;
    }

    size_t num_samples = sf_info.frames * sf_info.channels;
    thrust::host_vector<cufftReal> h_buffer(num_samples);

    sf_count_t num_read = sf_read_float(file, thrust::raw_pointer_cast(h_buffer.data()), num_samples);
    if (num_read != num_samples) {
        std::cerr << "Did not read expected number of samples." << std::endl;
        sf_close(file);
        return -1;
    }

    thrust::device_vector<cufftReal> d_buffer = h_buffer;
    thrust::device_vector<cufftComplex> d_output(num_samples);

    cufftHandle plan;
    cufftPlan1d(&plan, num_samples, CUFFT_R2C, 1);

    cufftExecR2C(plan, thrust::raw_pointer_cast(d_buffer.data()), thrust::raw_pointer_cast(d_output.data()));
    cudaDeviceSynchronize();

    cufftDestroy(plan);

    thrust::host_vector<cufftComplex> copied_buffer = d_output;

    cufftComplex fifteen = copied_buffer[15];

    sf_close(file);
    return 0;
}

void AudioVisualizer::addParams(po::options_description *desc) {

}
