//
// Created by leo on 3/12/24.
//

#include <iostream>
#include <cufft.h>
#include "AudioVisualizer.cuh"
#include "sndfile.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <QApplication>
#include "VisualizerUI.cuh"

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

    const int chunkSize = 2048;
    const int overlap = chunkSize / 2;
    const int stepSize = chunkSize - overlap;
    size_t num_samples = sf_info.frames * sf_info.channels;
    thrust::host_vector<cufftReal> h_buffer(num_samples);

    sf_count_t num_read = sf_read_float(file, thrust::raw_pointer_cast(h_buffer.data()), num_samples);
    if (num_read != num_samples) {
        std::cerr << "Did not read expected number of samples." << std::endl;
        sf_close(file);
        return -1;
    }

    cufftHandle plan;
    cufftPlan1d(&plan, chunkSize, CUFFT_R2C, 1);

    ChunkVector cached_chunks;

    for (int start = 0; start + chunkSize <= num_samples; start += stepSize) {
        thrust::device_vector<cufftReal> d_buffer(h_buffer.begin() + start, h_buffer.begin() + start + chunkSize);
        thrust::device_vector<cufftComplex> d_output(chunkSize);


        cufftExecR2C(plan, thrust::raw_pointer_cast(d_buffer.data()), thrust::raw_pointer_cast(d_output.data()));
        cudaDeviceSynchronize();

        thrust::host_vector<cufftComplex> copied_buffer(chunkSize);
        thrust::copy(d_output.begin(), d_output.end(), copied_buffer.begin());
        cached_chunks.push_back(copied_buffer);
    }

    cufftDestroy(plan);
    sf_close(file);

    char* args[] = { (char*) "CUDAVisualizer"};
    int argc = 1;
    QApplication app(argc, args);

    MainWindow mainWindow(cached_chunks);
    mainWindow.show();

    return app.exec();
}

void AudioVisualizer::addParams(po::options_description *desc) {

}
