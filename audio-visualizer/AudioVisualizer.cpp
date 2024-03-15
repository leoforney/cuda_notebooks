//
// Created by leo on 3/12/24.
//

#include <iostream>
#include <cufft.h>
#include "AudioVisualizer.h"
#include "sndfile.h"
#include <QApplication>
#include "VisualizerUI.h"
#include "AudioProcessor.cuh"

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

    std::vector<cufftReal> readData(num_samples);

    sf_count_t num_read = sf_read_float(file, readData.data(), num_samples);
    if (num_read != num_samples) {
        std::cerr << "Did not read expected number of samples." << std::endl;
        sf_close(file);
        return -1;
    }

    thrust::host_vector<cufftReal> h_buffer(num_samples);

    ChunkVector cached_chunks = transformAudio(readData);

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
