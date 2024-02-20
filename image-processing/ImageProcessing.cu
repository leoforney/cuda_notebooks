//
// Created by leo on 2/18/24.
//
#include <cstdlib>
#include <iostream>
#include <filesystem>
#include "ImageProcessing.cuh"
#include "../CImg.h"

__global__ void makeImageGrayscale(const float *imageData, int width, int height, float *outputData) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int idx = 3 * (row * width + col);

        float r = imageData[idx];
        float g = imageData[idx + 1];
        float b = imageData[idx + 2];

        float grayscaleValue =  0.299f * r + 0.587f * g + 0.114f * b;

        outputData[idx] = grayscaleValue;
        outputData[idx + 1] = grayscaleValue;
        outputData[idx + 2] = grayscaleValue;
    }
}

int ImageProcessing::main(const po::variables_map &vm) {
    std::string filename = "image.jpg";

    if (vm.count("file")) {
        filename = vm["file"].as<std::string>();
    }

    cimg_library::CImg<unsigned char> image(filename.c_str());

    int numPixels = image.width() * image.height();
    int numFloats = numPixels * 3;

    std::vector<float> flatData(numFloats);

    int floatIndex = 0;
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            // Fetch color components (assumes R, G, B order)
            unsigned char red = image.atXY(x, y, 0);
            unsigned char green = image.atXY(x, y, 1);
            unsigned char blue = image.atXY(x, y, 2);

            flatData[floatIndex++] = float(red) / 255.0f;
            flatData[floatIndex++] = float(green) / 255.0f;
            flatData[floatIndex++] = float(blue) / 255.0f;
        }
    }

    float *d_imgData;
    float *d_outputData;
    cudaMalloc(&d_imgData, flatData.size() * sizeof(float));
    cudaMalloc(&d_outputData, flatData.size() * sizeof(float));

    cudaMemcpy(d_imgData, flatData.data(), flatData.size() * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);

    dim3 gridSize((image.width() + blockSize.x - 1) / blockSize.x,
                  (image.height() + blockSize.y - 1) / blockSize.y);

    makeImageGrayscale<<<gridSize, blockSize>>>(d_imgData,  image.width(), image.height(), d_outputData);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    float *h_imgData = new float[flatData.size()];
    cudaMemcpy(h_imgData, d_outputData, flatData.size() * sizeof(float), cudaMemcpyDeviceToHost);

    floatIndex = 0;
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            unsigned char red = static_cast<unsigned char>(h_imgData[floatIndex++] * 255.0f);
            unsigned char green = static_cast<unsigned char>(h_imgData[floatIndex++] * 255.0f);
            unsigned char blue = static_cast<unsigned char>(h_imgData[floatIndex++] * 255.0f);

            image.atXY(x, y, 0, 0) = red;
            image.atXY(x, y, 0, 1) = green;
            image.atXY(x, y, 0, 2) = blue;
        }
    }

    delete[] h_imgData;

    cudaFree(d_imgData);
    cudaFree(d_outputData);

    std::filesystem::path p(filename);
    std::string outputImageName = p.stem().string() + "_grayscale" + p.extension().string();

    if (vm.count("output")) {
        outputImageName = vm["output"].as<std::string>();
    }

    image.save(outputImageName.c_str());
    return 0;
}

void ImageProcessing::addParams(po::options_description *desc) {
    desc->add_options()
            ("file,f", po::value<std::string>()->implicit_value("image.jpg"), "Input filename")
            ("output,of", po::value<std::string>(), "Output filename");
}

ImageProcessing::ImageProcessing() {
}