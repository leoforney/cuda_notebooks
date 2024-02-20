//
// Created by leo on 2/18/24.
//

#ifndef CUDA_NOTEBOOKS_IMAGEPROCESSING_CUH
#define CUDA_NOTEBOOKS_IMAGEPROCESSING_CUH

#include "boost/program_options/variables_map.hpp"
#include "boost/program_options/options_description.hpp"

namespace po = boost::program_options;

class ImageProcessing {
public:
    ImageProcessing();
    int main(const po::variables_map& vm);
    void addParams(po::options_description *desc);
};


#endif //CUDA_NOTEBOOKS_IMAGEPROCESSING_CUH
