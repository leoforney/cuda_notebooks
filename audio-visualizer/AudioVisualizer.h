//
// Created by leo on 3/12/24.
//

#ifndef CUDA_NOTEBOOKS_AUDIOVISUALIZER_H
#define CUDA_NOTEBOOKS_AUDIOVISUALIZER_H

#include "boost/program_options/variables_map.hpp"
#include "boost/program_options/options_description.hpp"

namespace po = boost::program_options;

class AudioVisualizer {
public:
    AudioVisualizer();
    int main(const po::variables_map& vm);
    void addParams(po::options_description *desc);
};


#endif //CUDA_NOTEBOOKS_AUDIOVISUALIZER_H
