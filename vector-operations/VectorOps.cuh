//
// Created by leo on 2/17/24.
//

#ifndef CUDA_NOTEBOOKS_VECTOROPS_CUH
#define CUDA_NOTEBOOKS_VECTOROPS_CUH


#include "boost/program_options/variables_map.hpp"
#include "boost/program_options/options_description.hpp"

namespace po = boost::program_options;


class VectorOps {
public:
    VectorOps();
    int main(const po::variables_map& vm);
    void addParams(po::options_description *desc);
};


#endif //CUDA_NOTEBOOKS_VECTOROPS_CUH