//
// Created by leo on 2/17/24.
//

#ifndef CUDA_NOTEBOOKS_LISTSORT_CUH
#define CUDA_NOTEBOOKS_LISTSORT_CUH

#include <curand.h>
#include <curand_kernel.h>

#include "boost/program_options/variables_map.hpp"
#include "boost/program_options/options_description.hpp"

namespace po = boost::program_options;


class ListSort {
public:
    ListSort();
    int main(const po::variables_map& vm);
    void addParams(po::options_description *desc);

    std::vector<float> sortElements(std::vector<float> elements);
};


#endif //CUDA_NOTEBOOKS_LISTSORT_CUH