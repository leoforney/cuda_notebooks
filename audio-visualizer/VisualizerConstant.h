//
// Created by leo on 3/15/24.
//

#ifndef CUDA_NOTEBOOKS_VISUALIZERCONSTANT_H
#define CUDA_NOTEBOOKS_VISUALIZERCONSTANT_H

#include <vector>
#include <cufft.h>

typedef std::vector<cufftComplex> ComplexVector;
typedef std::vector<ComplexVector> ChunkVector;

#endif //CUDA_NOTEBOOKS_VISUALIZERCONSTANT_H
