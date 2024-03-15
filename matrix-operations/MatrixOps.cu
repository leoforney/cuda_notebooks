//
// Created by leo on 3/4/24.
//

#include "MatrixOps.cuh"
#include <iostream>


void matrixMultiply(const std::vector<std::vector<int>>& A,
                    const std::vector<std::vector<int>>& B,
                    std::vector<std::vector<int>>& C) {
    int m = A.size();
    int n = A[0].size();
    int p = B[0].size();

    C = std::vector<std::vector<int>>(m, std::vector<int>(p, 0));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

MatrixOps::MatrixOps() {

}

int MatrixOps::main(const po::variables_map &vm) {
    std::vector<std::vector<int>> A = {{1, 2}, {3, 4}};
    std::vector<std::vector<int>> B = {{2, 0}, {1, 2}};
    std::vector<std::vector<int>> C;

    matrixMultiply(A, B, C);

    for (const auto &row : C) {
        for (int val : row) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }

    return 0;
}

void MatrixOps::addParams(po::options_description *desc) {

}
