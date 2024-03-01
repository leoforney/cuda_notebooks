//
// Created by leo on 2/29/24.
//

#include <iostream>
#include "LibListSort.cuh"
#include "ListSort.cuh"

JNIEXPORT jfloatArray JNICALL Java_ListSort_sort(JNIEnv *env, jobject, jfloatArray in) {
    const size_t n = env->GetArrayLength(in);

    jfloat *inArr = env->GetFloatArrayElements(in, NULL);
    ListSort listSort;

    std::vector<float> data;

    for (int i = 0; i < n; i++) {
        data.push_back(inArr[i]);
    }

    std::vector<float> sortedData = listSort.sortElements(data);

    jfloatArray out = env->NewFloatArray(n);
    jfloat* outArr = (jfloat*) malloc(sizeof(jfloat) * n);
    for (int i = 0; i < sortedData.size(); i++) {
        outArr[i] = sortedData[i];
    }
    env->SetFloatArrayRegion(out, 0, n, outArr);
    free(outArr);

    return out;
}