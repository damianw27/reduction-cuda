#ifndef CUDA_CHECK_ERRORS_METHOD
#define CUDA_CHECK_ERRORS_METHOD

#include <iostream>

using namespace std;

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename ERROR>
void check(ERROR err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        cerr << "CUDA error at: " << file << ":" << line << endl;
        cerr << cudaGetErrorString(err) << " " << func << endl;
        exit(1);
    }
}

#endif
