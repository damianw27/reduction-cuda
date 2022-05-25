// compile method normal: nvcc main.cu -o reduction  -lcudadevrt -lcudart_static -lrt -lpthread -ldl -L"/lib/x86_64-linux-gnu"
// compile method optimi: nvcc main.cu -o reduction  -lcudadevrt -lcudart_static -lrt -lpthread -ldl -L"/lib/x86_64-linux-gnu" -G

#include <iostream>
#include <random>
#include <vector>
#include <chrono>

#include "cuda_check_errors.cu"
#include "reduction.cu"
#include "mandelbrot.cu"

#define NUMBER double
#define BLOCK_SIZE 512
#define SAMPLE_SIZE 11

using namespace std;
using namespace chrono;

int sum_cpu(const int *dataIn, const int dataSize) {
    int resultSum = 0;

    for (int i = 0; i < dataSize; ++i) {
        resultSum += dataIn[i];
    }

    return resultSum;
}

int get_grid_size(int dataSize, int maxElementsPerBlock) {
    int gridSize;

    if (dataSize <= maxElementsPerBlock) {
        gridSize = (int) ceil(float(dataSize) / float(maxElementsPerBlock));
    } else {
        gridSize = dataSize / maxElementsPerBlock;

        if (dataSize % maxElementsPerBlock != 0) {
            gridSize++;
        }
    }

    return gridSize;
}

int sum_gpu(const int *deviceDataIn, const int dataSize, const int threadsCount, void (*reduce)(int *, const int *, int, int)) {
    int totalSum = 0;
    int maxElementsPerBlock = threadsCount;
    int gridSize = get_grid_size(dataSize, maxElementsPerBlock);
    int *deviceBlockSums;
    int *deviceTotalSum;
    int *deviceBlockSumsIn;

    checkCudaErrors(cudaMalloc(&deviceBlockSums, sizeof(int) * gridSize));
    checkCudaErrors(cudaMemset(deviceBlockSums, 0, sizeof(int) * gridSize));

    reduce<<<gridSize, threadsCount, sizeof(int) * threadsCount>>>(deviceBlockSums, deviceDataIn, dataSize, threadsCount);

    if (gridSize <= maxElementsPerBlock) {
        checkCudaErrors(cudaMalloc(&deviceTotalSum, sizeof(int)));
        checkCudaErrors(cudaMemset(deviceTotalSum, 0, sizeof(int)));
        reduce<<<1, threadsCount, sizeof(int) * threadsCount>>>(deviceTotalSum, deviceBlockSums, gridSize, threadsCount);
        checkCudaErrors(cudaMemcpy(&totalSum, deviceTotalSum, sizeof(int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(deviceTotalSum));
    } else {
        checkCudaErrors(cudaMalloc(&deviceBlockSumsIn, sizeof(int) * gridSize));
        checkCudaErrors(cudaMemcpy(deviceBlockSumsIn, deviceBlockSums, sizeof(int) * gridSize, cudaMemcpyDeviceToDevice));
        totalSum = sum_gpu(deviceBlockSumsIn, gridSize, threadsCount, reduce);
        checkCudaErrors(cudaFree(deviceBlockSumsIn));
    }

    checkCudaErrors(cudaFree(deviceBlockSums));
    return totalSum;
}

NUMBER median(const NUMBER *data, int length) {
    auto *location = new int[length];
    auto *sorted = new NUMBER[length];
    int currentValueScore;
    NUMBER currentValue;

    for (int outerIndex = 0; outerIndex < length; outerIndex++) {
        currentValueScore = 0;
        currentValue = data[outerIndex];

        for (int innerIndex = 0; innerIndex < length; innerIndex++) {
            if (data[innerIndex] < currentValue)
                currentValueScore++;
        }

        location[outerIndex] = currentValueScore;
    }

    for (int index = 0; index < length; index++) {
        sorted[location[index]] = data[index];
    }

    return sorted[length / 2];
}

NUMBER mean(const NUMBER *data, int length) {
    NUMBER sumOfElements = 0;

    for (int index = 0; index < length; index++) {
        sumOfElements += data[index];
    }

    return sumOfElements / (NUMBER) length;
}

NUMBER standard_deviation(const NUMBER *data, int length, NUMBER meanValue) {
    NUMBER result = 0.0;

    for (int index = 0; index < length; index++) {
        result += (data[index] - meanValue) * (data[index] - meanValue);
    }

    result = sqrt(result / NUMBER(length) / NUMBER(length - 1));
    return result;
}

inline NUMBER round(NUMBER val) {
    double fullValuePart = val * 1000.0;

    fullValuePart = fullValuePart < 0
            ? ceil(fullValuePart - 0.5)
            : floor(fullValuePart + 0.5);

    return fullValuePart / 1000.0;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        cout << " -- mandelbrot data reduction -- " << endl;
        cout << "ARG[1] - width of mandelbrot set" << endl;
        cout << "ARG[2] - height of mandelbrot set" << endl;
        cout << "Example usage: ./reduction 3000 3000 > output.txt" << endl;
        exit(1);
    }

    int width = stoi(argv[1]);
    int height = stoi(argv[2]);

    for (int i = 0; i < 20; i++) {
        calculate_mandelbrot_data<NUMBER>(width, height);
    }

    int *dataIn = calculate_mandelbrot_data<NUMBER>(width, height);
    int dataSize = width * height;

    cout << "SIZE OF VAR: " << sizeof(int) << endl;
    cout << "DATA SIZE: " << dataSize << endl;
    cout << "SUM CPU: " << sum_cpu(dataIn, dataSize) << endl;

    int *deviceDataIn;
    checkCudaErrors(cudaMalloc(&deviceDataIn, sizeof(int) * dataSize));
    checkCudaErrors(cudaMemcpy(deviceDataIn, dataIn, sizeof(int) * dataSize, cudaMemcpyHostToDevice));

    vector<void (*)(int *, const int *, int, int)> reductions = {
            reduce_0,
            reduce_1,
            reduce_2,
            reduce_3,
            reduce_4,
            reduce_5,
            reduce_shuffle,
    };

    const vector<string> reductionNames = {
            "Interleaved Addressing",
            "Interleaved Addressing (without bank conflicts)",
            "Sequential Addressing",
            "First Add During Load",
            "Unroll Last Warp",
            "Completely Unroll",
            "Shuffle",
    };

    cout << endl;
    cout << "method;\t 32;\t 64;\t 128;\t 256;\t 512;\t 1024;\t result;" << endl;

    int reductionIndex = 0;
    int reductionResult = 0;

    for (auto &reduce: reductions) {
        cout << reductionNames.at(reductionIndex) << "; ";

        for (int threadsCount = 32; threadsCount < 2048; threadsCount = 2 * threadsCount) {
            auto *sample = new NUMBER[SAMPLE_SIZE];

            for (int sampleIndex = 0; sampleIndex < SAMPLE_SIZE; sampleIndex++) {
                auto startTime = steady_clock::now();

                reductionResult = sum_gpu(deviceDataIn, dataSize, threadsCount, reduce);

                auto endTime = steady_clock::now();
                sample[sampleIndex] = duration<NUMBER, milli>(endTime - startTime).count();
            }

            NUMBER medianResult = median(sample, SAMPLE_SIZE);
            NUMBER meanResult = mean(sample, SAMPLE_SIZE);
            NUMBER sdResult = standard_deviation(sample, SAMPLE_SIZE, meanResult);

            cout << round(medianResult) << " (+/- " << round(sdResult) << "); ";
        }

        cout << reductionResult << "; " << endl;
        reductionIndex++;
    }

    cout << endl;
    cout << "threads count;\t";

    for (auto &strategyName: reductionNames) {
        cout << strategyName << ";\t";
    }

    cout << endl;

    for (int threadsCount = 32; threadsCount < 2048; threadsCount = 2 * threadsCount) {
        cout << threadsCount << "; ";

        for (auto &reduce: reductions) {
            auto *sample = new NUMBER[SAMPLE_SIZE];

            for (int sampleIndex = 0; sampleIndex < SAMPLE_SIZE; sampleIndex++) {
                auto startTime = steady_clock::now();

                sum_gpu(deviceDataIn, dataSize, threadsCount, reduce);

                auto endTime = steady_clock::now();
                sample[sampleIndex] = duration<NUMBER, milli>(endTime - startTime).count();
            }

            NUMBER medianResult = median(sample, SAMPLE_SIZE);
            NUMBER meanResult = mean(sample, SAMPLE_SIZE);
            NUMBER sdResult = standard_deviation(sample, SAMPLE_SIZE, meanResult);

            cout << round(medianResult) << " (+/- " << round(sdResult) << "); ";
        }

        cout << endl;
    }

    checkCudaErrors(cudaFree(deviceDataIn));
}
