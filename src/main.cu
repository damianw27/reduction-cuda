// compile method normal: nvcc main.cu -o reduction  -lcudadevrt -lcudart_static -lrt -lpthread -ldl -L"/lib/x86_64-linux-gnu"
// compile method optimi: nvcc main.cu -o reduction  -lcudadevrt -lcudart_static -lrt -lpthread -ldl -L"/lib/x86_64-linux-gnu" -G

#include <iostream>
#include <random>
#include <vector>
#include <chrono>

#include "cuda_check_errors.cu"
#include "reduction.cu"
#include "mandelbrot.cu"

#define NUMBER float
#define BLOCK_SIZE 512
#define SAMPLE_SIZE 11

using namespace std;
using namespace chrono;

unsigned int sum_cpu(const unsigned int *dataIn, const unsigned int dataSize) {
    unsigned int resultSum = 0;

    for (unsigned int i = 0; i < dataSize; ++i) {
        resultSum += dataIn[i];
    }

    return resultSum;
}

unsigned int get_grid_size(unsigned int dataSize, unsigned int maxElementsPerBlock) {
    unsigned int gridSize;

    if (dataSize <= maxElementsPerBlock) {
        gridSize = (unsigned int) ceil(float(dataSize) / float(maxElementsPerBlock));
    } else {
        gridSize = dataSize / maxElementsPerBlock;

        if (dataSize % maxElementsPerBlock != 0) {
            gridSize++;
        }
    }

    return gridSize;
}

unsigned int sum_gpu(const unsigned int *deviceDataIn, const unsigned int dataSize, const int threadsCount, void (*reduce)(unsigned int*, const unsigned int*, unsigned int, int)) {
    // wartość globalna sumy
    unsigned int totalSum = 0;

    // ustal ilość wątków oraz ilość bloków
    // jeżeli rozmiar danych wejściowych nie jest potęga dwójki, część danych musi zajmować cały blok
    // w związku z tym liczba bloków musi być najmniejsza liczba 2048 bloków większa niż rozmiar wejściowy
    unsigned int maxElementsPerBlock = threadsCount;
    unsigned int gridSize = get_grid_size(dataSize, maxElementsPerBlock);

    // przypisz pamięć do tablicy wyników sumy dla kolejnych bloków
    // rozmiar tablicy musi być taki sam jak wynik równania: numberOfBlocks / gridSize
    unsigned int *deviceBlockSums;
    checkCudaErrors(cudaMalloc(&deviceBlockSums, sizeof(unsigned int) * gridSize));
    checkCudaErrors(cudaMemset(deviceBlockSums, 0, sizeof(unsigned int) * gridSize));

    // zsumuj dane przypisane do kolejnych bloków
    reduce<<<gridSize, BLOCK_SIZE, sizeof(unsigned int) * BLOCK_SIZE>>>(deviceBlockSums, deviceDataIn, dataSize, threadsCount);

    // zsumuj kolejne bloki pamięci w celu uzyskania globalnej sumy
    // w przypadku ilości bloków mniejszej niż 2048 wykorzystaj redukcję do określenia globalnej sumy
    // w przypadku gdy bloków jest więcej, wywołaj metodę rekurencyjnie
    if (gridSize <= maxElementsPerBlock) {
        unsigned int *deviceTotalSum;
        checkCudaErrors(cudaMalloc(&deviceTotalSum, sizeof(unsigned int)));
        checkCudaErrors(cudaMemset(deviceTotalSum, 0, sizeof(unsigned int)));
        reduce<<<1, BLOCK_SIZE, sizeof(unsigned int) * BLOCK_SIZE>>>(deviceTotalSum, deviceBlockSums, gridSize, threadsCount);
        checkCudaErrors(cudaMemcpy(&totalSum, deviceTotalSum, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(deviceTotalSum));
    } else {
        unsigned int *deviceBlockSumsIn;
        checkCudaErrors(cudaMalloc(&deviceBlockSumsIn, sizeof(unsigned int) * gridSize));
        checkCudaErrors(cudaMemcpy(deviceBlockSumsIn, deviceBlockSums, sizeof(unsigned int) * gridSize, cudaMemcpyDeviceToDevice));
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

int main(int argc, char **argv) {
    if (argc != 3) {
        cout << " -- mandelbrot data reduction -- " << endl;
        cout << "ARG[1] - width of mandelbrot set" << endl;
        cout << "ARG[2] - height of mandelbrot set" << endl;
        cout << "Example usage: ./reduction 3000 3000 > output.txt" << endl;
        exit(1);
    }

    // odczytaj dane wprowadzone przez użytkownika
    unsigned int width = stoi(argv[1]);
    unsigned int height = stoi(argv[2]);

    for (int i = 0; i < 20; i++) {
        calculate_mandelbrot_data<NUMBER>(width, height);
    }

    // wylicz mandelbrota i zwróć zbiór
    unsigned int *dataIn = calculate_mandelbrot_data<NUMBER>(width, height);
    unsigned int dataSize = width * height;

    cout << "SIZE OF VAR: " << sizeof (unsigned int) << endl;
    cout << "DATA SIZE: " << dataSize << endl;
    cout << "SUM CPU: " << sum_cpu(dataIn, dataSize) << endl;

    // wprowadź dane z mandelbrota do gpu
    unsigned int *deviceDataIn;
    checkCudaErrors(cudaMalloc(&deviceDataIn, sizeof(unsigned int) * dataSize));
    checkCudaErrors(cudaMemcpy(deviceDataIn, dataIn, sizeof(unsigned int) * dataSize, cudaMemcpyHostToDevice));

    vector<void (*)(unsigned int*, const unsigned int*, unsigned int, int)> reductions = {
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

    unsigned int reductionIndex = 0;

    for (auto &reduce : reductions) {
        cout << reductionNames.at(reductionIndex) << "; ";

        unsigned int reductionResult = 0;

        for (int threadsCount = 32; threadsCount < 2048; threadsCount = 2 * threadsCount) {
            auto *sample = new NUMBER[SAMPLE_SIZE];

            for (int sampleIndex = 0; sampleIndex < SAMPLE_SIZE; sampleIndex++) {
                // rozpocznij badanie czasu wykonania dla próbki
                auto startTime = steady_clock::now();

                // wykonaj redukcje z dana ilością wątków
                reductionResult = sum_gpu(deviceDataIn, dataSize, threadsCount, reduce);

                // zakończ badanie czasu wykonani dla próbki
                auto endTime = steady_clock::now();

                // wylicz czas wykonania próbki i wprowadź do tablicy próbek
                sample[sampleIndex] = duration<NUMBER, milli>(endTime - startTime).count();
            }

            NUMBER medianResult = median(sample, SAMPLE_SIZE);
            NUMBER meanResult = mean(sample, SAMPLE_SIZE);
            NUMBER sdResult = standard_deviation(sample, SAMPLE_SIZE, meanResult);

            cout << meanResult << "; ";
        }

        cout << reductionResult << "; " << endl;
        reductionIndex++;
    }

    cout << endl;
    cout << "threads count;\t";

    for (auto &strategyName : reductionNames) {
        cout << strategyName << ";\t";
    }

    cout << endl;

    for (int threadsCount = 32; threadsCount < 2048; threadsCount = 2 * threadsCount) {
        cout << threadsCount << "; ";

        for (auto &reduce : reductions) {
            auto *sample = new NUMBER[SAMPLE_SIZE];

            for (int sampleIndex = 0; sampleIndex < SAMPLE_SIZE; sampleIndex++) {
                // rozpocznij badanie czasu wykonania dla próbki
                auto startTime = steady_clock::now();

                // wykonaj redukcje z dana ilością wątków
                sum_gpu(deviceDataIn, dataSize, threadsCount, reduce);

                // zakończ badanie czasu wykonani dla próbki
                auto endTime = steady_clock::now();

                // wylicz czas wykonania próbki i wprowadź do tablicy próbek
                sample[sampleIndex] = duration<NUMBER, milli>(endTime - startTime).count();
            }

            NUMBER medianResult = median(sample, SAMPLE_SIZE);
            NUMBER meanResult = mean(sample, SAMPLE_SIZE);
            NUMBER sdResult = standard_deviation(sample, SAMPLE_SIZE, meanResult);

            cout << meanResult << "; ";
        }

        cout << endl;
    }

    // zwolnij miejsce zaalokowane dla zmiennej deviceDataIn
    checkCudaErrors(cudaFree(deviceDataIn));
}
