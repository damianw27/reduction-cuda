#ifndef MANDELBROT_METHODS
#define MANDELBROT_METHODS

#include <iostream>

#define ITERATIONS_COUNT 256

using namespace std;

template<typename NUMBER>
__global__ void cuda_mandelbrot(NUMBER x0, NUMBER y0, NUMBER x1, NUMBER y1, int width, int height, int *dataOut) {
    int i;
    int threadId = int(threadIdx.x + blockIdx.x * blockDim.x);
    int size = height * width;
    NUMBER dX = (x1 - x0) / NUMBER(width - 1);
    NUMBER dY = (y1 - y0) / NUMBER(height - 1);
    NUMBER x, y, Zx, Zy, tZx, tZy;
    NUMBER tmpWidth, tmpHeight;

    if (threadId < size) {
        tmpWidth = (NUMBER) threadId / (NUMBER) size;
        tmpHeight = NUMBER(threadId % size);
        x = x0 + dX * tmpHeight;
        y = y0 + dY * tmpWidth;
        Zx = x;
        Zy = y;
        i = 0;

        while (i < ITERATIONS_COUNT && ((Zx * Zx + Zy * Zy) < 4)) {
            tZx = Zx * Zx - Zy * Zy + x;
            tZy = 2 * Zx * Zy + y;
            Zx = tZx;
            Zy = tZy;
            i++;
        }

        dataOut[threadId] = i;
    }
}

template<typename NUMBER>
__host__ int *calculate_mandelbrot_data(const int width, const int height) {
    const NUMBER x0 = -0.82;
    const NUMBER y0 = 0.1;
    const NUMBER x1 = -0.7;
    const NUMBER y1 = 0.22;

    // wylicz rozmiar macierzy wynikowej
    int size = width * height;

    // zalokuj pamiec na cpu
    auto *mandelDataHost = new int[size];

    // zalokuj pamiec na gpu
    int *mandelDataDevice;
    cudaMalloc(&mandelDataDevice, size * sizeof(int));

    // wykonaj generowanie mandelbrota
    for (int currentThreadsCount = 32; currentThreadsCount < 2048; currentThreadsCount = 2 * currentThreadsCount) {
        // wylicz konfigurację watkow wykorzystywanych w obliczeniach mandelbrota
        dim3 currentBlockDim(currentThreadsCount, 1, 1);
        dim3 currentGridDim(width * height / currentThreadsCount + 1, 1, 1);

        for (int index = 0; index < 15; index++) {
            cuda_mandelbrot<NUMBER><<<currentGridDim, currentBlockDim>>>(x0, y0, x1, y1, width, height, mandelDataDevice);
            cudaDeviceSynchronize();
        }
    }

    // wczytaj wyliczone dane z cuda do pamieci komputera
    cudaMemcpy(mandelDataHost, mandelDataDevice, size * sizeof(int), cudaMemcpyDeviceToHost);

    // usun
    cudaFree(&mandelDataDevice);

    return mandelDataHost;
}

#endif
