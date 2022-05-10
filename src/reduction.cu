#ifndef REDUCTION_METHODS
#define REDUCTION_METHODS

extern __shared__ volatile unsigned int sharedSumData[];

__global__ void reduce_0(unsigned int *dataOut, const unsigned int *dataIn, unsigned int dataSize, int threadsCount) {
    unsigned int threadId = threadIdx.x;
    unsigned int threadLocation = blockIdx.x * blockDim.x + threadIdx.x;

    sharedSumData[threadId] = 0;

    if (threadLocation < dataSize) {
        sharedSumData[threadId] = dataIn[threadLocation];
    }

    __syncthreads();

    for (unsigned int stepValue = 1; stepValue < blockDim.x; stepValue *= 2) {
        if (threadId % (2 * stepValue) == 0) {
            sharedSumData[threadId] += sharedSumData[threadId + stepValue];
        }
        __syncthreads();
    }

    if (threadId == 0) {
        dataOut[blockIdx.x] = sharedSumData[0];
    }
}

__global__ void reduce_1(unsigned int *dataOut, const unsigned int *dataIn, unsigned int dataSize, int threadsCount) {
    unsigned int threadId = threadIdx.x;
    unsigned int threadLocation = blockIdx.x * blockDim.x + threadIdx.x;

    sharedSumData[threadId] = 0;

    if (threadLocation < dataSize) {
        sharedSumData[threadId] = dataIn[threadLocation];
    }

    __syncthreads();

    for (unsigned int stepValue = 1; stepValue < blockDim.x; stepValue *= 2) {
        unsigned int index = 2 * stepValue * threadId;

        if (index < blockDim.x) {
            sharedSumData[index] += sharedSumData[index + stepValue];
        }

        __syncthreads();
    }

    if (threadId == 0) {
        dataOut[blockIdx.x] = sharedSumData[0];
    }
}

__global__ void reduce_2(unsigned int *dataOut, const unsigned int *dataIn, unsigned int dataSize, int threadsCount) {
    unsigned int threadId = threadIdx.x;
    unsigned int threadLocation = blockIdx.x * blockDim.x + threadIdx.x;

    sharedSumData[threadId] = 0;

    if (threadLocation < dataSize) {
        sharedSumData[threadId] = dataIn[threadLocation];
    }

    __syncthreads();

    for (unsigned int stepValue = blockDim.x / 2; stepValue > 0; stepValue >>= 1) {
        if (threadId < stepValue) {
            sharedSumData[threadId] += sharedSumData[threadId + stepValue];
        }

        __syncthreads();
    }

    if (threadId == 0) {
        dataOut[blockIdx.x] = sharedSumData[0];
    }
}

__global__ void reduce_3(unsigned int *dataOut, const unsigned int *dataIn, unsigned int dataSize, int threadsCount) {
    unsigned int threadId = threadIdx.x;
    unsigned int threadLocation = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    sharedSumData[threadId] = 0;

    if (threadLocation < dataSize) {
        sharedSumData[threadId] = dataIn[threadLocation] + dataIn[threadLocation + blockDim.x];
    }

    __syncthreads();

    for (unsigned int stepValue = blockDim.x / 2; stepValue > 0; stepValue >>= 1) {
        if (threadId < stepValue) {
            sharedSumData[threadId] += sharedSumData[threadId + stepValue];
        }
        __syncthreads();
    }

    if (threadId == 0) {
        dataOut[blockIdx.x] = sharedSumData[0];
    }
}

__global__ void reduce_4(unsigned int *dataOut, const unsigned int *dataIn, unsigned int dataSize, int threadsCount) {
    unsigned int threadId = threadIdx.x;
    unsigned int threadLocation = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    sharedSumData[threadId] = 0;

    if (threadLocation < dataSize) {
        sharedSumData[threadId] = dataIn[threadLocation] + dataIn[threadLocation + blockDim.x];
    }

    __syncthreads();

    for (unsigned int stepValue = blockDim.x / 2; stepValue > 32; stepValue >>= 1) {
        if (threadId < stepValue) {
            sharedSumData[threadId] += sharedSumData[threadId + stepValue];
        }

        __syncthreads();
    }

    if (threadId < 32) {
        sharedSumData[threadId] += sharedSumData[threadId + 32];
        sharedSumData[threadId] += sharedSumData[threadId + 16];
        sharedSumData[threadId] += sharedSumData[threadId + 8];
        sharedSumData[threadId] += sharedSumData[threadId + 4];
        sharedSumData[threadId] += sharedSumData[threadId + 2];
        sharedSumData[threadId] += sharedSumData[threadId + 1];
        __syncthreads();
    }

    if (threadId == 0) {
        dataOut[blockIdx.x] = sharedSumData[0];
    }
}

__global__ void reduce_5(unsigned int *dataOut, const unsigned int *dataIn, unsigned int dataSize, int threadsCount) {
    unsigned int threadId = threadIdx.x;
    unsigned int threadLocation = threadIdx.x + blockIdx.x * (blockDim.x * 2);

    sharedSumData[threadId] = 0;

    if (threadLocation < dataSize / 2) {
        sharedSumData[threadId] = dataIn[threadLocation] + dataIn[threadLocation + blockDim.x];
    }

    __syncthreads();

    if (threadsCount >= 512) {
        if (threadId < 256) {
            sharedSumData[threadId] += sharedSumData[threadId + 256];
        }

        __syncthreads();
    }

    if (threadsCount >= 256) {
        if (threadId < 128) {
            sharedSumData[threadId] += sharedSumData[threadId + 128];
        }

        __syncthreads();
    }

    if (threadsCount >= 128) {
        if (threadId < 64) {
            sharedSumData[threadId] += sharedSumData[threadId + 64];
        }

        __syncthreads();
    }

    if (threadId < 32) {
        if (threadsCount >= 64) {
            sharedSumData[threadId] += sharedSumData[threadId + 32];
        }

        if (threadsCount >= 32) {
            sharedSumData[threadId] += sharedSumData[threadId + 16];
        }

        if (threadsCount >= 16) {
            sharedSumData[threadId] += sharedSumData[threadId + 8];
        }

        if (threadsCount >= 8) {
            sharedSumData[threadId] += sharedSumData[threadId + 4];
        }

        if (threadsCount >= 4) {
            sharedSumData[threadId] += sharedSumData[threadId + 2];
        }

        if (threadsCount >= 2) {
            sharedSumData[threadId] += sharedSumData[threadId + 1];
        }
    }

    if (threadId == 0) {
        dataOut[blockIdx.x] = sharedSumData[0];
    }
}

__global__ void reduce_shuffle(unsigned int *dataOut, const unsigned int *dataIn, unsigned int dataSize, int threadsCount) {
    unsigned int threadId = threadIdx.x;
    unsigned int threadLocation = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int valueFromSharedMemory = 0.0f;

    unsigned mask = 0xFFFFFFFFU;

    unsigned int lane = threadIdx.x % warpSize;
    unsigned int warpId = threadIdx.x / warpSize;

    while (threadLocation < dataSize) {
        valueFromSharedMemory += dataIn[threadLocation];
        threadLocation += gridDim.x * blockDim.x;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        valueFromSharedMemory += __shfl_down_sync(mask, valueFromSharedMemory, offset);
    }

    if (lane == 0) {
        sharedSumData[warpId] = valueFromSharedMemory;
    }

    __syncthreads();

    if (warpId == 0) {
        valueFromSharedMemory = (threadId < blockDim.x / warpSize)
                ? sharedSumData[lane]
                : 0;

        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            valueFromSharedMemory += __shfl_down_sync(mask, valueFromSharedMemory, offset);
        }

        if (threadId == 0) {
            atomicAdd(reinterpret_cast<unsigned int *>(dataOut), valueFromSharedMemory);
        }
    }
}

#endif
