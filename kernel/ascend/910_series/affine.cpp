#include "kernel_operator.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#endif

using namespace AscendC;

class MiraAffine {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR v, GM_ADDR outA, GM_ADDR outV,
                              uint64_t planes, uint32_t tokens, uint32_t channels,
                              uint32_t order, uint32_t memory, uint32_t distance) {
        count = planes;
        T = tokens;
        C = channels;
        K = order;
        M = memory;
        stride = distance;
        matrix.SetGlobalBuffer((__gm__ float *)a);
        carry.SetGlobalBuffer((__gm__ float *)v);
        matrixOut.SetGlobalBuffer((__gm__ float *)outA);
        carryOut.SetGlobalBuffer((__gm__ float *)outV);
        pipe.InitBuffer(storage, (4 * K * 8 + 3 * K * M + M) * sizeof(float));
    }

    __aicore__ inline void Process() {
        auto buffer = storage.Get<float>();
        auto rightA = buffer;
        auto leftA = buffer[K * 8];
        auto resultA = buffer[2 * K * 8];
        auto tempA = buffer[3 * K * 8];
        auto rightV = buffer[4 * K * 8];
        auto leftV = buffer[4 * K * 8 + K * M];
        auto resultV = buffer[4 * K * 8 + 2 * K * M];
        auto tempV = buffer[4 * K * 8 + 3 * K * M];
        for (uint64_t plane = GetBlockIdx(); plane < count; plane += GetBlockNum()) {
            DataCopy(rightA, matrix[plane * K * 8], K * 8);
            DataCopy(rightV, carry[plane * K * M], K * M);
            PipeBarrier<PIPE_ALL>();
            uint32_t token = (plane / C) % T;
            if (token < stride) {
                DataCopy(matrixOut[plane * K * 8], rightA, K * 8);
                DataCopy(carryOut[plane * K * M], rightV, K * M);
                PipeBarrier<PIPE_ALL>();
                continue;
            }
            uint64_t previous = plane - uint64_t(stride) * C;
            DataCopy(leftA, matrix[previous * K * 8], K * 8);
            DataCopy(leftV, carry[previous * K * M], K * M);
            PipeBarrier<PIPE_ALL>();
            for (uint32_t row = 0; row < K; ++row) {
                Duplicate(resultA[row * 8], 0.0f, 8);
                DataCopy(resultV[row * M], rightV[row * M], M);
                PipeBarrier<PIPE_ALL>();
                for (uint32_t column = 0; column < K; ++column) {
                    float coefficient = rightA.GetValue(row * 8 + column);
                    Muls(tempA, leftA[column * 8], coefficient, 8);
                    Muls(tempV, leftV[column * M], coefficient, M);
                    PipeBarrier<PIPE_V>();
                    Add(resultA[row * 8], resultA[row * 8], tempA, 8);
                    Add(resultV[row * M], resultV[row * M], tempV, M);
                    PipeBarrier<PIPE_ALL>();
                }
            }
            DataCopy(matrixOut[plane * K * 8], resultA, K * 8);
            DataCopy(carryOut[plane * K * M], resultV, K * M);
            PipeBarrier<PIPE_ALL>();
        }
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> storage;
    GlobalTensor<float> matrix, carry, matrixOut, carryOut;
    uint64_t count;
    uint32_t T, C, K, M, stride;
};

extern "C" __global__ __aicore__ void mira_affine(
    GM_ADDR matrix, GM_ADDR carry, GM_ADDR matrixOut, GM_ADDR carryOut,
    uint64_t planes, uint32_t tokens, uint32_t channels, uint32_t order,
    uint32_t memory, uint32_t distance) {
#if defined(KERNEL_TASK_TYPE_DEFAULT) && defined(__CCE_AICORE__) && __CCE_AICORE__ >= 220
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#endif
    MiraAffine op;
    op.Init(matrix, carry, matrixOut, carryOut, planes, tokens, channels,
            order, memory, distance);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void mira_affine_launch(
    uint32_t blocks, void *stream, void *matrix, void *carry, void *matrixOut,
    void *carryOut, uint64_t planes, uint32_t tokens, uint32_t channels,
    uint32_t order, uint32_t memory, uint32_t distance) {
    mira_affine<<<blocks, nullptr, stream>>>(
        (uint8_t *)matrix, (uint8_t *)carry, (uint8_t *)matrixOut,
        (uint8_t *)carryOut, planes, tokens, channels, order, memory, distance);
}
#endif
