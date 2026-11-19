#pragma once

#include "compact_position_data.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace engine
{
    namespace cuda_kernels
    {

        // posições compactas --> entrada da rede [count, 19, 8, 8], na GPU
        void decodeCompact(const encoding::CompactPosition *positions, float *inputs, int count, cudaStream_t stream);

        // copia da saída de política só os logits dos lances legais de cada posição
        void gatherLegal(const float *policies, const uint16_t *policyIndices, float *legalLogits, int count, int slotsPerPosition, cudaStream_t stream);

    } // namespace cuda_kernels
} // namespace engine
