#include "inference_cuda_kernels.hpp"

#include <cuda_runtime.h>

namespace engine
{
    namespace cuda_kernels
    {
        namespace
        {
            // canais 0-11 peças, 12 en passant, 13-16 roques, 17-18 ataques
            constexpr int kChannels = 19;
            constexpr int kSquares = 64;
            constexpr int kPolicySize = 4224;

            // duas casas por byte (4 bits cada), casa par nos 4 bits baixos e ímpar nos altos
            __device__ __forceinline__ uint8_t pieceAt(const encoding::CompactPosition &position, int square)
            {
                const uint8_t packed = position.data[square >> 1];
                return (square & 1) != 0 ? packed >> 4 : packed & 0x0F;
            }

            // um bloco por posição, uma thread por casa
            // tabuleiro já espelhado, com 1-6 peças de quem joga, 7-12 do adversário e 0 vazia
            __global__ void decodeKernel(const encoding::CompactPosition *positions, float *output, int count)
            {
                const int batch = blockIdx.x;
                const int square = threadIdx.x;
                if (batch >= count || square >= kSquares)
                    return;

                const encoding::CompactPosition &position = positions[batch];
                float *sample = output + static_cast<size_t>(batch) * kChannels * kSquares;
                for (int channel = 0; channel < kChannels; ++channel)
                    sample[channel * kSquares + square] = 0.0f;

                // tabuleiro em memória compartilhada para ler casas vizinhas
                __shared__ uint8_t board[kSquares];
                board[square] = pieceAt(position, square);
                __syncthreads();

                // planos de peça, en passant e roque
                const uint8_t piece = board[square];
                if (piece >= 1 && piece <= 12)
                    sample[(piece - 1) * kSquares + square] = 1.0f;
                if (position.enPassant == square)
                    sample[12 * kSquares + square] = 1.0f;
                for (int plane = 0; plane < 4; ++plane)
                    if ((position.flags & (1U << plane)) != 0)
                        sample[(13 + plane) * kSquares + square] = 1.0f;

                // conta os atacantes da casa (white = quem joga, black = adversário)
                constexpr int knightDirections[8][2] = {{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {1, -2}, {1, 2}, {2, -1}, {2, 1}};
                constexpr int kingDirections[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

                const int rank = square >> 3;
                const int file = square & 7;
                float white = 0.0f;
                float black = 0.0f;

                // cavalos
                for (int index = 0; index < 8; ++index)
                {
                    const int r = rank + knightDirections[index][0];
                    const int f = file + knightDirections[index][1];
                    if (r < 0 || r > 7 || f < 0 || f > 7)
                        continue;
                    const uint8_t attacker = board[r * 8 + f];
                    white += attacker == 2 ? 1.0f : 0.0f;
                    black += attacker == 8 ? 1.0f : 0.0f;
                }

                // reis
                for (int index = 0; index < 8; ++index)
                {
                    const int r = rank + kingDirections[index][0];
                    const int f = file + kingDirections[index][1];
                    if (r < 0 || r > 7 || f < 0 || f > 7)
                        continue;
                    const uint8_t attacker = board[r * 8 + f];
                    white += attacker == 6 ? 1.0f : 0.0f;
                    black += attacker == 12 ? 1.0f : 0.0f;
                }

                // o peão de quem joga fica em rank - 1, o do adversário em rank + 1
                if (rank > 0)
                {
                    if (file > 0 && board[(rank - 1) * 8 + file - 1] == 1)
                        white += 1.0f;
                    if (file < 7 && board[(rank - 1) * 8 + file + 1] == 1)
                        white += 1.0f;
                }
                if (rank < 7)
                {
                    if (file > 0 && board[(rank + 1) * 8 + file - 1] == 7)
                        black += 1.0f;
                    if (file < 7 && board[(rank + 1) * 8 + file + 1] == 7)
                        black += 1.0f;
                }

                // nas peças deslizantes, só conta a primeira peça em cada direção
                // diagonal = bispo e dama, reta = torre e dama
                for (int direction = 0; direction < 8; ++direction)
                {
                    for (int distance = 1; distance < 8; ++distance)
                    {
                        const int r = rank + kingDirections[direction][0] * distance;
                        const int f = file + kingDirections[direction][1] * distance;
                        if (r < 0 || r > 7 || f < 0 || f > 7)
                            break;
                        const uint8_t attacker = board[r * 8 + f];
                        if (attacker == 0)
                            continue;

                        const bool diagonal = kingDirections[direction][0] != 0 && kingDirections[direction][1] != 0;
                        if (diagonal)
                        {
                            white += (attacker == 3 || attacker == 5) ? 1.0f : 0.0f;
                            black += (attacker == 9 || attacker == 11) ? 1.0f : 0.0f;
                        }
                        else
                        {
                            white += (attacker == 4 || attacker == 5) ? 1.0f : 0.0f;
                            black += (attacker == 10 || attacker == 11) ? 1.0f : 0.0f;
                        }
                        break;
                    }
                }

                sample[17 * kSquares + square] = white;
                sample[18 * kSquares + square] = black;
            }

            // copia só os logits dos lances legais da saída de política
            __global__ void gatherKernel(const float *policies, const uint16_t *indices, float *legalLogits, int totalSlots, int slotsPerPosition)
            {
                const int slot = blockIdx.x * blockDim.x + threadIdx.x;
                if (slot >= totalSlots)
                    return;
                const int batch = slot / slotsPerPosition;
                legalLogits[slot] = policies[static_cast<size_t>(batch) * kPolicySize + indices[slot]];
            }
        } // namespace

        // expande as posições compactas no tensor [count, 19, 8, 8]
        void decodeCompact(const encoding::CompactPosition *positions, float *inputs, int count, cudaStream_t stream)
        {
            decodeKernel<<<count, kSquares, 0, stream>>>(positions, inputs, count);
        }

        // junta os logits legais de cada posição
        void gatherLegal(const float *policies, const uint16_t *policyIndices, float *legalLogits, int count, int slotsPerPosition, cudaStream_t stream)
        {
            const int total = count * slotsPerPosition;
            constexpr int threads = 256;
            gatherKernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(policies, policyIndices, legalLogits, total, slotsPerPosition);
        }

    } // namespace cuda_kernels
} // namespace engine
