#include "board_encoder.hpp"
#include "compact_position.hpp"
#include "encoding.hpp"
#include "inference_cuda_kernels.hpp"

#include <cuda_runtime.h>

#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    // erro do CUDA --> exceção com o nome da operação
    void check(cudaError_t status, const char *operation)
    {
        if (status != cudaSuccess)
            throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }

    void appendFen(std::vector<chess::Board> &boards, const std::string &fen)
    {
        boards.emplace_back(fen);
    }
} // namespace

// compara a expansão das posições compactas na GPU com a codificação direta na CPU, valor por valor
int main(int argc, char *argv[])
{
    // sem GPU --> sai com 77, que o ctest conta como teste pulado
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
        return 77;

    std::vector<chess::Board> boards;
    appendFen(boards, chess::constants::STARTPOS);
    appendFen(boards, "rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq e6 0 2");
    appendFen(boards, "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");
    appendFen(boards, "7k/P7/8/8/8/8/7p/7K w - - 0 1");

    // posições geradas cobrem perda de roque, en passant, capturas e os dois lados, sem depender só do dataset
    for (int game = 0; game < 64; ++game)
    {
        chess::Board board(chess::constants::STARTPOS);
        for (int ply = 0; ply < 96; ++ply)
        {
            boards.push_back(board);
            chess::Movelist legal;
            chess::movegen::legalmoves(legal, board);
            if (legal.empty())
                break;
            const int selected = (game * 31 + ply * 17) % legal.size();
            board.makeMove(legal[selected]);
        }
    }

    // argumento opcional --> CSV de puzzles, com a FEN na segunda coluna
    if (argc > 1)
    {
        std::ifstream input(argv[1]);
        if (!input)
            throw std::runtime_error("could not open optional FEN sample");
        std::string line;
        while (std::getline(input, line))
        {
            const size_t firstComma = line.find(',');
            const size_t secondComma = line.find(',', firstComma + 1);
            if (firstComma == std::string::npos || secondComma == std::string::npos || line.rfind("PuzzleId,", 0) == 0)
                continue;
            appendFen(boards, line.substr(firstComma + 1, secondComma - firstComma - 1));
        }
    }

    // resultado esperado --> codificação direta na CPU (encodeBoard)
    std::vector<encoding::CompactPosition> compact;
    std::vector<float> expected;
    compact.reserve(boards.size());
    expected.resize(boards.size() * encoding::INPUT_SIZE);
    for (size_t index = 0; index < boards.size(); ++index)
    {
        compact.push_back(encoding::compactPosition(boards[index]));
        bool flip = false;
        encoding::encodeBoard(boards[index], expected.data() + index * encoding::INPUT_SIZE, flip);
    }

    encoding::CompactPosition *deviceCompact = nullptr;
    float *deviceOutput = nullptr;
    check(cudaMalloc(reinterpret_cast<void **>(&deviceCompact), compact.size() * sizeof(encoding::CompactPosition)), "cudaMalloc compact");
    check(cudaMalloc(reinterpret_cast<void **>(&deviceOutput), expected.size() * sizeof(float)), "cudaMalloc output");
    try
    {
        check(cudaMemcpy(deviceCompact, compact.data(), compact.size() * sizeof(encoding::CompactPosition), cudaMemcpyHostToDevice), "copy compact positions");
        engine::cuda_kernels::decodeCompact(deviceCompact, deviceOutput, static_cast<int>(compact.size()), nullptr);
        check(cudaGetLastError(), "launch decoder");
        std::vector<float> actual(expected.size());
        check(cudaMemcpy(actual.data(), deviceOutput, actual.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy decoded positions");

        for (size_t index = 0; index < actual.size(); ++index)
        {
            if (actual[index] != expected[index])
            {
                const size_t board = index / encoding::INPUT_SIZE;
                const size_t feature = index % encoding::INPUT_SIZE;
                throw std::runtime_error("CUDA compact decoder differs at board " + std::to_string(board) + ", feature " + std::to_string(feature) + ": " + boards[board].getFen());
            }
        }
    }
    catch (...)
    {
        cudaFree(deviceOutput);
        cudaFree(deviceCompact);
        throw;
    }
    cudaFree(deviceOutput);
    cudaFree(deviceCompact);
    return 0;
}
