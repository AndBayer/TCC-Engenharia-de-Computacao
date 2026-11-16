#include "inference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// compara o pipeline na GPU (CudaEvalService) com a rede rodada direto pelo ONNX Runtime; recebe o modelo .onnx como argumento
int main(int argc, char *argv[])
{
    if (argc != 2)
        throw std::invalid_argument("usage: tfird_cuda_inference_tests model.onnx");

    // índices de política fixos (0, 16, 32, ...), só para comparar os logits
    std::vector<uint16_t> indices(chess::constants::MAX_MOVES);
    for (size_t index = 0; index < indices.size(); ++index)
        indices[index] = static_cast<uint16_t>(index * 16);

    std::vector<chess::Board> boards = {
        chess::Board(chess::constants::STARTPOS),
        chess::Board("r4rk1/p3qpb1/2n3p1/3p3p/3B2N1/P2P3P/1P1Q1PP1/1R2R1K1 b - - 1 21"),
        chess::Board("rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq e6 0 2"),
        chess::Board("7k/P7/8/8/8/8/7p/7K w - - 0 1"),
    };

    std::unique_ptr<engine::NetEvaluator> network = engine::makeOnnxEvaluator(argv[1]);
    engine::SynchronousEvaluator host(*network);
    // lote máximo de 3 para as 4 posições, então a GPU roda um lote cheio e um incompleto
    engine::CudaEvalService cuda(argv[1], 3, 0);

    std::vector<engine::PositionEvalRequest> requests;
    requests.reserve(boards.size());
    for (const chess::Board &board : boards)
        requests.push_back({&board, indices});

    const std::vector<engine::EvalOutput> expectedBatch = host.evaluateBatch(requests);
    const std::vector<engine::EvalOutput> actualBatch = cuda.evaluateBatch(requests);

    if (expectedBatch.size() != boards.size() || actualBatch.size() != boards.size())
        throw std::runtime_error("grouped inference returned the wrong batch size");

    // maior diferença de logit e de valor entre os dois caminhos; tolerância de 1e-5
    float maxLogitDifference = 0.0f;
    float maxValueDifference = 0.0f;
    for (size_t boardIndex = 0; boardIndex < boards.size(); ++boardIndex)
    {
        const engine::EvalOutput &expected = expectedBatch[boardIndex];
        const engine::EvalOutput &actual = actualBatch[boardIndex];
        if (expected.legalLogits.size() != actual.legalLogits.size())
            throw std::runtime_error("CUDA inference returned the wrong logit count");

        for (size_t index = 0; index < expected.legalLogits.size(); ++index)
            maxLogitDifference = std::max(maxLogitDifference, std::abs(expected.legalLogits[index] - actual.legalLogits[index]));

        maxValueDifference = std::max(maxValueDifference, std::abs(expected.value - actual.value));
    }
    cuda.shutdown();

    std::cout << "max_logit_difference=" << maxLogitDifference << " max_value_difference=" << maxValueDifference << '\n';
    if (maxLogitDifference > 1e-5f || maxValueDifference > 1e-5f)
        throw std::runtime_error("CUDA inference differs beyond tolerance");
    return 0;
}
