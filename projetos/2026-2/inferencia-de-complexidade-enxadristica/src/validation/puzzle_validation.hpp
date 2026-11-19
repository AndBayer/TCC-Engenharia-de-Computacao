#pragma once

#include "inference.hpp"
#include "params.hpp"
#include "puzzle_sampling.hpp"
#include "search.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// resultado da busca num puzzle, gravado no CSV
struct AutonomousValidationResult
{
    std::string puzzleId;
    std::string fen; // posição onde a busca começou (depois do lance forçador)
    int rating = 0;
    double centipawns = 0.0; // avaliação da raiz em centipawns
    std::vector<engine::InstrumentalProfileNode> instrumentalProfile;
    std::uint64_t requestedSimulations = 0; // orçamento pedido para a busca
};

// joga o lance forçador do puzzle e roda a busca na posição resultante; nullopt se o puzzle é inválido
std::optional<AutonomousValidationResult>
validateAutonomousPuzzle(const SampledPuzzle &puzzle, const engine::SearchParams &params, engine::PositionEvaluator &evaluator);
