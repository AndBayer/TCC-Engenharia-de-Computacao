#include "puzzle_validation.hpp"

#include "instrumental_profile.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    // substitui a rede neural no teste --> todos os lances com a mesma chance (logit 0) e valor 0,125; conta as posições avaliadas
    class ConstantEvaluator final : public engine::PositionEvaluator
    {
    public:
        std::vector<engine::EvalOutput> evaluateBatch(std::span<const engine::PositionEvalRequest> requests) override
        {
            evaluations_ += requests.size();
            std::vector<engine::EvalOutput> outputs;
            outputs.reserve(requests.size());
            for (const auto &request : requests)
            {
                engine::EvalOutput output;
                output.legalLogits.assign(request.legalPolicyIndices.size(), 0.0f);
                output.value = 0.125f;
                outputs.push_back(std::move(output));
            }
            return outputs;
        }

        size_t evaluations() const { return evaluations_; }

    private:
        size_t evaluations_ = 0;
    };

    // puzzle de teste a partir da posição inicial, com 3 lances do resolvedor
    SampledPuzzle threeDecisionPuzzle()
    {
        return {
            "three-decisions",
            std::string(chess::constants::STARTPOS),
            {"e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"},
            1500,
        };
    }

    // posição depois do lance forçador (1º lance do puzzle, feito pelo adversário), onde a busca começa
    chess::Board autonomousBoard(const SampledPuzzle &puzzle)
    {
        chess::Board board(puzzle.fen);
        board.makeMove(chess::uci::uciToMove(board, puzzle.moves[0]));
        return board;
    }
} // namespace

// validar o puzzle tem que dar o mesmo resultado de uma busca direta na posição depois do lance forçador --> mesmo perfil e mesmo número de avaliações
int main()
{
    // busca curta (8 simulações) e uma posição por vez, para o resultado não variar
    engine::SearchParams params;
    params.simBudget = 8;
    params.batchSize = 1;
    params.gatherBatchSize = 1;
    params.infoIntervalMs = 0;
    params.maxNodes = 4096;
    const SampledPuzzle puzzle = threeDecisionPuzzle();

    ConstantEvaluator expectedAutonomousEvaluator;
    const engine::SearchResult expectedAutonomous = engine::runSearch(autonomousBoard(puzzle), params, expectedAutonomousEvaluator);
    ConstantEvaluator autonomousEvaluator;
    const auto autonomous = validateAutonomousPuzzle(puzzle, params, autonomousEvaluator);

    if (!autonomous || autonomous->requestedSimulations != 8 ||
        autonomous->instrumentalProfile.empty() ||
        autonomous->instrumentalProfile.front().nodeVisits != 8)
        throw std::runtime_error("autonomous mode changed its budget");

    if (engine::serializeInstrumentalProfile(autonomous->instrumentalProfile) != engine::serializeInstrumentalProfile(expectedAutonomous.instrumentalProfile))
        throw std::runtime_error("autonomous raw profile changed");

    if (autonomousEvaluator.evaluations() != expectedAutonomousEvaluator.evaluations())
        throw std::runtime_error("autonomous mode ran extra inference");

    return 0;
}
