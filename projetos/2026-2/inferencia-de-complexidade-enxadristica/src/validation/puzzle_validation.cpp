#include "puzzle_validation.hpp"

#include "chess.hpp"
#include "search.hpp"

namespace
{
    // posição onde a busca começa --> a do puzzle depois do lance forçador; nullopt se não há lance ou o lance é inválido
    std::optional<chess::Board> initialSearchBoard(const SampledPuzzle &puzzle)
    {
        if (puzzle.moves.empty())
            return std::nullopt;
        try
        {
            chess::Board board(puzzle.fen);
            const chess::Move initial = chess::uci::uciToMove(board, puzzle.moves.front());

            if (initial == chess::Move::NO_MOVE)
                return std::nullopt;

            // joga o lance forçador
            board.makeMove(initial);
            return board;
        }
        catch (...)
        {
            return std::nullopt;
        }
    }
} // namespace

// busca na posição depois do lance forçador; nullopt se o puzzle é inválido
std::optional<AutonomousValidationResult>
validateAutonomousPuzzle(const SampledPuzzle &puzzle, const engine::SearchParams &params, engine::PositionEvaluator &evaluator)
{
    const auto board = initialSearchBoard(puzzle);
    if (!board)
        return std::nullopt;

    const engine::SearchResult search = engine::runSearch(*board, params, evaluator);
    AutonomousValidationResult output;
    output.puzzleId = puzzle.id;
    output.fen = board->getFen();
    output.rating = puzzle.rating;
    output.centipawns = search.evalCp;
    output.instrumentalProfile = search.instrumentalProfile;
    output.requestedSimulations = static_cast<std::uint64_t>(params.simBudget);
    return output;
}
