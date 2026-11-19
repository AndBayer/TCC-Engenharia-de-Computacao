#include "puzzle_sampling.hpp"

#include <charconv>
#include <istream>
#include <ostream>
#include <random>
#include <string_view>

namespace
{
    // faixas de rating de 200 em 200, de 200-399 a 3200-3399
    constexpr int kBinLow = 200; // início da primeira faixa
    constexpr int kBinHigh = 3200; // início da última faixa
    constexpr int kBinWidth = 200; // largura de cada faixa
    constexpr int kBinCount = (kBinHigh - kBinLow) / kBinWidth + 1; // 16 faixas

    // corta o texto no separador; nos espaços ignora pedaços vazios, nas vírgulas mantém todos os campos
    std::vector<std::string_view> split(std::string_view text, char separator)
    {
        std::vector<std::string_view> fields;
        size_t start = 0;
        for (size_t end = text.find(separator); end != std::string_view::npos;
             end = text.find(separator, start))
        {
            if (separator != ' ' || end > start)
                fields.emplace_back(text.data() + start, end - start);
            start = end + 1;
        }
        if (start < text.size() || separator == ',')
            fields.emplace_back(text.data() + start, text.size() - start);
        return fields;
    }

    // faixa do rating (0 = 200-399, ..., 15 = 3200-3399); -1 fora das faixas
    int binOf(int rating)
    {
        const int lowerBound = (rating / kBinWidth) * kBinWidth;
        return lowerBound < kBinLow || lowerBound > kBinHigh ? -1 : (lowerBound - kBinLow) / kBinWidth;
    }
} // namespace

// sorteia perBin puzzles por faixa de rating (amostragem de reservatório)
std::vector<SampledPuzzle> samplePuzzles(std::istream &input, int perBin, std::uint64_t seed, std::ostream &progress)
{
    std::mt19937_64 rng(seed);
    std::vector<std::vector<SampledPuzzle>> reservoirs(kBinCount);
    std::vector<long long> seen(kBinCount, 0);
    for (auto &reservoir : reservoirs)
        reservoir.reserve(static_cast<size_t>(perBin));

    std::string line;
    size_t lineNumber = 0;
    while (std::getline(input, line))
    {
        ++lineNumber;
        if (line.empty() || (lineNumber == 1 && line.rfind("PuzzleId", 0) == 0))
            continue;
        const auto fields = split(line, ',');
        if (fields.size() < 4)
            continue;
        int rating = 0;
        std::from_chars(fields[3].data(), fields[3].data() + fields[3].size(), rating);
        const int bin = binOf(rating);
        // pula puzzle fora das faixas ou sem lance de solução
        const auto moves = split(fields[2], ' ');
        if (bin < 0 || moves.size() < 2)
            continue;

        SampledPuzzle puzzle;
        puzzle.id = fields[0];
        puzzle.fen = fields[1];
        puzzle.rating = rating;
        for (const auto move : moves)
            puzzle.moves.emplace_back(move);

        // mantém uma amostra uniforme da faixa
        const size_t index = static_cast<size_t>(bin);
        ++seen[index];
        auto &reservoir = reservoirs[index];
        if (static_cast<int>(reservoir.size()) < perBin)
            reservoir.push_back(std::move(puzzle));
        else
        {
            std::uniform_int_distribution<long long> distribution(0, seen[index] - 1);
            const long long replacement = distribution(rng);
            if (replacement < perBin)
                reservoir[static_cast<size_t>(replacement)] = std::move(puzzle);
        }
        if (lineNumber % 500000 == 0)
            progress << "\r[INFO] Read " << lineNumber << " lines..." << std::flush;
    }

    std::vector<SampledPuzzle> sampled;
    for (auto &reservoir : reservoirs)
        for (auto &puzzle : reservoir)
            sampled.push_back(std::move(puzzle));
    progress << "\r[INFO] Sampling complete (" << lineNumber << " lines, " << sampled.size() << " puzzles).\n";
    return sampled;
}
