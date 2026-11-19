#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

// puzzle sorteado do CSV do Lichess
struct SampledPuzzle
{
    std::string id;
    std::string fen; // posição antes do lance forçador
    std::vector<std::string> moves; // lance forçador seguido da solução, em UCI
    int rating = 0;
};

// sorteia perBin puzzles em cada faixa de 200 pontos de rating
std::vector<SampledPuzzle> samplePuzzles(std::istream &input, int perBin, std::uint64_t seed, std::ostream &progress);
