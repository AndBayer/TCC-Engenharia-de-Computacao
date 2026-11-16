#pragma once

#include "puzzle_validation.hpp"

#include <fstream>
#include <string>
#include <unordered_set>

inline constexpr const char *kAutonomousCsvHeaderLine = "puzzle_id,fen,rating,cp,instrumental_profile";

// double --> texto com todos os dígitos e ponto decimal
std::string serializeDouble(double value);

// grava o CSV linha a linha; ao abrir um CSV existente, mantém a primeira linha completa de cada id e marca esses ids (contains) para a busca pular
class ResumableCsvWriter
{
public:
    ResumableCsvWriter(const std::string &path, const std::string &header);

    bool contains(const std::string &id) const;
    size_t resumedCount() const;
    size_t discardedCount() const;
    void appendRow(const std::string &row);

private:
    std::unordered_set<std::string> resumedIds_;
    size_t discarded_ = 0;
    std::ofstream output_;
};

// CSV da validação com puzzles --> id, FEN, rating, cp e perfil instrumental
class PuzzleCsvWriter : public ResumableCsvWriter
{
public:
    explicit PuzzleCsvWriter(const std::string &path) : ResumableCsvWriter(path, kAutonomousCsvHeaderLine) {}

    void append(const AutonomousValidationResult &result);
};
