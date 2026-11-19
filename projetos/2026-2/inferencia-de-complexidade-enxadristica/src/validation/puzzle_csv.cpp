#include "puzzle_csv.hpp"

#include "instrumental_profile.hpp"

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iterator>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace
{
    namespace fs = std::filesystem;

    // linha completa --> mesmo número de campos do cabeçalho, nenhum campo vazio e perfil (último campo) legível; o id é o primeiro campo
    bool completeRow(const std::string &line, size_t expectedFields, std::string &id)
    {
        const auto fields = engine::detail::split(line, ',');
        if (fields.size() != expectedFields || std::ranges::any_of(fields, [](std::string_view field)
                                                                   { return field.empty(); }))
            return false;
        try
        {
            (void)engine::parseRawInstrumentalProfile(fields.back());
        }
        catch (const std::exception &)
        {
            return false;
        }
        id = std::string(fields.front());
        return true;
    }
} // namespace

std::string serializeDouble(double value)
{
    std::ostringstream serialized;
    serialized.imbue(std::locale::classic());
    serialized << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
    return serialized.str();
}

// CSV novo --> só o cabeçalho; CSV existente --> descarta linhas cortadas e repetidas, regrava o arquivo e continua anexando
ResumableCsvWriter::ResumableCsvWriter(const std::string &path, const std::string &header)
{
    const fs::path outputPath(path);
    if (!fs::exists(outputPath))
    {
        output_.open(outputPath, std::ios::binary | std::ios::trunc);
        output_ << header << '\n';
        output_.flush();
        if (!output_)
            throw std::runtime_error("Could not open CSV output: " + path);
        return;
    }

    std::ifstream previous(outputPath, std::ios::binary);
    if (!previous)
        throw std::runtime_error("Could not read existing CSV output: " + path);
    std::string content((std::istreambuf_iterator<char>(previous)), std::istreambuf_iterator<char>());
    previous.close();
    // tira bytes nulos, que podem sobrar no arquivo depois de uma queda durante a escrita
    content.erase(std::remove(content.begin(), content.end(), '\0'), content.end());
    // texto depois da última quebra de linha foi cortado no meio da escrita
    const size_t lastBreak = content.rfind('\n');
    content = lastBreak == std::string::npos ? std::string() : content.substr(0, lastBreak + 1);

    const size_t expectedFields = static_cast<size_t>(std::ranges::count(header, ',')) + 1;
    std::istringstream lines(content);
    std::string line;
    std::vector<std::string> kept;
    bool first = true;
    while (std::getline(lines, line))
    {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        // cabeçalho diferente --> CSV de outro formato, recusado sem alterar o arquivo
        if (first)
        {
            first = false;
            if (line != header)
                throw std::runtime_error("Incompatible CSV header; use a new output path: " + path);
            continue;
        }
        std::string id;
        if (completeRow(line, expectedFields, id) && resumedIds_.insert(id).second)
            kept.push_back(line);
        else
            ++discarded_;
    }

    // regrava o CSV só com as linhas mantidas, num arquivo temporário que depois substitui o original
    fs::path temporary = outputPath;
    temporary += ".tmp";
    {
        std::ofstream cleaned(temporary, std::ios::binary | std::ios::trunc);
        cleaned << header << '\n';
        for (const auto &row : kept)
            cleaned << row << '\n';
        cleaned.flush();
        if (!cleaned)
            throw std::runtime_error("Could not write temporary CSV file: " + temporary.string());
    }
    fs::rename(temporary, outputPath);
    output_.open(outputPath, std::ios::binary | std::ios::app);
    if (!output_)
        throw std::runtime_error("Could not open CSV output: " + path);
}

bool ResumableCsvWriter::contains(const std::string &id) const
{
    return resumedIds_.contains(id);
}

size_t ResumableCsvWriter::resumedCount() const
{
    return resumedIds_.size();
}

size_t ResumableCsvWriter::discardedCount() const
{
    return discarded_;
}

// grava a linha e descarrega na hora, então uma queda perde no máximo a linha em andamento
void ResumableCsvWriter::appendRow(const std::string &row)
{
    output_ << row << '\n';
    output_.flush();
    if (!output_)
        throw std::runtime_error("Could not write CSV row");
}

// uma linha por puzzle --> id, FEN, rating, cp e perfil instrumental
void PuzzleCsvWriter::append(const AutonomousValidationResult &result)
{
    appendRow(result.puzzleId + ',' + result.fen + ',' + std::to_string(result.rating) + ',' + serializeDouble(result.centipawns) + ',' + engine::serializeInstrumentalProfile(result.instrumentalProfile));
}
