#include "puzzle_csv.hpp"

#include "instrumental_profile.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace
{
    namespace fs = std::filesystem;

    // pasta temporária, apagada no fim do teste
    struct TemporaryDirectory
    {
        fs::path path = fs::temp_directory_path() / ("tfird_csv_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));

        TemporaryDirectory() { fs::create_directories(path); }
        ~TemporaryDirectory()
        {
            std::error_code ignored;
            fs::remove_all(path, ignored);
        }
    };

    std::string readFile(const fs::path &path)
    {
        std::ifstream input(path);
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }

    // nó de exemplo com dois filhos visitados e um sem visita (sem Q e V)
    engine::InstrumentalProfileNode rootNode()
    {
        engine::InstrumentalProfileNode root;
        root.nodeVisits = 8;
        root.children = {
            {"e2e4", 6, 0.5f, 0.75, 0.125f},
            {"d2d4", 2, 0.25f, -0.5, -0.75f},
            {"g1f3", 0, 0.25f, std::nullopt, std::nullopt},
        };
        return root;
    }
} // namespace

int main()
{
    TemporaryDirectory temporary;
    // texto --> perfil --> texto precisa voltar igual
    const std::string serialized = "8{e2e4:6:0.5:0.75:0.125|d2d4:2:0.25:-0.5:-0.75|g1f3:0:0.25:-:-}";
    const auto roundTrip = engine::parseRawInstrumentalProfile(serialized);
    if (roundTrip.size() != 1 || roundTrip[0].children.size() != 3 ||
        roundTrip[0].children[0].move != "e2e4" ||
        roundTrip[0].children[2].q || roundTrip[0].children[2].v ||
        engine::serializeInstrumentalProfile(roundTrip) != serialized)
        throw std::runtime_error("raw profile round-trip changed N/P/Q/V");

    // CSV gravado --> cabeçalho e uma linha por puzzle
    const fs::path autonomousPath = temporary.path / "autonomous.csv";
    AutonomousValidationResult autonomous;
    autonomous.puzzleId = "puzzle-1";
    autonomous.fen = "8/8/8/8/8/8/8/8 w - - 0 1";
    autonomous.rating = 1500;
    autonomous.centipawns = 35.0;
    autonomous.instrumentalProfile = {rootNode()};
    {
        PuzzleCsvWriter writer(autonomousPath.string());
        writer.append(autonomous);
    }
    const std::string autonomousCsv = readFile(autonomousPath);
    if (!autonomousCsv.starts_with(std::string(kAutonomousCsvHeaderLine) + "\n") ||
        !autonomousCsv.ends_with(",35," + serialized + "\n"))
        throw std::runtime_error("autonomous CSV header or row mismatch");

    // reabrir o CSV marca o puzzle já gravado
    PuzzleCsvWriter resumedAutonomous(autonomousPath.string());

    if (resumedAutonomous.resumedCount() != 1 ||
        !resumedAutonomous.contains("puzzle-1"))
        throw std::runtime_error("autonomous resume failed");

    // perfil vazio é gravado como '-'
    AutonomousValidationResult empty = autonomous;
    empty.puzzleId = "empty";
    empty.instrumentalProfile.clear();
    {
        PuzzleCsvWriter writer((temporary.path / "empty.csv").string());
        writer.append(empty);
    }
    if (!readFile(temporary.path / "empty.csv").ends_with(",-\n"))
        throw std::runtime_error("empty profile was not serialized as '-'");

    return 0;
}
