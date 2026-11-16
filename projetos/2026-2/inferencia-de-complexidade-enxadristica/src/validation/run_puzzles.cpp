// valida o motor numa amostra estratificada de puzzles do Lichess lidos do stdin
// amostragem, busca no puzzle e CSV ficam em outros arquivos; o run_puzzles só lê opções e coordena os workers

#include "inference.hpp"
#include "params.hpp"
#include "puzzle_csv.hpp"
#include "puzzle_sampling.hpp"
#include "puzzle_validation.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace
{
    // opções da linha de comando, com os valores padrão
    struct Options
    {
        std::string modelPath = "checkpoint_epoch20.onnx";
        std::string outputPath = "validacao_motor.csv";
        long long simulations = 1600;  // orçamento de simulações por puzzle
        int perBin = 200; // puzzles sorteados por faixa de rating
        std::uint64_t seed = 42; // semente do sorteio
        int batchSize = 512; // lote máximo da fila de inferência
        int workers = 16; // buscas em paralelo, uma por thread
        std::string inferenceMode = "auto"; // auto --> pipeline na GPU e, se falhar, ONNX Runtime pela fila; host --> ONNX Runtime pela fila; cuda --> só o pipeline na GPU
        int gatherBatchSize = 32; // folhas por lote em cada árvore
    };

    void printUsage(std::ostream &output)
    {
        output << "Usage: zstd -dc lichess_db_puzzle.csv.zst | run_puzzles\n"
                  "       [--model <.onnx>] [--output <csv>] [--sims N]\n"
                  "       [--per-bin K] [--seed S] [--batch B] [--workers W]\n"
                  "       [--inference auto|host|cuda]\n"
                  "       [--gather-batch N]\n"
                  "Reads Lichess puzzles from stdin, samples K puzzles per 200-point\n"
                  "rating bin and writes raw search profiles to CSV. --sims is\n"
                  "the search budget per puzzle.\n";
    }

    Options parseOptions(int argc, char *argv[])
    {
        Options options;
        for (int index = 1; index < argc; ++index)
        {
            const std::string argument = argv[index];
            const auto value = [&](const char *flag) -> std::string
            {
                if (index + 1 >= argc)
                    throw std::invalid_argument(std::string("Missing value for ") + flag);
                return argv[++index];
            };

            if (argument == "--model" || argument == "-m")
                options.modelPath = value("--model");
            else if (argument == "--output" || argument == "-o")
                options.outputPath = value("--output");
            else if (argument == "--sims")
                options.simulations = std::stoll(value("--sims"));
            else if (argument == "--per-bin")
                options.perBin = std::stoi(value("--per-bin"));
            else if (argument == "--seed")
                options.seed = std::stoull(value("--seed"));
            else if (argument == "--batch")
                options.batchSize = std::clamp(std::stoi(value("--batch")), 1, 1024);
            else if (argument == "--workers")
                options.workers = std::max(1, std::stoi(value("--workers")));
            else if (argument == "--gather-batch")
                options.gatherBatchSize = std::clamp(std::stoi(value("--gather-batch")), 1, 1024);
            else if (argument == "--inference")
            {
                options.inferenceMode = value("--inference");
                if (options.inferenceMode != "auto" && options.inferenceMode != "host" && options.inferenceMode != "cuda")
                    throw std::invalid_argument("--inference must be auto, host, or cuda");
            }
            else if (argument == "--help" || argument == "-h")
            {
                printUsage(std::cout);
                std::exit(0);
            }
            else
                throw std::invalid_argument("Unknown argument: " + argument);
        }
        if (options.simulations <= 0 || options.perBin <= 0)
            throw std::invalid_argument("--sims and --per-bin must be positive");
        return options;
    }
} // namespace

int main(int argc, char *argv[])
{
    std::ios_base::sync_with_stdio(false);

    try
    {
        const Options options = parseOptions(argc, argv);
        std::vector<SampledPuzzle> puzzles = samplePuzzles(std::cin, options.perBin, options.seed, std::cerr);

        engine::SearchParams searchParams;
        searchParams.simBudget = options.simulations;
        searchParams.batchSize = options.batchSize;
        searchParams.gatherBatchSize = options.gatherBatchSize;
        std::cerr << "[INFO] Search: one coordinator per tree (gather=" << options.gatherBatchSize << ").\n";
        std::cerr << "[INFO] Budget: " << options.simulations << " simulations per puzzle.\n";

        // um avaliador para todas as buscas --> pipeline na GPU ou fila do EvalService com o ONNX Runtime
        std::unique_ptr<engine::NetEvaluator> network;
        std::unique_ptr<engine::PositionEvaluator> evaluator;
        if (options.inferenceMode != "host")
        {
            try
            {
                evaluator = std::make_unique<engine::CudaEvalService>(options.modelPath, options.batchSize, searchParams.batchTimeoutUs);
                std::cerr << "[INFO] Inference pipeline: cuda-resident (CUDA provider).\n";
            }
            catch (const std::exception &error)
            {
                if (options.inferenceMode == "cuda")
                    throw;
                std::cerr << "[WARN] CUDA-resident inference unavailable: " << error.what() << "\n";
            }
        }
        if (!evaluator)
        {
            network = engine::makeOnnxEvaluator(options.modelPath);
            evaluator = std::make_unique<engine::EvalService>(*network, options.batchSize, searchParams.batchTimeoutUs);
            std::cerr << "[INFO] Inference pipeline: host-staged.\n";
        }

        // CSV retomável --> puzzles já gravados são pulados
        PuzzleCsvWriter csv(options.outputPath);
        if (csv.resumedCount() > 0)
            std::cerr << "[INFO] Resume: kept " << csv.resumedCount() << " complete rows.\n";

        std::atomic<size_t> nextIndex{0};
        std::atomic<size_t> completed{0};
        std::atomic<size_t> skipped{0};
        std::atomic<unsigned long long> fixedNodeSearches{0};
        std::atomic<unsigned long long> requestedSimulations{0};
        std::mutex csvMutex;

        // cada worker pega o próximo puzzle da lista e roda a busca dele
        const auto worker = [&]()
        {
            for (;;)
            {
                const size_t index = nextIndex.fetch_add(1, std::memory_order_relaxed);
                if (index >= puzzles.size())
                    return;
                const SampledPuzzle &puzzle = puzzles[index];
                if (csv.contains(puzzle.id))
                {
                    completed.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }

                auto result = validateAutonomousPuzzle(puzzle, searchParams, *evaluator);
                if (!result)
                {
                    skipped.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                fixedNodeSearches.fetch_add(1, std::memory_order_relaxed);
                requestedSimulations.fetch_add(result->requestedSimulations, std::memory_order_relaxed);
                {
                    std::lock_guard<std::mutex> lock(csvMutex);
                    csv.append(*result);
                }

                const size_t count = completed.fetch_add(1, std::memory_order_relaxed) + 1;
                if (count % 50 == 0)
                {
                    std::lock_guard<std::mutex> lock(csvMutex);
                    std::cerr << "\r[INFO] " << count << '/' << puzzles.size() << " puzzles evaluated..." << std::flush;
                }
            }
        };

        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(options.workers));
        for (int index = 0; index < options.workers; ++index)
            workers.emplace_back(worker);
        for (auto &workerThread : workers)
            workerThread.join();
        evaluator.reset();

        const unsigned long long searches = fixedNodeSearches.load();
        std::cerr << "\r[INFO] Complete: " << completed.load() << " evaluated, " << skipped.load() << " skipped. Output: " << options.outputPath << '\n';
        std::cerr << "[INFO] Fixed-node searches: " << searches << "; requested simulations: " << requestedSimulations.load() << ".\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "[ERROR] " << error.what() << '\n';
        printUsage(std::cerr);
        return 2;
    }
}
