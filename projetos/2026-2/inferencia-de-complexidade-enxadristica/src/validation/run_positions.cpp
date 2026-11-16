// roda a busca em posições lidas do stdin, linhas CSV "position_id,fen" (cabeçalho opcional)
// diferente do run_puzzles, sem amostragem por rating e sem aplicar lance antes da busca
// saída em "position_id,fen,cp,instrumental_profile", uma linha por posição, gravada na hora
// ao reiniciar, mantém as linhas completas e refaz as cortadas ou corrompidas

#include "chess.hpp"
#include "inference.hpp"
#include "instrumental_profile.hpp"
#include "params.hpp"
#include "puzzle_csv.hpp"
#include "search.hpp"

#include <algorithm>
#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace
{
    constexpr const char *kHeader = "position_id,fen,cp,instrumental_profile";

    // opções da linha de comando, com os valores padrão
    struct Options
    {
        std::string modelPath = "model.onnx";
        std::string outputPath = "posicoes_busca.csv";
        long long simulations = 2048;  // orçamento de simulações por posição
        int batchSize = 512; // lote máximo da fila de inferência
        int workers = 16; // buscas em paralelo, uma por thread
        std::string inferenceMode = "auto"; // auto --> pipeline na GPU e, se falhar, ONNX Runtime pela fila; host --> ONNX Runtime pela fila; cuda --> só o pipeline na GPU
        int gatherBatchSize = 32; // folhas por lote em cada árvore
    };

    struct Position
    {
        std::string id;
        std::string fen;
    };

    void printUsage(std::ostream &output)
    {
        output << "Usage: type posicoes.csv | run_positions\n"
                  "       [--model <.onnx>] [--output <csv>] [--sims N]\n"
                  "       [--batch B] [--workers W] [--gather-batch N]\n"
                  "       [--inference auto|host|cuda]\n"
                  "Reads \"position_id,fen\" lines from stdin and writes one raw\n"
                  "search profile per position to CSV.\n";
    }

    Options parseOptions(int argc, char *argv[])
    {
        Options options;
        for (int index = 1; index < argc; ++index)
        {
            const std::string argument = argv[index];
            const auto value = [&](const char *name) -> std::string
            {
                if (index + 1 >= argc)
                    throw std::invalid_argument(std::string("missing value for ") + name);
                return argv[++index];
            };
            if (argument == "--model" || argument == "-m")
                options.modelPath = value("--model");
            else if (argument == "--output" || argument == "-o")
                options.outputPath = value("--output");
            else if (argument == "--sims")
                options.simulations = std::stoll(value("--sims"));
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
                throw std::invalid_argument("unknown option: " + argument);
        }
        if (options.simulations <= 0)
            throw std::invalid_argument("--sims must be positive");
        return options;
    }

    // lê as linhas id,fen; pula linha vazia, linha sem vírgula e o cabeçalho
    std::vector<Position> readPositions(std::istream &input)
    {
        std::vector<Position> positions;
        std::string line;
        while (std::getline(input, line))
        {
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            const size_t comma = line.find(',');
            if (line.empty() || comma == std::string::npos)
                continue;
            Position position{line.substr(0, comma), line.substr(comma + 1)};
            if (position.id == "position_id")
                continue;
            positions.push_back(std::move(position));
        }
        std::cerr << "[INFO] Read " << positions.size() << " positions.\n";
        return positions;
    }
} // namespace

int main(int argc, char *argv[])
{
    std::ios_base::sync_with_stdio(false);

    try
    {
        const Options options = parseOptions(argc, argv);
        const std::vector<Position> positions = readPositions(std::cin);

        engine::SearchParams searchParams;
        searchParams.simBudget = options.simulations;
        searchParams.batchSize = options.batchSize;
        searchParams.gatherBatchSize = options.gatherBatchSize;
        std::cerr << "[INFO] Budget: " << options.simulations << " simulations per position.\n";

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

        ResumableCsvWriter csv(options.outputPath, kHeader);
        if (csv.resumedCount() > 0 || csv.discardedCount() > 0)
            std::cerr << "[INFO] Resume: kept " << csv.resumedCount() << " complete rows, discarded " << csv.discardedCount() << " incomplete.\n";

        std::atomic<size_t> nextIndex{0};
        std::atomic<size_t> completed{0};
        std::atomic<size_t> skipped{0};
        std::mutex csvMutex;

        // cada worker pega a próxima posição da lista e roda a busca dela
        const auto worker = [&]()
        {
            for (;;)
            {
                const size_t index = nextIndex.fetch_add(1, std::memory_order_relaxed);
                if (index >= positions.size())
                    return;
                const Position &position = positions[index];
                if (csv.contains(position.id))
                {
                    completed.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }

                std::string fen;
                engine::SearchResult search;
                try
                {
                    const chess::Board board(position.fen);
                    search = engine::runSearch(board, searchParams, *evaluator);
                    fen = board.getFen();
                }
                catch (const std::exception &error)
                {
                    std::lock_guard<std::mutex> lock(csvMutex);
                    std::cerr << "\n[WARN] " << position.id << " skipped: " << error.what() << '\n';
                    skipped.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }

                std::lock_guard<std::mutex> lock(csvMutex);
                csv.appendRow(position.id + ',' + fen + ',' + serializeDouble(search.evalCp) + ',' + engine::serializeInstrumentalProfile(search.instrumentalProfile));
                const size_t count = completed.fetch_add(1, std::memory_order_relaxed) + 1;
                if (count % 50 == 0)
                    std::cerr << "\r[INFO] " << count << '/' << positions.size() << " positions evaluated..." << std::flush;
            }
        };

        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(options.workers));
        for (int index = 0; index < options.workers; ++index)
            workers.emplace_back(worker);
        for (auto &workerThread : workers)
            workerThread.join();
        evaluator.reset();

        std::cerr << "\r[INFO] Complete: " << completed.load() << " evaluated, " << skipped.load() << " skipped. Output: " << options.outputPath << '\n';
        // retorna 1 se alguma posição foi pulada
        return skipped.load() == 0 ? 0 : 1;
    }
    catch (const std::exception &error)
    {
        std::cerr << "[ERROR] " << error.what() << '\n';
        return 1;
    }
}
