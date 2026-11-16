// NOMINMAX impede o <windows.h> (inclusive via onnxruntime) de definir min/max, que colidem com std::min/std::max
// precisa vir antes de qualquer include
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "chess.hpp"
#include "board_encoder.hpp"
#include "encoding.hpp"
#include <algorithm>
#include <atomic>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <memory>
#include <filesystem>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <string_view>
#include <charconv>
#include <cstdlib>
#include <execution>
#include <numeric>
#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#include <windows.h>
#endif

// uciToIndex e constantes vêm de shared/include/encoding.hpp; o tabuleiro é codificado pelo encodeBoard do motor
using encoding::BOARD_SQ;
using encoding::CHANNELS;
using encoding::POLICY_SIZE;
using encoding::uciToIndex;

constexpr int BATCH_SIZE = 1024;  // puzzles por lote lido e posições por chamada da rede
constexpr int MAX_QUEUE_SIZE = 4; // lotes esperando em cada fila, no máximo

// separa a linha do CSV nas vírgulas
static std::vector<std::string_view> split_csv_fast(std::string_view line)
{
    std::vector<std::string_view> out;
    out.reserve(5);
    size_t start = 0, pos = line.find(',');
    while (pos != std::string_view::npos)
    {
        out.emplace_back(line.data() + start, pos - start);
        start = pos + 1;
        pos = line.find(',', start);
    }
    out.emplace_back(line.data() + start, line.size() - start);
    return out;
}

// separa a lista de lances nos espaços
static std::vector<std::string_view> split_spaces_fast(std::string_view str)
{
    std::vector<std::string_view> out;
    size_t start = 0, pos = str.find(' ');
    while (pos != std::string_view::npos)
    {
        if (pos > start)
            out.emplace_back(str.data() + start, pos - start);
        start = pos + 1;
        pos = str.find(' ', start);
    }
    if (start < str.size())
        out.emplace_back(str.data() + start, str.size() - start);
    return out;
}

// uma decisão do resolvedor na linha do puzzle, com posição antes do lance, lances legais,
// gabarito e, se o gabarito for mate, os outros mates aceitos (regra do Lichess)
struct SolverStep
{
    std::vector<float> input; // CHANNELS*BOARD_SQ (posição antes do lance do resolvedor)
    bool flip;
    chess::Movelist legal_moves;
    std::string gabarito;                 // lance da solução do Lichess (UCI)
    std::vector<uint8_t> equivalent_mate; // alinhado com legal_moves; vazio se não é mate
    std::vector<int32_t> policy_idx;      // índice na política de cada lance legal (calculado antes)
    int gabarito_mi = -1;                 // posição do gabarito em legal_moves
};

// puzzle lido do CSV
struct PuzzleMeta
{
    std::string puzzle_id;
    int rating;
    int solution_length;           // número de meios-lances da solução (sem o lance forçador)
    std::vector<SolverStep> steps; // uma entrada por decisão do resolvedor
};

// métrica por (puzzle, modelo) --> lances do resolvedor em que o lance de maior logit da política bate com o gabarito
// (cada decisão avaliada na posição do gabarito); completed = (solved_moves == total)
struct PuzzleResult
{
    uint16_t solved_moves; // lances do resolvedor acertados ao longo da linha (0 a total)
};

// lote de puzzles para a GPU
struct BatchJob
{
    std::vector<PuzzleMeta> batched_meta;
};

// lote já avaliado, esperando a gravação nos CSVs
struct WriterJob
{
    std::vector<PuzzleMeta> batched_meta;
    // [puzzle_idx][model_idx] --> resultado
    std::vector<std::vector<PuzzleResult>> batch_results;
};

// uma sessão do ONNX Runtime por modelo, no provedor CUDA
class BenchmarkEngine
{
    Ort::Env env;
    Ort::Session session{nullptr};

public:
    BenchmarkEngine(const std::string &model_path) : env(ORT_LOGGING_LEVEL_WARNING, "bench")
    {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        // HEURISTIC em vez do EXHAUSTIVE, pois o exhaustive refaz o benchmark do cuDNN a cada
        // tamanho de lote novo (segundos por tamanho) e derruba o desempenho
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
        session_options.AppendExecutionProvider_CUDA(cuda_options);

#ifdef _WIN32
        session = Ort::Session(env, std::wstring(model_path.begin(), model_path.end()).c_str(), session_options);
#else
        session = Ort::Session(env, model_path.c_str(), session_options);
#endif
    }

    // resolve o puzzle inteiro pelo lance de maior logit da política, para todos os puzzles do lote
    // toda decisão é avaliada na posição do gabarito, então os passos são independentes
    // a lista (puzzle, passo) é fatiada em lotes sempre cheios, sem sobras que subutilizam a GPU
    void evaluateBatch(const std::vector<PuzzleMeta> &meta_batch, std::vector<std::vector<PuzzleResult>> &batch_results, int model_index)
    {
        const int n = static_cast<int>(meta_batch.size());
        std::vector<uint16_t> solved_count(static_cast<size_t>(n), 0);

        std::vector<std::pair<int, int>> flat; // (puzzle_idx, step_idx)
        {
            size_t total = 0;
            for (const auto &m : meta_batch)
                total += m.steps.size();
            flat.reserve(total);
        }
        for (int i = 0; i < n; ++i)
            for (size_t s = 0; s < meta_batch[static_cast<size_t>(i)].steps.size(); ++s)
                flat.emplace_back(i, static_cast<int>(s));

        // entrada montada na CPU; o ONNX Runtime copia para a GPU
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const char *in_names[] = {"board"};
        const char *out_names[] = {"policy", "value"};

        std::vector<float> inputs;
        inputs.reserve(static_cast<size_t>(BATCH_SIZE) * CHANNELS * BOARD_SQ);

        for (size_t base = 0; base < flat.size(); base += BATCH_SIZE)
        {
            const int cbs = static_cast<int>((std::min)(static_cast<size_t>(BATCH_SIZE), flat.size() - base));

            inputs.clear();
            for (int j = 0; j < cbs; ++j)
            {
                const auto &ps = flat[base + static_cast<size_t>(j)];
                const auto &in = meta_batch[static_cast<size_t>(ps.first)].steps[static_cast<size_t>(ps.second)].input;
                inputs.insert(inputs.end(), in.begin(), in.end());
            }

            // lote sempre com BATCH_SIZE, e o último é completado com zeros
            // tamanhos variáveis fazem o cuDNN replanejar as convoluções; as saídas do preenchimento são ignoradas
            inputs.resize(static_cast<size_t>(BATCH_SIZE) * CHANNELS * BOARD_SQ, 0.0f);

            std::array<int64_t, 4> shape = {BATCH_SIZE, CHANNELS, 8, 8};
            auto tensor = Ort::Value::CreateTensor<float>(mem, inputs.data(), static_cast<size_t>(BATCH_SIZE) * CHANNELS * BOARD_SQ, shape.data(), shape.size());
            auto outputs = session.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 2);
            const float *policy_batch = outputs[0].GetTensorData<float>();

            for (int j = 0; j < cbs; ++j)
            {
                const float *policy = policy_batch + static_cast<size_t>(j) * POLICY_SIZE;
                const int i = flat[base + static_cast<size_t>(j)].first;
                const auto &step = meta_batch[static_cast<size_t>(i)].steps[static_cast<size_t>(flat[base + static_cast<size_t>(j)].second)];

                // argmax da política nos lances legais (lance de maior logit), com índices já calculados
                int best_mi = -1;
                float best_logit = -1e30f;
                for (size_t mi = 0; mi < step.policy_idx.size(); ++mi)
                {
                    const int idx = step.policy_idx[mi];
                    if (idx < 0 || idx >= POLICY_SIZE)
                        continue;
                    if (policy[idx] > best_logit)
                    {
                        best_logit = policy[idx];
                        best_mi = static_cast<int>(mi);
                    }
                }

                // aceita se bate com o gabarito ou é mate equivalente (regra do Lichess)
                const bool accepted =
                    best_mi >= 0 &&
                    (best_mi == step.gabarito_mi ||
                     (!step.equivalent_mate.empty() && step.equivalent_mate[static_cast<size_t>(best_mi)]));
                if (accepted)
                    ++solved_count[static_cast<size_t>(i)]; // conta a decisão acertada
            }
        }

        for (int i = 0; i < n; ++i)
            batch_results[static_cast<size_t>(i)][static_cast<size_t>(model_index)].solved_moves = solved_count[static_cast<size_t>(i)];
    }
};

int main(int argc, char *argv[])
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    // argumentos da linha de comando
    std::string model_dir = ".";
    std::string dll_dir;
    std::string input_path;
    std::string output_dir = "."; // pasta dos CSVs, um por modelo

    auto print_usage = [&](std::ostream &os)
    {
        os << "Uso: model_benchmark [--models <dir>] [--cudnn <dir>] [--input <arquivo.csv>] [--outdir <dir>]\n"
           << "  -m, --models  Pasta com os .onnx\n"
           << "  -c, --cudnn   Pasta com o cudnn64_9.dll\n"
           << "  -i, --input   CSV de puzzles (default: stdin)\n"
           << "  -o, --outdir  Pasta de saida (1 CSV por modelo) (default: .)\n"
           << "  -h, --help    Mostra esta ajuda\n";
    };

    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto need_value = [&](const std::string &flag) -> std::string
        {
            if (i + 1 >= argc)
            {
                std::cerr << "[ERRO] Faltou valor para " << flag << std::endl;
                std::exit(2);
            }
            return argv[++i];
        };

        if (a == "-m" || a == "--models")
            model_dir = need_value(a);
        else if (a == "-c" || a == "--cudnn")
            dll_dir = need_value(a);
        else if (a == "-i" || a == "--input")
            input_path = need_value(a);
        else if (a == "-o" || a == "--outdir")
            output_dir = need_value(a);
        else if (a == "-h" || a == "--help")
        {
            print_usage(std::cout);
            return 0;
        }
        else
        {
            std::cerr << "[ERRO] Argumento desconhecido: " << a << "\n";
            print_usage(std::cerr);
            return 2;
        }
    }

    // pasta extra com as DLLs do CUDA/cuDNN
    if (!dll_dir.empty())
    {
#ifdef _WIN32
        const char *old_path = std::getenv("PATH");
        const std::string new_path = dll_dir + ";" + (old_path ? old_path : "");
        SetEnvironmentVariableA("PATH", new_path.c_str());
        _putenv_s("PATH", new_path.c_str());
#endif
        std::cerr << "[INFO] Diretorio de DLLs (CUDA/cuDNN): " << dll_dir << std::endl;
    }

    std::cerr << "[INFO] Iniciando benchmark de PUZZLE INTEIRO (policy greedy) (modelos em: " << model_dir << ")" << std::endl;

    if (!std::filesystem::is_directory(model_dir))
    {
        std::cerr << "[ERRO] Diretorio nao encontrado: " << model_dir << std::endl;
        return 1;
    }

    std::vector<std::string> model_names;
    for (const auto &entry : std::filesystem::directory_iterator(model_dir))
    {
        if (entry.path().extension() == ".onnx")
        {
            model_names.push_back(entry.path().filename().string());
        }
    }
    if (model_names.empty())
    {
        std::cerr << "[ERRO] Nenhum .onnx encontrado em: " << model_dir << std::endl;
        return 1;
    }
    // ordena por número de época
    auto epoch_of = [](const std::string &n)
    {
        size_t p = n.find("epoch");
        if (p == std::string::npos)
            return 0;
        int e = 0;
        std::from_chars(n.data() + p + 5, n.data() + n.size(), e);
        return e;
    };
    std::sort(model_names.begin(), model_names.end(), [&](const std::string &a, const std::string &b)
              { return epoch_of(a) < epoch_of(b); });

    // carrega todos os modelos antes de ler os puzzles
    std::vector<std::unique_ptr<BenchmarkEngine>> engines;
    for (size_t i = 0; i < model_names.size(); ++i)
    {
        const std::string model_path = (std::filesystem::path(model_dir) / model_names[i]).string();
        std::cerr << "[INFO] Carregando: " << model_path << std::endl;
        engines.push_back(std::make_unique<BenchmarkEngine>(model_path));
    }

    // um CSV por modelo em output_dir, com nome <base>.csv
    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    std::vector<std::ofstream> csv_outs(model_names.size());
    for (size_t e = 0; e < model_names.size(); ++e)
    {
        const std::string base = model_names[e].substr(0, model_names[e].size() - 5);
        const std::string path = (std::filesystem::path(output_dir) / (base + ".csv")).string();
        csv_outs[e].open(path);
        if (!csv_outs[e])
        {
            std::cerr << "[ERRO] Nao foi possivel criar o CSV: " << path << std::endl;
            return 1;
        }
        csv_outs[e] << "PuzzleId,Rating,total_moves,solved_moves,completed\n";
    }

    // pipeline --> linhas --> parsers --> GPU --> writer
    std::queue<BatchJob> gpu_queue;
    std::mutex gpu_mtx;
    std::condition_variable cv_gpu_consumer, cv_gpu_producer;
    bool reader_finished = false;

    std::queue<WriterJob> writer_queue;
    std::mutex writer_mtx;
    std::condition_variable cv_writer;
    bool gpu_finished = false;

    int total_puzzles_processed = 0;

    // thread da GPU --> pega um lote, avalia em todos os modelos e passa para o writer
    auto gpu_worker = [&]()
    {
        while (true)
        {
            BatchJob job;
            {
                std::unique_lock<std::mutex> lock(gpu_mtx);
                cv_gpu_consumer.wait(lock, [&]
                                     { return !gpu_queue.empty() || reader_finished; });
                if (gpu_queue.empty() && reader_finished)
                    break;
                job = std::move(gpu_queue.front());
                gpu_queue.pop();
            }
            cv_gpu_producer.notify_one();

            if (job.batched_meta.empty())
                continue;
            int cbs = static_cast<int>(job.batched_meta.size());
            std::vector<std::vector<PuzzleResult>> batch_results(cbs, std::vector<PuzzleResult>(engines.size()));

            for (size_t e = 0; e < engines.size(); ++e)
                engines[e]->evaluateBatch(job.batched_meta, batch_results, static_cast<int>(e));
            {
                std::unique_lock<std::mutex> lock(writer_mtx);
                writer_queue.push({std::move(job.batched_meta), std::move(batch_results)});
            }
            cv_writer.notify_one();
        }
        {
            std::unique_lock<std::mutex> lock(writer_mtx);
            gpu_finished = true;
        }
        cv_writer.notify_all();
    };

    // thread do writer --> grava uma linha por puzzle no CSV de cada modelo
    auto writer_worker = [&]()
    {
        while (true)
        {
            WriterJob wjob;
            {
                std::unique_lock<std::mutex> lock(writer_mtx);
                cv_writer.wait(lock, [&]
                               { return !writer_queue.empty() || gpu_finished; });
                if (writer_queue.empty() && gpu_finished)
                    break;
                wjob = std::move(writer_queue.front());
                writer_queue.pop();
            }

            // cada modelo grava no seu próprio arquivo
            const size_t cbs = wjob.batched_meta.size();
            std::vector<size_t> eidx(engines.size());
            std::iota(eidx.begin(), eidx.end(), size_t{0});
            std::for_each(std::execution::par, eidx.begin(), eidx.end(), [&](size_t e)
                          {
                auto &os = csv_outs[e];
                for (size_t b = 0; b < cbs; ++b)
                {
                    const auto &m = wjob.batched_meta[b];
                    const auto &r = wjob.batch_results[b][e];
                    const int total_moves = static_cast<int>(m.steps.size());
                    const int solved_moves = static_cast<int>(r.solved_moves);
                    const int completed = (solved_moves == total_moves) ? 1 : 0;
                    os << m.puzzle_id << "," << m.rating << "," << total_moves
                       << "," << solved_moves << "," << completed << "\n";
                } });

            total_puzzles_processed += static_cast<int>(cbs);
            std::cerr << "\rPuzzles processados: " << total_puzzles_processed << std::flush;
        }
    };

    std::thread t_gpu(gpu_worker);
    std::thread t_writer(writer_worker);

    // bloco de linhas cruas do CSV para um parser
    struct ParseJob
    {
        std::vector<std::string> lines;
    };
    std::queue<ParseJob> parse_queue;
    std::mutex parse_mtx;
    std::condition_variable cv_parse_consumer, cv_parse_producer;
    bool lines_finished = false;
    std::atomic<int> total_skipped{0};

    // parsers --> núcleos da CPU menos 4, no mínimo 2
    const unsigned hw = std::thread::hardware_concurrency();
    const int n_parsers = static_cast<int>((std::max)(2u, hw > 4 ? hw - 4 : 2u));

    // parser --> transforma as linhas do CSV em puzzles com as decisões do resolvedor já codificadas
    auto parse_worker = [&]()
    {
        std::vector<PuzzleMeta> local;
        local.reserve(BATCH_SIZE);
        for (;;)
        {
            ParseJob pj;
            {
                std::unique_lock<std::mutex> lock(parse_mtx);
                cv_parse_consumer.wait(lock, [&]
                                       { return !parse_queue.empty() || lines_finished; });
                if (parse_queue.empty() && lines_finished)
                    break;
                pj = std::move(parse_queue.front());
                parse_queue.pop();
            }
            cv_parse_producer.notify_one();

            local.clear();
            for (const std::string &line : pj.lines)
            {
                auto tokens = split_csv_fast(line);
                if (tokens.size() < 4)
                {
                    total_skipped++;
                    continue;
                }

                std::string puzzle_id(tokens[0]);
                std::string_view fen = tokens[1];
                std::string_view moves_str = tokens[2];
                int rating = 0;
                std::from_chars(tokens[3].data(), tokens[3].data() + tokens[3].size(), rating);

                auto moves = split_spaces_fast(moves_str);
                if (moves.size() < 2)
                {
                    total_skipped++;
                    continue;
                }

                const int solution_length = static_cast<int>(moves.size()) - 1;

                // moves[0] é o lance forçador, jogado antes da primeira decisão do resolvedor
                std::string fen_str(fen);
                chess::Board board(fen_str);
                chess::Move setup = chess::uci::uciToMove(board, std::string(moves[0]));
                if (setup == chess::Move::NO_MOVE)
                {
                    total_skipped++;
                    continue;
                }
                board.makeMove(setup);

                // monta a linha com um SolverStep por lance ímpar (do resolvedor);
                // os lances pares (do adversário) são forçados e só avançam o tabuleiro
                PuzzleMeta meta;
                meta.puzzle_id = std::move(puzzle_id);
                meta.rating = rating;
                meta.solution_length = solution_length;

                bool ok = true;
                for (size_t k = 1; k < moves.size(); ++k)
                {
                    if (k % 2 == 1) // lance do resolvedor
                    {
                        SolverStep step;
                        step.input.resize(static_cast<size_t>(CHANNELS) * BOARD_SQ);
                        bool flip = false;
                        if (!encoding::encodeBoard(board, step.input.data(), flip))
                        {
                            ok = false;
                            break;
                        }
                        step.flip = flip;
                        chess::movegen::legalmoves(step.legal_moves, board);
                        step.gabarito = std::string(moves[k]);

                        chess::Move correct = chess::uci::uciToMove(board, step.gabarito);
                        if (correct == chess::Move::NO_MOVE)
                        {
                            ok = false;
                            break;
                        }

                        // calcula antes os índices da política (sem strings no caminho da GPU)
                        step.policy_idx.resize(static_cast<size_t>(step.legal_moves.size()));
                        for (int mi = 0; mi < step.legal_moves.size(); ++mi)
                        {
                            const std::string uci = chess::uci::moveToUci(step.legal_moves[mi]);
                            step.policy_idx[static_cast<size_t>(mi)] = uciToIndex(uci, flip);
                            if (step.legal_moves[mi] == correct)
                                step.gabarito_mi = mi;
                        }

                        // pela regra do Lichess, se o gabarito é mate imediato, qualquer mate vale
                        chess::Board after = board;
                        after.makeMove(correct);
                        auto [reason0, result0] = after.isGameOver();
                        if (reason0 == chess::GameResultReason::CHECKMATE)
                        {
                            step.equivalent_mate.assign(static_cast<size_t>(step.legal_moves.size()), 0);
                            for (int mi = 0; mi < step.legal_moves.size(); ++mi)
                            {
                                if (step.legal_moves[mi] == correct)
                                    continue;
                                chess::Board alt = board;
                                alt.makeMove(step.legal_moves[mi]);
                                auto [reason1, result1] = alt.isGameOver();
                                if (reason1 == chess::GameResultReason::CHECKMATE)
                                    step.equivalent_mate[static_cast<size_t>(mi)] = 1;
                            }
                        }

                        meta.steps.push_back(std::move(step));
                        board.makeMove(correct); // avança pelo gabarito
                    }
                    else // lance do adversário (forçado)
                    {
                        chess::Move opp = chess::uci::uciToMove(board, std::string(moves[k]));
                        if (opp == chess::Move::NO_MOVE)
                        {
                            ok = false;
                            break;
                        }
                        board.makeMove(opp);
                    }
                }

                if (!ok || meta.steps.empty())
                {
                    total_skipped++;
                    continue;
                }

                local.push_back(std::move(meta));
            } // fim do bloco de linhas

            if (local.empty())
                continue;
            {
                std::unique_lock<std::mutex> lock(gpu_mtx);
                cv_gpu_producer.wait(lock, [&]
                                     { return gpu_queue.size() < MAX_QUEUE_SIZE; });
                gpu_queue.push({std::move(local)});
            }
            cv_gpu_consumer.notify_one();
            local = std::vector<PuzzleMeta>();
            local.reserve(BATCH_SIZE);
        }
    };

    std::vector<std::thread> parsers;
    parsers.reserve(static_cast<size_t>(n_parsers));
    for (int p = 0; p < n_parsers; ++p)
        parsers.emplace_back(parse_worker);

    // puzzles vêm do arquivo (--input) ou do stdin, só linhas cruas
    std::ifstream input_file;
    if (!input_path.empty())
    {
        input_file.open(input_path);
        if (!input_file)
        {
            std::cerr << "[ERRO] Nao foi possivel abrir o CSV: " << input_path << std::endl;
            return 1;
        }
        std::cerr << "[INFO] Lendo puzzles de: " << input_path << std::endl;
    }
    std::istream &in = input_path.empty() ? std::cin : input_file;

    std::string line;
    size_t line_no = 0;
    ParseJob chunk;
    chunk.lines.reserve(BATCH_SIZE);
    while (std::getline(in, line))
    {
        line_no++;
        // pula linha vazia e o cabeçalho
        if (line.empty() || (line_no == 1 && line.rfind("PuzzleId", 0) == 0))
            continue;
        chunk.lines.push_back(std::move(line));
        if (chunk.lines.size() == BATCH_SIZE)
        {
            {
                std::unique_lock<std::mutex> lock(parse_mtx);
                cv_parse_producer.wait(lock, [&]
                                       { return parse_queue.size() < MAX_QUEUE_SIZE; });
                parse_queue.push(std::move(chunk));
            }
            cv_parse_consumer.notify_one();
            chunk = ParseJob{};
            chunk.lines.reserve(BATCH_SIZE);
        }
    }
    if (!chunk.lines.empty())
    {
        {
            std::unique_lock<std::mutex> lock(parse_mtx);
            parse_queue.push(std::move(chunk));
        }
        cv_parse_consumer.notify_one();
    }

    // encerramento em cascata (linhas --> parsers --> GPU --> writer)
    {
        std::unique_lock<std::mutex> lock(parse_mtx);
        lines_finished = true;
    }
    cv_parse_consumer.notify_all();
    for (auto &t : parsers)
        t.join();

    {
        std::unique_lock<std::mutex> lock(gpu_mtx);
        reader_finished = true;
    }
    cv_gpu_consumer.notify_all();

    t_gpu.join();
    t_writer.join();
    for (auto &f : csv_outs)
        f.close();

    std::cerr << "\n\n======================================================\n";
    std::cerr << "Puzzles processados:  " << total_puzzles_processed << "\n";
    std::cerr << "Puzzles ignorados:    " << total_skipped << "\n";
    std::cerr << "Modelos avaliados:    " << model_names.size() << "\n";
    std::cerr << "Metrica: resolucao do PUZZLE INTEIRO (policy greedy, regra do mate)\n";
    std::cerr << "Saida: " << model_names.size() << " CSV(s) em " << output_dir << "  (colunas: total_moves, solved_moves, completed)\n";
    std::cerr << "======================================================\n\n";

    return 0;
}
