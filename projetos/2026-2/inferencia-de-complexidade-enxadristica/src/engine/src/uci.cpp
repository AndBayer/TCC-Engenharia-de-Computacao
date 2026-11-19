#include "uci.hpp"

#include "inference.hpp"
#include "instrumental_profile.hpp"
#include "node.hpp"
#include "params.hpp"
#include "search.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <atomic>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace engine
{

    namespace
    {

        // estado do motor entre comandos --> opções, modelo carregado, posição e busca em andamento
        struct UciState
        {
            SearchParams params;
            std::string loadedModel;                        // caminho do modelo já carregado
            std::unique_ptr<NetEvaluator> net;              // rede pelo ONNX Runtime, quando não há pipeline na GPU
            std::unique_ptr<CudaEvalService> cudaEvaluator; // pipeline na GPU
            int loadedCudaBatch = 0;                        // lote usado ao criar o pipeline na GPU
            std::string inferenceStatus;                    // modo de inferência mostrado no info string
            std::string inferenceWarning;                   // motivo de o pipeline na GPU não ter sido criado

            chess::Board board{chess::constants::STARTPOS};

            std::atomic<bool> stopFlag{false};
            std::thread searchThread; // a busca roda em outra thread para o loop seguir lendo comandos
            bool infiniteSearch = false; // a busca atual veio de go infinite
            std::string exeDir; // pasta do executável, onde o modelo também é procurado

            // trava a saída, pois a busca e o loop de comandos podem escrever juntos (readyok durante a busca)
            std::mutex outMutex;

            // espera a busca terminar
            void joinSearch()
            {
                if (searchThread.joinable())
                    searchThread.join();
                infiniteSearch = false;
            }

            // busca com orçamento termina sozinha e é esperada (comandos enviados por pipe seguem determinísticos)
            // go infinite só termina com stop
            void finishSearch()
            {
                if (infiniteSearch)
                    stopFlag.store(true);
                joinSearch();
            }

            // carrega a inferência quando precisa --> pipeline na GPU ou, se falhar, ONNX Runtime direto (GPU ou CPU)
            bool ensureInference()
            {
                namespace fs = std::filesystem;
                fs::path p(params.modelPath);
                if (p.is_relative())
                {
                    // procura no diretório atual e depois na pasta do executável
                    if (!fs::exists(p) && !exeDir.empty() && fs::exists(fs::path(exeDir) / p))
                        p = fs::path(exeDir) / p;
                }
                const std::string resolved = p.string();
                if (resolved == loadedModel && ((cudaEvaluator && loadedCudaBatch == params.batchSize) || net))
                    return false;

                cudaEvaluator.reset();
                net.reset();
                inferenceWarning.clear();
                std::string reason;
                const bool cuda = cudaLooksAvailable(exeDir, reason);
                if (cuda)
                {
                    try
                    {
                        cudaEvaluator = std::make_unique<CudaEvalService>(resolved, params.batchSize, params.batchTimeoutUs);
                        loadedCudaBatch = params.batchSize;
                        inferenceStatus = "cuda-resident (CUDA provider)";
                    }
                    catch (const std::exception &error)
                    {
                        inferenceWarning = error.what();
                    }
                }
                else
                    inferenceWarning = reason;
                if (!cudaEvaluator)
                {
                    net = makeOnnxEvaluator(resolved, cuda);
                    loadedCudaBatch = 0;
                    inferenceStatus = cuda ? "host-staged" : "host-staged (CPU)";
                }
                loadedModel = resolved;
                return true;
            }
        };

        // separa a linha em palavras
        std::vector<std::string> tokenize(const std::string &line)
        {
            std::vector<std::string> out;
            std::istringstream iss(line);
            std::string tok;
            while (iss >> tok)
                out.push_back(tok);
            return out;
        }

        // valida a FEN antes de montar o tabuleiro, pois a biblioteca não confere
        // e uma FEN corrompida derrubaria o processo no go
        void validateFen(const std::string &fen)
        {
            std::istringstream fields(fen);
            std::string placement, side, castling = "-", ep = "-", halfmove = "0", fullmove = "1";
            if (!(fields >> placement))
                throw std::invalid_argument("empty FEN");

            int rank = 7, file = 0;
            int kings[2] = {0, 0}, pieces[2] = {0, 0}, pawns[2] = {0, 0};
            char grid[8][8] = {};
            for (const char c : placement)
            {
                if (c == '/')
                {
                    if (file != 8 || rank == 0)
                        throw std::invalid_argument("each rank must have 8 squares");
                    --rank;
                    file = 0;
                }
                else if (c >= '1' && c <= '8')
                    file += c - '0';
                else if (std::string_view("pnbrqkPNBRQK").find(c) != std::string_view::npos)
                {
                    if (file >= 8)
                        throw std::invalid_argument("each rank must have 8 squares");
                    const int color = std::isupper(static_cast<unsigned char>(c)) ? 0 : 1;
                    const char piece = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                    ++pieces[color];
                    if (piece == 'k')
                        ++kings[color];
                    if (piece == 'p')
                    {
                        ++pawns[color];
                        if (rank == 0 || rank == 7)
                            throw std::invalid_argument("pawn on the first or last rank");
                    }
                    grid[rank][file++] = c;
                }
                else
                    throw std::invalid_argument(std::string("invalid character '") + c + "' in piece placement");
                if (file > 8)
                    throw std::invalid_argument("each rank must have 8 squares");
            }
            if (rank != 0 || file != 8)
                throw std::invalid_argument("the board must have 8 ranks of 8 squares");

            if (!(fields >> side))
                throw std::invalid_argument("missing side to move");
            fields >> castling >> ep >> halfmove >> fullmove;
            std::string extra;
            if (fields >> extra)
                throw std::invalid_argument("too many fields");
            if (kings[0] != 1 || kings[1] != 1)
                throw std::invalid_argument("each side must have exactly one king");
            if (pieces[0] > 16 || pieces[1] > 16 || pawns[0] > 8 || pawns[1] > 8)
                throw std::invalid_argument("too many pieces");

            if (side != "w" && side != "b")
                throw std::invalid_argument("side to move must be w or b");

            if (castling != "-")
            {
                const std::string order = "KQkq";
                size_t last = 0;
                for (const char c : castling)
                {
                    const size_t at = order.find(c);
                    if (at == std::string::npos || at < last) // ordem KQkq, sem repetição
                        throw std::invalid_argument("invalid castling rights");
                    last = at + 1;
                }
                const auto need = [&](char right, int r, int kingFile, int rookFile, char king, char rook)
                {
                    if (castling.find(right) != std::string::npos &&
                        (grid[r][kingFile] != king || grid[r][rookFile] != rook))
                        throw std::invalid_argument(std::string("castling right ") + right + " without king and rook on their squares");
                };
                need('K', 0, 4, 7, 'K', 'R');
                need('Q', 0, 4, 0, 'K', 'R');
                need('k', 7, 4, 7, 'k', 'r');
                need('q', 7, 4, 0, 'k', 'r');
            }

            if (ep != "-" && (ep.size() != 2 || ep[0] < 'a' || ep[0] > 'h' ||
                              ep[1] != (side == "w" ? '6' : '3')))
                throw std::invalid_argument("invalid en passant square");

            for (const std::string *counter : {&halfmove, &fullmove})
                if (counter->empty() || counter->find_first_not_of("0123456789") != std::string::npos)
                    throw std::invalid_argument("move counters must be non-negative integers");

            // o lado que acabou de jogar não pode ter o rei em xeque
            const chess::Board board(fen);
            const chess::Color waiting = ~board.sideToMove();
            if (board.isAttacked(board.kingSq(waiting), board.sideToMove()))
                throw std::invalid_argument("the side not to move is in check");
        }

        // position startpos|fen <FEN> [moves <lances>]
        void cmdPosition(UciState &st, const std::vector<std::string> &tok)
        {
            // monta a posição numa cópia; se der erro, o tabuleiro anterior fica intacto
            chess::Board board;
            size_t i = 1;
            if (i < tok.size() && tok[i] == "startpos")
            {
                board = chess::Board(chess::constants::STARTPOS);
                ++i;
            }
            else if (i < tok.size() && tok[i] == "fen")
            {
                ++i;
                std::string fen;
                while (i < tok.size() && tok[i] != "moves")
                {
                    if (!fen.empty())
                        fen += ' ';
                    fen += tok[i++];
                }
                try
                {
                    validateFen(fen);
                }
                catch (const std::invalid_argument &e)
                {
                    throw std::invalid_argument("invalid FEN \"" + fen + "\": " + e.what());
                }
                board = chess::Board(fen);
            }
            else
            {
                throw std::invalid_argument("position requires startpos or fen");
            }
            if (i < tok.size() && tok[i] == "moves")
            {
                ++i;
                for (; i < tok.size(); ++i)
                {
                    chess::Movelist legal;
                    chess::movegen::legalmoves(legal, board);
                    const chess::Move mv = chess::uci::uciToMove(board, tok[i]);
                    if (mv == chess::Move::NO_MOVE || std::find(legal.begin(), legal.end(), mv) == legal.end())
                        throw std::invalid_argument("illegal move \"" + tok[i] + "\" in position command");
                    board.makeMove(mv);
                }
            }
            st.board = board;
        }

        void cmdSetOption(UciState &st, std::ostream &out, const std::vector<std::string> &tok)
        {
            // setoption name <Nome> value <Valor...>
            std::string name, value;
            size_t i = 1;
            if (i < tok.size() && tok[i] == "name")
                ++i;
            while (i < tok.size() && tok[i] != "value")
                name += (name.empty() ? "" : " ") + tok[i++];
            if (i < tok.size() && tok[i] == "value")
                ++i;
            while (i < tok.size())
                value += (value.empty() ? "" : " ") + tok[i++];

            if (name.empty())
            {
                out << "info string error: setoption requires an option name" << std::endl;
                return;
            }

            auto &p = st.params;
            try
            {
                // nomes no padrão Lc0
                if (name == "WeightsFile")
                {
                    if (value.empty())
                        throw std::invalid_argument("empty model path");
                    p.modelPath = value;
                }
                else if (name == "MinibatchSize")
                    p.batchSize = std::clamp(std::stoi(value), 1, 1024);
                else if (name == "GatherBatchSize")
                    p.gatherBatchSize = std::clamp(std::stoi(value), 1, 1024);
                else if (name == "MultiPV")
                    p.multiPV = std::clamp(std::stoi(value), 1, 64);
                else if (name == "CPuct")
                {
                    const float cpuct = std::stof(value);
                    if (!std::isfinite(cpuct) || cpuct <= 0.0f)
                        throw std::invalid_argument("CPuct must be finite and positive");
                    p.cpuct = cpuct;
                }
                else if (name == "VirtualLoss")
                {
                    const float virtualLoss = std::stof(value);
                    if (!std::isfinite(virtualLoss) || virtualLoss < 0.0f)
                        throw std::invalid_argument("VirtualLoss must be finite and non-negative");
                    p.virtualLoss = virtualLoss;
                }
                else if (name == "RamLimitMb")
                {
                    // memória no padrão Lc0, convertida de MB em número de nós da árvore (sizeof(Node) por nó)
                    const unsigned long long mb = std::stoull(value);
                    p.maxNodes = static_cast<size_t>(mb * 1024ull * 1024ull / sizeof(Node));
                }
                else
                    out << "info string error: unknown option: " << name << std::endl;
            }
            catch (const std::exception &)
            {
                out << "info string error: invalid value for " << name << ": " << value
                    << std::endl;
            }
        }

        // opções mostradas na resposta ao comando uci
        void printOptions(std::ostream &out, const SearchParams &p)
        {
            out << "option name WeightsFile type string default " << p.modelPath << '\n'
                << "option name MinibatchSize type spin default " << p.batchSize
                << " min 1 max 1024\n"
                << "option name GatherBatchSize type spin default "
                << p.gatherBatchSize << " min 1 max 1024\n"
                << "option name MultiPV type spin default 1 min 1 max 64\n"
                << "option name CPuct type string default " << p.cpuct << "\n"
                << "option name VirtualLoss type string default " << p.virtualLoss << "\n"
                << "option name RamLimitMb type spin default "
                << (p.maxNodes * sizeof(Node) / (1024ull * 1024ull))
                << " min 64 max 65536\n";
        }

        // go [nodes N] [movetime ms] [infinite] --> roda a busca numa thread separada
        void cmdGo(UciState &st, std::ostream &out, const std::vector<std::string> &tok)
        {
            SearchParams prm = st.params;
            prm.movetimeMs = 0;
            bool infinite = false;
            for (size_t i = 1; i + 1 <= tok.size(); ++i)
            {
                if (tok[i] == "nodes" && i + 1 < tok.size())
                {
                    prm.simBudget = std::stoll(tok[++i]);
                    if (prm.simBudget <= 0)
                        throw std::invalid_argument("nodes must be greater than zero");
                }
                else if (tok[i] == "movetime" && i + 1 < tok.size())
                {
                    prm.movetimeMs = std::stoll(tok[++i]);
                    if (prm.movetimeMs <= 0)
                        throw std::invalid_argument("movetime must be greater than zero");
                }
                else if (tok[i] == "infinite")
                    infinite = true;
            }
            // infinite --> sem limite de nós
            if (infinite)
                prm.simBudget = std::numeric_limits<long long>::max();
            if (prm.movetimeMs > 0 && !infinite && prm.simBudget == st.params.simBudget)
                prm.simBudget = std::numeric_limits<long long>::max(); // movetime sem nodes --> só o tempo limita a busca

            try
            {
                const bool initialized = st.ensureInference();
                if (initialized)
                {
                    if (!st.inferenceWarning.empty())
                        out << "info string warning: CUDA-resident inference unavailable: " << st.inferenceWarning << std::endl;
                    out << "info string inference pipeline " << st.inferenceStatus << std::endl;
                }
            }
            catch (const std::exception &e)
            {
                out << "info string error: failed to load model: " << e.what() << std::endl;
                out << "bestmove 0000" << std::endl;
                return;
            }

            st.stopFlag.store(false);
            st.infiniteSearch = infinite;
            chess::Board board = st.board;
            NetEvaluator *net = st.net.get();
            PositionEvaluator *cudaEvaluator = st.cudaEvaluator.get();
            std::atomic<bool> *stop = &st.stopFlag;

            // tabuleiro e opções vão por cópia, pois cmdGo retorna antes de a busca terminar
            st.searchThread = std::thread([&st, &out, board, prm, net, cudaEvaluator, stop]
                                          {
        try
        {
        // linha info periódica durante a busca e final antes do bestmove
        auto printInfo = [&out, &prm, &st](const SearchResult &r) {
            std::lock_guard<std::mutex> lock(st.outMutex);
            const long long ms = std::max<long long>(1, r.elapsedMs);
            const long long nps = r.simulations * 1000 / ms;
            // linhas info só com tokens UCI padrão, pois GUIs como En Croissant e Arena quebram com
            // tokens desconhecidos antes do pv; métricas próprias vão num info string separado
            const int kPV = std::min<int>(prm.multiPV, static_cast<int>(r.rootMoves.size()));
            if (kPV > 1)
            {
                for (int i = 0; i < kPV; ++i)
                {
                    const MoveStat &m = r.rootMoves[static_cast<size_t>(i)];
                    if (m.pv.empty())
                        break; // candidato sem visitas não gera linha
                    out << "info depth " << r.avgDepth << " seldepth " << r.maxDepth
                        << " multipv " << (i + 1)
                        << " nodes " << r.simulations << " nps " << nps
                        << " time " << r.elapsedMs << " score cp "
                        << static_cast<long long>(m.evalCp) << " pv";
                    for (const auto &mv : m.pv)
                        out << ' ' << mv;
                    out << std::endl;
                }
            }
            else
            {
                out << "info depth " << r.avgDepth << " seldepth " << r.maxDepth
                    << " nodes " << r.simulations
                    << " nps " << nps << " time " << r.elapsedMs << " score cp "
                    << static_cast<long long>(r.evalCp);
                if (!r.pv.empty())
                {
                    out << " pv";
                    for (const auto &m : r.pv)
                        out << ' ' << m;
                }
                out << std::endl;
            }
        };

        // sem o pipeline na GPU, a rede roda na própria thread da busca
        std::unique_ptr<SynchronousEvaluator> hostEvaluator;
        PositionEvaluator *evaluator = cudaEvaluator;
        if (!evaluator)
        {
            hostEvaluator = std::make_unique<SynchronousEvaluator>(*net);
            evaluator = hostEvaluator.get();
        }
        SearchResult r = runSearch(board, prm, *evaluator, stop, printInfo);

        printInfo(r);
        std::lock_guard<std::mutex> lock(st.outMutex);
        // perfil instrumental bruto ao longo da PV
        out << "info string instrumental_profile " << serializeInstrumentalProfile(r.instrumentalProfile) << std::endl;
        // padrão UCI é bestmove <lance> ponder <resposta esperada>
        // o ponder é o segundo lance da PV, a resposta prevista do adversário
        out << "bestmove " << (r.bestMove.empty() ? "0000" : r.bestMove);
        if (r.pv.size() >= 2)
            out << " ponder " << r.pv[1];
        out << std::endl;
        }
        catch (const std::exception &e)
        {
            std::lock_guard<std::mutex> lock(st.outMutex);
            out << "info string error: search failed: " << e.what() << std::endl;
            out << "bestmove 0000" << std::endl;
        }
        catch (...)
        {
            std::lock_guard<std::mutex> lock(st.outMutex);
            out << "info string error: search failed: unknown error" << std::endl;
            out << "bestmove 0000" << std::endl;
        } });
        }

    } // namespace

    int uciLoop(std::istream &in, std::ostream &out, const std::string &exeDir)
    {
        UciState st;
        st.exeDir = exeDir;

        // um comando por linha
        std::string line;
        while (std::getline(in, line))
        {
            auto tok = tokenize(line);
            if (tok.empty())
                continue;
            const std::string &cmd = tok[0];

            try
            {
                if (cmd == "uci")
                {
                    std::lock_guard<std::mutex> lock(st.outMutex);
                    out << "id name tfird\n" << "id author Lucas Franco de Mello\n";
                    printOptions(out, st.params);
                    out << "uciok" << std::endl;
                }
                else if (cmd == "isready")
                {
                    // durante a busca o modelo já está carregado, então responde na hora
                    if (st.searchThread.joinable())
                    {
                        std::lock_guard<std::mutex> lock(st.outMutex);
                        out << "readyok" << std::endl;
                        continue;
                    }
                    try
                    {
                        const bool initialized = st.ensureInference();
                        if (initialized)
                        {
                            if (!st.inferenceWarning.empty())
                                out << "info string warning: CUDA-resident inference unavailable: " << st.inferenceWarning << std::endl;
                            out << "info string inference pipeline " << st.inferenceStatus << std::endl;
                        }
                        out << "readyok" << std::endl;
                    }
                    catch (const std::exception &e)
                    {
                        out << "info string error: failed to load model: " << e.what() << std::endl;
                        out << "readyok" << std::endl;
                    }
                }
                else if (cmd == "ucinewgame")
                {
                    // ucinewgame, position, setoption e go esperam a busca atual (go infinite é parado)
                    st.finishSearch();
                    st.board = chess::Board(chess::constants::STARTPOS);
                }
                else if (cmd == "position")
                {
                    st.finishSearch();
                    cmdPosition(st, tok);
                }
                else if (cmd == "setoption")
                {
                    st.finishSearch();
                    cmdSetOption(st, out, tok);
                }
                else if (cmd == "go")
                {
                    st.finishSearch();
                    cmdGo(st, out, tok);
                }
                else if (cmd == "stop")
                {
                    st.stopFlag.store(true);
                    st.joinSearch();
                }
                else if (cmd == "quit")
                {
                    // busca com orçamento é concluída (determinística com stdin por pipe);
                    // busca infinita é interrompida, pois só terminaria com stop
                    st.finishSearch();
                    break;
                }
                else
                {
                    std::lock_guard<std::mutex> lock(st.outMutex);
                    out << "info string error: unknown command: " << cmd << std::endl;
                }
            }
            catch (const std::exception &e)
            {
                std::lock_guard<std::mutex> lock(st.outMutex);
                out << "info string error: " << e.what() << std::endl;
                if (cmd == "go")
                    out << "bestmove 0000" << std::endl;
            }
            catch (...)
            {
                std::lock_guard<std::mutex> lock(st.outMutex);
                out << "info string error: command failed: unknown error" << std::endl;
                if (cmd == "go")
                    out << "bestmove 0000" << std::endl;
            }
        }
        // fim da entrada --> para a busca que ainda estiver rodando
        st.stopFlag.store(true);
        st.joinSearch();
        return 0;
    }

} // namespace engine
