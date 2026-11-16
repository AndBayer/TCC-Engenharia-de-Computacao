#include "search.hpp"

#include "encoding.hpp"
#include "node.hpp"
#include "search_metrics.hpp"
#include "tree.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <optional>

namespace engine
{
    namespace
    {
        // um passo do caminho da descida, da raiz até a folha
        struct PathEntry
        {
            int32_t nodeIdx;
        };

        // lances legais da folha e o índice de cada um na saída de política
        struct PreparedLeaf
        {
            chess::Movelist legalMoves;
            std::vector<uint16_t> policyIndices;
        };

        // folha coletada que espera a rede --> tabuleiro, caminho e nó
        struct GatheredLeaf
        {
            chess::Board board;
            std::vector<PathEntry> path;
            int32_t nodeIdx;
            PreparedLeaf prepared;
        };

        // resultado de uma descida --> folha para a rede, terminal já contado ou colisão
        enum class GatherStatus
        {
            PendingNetwork,
            CompletedTerminal,
            Collision
        };

        // valor exato se o jogo acabou; nullopt se não acabou
        std::optional<float> terminalValue(const chess::Board &board)
        {
            // mate = -1 para quem joga; empate = 0
            auto [reason, result] = board.isGameOver();
            if (reason == chess::GameResultReason::NONE)
                return std::nullopt;
            if (reason == chess::GameResultReason::CHECKMATE)
                return -1.0f;
            return 0.0f;
        }

        // gera os lances legais da folha antes de mandar a posição para a rede
        PreparedLeaf prepareLeaf(const chess::Board &board)
        {
            PreparedLeaf prepared;
            chess::movegen::legalmoves(prepared.legalMoves, board);
            prepared.policyIndices.reserve(prepared.legalMoves.size());

            // acha o índice de cada lance legal na saída de política
            const bool flip = board.sideToMove() == chess::Color::BLACK;
            for (int index = 0; index < prepared.legalMoves.size(); ++index)
            {
                const chess::Move move = prepared.legalMoves[index];
                int policyIndex = -1;
                if (move.typeOf() == chess::Move::CASTLING)
                {
                    // no roque a rede aprendeu e1h1 (dataset), não e1g1
                    // o Move já guarda e1h1, então usa as casas dele
                    int from = move.from().index();
                    int to = move.to().index();
                    if (flip)
                    {
                        from ^= 56;
                        to ^= 56;
                    }
                    policyIndex = to * 64 + from;
                }
                else
                {
                    policyIndex = encoding::uciToIndex(chess::uci::moveToUci(move), flip);
                }

                if (policyIndex < 0 || policyIndex >= encoding::POLICY_SIZE)
                    throw std::runtime_error("legal move could not be mapped to the policy");
                prepared.policyIndices.push_back(static_cast<uint16_t>(policyIndex));
            }
            return prepared;
        }

        // expande o nó --> cria um filho por lance legal, com o prior vindo da política da rede; retorna false se a árvore encheu
        // a raiz chega Unexpanded, antes da busca; as folhas do lote chegam em Expanding
        bool expandNode(Tree &tree, int32_t nodeIdx, const chess::Board &board, const PreparedLeaf &prepared, const std::vector<float> &legalPolicy, float value)
        {
            const chess::Movelist &legal = prepared.legalMoves;

            Node &node = tree.at(nodeIdx);
            if (legal.empty())
            {
                // não deveria acontecer (terminais são tratados antes), mas por segurança
                node.stateValue = terminalValue(board).value_or(0.0f);
                node.state = NodeState::Terminal;
                return true;
            }

            if (legalPolicy.size() != static_cast<size_t>(legal.size()))
                throw std::runtime_error("inference returned the wrong number of legal logits");

            // softmax só nos lances legais --> logits viram chances que somam 1 (subtrai o máximo para o exp não estourar)
            std::array<float, chess::constants::MAX_MOVES> logits{};
            float maxLogit = -1e30f;
            for (int i = 0; i < legal.size(); ++i)
            {
                // os logits já vêm na ordem dos lances legais
                const float lg = legalPolicy[static_cast<size_t>(i)];
                logits[static_cast<size_t>(i)] = lg;
                maxLogit = std::max(maxLogit, lg);
            }
            float z = 0.0f;
            for (int i = 0; i < legal.size(); ++i)
            {
                float &lg = logits[static_cast<size_t>(i)];
                lg = std::exp(lg - maxLogit);
                z += lg;
            }
            if (!(z > 0.0f) || !std::isfinite(z))
            { // se falhar, usa prior uniforme
                std::fill_n(logits.begin(), legal.size(), 1.0f);
                z = static_cast<float>(legal.size());
            }

            int32_t base = tree.allocChildren(legal.size());
            if (base < 0)
                return false; // árvore cheia encerra a busca

            for (int i = 0; i < legal.size(); ++i)
            {
                Node &child = tree.at(base + i);
                child.move = legal[i];
                child.P = logits[static_cast<size_t>(i)] / z;
            }
            node.firstChild = base;
            node.numChildren = legal.size();
            node.stateValue = value;
            node.state = NodeState::Expanded;
            return true;
        }

        // o PUCT escolhe o filho com maior q + c_puct * P * sqrt(N_pai) / (1 + N) --> valor médio mais um bônus para lance com prior alto e poucas visitas
        // q = -(W + VL) / (N + VL), na visão do pai (daí o sinal) e contando o virtual loss
        int32_t selectChild(Tree &tree, const Node &parent, float cpuct)
        {
            const float parentN = static_cast<float>(parent.N) + parent.VL;
            const float sqrtParent = std::sqrt(std::max(1.0f, parentN));

            int32_t best = -1;
            float bestScore = -1e30f;
            for (int i = 0; i < parent.numChildren; ++i)
            {
                const int32_t ci = parent.firstChild + i;
                const Node &child = tree.at(ci);
                const float n = static_cast<float>(child.N);
                const float vl = child.VL;
                const float nEff = n + vl;
                // filho nunca visitado tem q = 0 (neutro)
                const float q = (nEff > 0.0f) ? -(child.W + vl) / nEff : 0.0f;
                const float u = cpuct * child.P * sqrtParent / (1.0f + nEff);
                const float score = q + u;
                if (score > bestScore)
                {
                    bestScore = score;
                    best = ci;
                }
            }
            return best;
        }

        // retropropagação --> soma o valor da folha em cada nó do caminho, trocando o sinal a cada nível (a vez alterna), e tira o virtual loss
        // vLeaf está na visão de quem joga no último nó do caminho
        void backpropagate(Tree &tree, const std::vector<PathEntry> &path, float vLeaf, float virtualLoss, bool usedVL)
        {
            float v = vLeaf;
            for (auto it = path.rbegin(); it != path.rend(); ++it)
            {
                Node &node = tree.at(it->nodeIdx);
                ++node.N;
                node.W += v;
                if (usedVL)
                    node.VL -= virtualLoss;
                v = -v;
            }
        }

        // tira o virtual loss do caminho de uma descida que não vai ser avaliada
        void revertVirtualLoss(Tree &tree, const std::vector<PathEntry> &path, float virtualLoss)
        {
            for (const auto &e : path)
                tree.at(e.nodeIdx).VL -= virtualLoss;
        }

        // estado da busca
        struct SharedState
        {
            Tree *tree = nullptr;
            const chess::Board *rootBoard = nullptr;
            const SearchParams *params = nullptr;
            const std::atomic<bool> *externalStop = nullptr; // comando stop do UCI

            bool stop = false; // parada interna, quando a árvore enche
            long long simulations = 0;
            long long collisions = 0;
            int maxDepth = 0;
            long long depthSum = 0; // soma das profundidades, para a média
            bool treeFull = false;
            std::chrono::steady_clock::time_point deadline{}; // fim do movetime
            bool hasDeadline = false;

            std::chrono::steady_clock::time_point t0; // início da busca
        };

        // resultado da busca até o momento (para o info periódico e para o resultado final)
        // só o coordenador mexe na árvore, então ela não muda durante a leitura
        SearchResult buildSnapshot(Tree &tree, const SharedState &st, const SearchParams &params, std::chrono::steady_clock::time_point t0)
        {
            SearchResult res;
            const auto now = std::chrono::steady_clock::now();
            res.elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count();
            res.simulations = st.simulations;
            res.collisions = st.collisions;
            res.maxDepth = st.maxDepth;
            res.avgDepth = (res.simulations > 0) ? static_cast<int>(st.depthSum / res.simulations) : 0;
            res.treeNodes = tree.size();
            res.treeFull = st.treeFull;

            populateSearchMetrics(tree, params, res);
            return res;
        }

        // para por árvore cheia, comando stop, orçamento ou movetime
        bool shouldStop(const SharedState &st)
        {
            if (st.stop)
                return true;
            if (st.externalStop && st.externalStop->load(std::memory_order_relaxed))
                return true;
            if (st.simulations >= st.params->simBudget)
                return true;
            if (st.hasDeadline && std::chrono::steady_clock::now() >= st.deadline)
                return true;
            return false;
        }

        // profundidade da descida --> soma para a média (depth) e máximo (seldepth)
        void recordDepth(SharedState &st, const std::vector<PathEntry> &path)
        {
            const int depth = static_cast<int>(path.size()) - 1;
            st.depthSum += depth;
            st.maxDepth = std::max(st.maxDepth, depth);
        }

        // desce da raiz até uma folha (nó sem filhos); fim de jogo já retropropaga, folha nova vai para o lote
        GatherStatus gatherOneLeaf(SharedState &st, std::vector<GatheredLeaf> &pending)
        {
            Tree &tree = *st.tree;
            const SearchParams &params = *st.params;
            const float virtualLoss = params.virtualLoss;

            chess::Board board = *st.rootBoard;
            std::vector<PathEntry> path;
            path.reserve(64);
            path.push_back({0});
            // virtual loss em cada nó do caminho, para a próxima descida do lote evitar o mesmo ramo
            tree.root().VL += virtualLoss;

            int32_t current = 0;
            while (true)
            {
                Node &node = tree.at(current);
                const NodeState state = node.state;

                // fim de jogo --> retropropaga o valor exato, sem a rede
                if (state == NodeState::Terminal)
                {
                    backpropagate(tree, path, node.stateValue, virtualLoss, true);
                    ++st.simulations;
                    recordDepth(st, path);
                    return GatherStatus::CompletedTerminal;
                }

                // nó com filhos --> desce pelo filho escolhido no PUCT
                if (state == NodeState::Expanded)
                {
                    const int32_t next = selectChild(tree, node, params.cpuct);
                    // nenhum filho escolhido (não deveria acontecer) --> conta como empate
                    if (next < 0)
                    {
                        backpropagate(tree, path, 0.0f, virtualLoss, true);
                        ++st.simulations;
                        recordDepth(st, path);
                        return GatherStatus::CompletedTerminal;
                    }
                    board.makeMove(tree.at(next).move);
                    path.push_back({next});
                    tree.at(next).VL += virtualLoss;
                    current = next;
                    continue;
                }

                // nó já esperando a rede no lote atual --> colisão, desfaz o virtual loss
                if (node.state != NodeState::Unexpanded)
                {
                    revertVirtualLoss(tree, path, virtualLoss);
                    ++st.collisions;
                    return GatherStatus::Collision;
                }
                // folha nova --> marca como esperando a rede
                node.state = NodeState::Expanding;

                // folha que é fim de jogo vira Terminal e retropropaga o valor exato
                if (const auto value = terminalValue(board))
                {
                    node.stateValue = *value;
                    node.state = NodeState::Terminal;
                    backpropagate(tree, path, node.stateValue, virtualLoss, true);
                    ++st.simulations;
                    recordDepth(st, path);
                    return GatherStatus::CompletedTerminal;
                }

                // folha normal vai para o lote; se der erro, desfaz a marcação e o virtual loss
                try
                {
                    PreparedLeaf prepared = prepareLeaf(board);
                    pending.push_back(GatheredLeaf{board, path, current, std::move(prepared)});
                }
                catch (...)
                {
                    node.state = NodeState::Unexpanded;
                    revertVirtualLoss(tree, path, virtualLoss);
                    throw;
                }
                return GatherStatus::PendingNetwork;
            }
        }

        // desfaz a coleta das folhas (erro ou árvore cheia)
        void rollbackGatheredLeaves(SharedState &st, std::vector<GatheredLeaf> &pending, size_t first = 0)
        {
            Tree &tree = *st.tree;
            for (size_t index = first; index < pending.size(); ++index)
            {
                tree.at(pending[index].nodeIdx).state = NodeState::Unexpanded;
                revertVirtualLoss(tree, pending[index].path, st.params->virtualLoss);
            }
        }

        // avalia o lote na rede, expande e retropropaga
        void evaluateGatheredLeaves(SharedState &st, PositionEvaluator &evaluator, std::vector<GatheredLeaf> &pending)
        {
            std::vector<PositionEvalRequest> requests;
            requests.reserve(pending.size());
            for (const GatheredLeaf &leaf : pending)
                requests.push_back(PositionEvalRequest{&leaf.board, leaf.prepared.policyIndices});

            std::vector<EvalOutput> outputs;
            try
            {
                outputs = evaluator.evaluateBatch(requests);
            }
            catch (...)
            {
                rollbackGatheredLeaves(st, pending);
                throw;
            }

            if (outputs.size() != pending.size())
            {
                rollbackGatheredLeaves(st, pending);
                throw std::runtime_error("inference returned the wrong gathered batch size");
            }

            Tree &tree = *st.tree;
            // expande cada folha com a resposta da rede e retropropaga o valor
            for (size_t index = 0; index < pending.size(); ++index)
            {
                bool expanded = false;
                try
                {
                    expanded = expandNode(tree, pending[index].nodeIdx, pending[index].board, pending[index].prepared, outputs[index].legalLogits, outputs[index].value);
                }
                catch (...)
                {
                    rollbackGatheredLeaves(st, pending, index);
                    throw;
                }

                // árvore cheia --> desfaz a folha atual e as seguintes e para a busca
                if (!expanded)
                {
                    rollbackGatheredLeaves(st, pending, index);
                    st.treeFull = true;
                    st.stop = true;
                    return;
                }

                backpropagate(tree, pending[index].path, outputs[index].value, st.params->virtualLoss, true);
                ++st.simulations;
                recordDepth(st, pending[index].path);
            }
        }

        // laço principal, que coleta um lote de folhas e avalia
        void runSearchLoop(SharedState &st, PositionEvaluator &evaluator, const InfoCallback &onInfo)
        {
            const int gatherSize = std::max(1, st.params->gatherBatchSize);
            // limite de tentativas por rodada, para não travar em colisões
            const int attemptLimit = std::max(8, 4 * gatherSize);
            const auto infoInterval = std::chrono::milliseconds(st.params->infoIntervalMs);
            auto nextInfo = std::chrono::steady_clock::now() + infoInterval;
            std::vector<GatheredLeaf> pending;
            pending.reserve(static_cast<size_t>(gatherSize));

            while (!shouldStop(st))
            {
                pending.clear();
                const long long simulationsAtStart = st.simulations;
                int attempts = 0;
                // coleta folhas até encher o lote, sem passar do orçamento e parando no stop ou no movetime
                while (static_cast<int>(pending.size()) < gatherSize && attempts < attemptLimit)
                {
                    const long long completed = st.simulations;
                    if (completed + static_cast<long long>(pending.size()) >=
                        st.params->simBudget)
                        break;
                    if (st.stop ||
                        (st.externalStop &&
                         st.externalStop->load(std::memory_order_relaxed)) ||
                        (st.hasDeadline &&
                         std::chrono::steady_clock::now() >= st.deadline))
                        break;

                    ++attempts;
                    (void)gatherOneLeaf(st, pending);
                }

                if (!pending.empty())
                    evaluateGatheredLeaves(st, evaluator, pending);

                // o coordenador é o único que escreve na árvore e o lote já foi expandido, retropropagado e contado, então o resultado parcial não disputa a árvore
                if (onInfo && st.params->infoIntervalMs > 0 &&
                    std::chrono::steady_clock::now() >= nextInfo)
                {
                    onInfo(buildSnapshot(*st.tree, st, *st.params, st.t0));
                    nextInfo = std::chrono::steady_clock::now() + infoInterval;
                }

                if (!pending.empty())
                    continue;

                // com lote vazio, só continua se algum terminal foi contado
                if (shouldStop(st))
                    break;
                if (st.simulations > simulationsAtStart)
                    continue;
                if (attempts >= attemptLimit)
                    throw std::runtime_error("search could not collect an available leaf");
            }
        }

    } // namespace

    SearchResult runSearch(const chess::Board &rootBoard, const SearchParams &params, PositionEvaluator &evaluator, const std::atomic<bool> *stopFlag, const InfoCallback &onInfo)
    {
        const auto t0 = std::chrono::steady_clock::now();

        Tree tree(params.maxNodes);
        SharedState st;
        st.tree = &tree;
        st.rootBoard = &rootBoard;
        st.params = &params;
        st.externalStop = stopFlag;
        st.t0 = t0;
        // movetime vira um prazo fixo a partir do início
        if (params.movetimeMs > 0)
        {
            st.deadline = t0 + std::chrono::milliseconds(params.movetimeMs);
            st.hasDeadline = true;
        }

        // expansão da raiz (síncrona, lote 1)
        {
            chess::Board board = rootBoard;
            Node &root = tree.root();
            if (const auto value = terminalValue(board))
            {
                root.stateValue = *value;
                root.state = NodeState::Terminal;
            }
            else
            {
                PreparedLeaf prepared = prepareLeaf(board);
                EvalOutput rootOut = evaluator.evaluate(board, prepared.policyIndices);
                expandNode(tree, 0, board, prepared, rootOut.legalLogits, rootOut.value);
                // a avaliação da raiz conta como a primeira visita
                ++root.N;
                root.W += rootOut.value;
            }
        }

        const bool rootTerminal = tree.root().state == NodeState::Terminal;

        // raiz terminal (mate ou empate) não tem busca
        if (!rootTerminal)
        {
            runSearchLoop(st, evaluator, onInfo);
        }

        // saída com melhor lance, avaliação e perfil instrumental
        SearchResult res = buildSnapshot(tree, st, params, t0);

        return res;
    }

} // namespace engine
