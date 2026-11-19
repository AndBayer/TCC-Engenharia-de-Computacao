#include "inference.hpp"
#include "instrumental_profile.hpp"
#include "node.hpp"
#include "params.hpp"
#include "search.hpp"

#include <atomic>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

namespace
{
    // guardar o V (valor dado pela rede) no perfil não pode aumentar o nó da árvore (36 bytes)
    static_assert(sizeof(engine::Node) == 36, "instrumental V must not increase Node size");

    // substitui a rede neural nos testes --> mesmo logit para todos os lances (mesma chance para cada um) e valor 0
    class ConstantEvaluator : public engine::PositionEvaluator
    {
    public:
        explicit ConstantEvaluator(float logit = 0.0f) : logit_(logit) {}

        std::vector<engine::EvalOutput> evaluateBatch(std::span<const engine::PositionEvalRequest> requests) override
        {
            std::vector<engine::EvalOutput> outputs;
            outputs.reserve(requests.size());
            for (const auto &request : requests)
            {
                engine::EvalOutput output;
                output.legalLogits.assign(request.legalPolicyIndices.size(), logit_);
                outputs.push_back(std::move(output));
            }
            return outputs;
        }

    private:
        float logit_;
    };

    // lances com chances diferentes --> logit cresce com o índice do lance, somado a um deslocamento fixo
    class IndexedLogitEvaluator final : public engine::PositionEvaluator
    {
    public:
        explicit IndexedLogitEvaluator(float offset) : offset_(offset) {}

        std::vector<engine::EvalOutput> evaluateBatch(std::span<const engine::PositionEvalRequest> requests) override
        {
            std::vector<engine::EvalOutput> outputs;
            outputs.reserve(requests.size());
            for (const auto &request : requests)
            {
                engine::EvalOutput output;
                output.legalLogits.reserve(request.legalPolicyIndices.size());
                for (const uint16_t index : request.legalPolicyIndices)
                    output.legalLogits.push_back(offset_ + static_cast<float>(index) * 0.001f);
                outputs.push_back(std::move(output));
            }
            return outputs;
        }

    private:
        float offset_;
    };

    // igual ao ConstantEvaluator, mas espera um tempo antes de responder, como uma rede lenta
    class SlowConstantEvaluator final : public ConstantEvaluator
    {
    public:
        explicit SlowConstantEvaluator(std::chrono::milliseconds delay) : delay_(delay) {}

        std::vector<engine::EvalOutput> evaluateBatch(std::span<const engine::PositionEvalRequest> requests) override
        {
            std::this_thread::sleep_for(delay_);
            return ConstantEvaluator::evaluateBatch(requests);
        }

    private:
        std::chrono::milliseconds delay_;
    };

    // simula erro da rede --> responde a primeira chamada e falha nas seguintes
    class FailingEvaluator final : public ConstantEvaluator
    {
    public:
        std::vector<engine::EvalOutput> evaluateBatch(std::span<const engine::PositionEvalRequest> requests) override
        {
            if (calls_.fetch_add(1, std::memory_order_relaxed) > 0)
                throw std::runtime_error("expected inference failure");
            return ConstantEvaluator::evaluateBatch(requests);
        }

    private:
        std::atomic<int> calls_{0};
    };

    // devolve os valores da lista na ordem dos pedidos; falha se pedirem mais
    class SequencedEvaluator final : public engine::PositionEvaluator
    {
    public:
        explicit SequencedEvaluator(std::vector<float> values) : values_(std::move(values)) {}

        std::vector<engine::EvalOutput> evaluateBatch(std::span<const engine::PositionEvalRequest> requests) override
        {
            std::vector<engine::EvalOutput> outputs;
            outputs.reserve(requests.size());
            for (const auto &request : requests)
            {
                if (next_ >= values_.size())
                    throw std::runtime_error("unexpected additional inference");
                engine::EvalOutput output;
                output.legalLogits.assign(request.legalPolicyIndices.size(), 0.0f);
                output.value = values_[next_++];
                outputs.push_back(std::move(output));
            }
            return outputs;
        }

        size_t evaluations() const { return next_; }

    private:
        std::vector<float> values_;
        size_t next_ = 0;
    };

    // quase toda a chance num lance só --> logit 100 para ele e -100 para os outros; valor 0,25; conta as avaliações
    class PreferMoveEvaluator final : public engine::PositionEvaluator
    {
    public:
        explicit PreferMoveEvaluator(uint16_t preferred) : preferred_(preferred) {}

        std::vector<engine::EvalOutput> evaluateBatch(std::span<const engine::PositionEvalRequest> requests) override
        {
            std::vector<engine::EvalOutput> outputs;
            outputs.reserve(requests.size());
            for (const auto &request : requests)
            {
                engine::EvalOutput output;
                output.legalLogits.reserve(request.legalPolicyIndices.size());
                for (const uint16_t index : request.legalPolicyIndices)
                    output.legalLogits.push_back(index == preferred_ ? 100.0f : -100.0f);
                output.value = 0.25f;
                outputs.push_back(std::move(output));
                ++evaluations_;
            }
            return outputs;
        }

        size_t evaluations() const { return evaluations_; }

    private:
        uint16_t preferred_;
        size_t evaluations_ = 0;
    };

    // substitui a rede ONNX no teste do EvalService --> cada logit igual ao seu índice na saída e valor 0; conta as chamadas
    class CountingNetwork final : public engine::NetEvaluator
    {
    public:
        void evaluateBatch(const float *, int count, float *policies, float *values) override
        {
            calls.fetch_add(1, std::memory_order_relaxed);
            for (int batch = 0; batch < count; ++batch)
                for (int policy = 0; policy < encoding::POLICY_SIZE; ++policy)
                    policies[static_cast<size_t>(batch) * encoding::POLICY_SIZE + policy] = static_cast<float>(policy);
            std::fill(values, values + count, 0.0f);
        }

        std::atomic<int> calls{0};
    };

    // busca na posição inicial com o ConstantEvaluator
    engine::SearchResult runConstantSearch(int gatherBatch, long long simBudget = 64)
    {
        engine::SearchParams params;
        params.simBudget = simBudget;
        params.gatherBatchSize = gatherBatch;
        params.infoIntervalMs = 0;
        params.maxNodes = 4096;
        ConstantEvaluator evaluator;
        return engine::runSearch(chess::Board(chess::constants::STARTPOS), params, evaluator);
    }

    // iguais a menos de erro de arredondamento
    void expectNear(double actual, double expected, const char *label)
    {
        const double scale = std::max({1.0, std::abs(actual), std::abs(expected)});
        const double tolerance = 32.0 * std::numeric_limits<double>::epsilon() * scale;
        if (std::abs(actual - expected) > tolerance)
            throw std::runtime_error(std::string(label) + " differs");
    }

    // iguais com tolerância absoluta
    void expectApprox(double actual, double expected, const char *label, double tolerance = 1e-6)
    {
        if (std::abs(actual - expected) > tolerance)
            throw std::runtime_error(std::string(label) + " differs");
    }

    // 0 simulações --> só avalia a raiz, para ler os priors (a chance que a rede dá a cada lance)
    engine::SearchResult runPolicyOnlySearch(engine::PositionEvaluator &evaluator)
    {
        engine::SearchParams params;
        params.simBudget = 0;
        params.gatherBatchSize = 1;
        params.infoIntervalMs = 0;
        params.maxNodes = 4096;
        return engine::runSearch(chess::Board(chess::constants::STARTPOS), params, evaluator);
    }

    // soma dos priors da raiz
    double priorTotal(const engine::SearchResult &result)
    {
        double total = 0.0;
        for (const auto &move : result.rootMoves)
            total += move.prior;
        return total;
    }

    // priors da raiz somam 1; logits iguais ou -infinito dão prior uniforme; somar uma constante aos logits não muda os priors
    void expectNormalizedPolicy()
    {
        ConstantEvaluator uniformEvaluator;
        const engine::SearchResult uniform = runPolicyOnlySearch(uniformEvaluator);
        
        if (uniform.rootMoves.size() != 20)
            throw std::runtime_error("start position legal move count changed");
        expectApprox(priorTotal(uniform), 1.0, "uniform policy total");
        for (const auto &move : uniform.rootMoves)
            expectApprox(move.prior, 1.0 / 20.0, "uniform legal prior");

        ConstantEvaluator fallbackEvaluator(-std::numeric_limits<float>::infinity());
        const engine::SearchResult fallback = runPolicyOnlySearch(fallbackEvaluator);

        expectApprox(priorTotal(fallback), 1.0, "fallback policy total");
        for (const auto &move : fallback.rootMoves)
            expectApprox(move.prior, 1.0 / 20.0, "fallback legal prior");

        IndexedLogitEvaluator baseEvaluator(0.0f);
        IndexedLogitEvaluator shiftedEvaluator(17.0f);
        const engine::SearchResult base = runPolicyOnlySearch(baseEvaluator);
        const engine::SearchResult shifted = runPolicyOnlySearch(shiftedEvaluator);
        expectApprox(priorTotal(base), 1.0, "indexed policy total");
        expectApprox(priorTotal(shifted), 1.0, "shifted policy total");
        for (const auto &left : base.rootMoves)
        {
            const auto right = std::find_if(shifted.rootMoves.begin(), shifted.rootMoves.end(), [&](const engine::MoveStat &candidate)
                                            { return candidate.uci == left.uci; });
            if (right == shifted.rootMoves.end())
                throw std::runtime_error("shifted policy lost a legal move");
            expectApprox(left.prior, right->prior, "policy logit-shift invariance", 2e-6);
        }
    }

    // confere o perfil contra as estatísticas da raiz e a PV (sequência dos lances mais visitados), e a conversão para texto e de volta
    void expectRawProfile(const engine::SearchResult &result)
    {
        const auto &root = result.instrumentalProfile.front();
        if (root.children.size() != result.rootMoves.size())
            throw std::runtime_error("root raw profile omitted legal children");

        if (root.children.empty() || root.children.front().move != result.bestMove)
            throw std::runtime_error("first raw child is not the root PV continuation");
        if (result.instrumentalProfile.size() != result.pv.size())
            throw std::runtime_error("raw profile and PV node counts differ");
        for (size_t point = 0; point < result.instrumentalProfile.size(); ++point)
            if (result.instrumentalProfile[point].children.empty() ||
                result.instrumentalProfile[point].children.front().move != result.pv[point])
                throw std::runtime_error("first raw child is not the PV continuation");

        bool sawUnvisited = false;
        for (const auto &child : root.children)
        {
            const auto rootMove = std::find_if(result.rootMoves.begin(), result.rootMoves.end(), [&](const engine::MoveStat &move)
                                               { return move.uci == child.move; });
            if (rootMove == result.rootMoves.end() ||
                rootMove->visits != child.visits ||
                rootMove->prior != child.prior)
                throw std::runtime_error("root raw child differs from root move statistics");
            if (child.visits == 0)
            {
                sawUnvisited = true;
                if (child.q || child.v)
                    throw std::runtime_error("unvisited raw child has Q or V");
            }
            else
            {
                if (!child.q)
                    throw std::runtime_error("visited raw child has no Q");
                expectNear(*child.q, rootMove->q, "parent-perspective child Q");
            }
        }
        if (result.simulations == 1 && !sawUnvisited)
            throw std::runtime_error("low-budget raw profile omitted unvisited children");

        const std::string serialized = engine::serializeInstrumentalProfile(result.instrumentalProfile);
        const auto roundTrip = engine::parseRawInstrumentalProfile(serialized);
        if (roundTrip.size() != result.instrumentalProfile.size())
            throw std::runtime_error("raw profile round-trip changed node count");
        for (size_t index = 0; index < roundTrip.size(); ++index)
        {
            if (roundTrip[index].children.size() != result.instrumentalProfile[index].children.size())
                throw std::runtime_error("round-trip changed child count");
            for (size_t child = 0; child < roundTrip[index].children.size(); ++child)
            {
                const auto &actual = roundTrip[index].children[child];
                const auto &expected = result.instrumentalProfile[index].children[child];
                if (actual.move != expected.move ||
                    actual.visits != expected.visits ||
                    actual.prior != expected.prior ||
                    actual.q != expected.q ||
                    actual.v != expected.v)
                    throw std::runtime_error("round-trip changed raw child statistics");
            }
        }
    }

    // a busca faz todas as simulações pedidas e as visitas do perfil batem com os contadores da árvore
    void expectBudget(int gatherBatch, long long simBudget = 64)
    {
        const engine::SearchResult result = runConstantSearch(gatherBatch, simBudget);

        if (result.simulations != simBudget)
            throw std::runtime_error("search did not complete its simulation budget");
        if (result.instrumentalProfile.empty())
            throw std::runtime_error("search did not produce an instrumental profile");
        if (result.instrumentalProfile.front().nodeVisits !=
            static_cast<uint32_t>(result.simulations))
            throw std::runtime_error("root profile visits differ from completed simulations");
        if (result.instrumentalProfile.size() > 1 &&
            result.instrumentalProfile[1].nodeVisits != result.rootMoves.front().visits)
            throw std::runtime_error("non-root profile visits differ from the tree counter");
        for (const auto &point : result.instrumentalProfile)
        {
            double total = 0.0;
            for (const auto &child : point.children)
                total += child.prior;
            expectApprox(total, 1.0, "expanded node policy total", 2e-6);
        }
        expectRawProfile(result);
    }

    // o info periódico (resultado parcial durante a busca) não pode pegar um lote aplicado pela metade nem mudar o resultado final
    void expectInlineSnapshots()
    {
        engine::SearchParams params;
        params.simBudget = 512;
        params.gatherBatchSize = 32;
        params.infoIntervalMs = 1;
        params.maxNodes = 100'000;

        SlowConstantEvaluator evaluator(std::chrono::milliseconds(1));
        int callbacks = 0;
        const engine::SearchResult withCallback = engine::runSearch(chess::Board(chess::constants::STARTPOS), params, evaluator, nullptr, [&](const engine::SearchResult &snapshot)
                                                                    {
                ++callbacks;
                if (snapshot.simulations <= 0 || snapshot.simulations > params.simBudget)
                    throw std::runtime_error("snapshot has invalid simulation count");
                uint64_t rootChildrenVisits = 0;
                for (const auto &move : snapshot.rootMoves)
                    rootChildrenVisits += move.visits;
                if (rootChildrenVisits != static_cast<uint64_t>(snapshot.simulations))
                    throw std::runtime_error( "snapshot observed a partially applied batch");
                if (snapshot.instrumentalProfile.empty() || snapshot.instrumentalProfile.front().nodeVisits != static_cast<uint32_t>(snapshot.simulations))
                    throw std::runtime_error( "snapshot profile differs from completed simulations"); });

        if (withCallback.simulations != params.simBudget)
            throw std::runtime_error("callback changed the fixed node budget");
        if (withCallback.treeNodes <= 4096)
            throw std::runtime_error("snapshot test did not cross an arena block");
        if (callbacks == 0)
            throw std::runtime_error("inline snapshot callback was not invoked");

        params.infoIntervalMs = 0;
        ConstantEvaluator baselineEvaluator;
        const engine::SearchResult withoutCallback = engine::runSearch(chess::Board(chess::constants::STARTPOS), params, baselineEvaluator);
        if (withoutCallback.simulations != withCallback.simulations ||
            withoutCallback.bestMove != withCallback.bestMove ||
            withoutCallback.pv != withCallback.pv ||
            withoutCallback.treeNodes != withCallback.treeNodes ||
            withoutCallback.qRoot != withCallback.qRoot ||
            withoutCallback.rootMoves.size() != withCallback.rootMoves.size())
            throw std::runtime_error("inline snapshots changed fixed-budget search");
        for (size_t index = 0; index < withoutCallback.rootMoves.size(); ++index)
        {
            const auto &left = withoutCallback.rootMoves[index];
            const auto &right = withCallback.rootMoves[index];
            if (left.uci != right.uci ||
                left.visits != right.visits ||
                left.prior != right.prior ||
                left.q != right.q)
                throw std::runtime_error("inline snapshots changed root statistics");
        }
    }

    // stop e limite de tempo (movetime) interrompem a busca entre lotes
    void expectStopAndMovetimeResponsive()
    {
        engine::SearchParams params;
        params.simBudget = 10'000;
        params.gatherBatchSize = 8;
        params.infoIntervalMs = 1;
        params.maxNodes = 200'000;

        std::atomic<bool> stop{false};
        SlowConstantEvaluator stoppedEvaluator(std::chrono::milliseconds(2));
        int callbacks = 0;
        const engine::SearchResult stopped = engine::runSearch(chess::Board(chess::constants::STARTPOS), params, stoppedEvaluator, &stop, [&](const engine::SearchResult &)
                                                               {
                ++callbacks;
                stop.store(true, std::memory_order_relaxed); });

        if (callbacks == 0 || stopped.simulations <= 0 || stopped.simulations >= params.simBudget)
            throw std::runtime_error("stop was not honored between batches");

        params.infoIntervalMs = 0;
        params.movetimeMs = 10;
        SlowConstantEvaluator timedEvaluator(std::chrono::milliseconds(2));
        const engine::SearchResult timed = engine::runSearch(chess::Board(chess::constants::STARTPOS), params, timedEvaluator);

        if (timed.simulations >= params.simBudget || timed.elapsedMs > 1000)
            throw std::runtime_error("movetime was not honored between batches");
    }

    // coletando uma folha por vez (gather 1), duas buscas iguais dão o mesmo resultado
    void expectBatchOneDeterministic()
    {
        const engine::SearchResult first = runConstantSearch(1);
        const engine::SearchResult second = runConstantSearch(1);
        if (first.bestMove != second.bestMove ||
            first.pv != second.pv ||
            first.simulations != second.simulations ||
            first.treeNodes != second.treeNodes ||
            first.maxDepth != second.maxDepth ||
            first.avgDepth != second.avgDepth ||
            first.qRoot != second.qRoot ||
            first.rootMoves.size() != second.rootMoves.size() ||
            first.instrumentalProfile.size() != second.instrumentalProfile.size())
            throw std::runtime_error("gather batch one search is not deterministic");

        for (size_t index = 0; index < first.rootMoves.size(); ++index)
        {
            const auto &left = first.rootMoves[index];
            const auto &right = second.rootMoves[index];
            if (left.uci != right.uci || left.visits != right.visits ||
                left.prior != right.prior || left.q != right.q)
                throw std::runtime_error("gather batch one root statistics differ");
        }
        for (size_t index = 0;
             index < first.instrumentalProfile.size(); ++index)
        {
            const auto &left = first.instrumentalProfile[index];
            const auto &right = second.instrumentalProfile[index];
            if (left.nodeVisits != right.nodeVisits || left.children.size() != right.children.size())
                throw std::runtime_error("gather batch one profile differs");
            for (size_t child = 0; child < left.children.size(); ++child)
            {
                const auto &leftChild = left.children[child];
                const auto &rightChild = right.children[child];
                if (leftChild.move != rightChild.move ||
                    leftChild.visits != rightChild.visits ||
                    leftChild.prior != rightChild.prior ||
                    leftChild.q != rightChild.q ||
                    leftChild.v != rightChild.v)
                    throw std::runtime_error("gather batch one raw profile child differs");
            }
        }
    }

    // o V do perfil é o valor que a rede deu na primeira avaliação do filho, sem chamar a rede de novo
    void expectCapturedInitialValueWithoutExtraInference()
    {
        engine::SearchParams params;
        params.simBudget = 1;
        params.gatherBatchSize = 1;
        params.infoIntervalMs = 0;
        params.maxNodes = 4096;
        SequencedEvaluator evaluator({0.25f, 0.5f});
        const engine::SearchResult result = engine::runSearch(chess::Board(chess::constants::STARTPOS), params, evaluator);

        if (evaluator.evaluations() != 2)
            throw std::runtime_error("instrumentation executed an additional inference");
        const auto &continuation = result.instrumentalProfile.front().children.front();
        if (!continuation.v)
            throw std::runtime_error("neural child lost its initial V");
        expectNear(*continuation.v, -0.5, "parent-perspective initial V");
        expectNear(*continuation.q, -0.5, "parent-perspective one-visit Q");
    }

    // lance que dá mate não passa pela rede, então não tem V; Q = 1 (vitória) para quem jogou
    void expectTerminalChildHasNoInitialValue()
    {
        const std::string mateMove = "f7g7";
        chess::Board board("7k/5Q2/7K/8/8/8/8/8 w - - 0 1");
        chess::Board after = board;
        after.makeMove(chess::uci::uciToMove(after, mateMove));
        if (after.isGameOver().first != chess::GameResultReason::CHECKMATE)
            throw std::runtime_error("terminal V test move is not checkmate");

        const int policyIndex = encoding::uciToIndex(mateMove, false);
        if (policyIndex < 0)
            throw std::runtime_error("terminal V test move has no policy index");

        PreferMoveEvaluator evaluator(static_cast<uint16_t>(policyIndex));
        engine::SearchParams params;
        params.simBudget = 1;
        params.gatherBatchSize = 1;
        params.infoIntervalMs = 0;
        params.maxNodes = 4096;
        const engine::SearchResult result = engine::runSearch(board, params, evaluator);

        if (evaluator.evaluations() != 1)
            throw std::runtime_error("terminal child triggered neural inference");
        const auto &continuation = result.instrumentalProfile.front().children.front();
        if (continuation.move != mateMove || !continuation.q || continuation.v)
            throw std::runtime_error("terminal child V absence was not preserved");
        expectNear(*continuation.q, 1.0, "terminal child Q");
    }
} // namespace

int main()
{
    expectNormalizedPolicy();
    expectBudget(1, 1);
    expectBudget(1);
    expectBudget(7);
    expectBudget(32);
    expectBatchOneDeterministic();
    expectCapturedInitialValueWithoutExtraInference();
    expectTerminalChildHasNoInitialValue();
    expectInlineSnapshots();
    expectStopAndMovetimeResponsive();

    // erro da rede chega a quem chamou a busca
    engine::SearchParams params;
    params.simBudget = 64;
    params.gatherBatchSize = 8;
    params.infoIntervalMs = 0;
    params.maxNodes = 4096;
    FailingEvaluator failing;
    bool propagated = false;
    try
    {
        (void)engine::runSearch(chess::Board(chess::constants::STARTPOS), params, failing);
    }
    catch (const std::runtime_error &)
    {
        propagated = true;
    }
    if (!propagated)
        throw std::runtime_error("inference failure was not propagated");

    // posição já encerrada (afogamento) --> sem simulações e sem perfil
    ConstantEvaluator terminalEvaluator;
    const engine::SearchResult terminal = engine::runSearch(chess::Board("7k/5Q2/7K/8/8/8/8/8 b - - 0 1"), params, terminalEvaluator);

    if (terminal.simulations != 0)
        throw std::runtime_error("terminal root should not run simulations");
    if (!terminal.instrumentalProfile.empty())
        throw std::runtime_error("terminal root should not produce root metrics");

    // busca já parada antes de começar --> sem simulações e sem perfil
    std::atomic<bool> stopped{true};
    ConstantEvaluator stoppedEvaluator;
    const engine::SearchResult stoppedBeforeVisits = engine::runSearch(chess::Board(chess::constants::STARTPOS), params, stoppedEvaluator, &stopped);

    if (stoppedBeforeVisits.simulations != 0)
        throw std::runtime_error("pre-stopped search completed simulations");
    if (!stoppedBeforeVisits.instrumentalProfile.empty())
        throw std::runtime_error("root without child visits should not produce root metrics");

    // EvalService (fila que junta os pedidos num lote para a rede) com 8 clientes pedindo ao mesmo tempo
    CountingNetwork network;
    engine::EvalService service(network, 8, 10'000);
    std::vector<std::thread> clients;
    for (int index = 0; index < 8; ++index)
    {
        clients.emplace_back([&]
                             { const engine::EvalOutput output = service.evaluate( chess::Board(chess::constants::STARTPOS), std::array<uint16_t, 2>{0, 1});
            if (output.legalLogits.size() != 2)
                throw std::runtime_error("batched service returned invalid legal logits"); });
    }
    for (auto &client : clients)
        client.join();
    if (network.calls.load(std::memory_order_relaxed) < 1)
        throw std::runtime_error("batched service did not call the network");

    // pedido com 10 posições volta na mesma ordem
    std::vector<chess::Board> boards(10, chess::Board(chess::constants::STARTPOS));
    std::vector<std::array<uint16_t, 1>> policyIndices(10);
    std::vector<engine::PositionEvalRequest> requests;
    requests.reserve(10);
    for (uint16_t index = 0; index < 10; ++index)
    {
        policyIndices[index][0] = index;
        requests.push_back({&boards[index], policyIndices[index]});
    }
    const std::vector<engine::EvalOutput> ordered = service.evaluateBatch(requests);
    if (ordered.size() != requests.size())
        throw std::runtime_error("grouped service returned the wrong batch size");
    for (size_t index = 0; index < ordered.size(); ++index)
        if (ordered[index].legalLogits.size() != 1 ||
            ordered[index].legalLogits[0] != static_cast<float>(index))
            throw std::runtime_error("grouped service changed result order");

    // depois do shutdown, pedido novo é recusado
    service.shutdown();

    bool rejected = false;
    try
    {
        (void)service.evaluate(chess::Board(chess::constants::STARTPOS), std::array<uint16_t, 1>{0});
    }
    catch (const std::runtime_error &)
    {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("stopped evaluation service accepted a request");
    return 0;
}
