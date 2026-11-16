#include "inference.hpp"

#include "board_encoder.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace engine
{

    namespace
    {
        // pedido sem tabuleiro ou com índice fora da saída da rede --> exceção
        void validateRequest(const PositionEvalRequest &request)
        {
            if (!request.board)
                throw std::invalid_argument("evaluation request has no board");
            for (const uint16_t index : request.legalPolicyIndices)
                if (index >= encoding::POLICY_SIZE)
                    throw std::out_of_range("legal policy index is outside the network output");
        }

        // pega só os logits dos lances legais
        EvalOutput gatherLegalOutput(const float *policy, float value, std::span<const uint16_t> legalPolicyIndices)
        {
            EvalOutput output;
            output.legalLogits.reserve(legalPolicyIndices.size());
            for (const uint16_t index : legalPolicyIndices)
            {
                if (index >= encoding::POLICY_SIZE)
                    throw std::out_of_range("legal policy index is outside the network output");
                output.legalLogits.push_back(policy[index]);
            }
            output.value = value;
            return output;
        }
    } // namespace

    EvalOutput PositionEvaluator::evaluate(const chess::Board &board, std::span<const uint16_t> legalPolicyIndices)
    {
        const std::array<PositionEvalRequest, 1> requests = {
            PositionEvalRequest{&board, legalPolicyIndices}};
        std::vector<EvalOutput> outputs = evaluateBatch(requests);
        if (outputs.size() != 1)
            throw std::runtime_error("inference returned the wrong batch size");
        return std::move(outputs.front());
    }

    // roda a rede direto, sem fila
    std::vector<EvalOutput> SynchronousEvaluator::evaluateBatch(std::span<const PositionEvalRequest> requests)
    {
        if (requests.empty())
            return {};

        std::vector<float> inputs(requests.size() * encoding::INPUT_SIZE);
        for (size_t index = 0; index < requests.size(); ++index)
        {
            validateRequest(requests[index]);
            bool flip = false;
            if (!encoding::encodeBoard(*requests[index].board, inputs.data() + index * encoding::INPUT_SIZE, flip))
                throw std::runtime_error("could not encode board for inference");
        }

        std::vector<float> policies(requests.size() * encoding::POLICY_SIZE);
        std::vector<float> values(requests.size());
        network_.evaluateBatch(inputs.data(), static_cast<int>(requests.size()), policies.data(), values.data());

        std::vector<EvalOutput> outputs;
        outputs.reserve(requests.size());
        for (size_t index = 0; index < requests.size(); ++index)
            outputs.push_back(gatherLegalOutput(policies.data() + index * encoding::POLICY_SIZE, values[index], requests[index].legalPolicyIndices));
        return outputs;
    }

    // fila que junta pedidos de várias buscas num lote só
    class EvalService::Impl
    {
        // um pedido por posição
        struct Request
        {
            Request(std::vector<float> encoded, std::vector<uint16_t> indices) : input(std::move(encoded)), legalPolicyIndices(std::move(indices)) {}

            std::vector<float> input;
            std::vector<uint16_t> legalPolicyIndices;
            EvalOutput output;
            std::exception_ptr error;
            std::mutex mutex;
            std::condition_variable completed;
            bool done = false;
        };

    public:
        Impl(NetEvaluator &network, int batchSize, int batchTimeoutUs)
            : network_(network), batchSize_(std::max(1, batchSize)),
              timeout_(std::chrono::microseconds(std::max(0, batchTimeoutUs))),
              worker_([this]
                      { run(); })
        {
        }

        ~Impl() { shutdown(); }

        // enfileira as posições já codificadas e espera as respostas da thread do lote
        std::vector<EvalOutput> evaluateBatch(std::vector<std::vector<float>> inputs, std::vector<std::vector<uint16_t>> legalPolicyIndices)
        {
            if (inputs.size() != legalPolicyIndices.size())
                throw std::invalid_argument("evaluation batch fields have different sizes");
            if (inputs.empty())
                return {};

            std::vector<std::shared_ptr<Request>> requests;
            requests.reserve(inputs.size());
            for (size_t index = 0; index < inputs.size(); ++index)
                requests.push_back(std::make_shared<Request>(std::move(inputs[index]), std::move(legalPolicyIndices[index])));
            {
                std::lock_guard<std::mutex> lock(queueMutex_);
                if (stopping_)
                    throw std::runtime_error("evaluation service is shutting down");
                for (const auto &request : requests)
                    queue_.push_back(request);
            }
            pending_.notify_all();

            // espera as respostas
            std::vector<EvalOutput> outputs;
            outputs.reserve(requests.size());
            for (const auto &request : requests)
            {
                std::unique_lock<std::mutex> lock(request->mutex);
                request->completed.wait(lock, [&]
                                        { return request->done; });
                if (request->error)
                    std::rethrow_exception(request->error);
                outputs.push_back(std::move(request->output));
            }
            return outputs;
        }

        // recusa pedidos novos; a thread do lote termina depois de atender os pendentes
        void shutdown()
        {
            bool notify = false;
            {
                std::lock_guard<std::mutex> lock(queueMutex_);
                if (!stopping_)
                {
                    stopping_ = true;
                    notify = true;
                }
            }
            if (notify)
                pending_.notify_all();
            if (worker_.joinable())
                worker_.join();
        }

    private:
        // entrega a resposta e acorda quem espera o pedido
        void complete(const std::shared_ptr<Request> &request, EvalOutput output)
        {
            {
                std::lock_guard<std::mutex> lock(request->mutex);
                request->output = std::move(output);
                request->done = true;
            }
            request->completed.notify_one();
        }

        // entrega o erro e acorda quem espera o pedido
        void fail(const std::shared_ptr<Request> &request, std::exception_ptr error)
        {
            {
                std::lock_guard<std::mutex> lock(request->mutex);
                request->error = error;
                request->done = true;
            }
            request->completed.notify_one();
        }

        // junta pedidos até encher o lote ou passar o tempo limite e roda a rede
        void run()
        {
            std::vector<std::shared_ptr<Request>> batch;
            std::vector<float> inputs;
            std::vector<float> policies;
            std::vector<float> values;

            for (;;)
            {
                batch.clear();
                {
                    std::unique_lock<std::mutex> lock(queueMutex_);
                    pending_.wait(lock, [&]
                                  { return stopping_ || !queue_.empty(); });
                    if (stopping_ && queue_.empty())
                        break;

                    const auto deadline = std::chrono::steady_clock::now() + timeout_;
                    while (!stopping_ && static_cast<int>(queue_.size()) < batchSize_)
                    {
                        if (pending_.wait_until(lock, deadline) == std::cv_status::timeout)
                            break;
                    }

                    const int count = std::min<int>(batchSize_, queue_.size());
                    for (int index = 0; index < count; ++index)
                    {
                        batch.push_back(std::move(queue_.front()));
                        queue_.pop_front();
                    }
                }

                const int count = static_cast<int>(batch.size());
                inputs.resize(static_cast<size_t>(count) * encoding::INPUT_SIZE);
                policies.resize(static_cast<size_t>(count) * encoding::POLICY_SIZE);
                values.resize(static_cast<size_t>(count));
                for (int index = 0; index < count; ++index)
                {
                    std::copy(batch[index]->input.begin(), batch[index]->input.end(), inputs.begin() + static_cast<size_t>(index) * encoding::INPUT_SIZE);
                }

                try
                {
                    network_.evaluateBatch(inputs.data(), count, policies.data(), values.data());
                    for (int index = 0; index < count; ++index)
                    {
                        EvalOutput output = gatherLegalOutput(policies.data() + static_cast<size_t>(index) * encoding::POLICY_SIZE, values[static_cast<size_t>(index)], batch[index]->legalPolicyIndices);
                        complete(batch[index], std::move(output));
                    }
                }
                catch (...)
                {
                    // erro no lote --> todos os pedidos do lote recebem o erro
                    const std::exception_ptr error = std::current_exception();
                    for (const auto &request : batch)
                        fail(request, error);
                }
            }
        }

        NetEvaluator &network_;
        int batchSize_;
        std::chrono::microseconds timeout_;
        std::mutex queueMutex_;
        std::condition_variable pending_;
        std::deque<std::shared_ptr<Request>> queue_;
        bool stopping_ = false;
        std::thread worker_;
    };

    EvalService::EvalService(NetEvaluator &network, int batchSize, int batchTimeoutUs) : impl_(std::make_unique<Impl>(network, batchSize, batchTimeoutUs))
    {
    }

    EvalService::~EvalService() = default;

    // codifica na thread de quem pediu e manda para a fila
    std::vector<EvalOutput> EvalService::evaluateBatch(std::span<const PositionEvalRequest> requests)
    {
        std::vector<std::vector<float>> inputs;
        std::vector<std::vector<uint16_t>> legalPolicyIndices;
        inputs.reserve(requests.size());
        legalPolicyIndices.reserve(requests.size());
        for (const PositionEvalRequest &request : requests)
        {
            validateRequest(request);
            inputs.emplace_back(encoding::INPUT_SIZE);
            bool flip = false;
            if (!encoding::encodeBoard(*request.board, inputs.back().data(), flip))
                throw std::runtime_error("could not encode board for inference");
            legalPolicyIndices.emplace_back(request.legalPolicyIndices.begin(), request.legalPolicyIndices.end());
        }
        return impl_->evaluateBatch(std::move(inputs), std::move(legalPolicyIndices));
    }

    void EvalService::shutdown()
    {
        impl_->shutdown();
    }

} // namespace engine
