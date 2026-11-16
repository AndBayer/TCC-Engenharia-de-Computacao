#include "inference.hpp"

#include "compact_position.hpp"
#include "inference_cuda_kernels.hpp"

#include <onnxruntime_cxx_api.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace engine
{
    namespace
    {
        // vagas por posição para os lances legais (MAX_MOVES da biblioteca de xadrez)
        constexpr int kLegalSlots = chess::constants::MAX_MOVES;

        // erro do CUDA --> exceção com o nome da operação
        void checkCuda(cudaError_t status, const char *operation)
        {
            if (status != cudaSuccess)
                throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
        }

        // o lançamento do kernel não retorna erro, então lê o último erro do CUDA
        void checkKernel(const char *operation)
        {
            checkCuda(cudaGetLastError(), operation);
        }

        // stream do CUDA (fila de tarefas da GPU) onde as cópias, os kernels e a rede rodam em ordem
        class CudaStream
        {
        public:
            CudaStream()
            {
                checkCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
            }

            ~CudaStream()
            {
                if (stream_)
                    cudaStreamDestroy(stream_);
            }

            CudaStream(const CudaStream &) = delete;
            CudaStream &operator=(const CudaStream &) = delete;

            cudaStream_t get() const { return stream_; }

        private:
            cudaStream_t stream_ = nullptr;
        };

        // vetor na memória da GPU, liberado no destrutor
        template <typename T>
        class DeviceBuffer
        {
        public:
            DeviceBuffer() = default;
            explicit DeviceBuffer(size_t count) { allocate(count); }
            ~DeviceBuffer()
            {
                if (data_)
                    cudaFree(data_);
            }

            DeviceBuffer(const DeviceBuffer &) = delete;
            DeviceBuffer &operator=(const DeviceBuffer &) = delete;

            void allocate(size_t count)
            {
                if (data_)
                    throw std::logic_error("CUDA buffer was allocated twice");
                checkCuda(cudaMalloc(reinterpret_cast<void **>(&data_), count * sizeof(T)), "cudaMalloc");
                count_ = count;
            }

            T *data() { return data_; }
            const T *data() const { return data_; }
            size_t size() const { return count_; }

        private:
            T *data_ = nullptr;
            size_t count_ = 0;
        };

        // vetor na RAM travada (pinned), que permite cópia assíncrona com a GPU
        template <typename T>
        class PinnedBuffer
        {
        public:
            PinnedBuffer() = default;
            explicit PinnedBuffer(size_t count) { allocate(count); }
            ~PinnedBuffer()
            {
                if (data_)
                    cudaFreeHost(data_);
            }

            PinnedBuffer(const PinnedBuffer &) = delete;
            PinnedBuffer &operator=(const PinnedBuffer &) = delete;

            void allocate(size_t count)
            {
                if (data_)
                    throw std::logic_error("pinned buffer was allocated twice");
                checkCuda(cudaHostAlloc(reinterpret_cast<void **>(&data_), count * sizeof(T), cudaHostAllocDefault), "cudaHostAlloc");
                count_ = count;
            }

            T *data() { return data_; }
            const T *data() const { return data_; }
            size_t size() const { return count_; }

        private:
            T *data_ = nullptr;
            size_t count_ = 0;
        };
    } // namespace

    // fila de pedidos e uma thread que monta os lotes; a posição vai compacta e é expandida na GPU
    class CudaEvalService::Impl
    {
        // pedido na fila --> posição, índices legais e a resposta, preenchida pela thread do lote
        struct Request
        {
            Request(encoding::CompactPosition compact, std::vector<uint16_t> indices) : position(compact), legalPolicyIndices(std::move(indices)) {}

            encoding::CompactPosition position;
            std::vector<uint16_t> legalPolicyIndices;
            EvalOutput output;
            std::exception_ptr error;
            std::mutex mutex;
            std::condition_variable completed;
            bool done = false;
        };

    public:
        Impl(const std::string &modelPath, int batchSize, int batchTimeoutUs)
            : stream_(), env_(ORT_LOGGING_LEVEL_WARNING, "engine-cuda"),
              batchSize_(std::max(1, batchSize)),
              timeout_(std::chrono::microseconds(std::max(0, batchTimeoutUs))),
              hostPositions_(static_cast<size_t>(batchSize_)),
              hostIndices_(static_cast<size_t>(batchSize_) * kLegalSlots),
              hostLegalLogits_(static_cast<size_t>(batchSize_) * kLegalSlots),
              hostValues_(static_cast<size_t>(batchSize_)),
              devicePositions_(static_cast<size_t>(batchSize_)),
              deviceIndices_(static_cast<size_t>(batchSize_) * kLegalSlots),
              deviceInputs_(static_cast<size_t>(batchSize_) * encoding::INPUT_SIZE),
              devicePolicies_(static_cast<size_t>(batchSize_) * encoding::POLICY_SIZE),
              deviceLegalLogits_(static_cast<size_t>(batchSize_) * kLegalSlots),
              deviceValues_(static_cast<size_t>(batchSize_))
        {
            Ort::SessionOptions options;
            options.SetIntraOpNumThreads(1);
            options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            // o ONNX Runtime roda a rede no mesmo stream das cópias e dos kernels
            OrtCUDAProviderOptions cudaOptions{};
            cudaOptions.device_id = 0;
            cudaOptions.has_user_compute_stream = 1;
            cudaOptions.user_compute_stream = stream_.get();
            options.AppendExecutionProvider_CUDA(cudaOptions);

            // no Windows o ONNX Runtime recebe o caminho em wchar_t
#ifdef _WIN32
            session_ = Ort::Session(env_, std::wstring(modelPath.begin(), modelPath.end()).c_str(), options);
#else
            session_ = Ort::Session(env_, modelPath.c_str(), options);
#endif
            // entrada e saídas da rede ligadas direto aos buffers da GPU, sem passar pela CPU
            cudaMemory_ = Ort::MemoryInfo("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
            binding_ = Ort::IoBinding(session_);

            // testa o caminho completo antes de liberar o serviço
            hostPositions_.data()[0] = encoding::compactPosition(chess::Board(chess::constants::STARTPOS));

            std::fill_n(hostIndices_.data(), kLegalSlots, uint16_t{0});
            execute(1);

            // thread que monta os lotes e roda a rede
            worker_ = std::thread([this]
                                  { run(); });
        }

        ~Impl() { shutdown(); }

        // enfileira as posições e espera as respostas da thread do lote
        std::vector<EvalOutput> evaluateBatch(std::span<const PositionEvalRequest> inputs)
        {
            if (inputs.empty())
                return {};

            // confere os pedidos e compacta as posições antes de entrar na fila
            std::vector<std::shared_ptr<Request>> requests;
            requests.reserve(inputs.size());
            for (const PositionEvalRequest &input : inputs)
            {
                if (!input.board)
                    throw std::invalid_argument("evaluation request has no board");
                if (input.legalPolicyIndices.size() > kLegalSlots)
                    throw std::invalid_argument("position has too many legal moves");
                for (const uint16_t index : input.legalPolicyIndices)
                    if (index >= encoding::POLICY_SIZE)
                        throw std::out_of_range("legal policy index is outside the network output");
                requests.push_back(std::make_shared<Request>(encoding::compactPosition(*input.board), std::vector<uint16_t>(input.legalPolicyIndices.begin(), input.legalPolicyIndices.end())));
            }
            {
                std::lock_guard<std::mutex> lock(queueMutex_);
                if (stopping_)
                    throw std::runtime_error("CUDA evaluation service is shutting down");
                for (const auto &request : requests)
                    queue_.push_back(request);
            }
            pending_.notify_all();

            // espera cada resposta, na ordem dos pedidos
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
        // copia o lote para a GPU, expande as posições, roda a rede, separa os logits dos lances legais e copia de volta
        void execute(int count)
        {
            const size_t positionBytes = static_cast<size_t>(count) * sizeof(encoding::CompactPosition);
            const size_t legalSlots = static_cast<size_t>(count) * kLegalSlots;
            checkCuda(cudaMemcpyAsync(devicePositions_.data(), hostPositions_.data(), positionBytes, cudaMemcpyHostToDevice, stream_.get()), "copy compact positions to CUDA");
            checkCuda(cudaMemcpyAsync(deviceIndices_.data(), hostIndices_.data(), legalSlots * sizeof(uint16_t), cudaMemcpyHostToDevice, stream_.get()), "copy legal policy indices to CUDA");

            cuda_kernels::decodeCompact(devicePositions_.data(), deviceInputs_.data(), count, stream_.get());
            checkKernel("decode compact positions");

            // tensores do ONNX Runtime criados sobre os buffers que já estão na GPU
            const std::array<int64_t, 4> inputShape = {count, encoding::CHANNELS, 8, 8};

            const std::array<int64_t, 2> policyShape = {count, encoding::POLICY_SIZE};

            const std::array<int64_t, 2> valueShape = {count, 1};

            Ort::Value input = Ort::Value::CreateTensor<float>(cudaMemory_, deviceInputs_.data(), static_cast<size_t>(count) * encoding::INPUT_SIZE, inputShape.data(), inputShape.size());
            Ort::Value policy = Ort::Value::CreateTensor<float>(cudaMemory_, devicePolicies_.data(), static_cast<size_t>(count) * encoding::POLICY_SIZE, policyShape.data(), policyShape.size());
            Ort::Value value = Ort::Value::CreateTensor<float>(cudaMemory_, deviceValues_.data(), static_cast<size_t>(count), valueShape.data(), valueShape.size());

            binding_.ClearBoundInputs();
            binding_.ClearBoundOutputs();
            binding_.BindInput("board", input);
            binding_.BindOutput("policy", policy);
            binding_.BindOutput("value", value);
            session_.Run(Ort::RunOptions{nullptr}, binding_);

            // só os logits dos lances legais voltam para a CPU, não a saída inteira de 4224
            cuda_kernels::gatherLegal(devicePolicies_.data(), deviceIndices_.data(), deviceLegalLogits_.data(), count, kLegalSlots, stream_.get());
            checkKernel("gather legal policy logits");

            checkCuda(cudaMemcpyAsync(hostLegalLogits_.data(), deviceLegalLogits_.data(), legalSlots * sizeof(float), cudaMemcpyDeviceToHost, stream_.get()), "copy legal logits from CUDA");
            checkCuda(cudaMemcpyAsync(hostValues_.data(), deviceValues_.data(), static_cast<size_t>(count) * sizeof(float), cudaMemcpyDeviceToHost, stream_.get()), "copy values from CUDA");
            // espera o stream terminar antes de ler os resultados na CPU
            checkCuda(cudaStreamSynchronize(stream_.get()), "synchronize CUDA inference");
        }

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

        // thread do lote --> espera pedidos, junta até batchSize ou até o tempo limite e avalia o lote
        void run()
        {
            std::vector<std::shared_ptr<Request>> batch;
            batch.reserve(static_cast<size_t>(batchSize_));
            for (;;)
            {
                batch.clear();
                {
                    std::unique_lock<std::mutex> lock(queueMutex_);
                    pending_.wait(lock, [&]
                                  { return stopping_ || !queue_.empty(); });
                    if (stopping_ && queue_.empty())
                        break;

                    // espera o lote encher por no máximo batchTimeoutUs
                    const auto deadline = std::chrono::steady_clock::now() + timeout_;
                    while (!stopping_ && static_cast<int>(queue_.size()) < batchSize_)
                        if (pending_.wait_until(lock, deadline) == std::cv_status::timeout)
                            break;

                    const int count = std::min<int>(batchSize_, queue_.size());
                    for (int index = 0; index < count; ++index)
                    {
                        batch.push_back(std::move(queue_.front()));
                        queue_.pop_front();
                    }
                }

                try
                {
                    const int count = static_cast<int>(batch.size());

                    // vagas sem lance ficam com índice 0; na volta só os legalCount primeiros logits são usados
                    std::fill_n(hostIndices_.data(), static_cast<size_t>(count) * kLegalSlots, uint16_t{0});

                    for (int index = 0; index < count; ++index)
                    {
                        hostPositions_.data()[index] = batch[index]->position;
                        std::copy(batch[index]->legalPolicyIndices.begin(), batch[index]->legalPolicyIndices.end(), hostIndices_.data() + static_cast<size_t>(index) * kLegalSlots);
                    }

                    execute(count);
                    for (int index = 0; index < count; ++index)
                    {
                        EvalOutput output;
                        const size_t legalCount = batch[index]->legalPolicyIndices.size();
                        const float *first = hostLegalLogits_.data() + static_cast<size_t>(index) * kLegalSlots;
                        output.legalLogits.assign(first, first + legalCount);
                        output.value = hostValues_.data()[index];
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

        CudaStream stream_;
        Ort::Env env_;
        Ort::Session session_{nullptr};
        Ort::MemoryInfo cudaMemory_{nullptr};
        Ort::IoBinding binding_{nullptr};
        int batchSize_;
        std::chrono::microseconds timeout_;

        // buffers do tamanho do lote máximo, alocados uma vez no construtor
        PinnedBuffer<encoding::CompactPosition> hostPositions_;
        PinnedBuffer<uint16_t> hostIndices_;
        PinnedBuffer<float> hostLegalLogits_;
        PinnedBuffer<float> hostValues_;
        DeviceBuffer<encoding::CompactPosition> devicePositions_;
        DeviceBuffer<uint16_t> deviceIndices_;
        DeviceBuffer<float> deviceInputs_;
        DeviceBuffer<float> devicePolicies_;
        DeviceBuffer<float> deviceLegalLogits_;
        DeviceBuffer<float> deviceValues_;

        std::mutex queueMutex_;
        std::condition_variable pending_;
        std::deque<std::shared_ptr<Request>> queue_;
        bool stopping_ = false;
        std::thread worker_;
    };

    CudaEvalService::CudaEvalService(const std::string &modelPath, int batchSize, int batchTimeoutUs)
        : impl_(std::make_unique<Impl>(modelPath, batchSize, batchTimeoutUs))
    {
    }

    CudaEvalService::~CudaEvalService() = default;

    std::vector<EvalOutput> CudaEvalService::evaluateBatch(std::span<const PositionEvalRequest> requests)
    {
        return impl_->evaluateBatch(requests);
    }

    void CudaEvalService::shutdown()
    {
        impl_->shutdown();
    }

} // namespace engine
