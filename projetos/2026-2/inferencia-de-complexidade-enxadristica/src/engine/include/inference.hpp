#pragma once

#include "chess.hpp"
#include "encoding.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace engine
{

    // resposta da rede para uma posição --> uma nota para cada lance legal e o valor da posição
    struct EvalOutput
    {
        // logits (as notas dos lances) na mesma ordem dos índices legais passados a evaluate()
        std::vector<float> legalLogits;
        float value = 0.0f; // valor da rede em [-1, 1], na visão de quem joga (1 = ganho, -1 = perdido)
    };

    // pedido de avaliação --> tabuleiro e índices dos lances legais na saída de política
    struct PositionEvalRequest
    {
        const chess::Board *board = nullptr;
        std::span<const uint16_t> legalPolicyIndices;
    };

    // interface da rede --> avalia um lote de entradas já codificadas e devolve a política e o valor de cada uma
    class NetEvaluator
    {
    public:
        virtual ~NetEvaluator() = default;
        virtual void evaluateBatch(const float *inputs, int count, float *policies, float *values) = 0;
    };

    // com tryCuda = false, cria a sessão direto na CPU
    std::unique_ptr<NetEvaluator> makeOnnxEvaluator(const std::string &modelPath, bool tryCuda = true);

    // diz se o CUDA pode funcionar (provedor do ONNX Runtime e driver NVIDIA); não inicializa o CUDA
    bool cudaLooksAvailable(const std::string &exeDir, std::string &reason);

    // interface usada pela busca; bloqueia só quem pediu, e o pedido pode ir no lote de outras buscas
    class PositionEvaluator
    {
    public:
        virtual ~PositionEvaluator() = default;
        virtual std::vector<EvalOutput> evaluateBatch(std::span<const PositionEvalRequest> requests) = 0;

        // avalia uma posição só (lote de 1)
        EvalOutput evaluate(const chess::Board &board, std::span<const uint16_t> legalPolicyIndices);
    };

    // caminho determinístico, com uma thread e sem fila nem tempo limite
    class SynchronousEvaluator final : public PositionEvaluator
    {
    public:
        explicit SynchronousEvaluator(NetEvaluator &network) : network_(network) {}
        std::vector<EvalOutput> evaluateBatch(std::span<const PositionEvalRequest> requests) override;

    private:
        NetEvaluator &network_;
    };

    // fila de inferência compartilhada; a sincronização fica no .cpp
    class EvalService final : public PositionEvaluator
    {
    public:
        EvalService(NetEvaluator &network, int batchSize, int batchTimeoutUs);
        ~EvalService() override;

        EvalService(const EvalService &) = delete;
        EvalService &operator=(const EvalService &) = delete;

        std::vector<EvalOutput> evaluateBatch(std::span<const PositionEvalRequest> requests) override;
        // encerra a fila depois de atender os pedidos pendentes
        void shutdown();

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    // pipeline na GPU que compacta as posições na CPU, expande os canais em CUDA, roda a rede no ONNX Runtime e devolve só os logits legais
    class CudaEvalService final : public PositionEvaluator
    {
    public:
        CudaEvalService(const std::string &modelPath, int batchSize, int batchTimeoutUs);
        ~CudaEvalService() override;

        CudaEvalService(const CudaEvalService &) = delete;
        CudaEvalService &operator=(const CudaEvalService &) = delete;

        std::vector<EvalOutput> evaluateBatch(std::span<const PositionEvalRequest> requests) override;
        // encerra a fila depois de atender os pedidos pendentes
        void shutdown();

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace engine
