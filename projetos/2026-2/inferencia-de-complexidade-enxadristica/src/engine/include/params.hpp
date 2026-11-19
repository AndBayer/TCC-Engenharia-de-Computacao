#pragma once

#include <algorithm>
#include <cstdint>
#include <string>

namespace engine
{

    struct SearchParams
    {
        // orçamento da busca --> cada simulação desce da raiz até uma posição ainda não avaliada (ou fim de jogo) e soma o valor encontrado nos nós do caminho
        long long simBudget = 1600; // simulações
        long long movetimeMs = 0;   // 0 = sem limite de tempo

        // peso do prior frente ao valor médio no PUCT, a regra que escolhe o lance a explorar
        float cpuct = 2.5f;

        // lote (batch) --> posições avaliadas juntas pela rede
        int batchSize = 32; // lote máximo da fila
        int gatherBatchSize = 32; // folhas que cada busca junta antes de chamar a rede
        int batchTimeoutUs = 1500; // espera máxima, em microssegundos, para a fila completar o lote
        float virtualLoss = 3.0f; // perda provisória no caminho de uma folha que espera a rede, para as outras folhas do lote saírem de outros ramos

        // sem aleatoriedade na busca, pois o ruído de Dirichlet (sorteio que varia os lances) só serve para autojogo, não para medir

        // intervalo das linhas info (progresso no protocolo UCI) durante a busca, em ms; 0 = só no final
        int infoIntervalMs = 1000;

        // quantas melhores linhas da raiz mostrar (MultiPV)
        int multiPV = 1;

        // limite de memória da árvore, em número de nós
        size_t maxNodes = 80'000'000;

        // modelo da rede (.onnx)
        std::string modelPath = "model.onnx";
    };

} // namespace engine
