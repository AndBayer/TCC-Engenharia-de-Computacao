#pragma once

#include "chess.hpp"

#include <cstdint>

namespace engine
{

    // Unexpanded --> sem filhos; Expanding --> esperando a rede; Expanded --> com filhos; Terminal --> fim de jogo
    enum class NodeState : uint8_t
    {
        Unexpanded = 0,
        Expanding = 1,
        Expanded = 2,
        Terminal = 3
    };

    // nó da árvore --> uma posição; os filhos são as posições depois de cada lance legal
    struct Node
    {
        uint32_t N = 0;  // visitas --> simulações concluídas que incluíram o nó no caminho
        float W = 0.0f;  // soma das avaliações das simulações, na visão de quem joga na posição do nó
        float VL = 0.0f; // virtual loss pendente --> perda provisória enquanto o nó está no caminho de uma folha que espera a rede
        float P = 0.0f;  // prior --> chance que a rede dá ao lance (move), entre os lances legais do pai (somam 1)

        // avaliação da rede ao expandir o nó (Expanded) ou valor exato (Terminal)
        // sempre na visão de quem joga na posição do nó
        float stateValue = 0.0f;

        // filhos ficam em sequência na árvore, a partir de firstChild
        int32_t firstChild = -1;
        int32_t numChildren = 0;
        chess::Move move = chess::Move::NO_MOVE; // lance que leva do pai ao nó
        NodeState state = NodeState::Unexpanded;

        // Q --> valor médio (W / N); 0 sem visitas
        float q() const
        {
            return N ? W / static_cast<float>(N) : 0.0f;
        }
    };

} // namespace engine
