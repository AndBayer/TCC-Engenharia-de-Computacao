#pragma once

#include "node.hpp"

#include <deque>
#include <vector>

namespace engine
{

    // guarda todos os nós da busca; cada nó acha os filhos por firstChild e numChildren
    class Tree
    {
    public:
        explicit Tree(size_t maxNodes) : capacity_(maxNodes)
        {
            blocks_.emplace_back(BLOCK);
            allocChildren(1); // raiz = índice 0
        }

        // reserva n nós e devolve o índice do primeiro; -1 se a árvore encheu
        int32_t allocChildren(int n)
        {
            const size_t start = cursor_;
            if (start + static_cast<size_t>(n) > capacity_)
                return -1;

            const size_t lastIdx = start + static_cast<size_t>(n) - 1;
            while (lastIdx / BLOCK >= blocks_.size())
                blocks_.emplace_back(BLOCK);

            cursor_ = start + static_cast<size_t>(n);
            return static_cast<int32_t>(start);
        }

        // índice --> nó; os nós ficam em blocos de 4096, então um bloco novo não move os nós já criados
        Node &at(int32_t idx)
        {
            return blocks_[static_cast<size_t>(idx) / BLOCK][static_cast<size_t>(idx) % BLOCK];
        }

        const Node &at(int32_t idx) const
        {
            return blocks_[static_cast<size_t>(idx) / BLOCK][static_cast<size_t>(idx) % BLOCK];
        }

        Node &root() { return at(0); }
        size_t size() const { return cursor_; }
        bool full() const { return size() >= capacity_; }

    private:
        static constexpr size_t BLOCK = 4096;

        std::deque<std::vector<Node>> blocks_;
        size_t cursor_ = 0;
        size_t capacity_;
    };

} // namespace engine
