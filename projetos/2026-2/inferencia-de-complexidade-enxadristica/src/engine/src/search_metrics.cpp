#include "search_metrics.hpp"

#include "node.hpp"
#include "tree.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace
{
    // Q --> centipawns, inversa do tanh(cp/600)
    // limitar q evita atanh(1) = infinito
    double valueToCp(double q)
    {
        q = std::clamp(q, -0.9999, 0.9999);
        return 600.0 * std::atanh(q);
    }
}

namespace engine
{
    // monta o resultado a partir da árvore
    void populateSearchMetrics(Tree &tree, const SearchParams &params, SearchResult &res)
    {
        Node &root = tree.root();
        // raiz terminal --> só o valor exato, sem lances
        if (root.state == NodeState::Terminal)
        {
            res.qRoot = root.stateValue;
            res.evalCp = valueToCp(res.qRoot);
            return;
        }

        // estatísticas de cada lance da raiz e o lance mais visitado
        uint32_t bestN = 0;
        int32_t bestIdx = -1;
        for (int i = 0; i < root.numChildren; ++i)
        {
            const int32_t childIdx = root.firstChild + i;
            const Node &child = tree.at(childIdx);
            const uint32_t n = child.N;
            MoveStat move;
            move.uci = chess::uci::moveToUci(child.move);
            move.visits = n;
            move.prior = child.P;
            // Q do filho é do adversário, então troca o sinal
            move.q = n > 0 ? -static_cast<double>(child.q()) : 0.0;
            move.evalCp = n > 0 ? valueToCp(move.q) : 0.0;
            move.nodeIdx = childIdx;
            res.rootMoves.push_back(std::move(move));

            if (n > bestN || bestIdx < 0)
            {
                bestN = n;
                bestIdx = childIdx;
            }
        }

        std::sort(res.rootMoves.begin(), res.rootMoves.end(), [](const MoveStat &a, const MoveStat &b)
                  { return a.visits > b.visits; });

        // MultiPV segue o filho mais visitado a partir de cada lance
        const int multiPvCount = std::min<int>(params.multiPV, static_cast<int>(res.rootMoves.size()));
        for (int i = 0; i < multiPvCount; ++i)
        {
            MoveStat &move = res.rootMoves[static_cast<size_t>(i)];
            if (move.visits == 0 || move.nodeIdx < 0)
                break;
            move.pv.push_back(move.uci);
            int32_t current = move.nodeIdx;
            // cada linha do MultiPV tem no máximo 24 lances
            for (int ply = 1; ply < 24; ++ply)
            {
                Node &node = tree.at(current);
                if (node.state != NodeState::Expanded || node.numChildren <= 0)
                    break;
                uint32_t mostVisits = 0;
                int32_t next = -1;
                for (int child = 0; child < node.numChildren; ++child)
                {
                    const int32_t childIdx = node.firstChild + child;
                    const uint32_t n = tree.at(childIdx).N;
                    if (n > mostVisits)
                    {
                        mostVisits = n;
                        next = childIdx;
                    }
                }
                if (next < 0 || mostVisits == 0)
                    break;
                move.pv.push_back(chess::uci::moveToUci(tree.at(next).move));
                current = next;
            }
        }

        if (bestIdx >= 0)
            res.bestMove = chess::uci::moveToUci(tree.at(bestIdx).move);

        // Q da raiz --> média de todos os valores retropropagados, incluindo a avaliação inicial
        const uint32_t rootVisits = root.N;
        res.qRoot = rootVisits ? root.W / static_cast<double>(rootVisits) : 0.0;
        res.evalCp = valueToCp(res.qRoot);

        // PV e perfil instrumental, seguindo o filho mais visitado (no máximo 256 nós)
        int32_t current = 0;
        for (int ply = 0; ply < 256; ++ply)
        {
            Node &node = tree.at(current);
            if (node.state != NodeState::Expanded || node.numChildren <= 0)
                break;

            uint32_t mostVisits = 0;
            int32_t next = -1;
            int selectedChild = -1;
            InstrumentalProfileNode profileNode;
            profileNode.children.reserve(static_cast<size_t>(node.numChildren));
            for (int child = 0; child < node.numChildren; ++child)
            {
                const int32_t childIdx = node.firstChild + child;
                const Node &candidate = tree.at(childIdx);
                const uint32_t n = candidate.N;
                InstrumentalChildStats childStats;
                childStats.move = chess::uci::moveToUci(candidate.move);
                childStats.visits = n;
                childStats.prior = candidate.P;
                if (n > 0)
                    childStats.q = -static_cast<double>(candidate.q());
                // V vem da rede, então nó terminal não tem
                const NodeState candidateState = candidate.state;
                if (n > 0 && candidateState == NodeState::Expanded)
                    childStats.v = -candidate.stateValue;
                profileNode.children.push_back(std::move(childStats));
                if (n > mostVisits)
                {
                    mostVisits = n;
                    next = childIdx;
                    selectedChild = child;
                }
            }
            if (next < 0 || mostVisits == 0)
                break;

            // a raiz conta a avaliação inicial como visita, então tira 1
            const uint32_t rawVisits = node.N;
            const uint32_t nodeVisitsCount = (ply == 0 && rawVisits > 0) ? rawVisits - 1 : rawVisits;
            profileNode.nodeVisits = nodeVisitsCount;
            // lance mais visitado primeiro
            std::rotate(profileNode.children.begin(), profileNode.children.begin() + selectedChild, profileNode.children.begin() + selectedChild + 1);
            res.instrumentalProfile.push_back(std::move(profileNode));
            res.pv.push_back(chess::uci::moveToUci(tree.at(next).move));
            current = next;
        }
    }

} // namespace engine
