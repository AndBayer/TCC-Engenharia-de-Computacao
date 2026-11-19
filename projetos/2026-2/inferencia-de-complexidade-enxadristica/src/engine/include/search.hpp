#pragma once

#include "chess.hpp"
#include "inference.hpp"
#include "params.hpp"

#include <atomic>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace engine
{

    // estatísticas de um lance da raiz
    struct MoveStat
    {
        std::string uci;
        uint32_t visits = 0;
        float prior = 0.0f;
        double q = 0.0; // na visão de quem joga na raiz
        double evalCp = 0.0; // avaliação em centipawns = 600*atanh(q), com q limitado para não dar infinito

        // pv só é preenchida para as multiPV melhores linhas
        std::vector<std::string> pv; // variante a partir do lance (inclui ele)
        int32_t nodeIdx = -1; // índice do filho na árvore (uso interno)
    };

    // um filho no perfil instrumental --> lance, visitas, prior, Q e V (Q e V podem faltar)
    struct InstrumentalChildStats
    {
        std::string move;
        uint32_t visits = 0;
        float prior = 0.0f;
        std::optional<double> q;
        std::optional<float> v;
    };

    // um nó da PV no perfil instrumental --> visitas do nó e todos os filhos legais
    struct InstrumentalProfileNode
    {
        uint32_t nodeVisits = 0;
        std::vector<InstrumentalChildStats> children;
    };

    // resultado da busca, usado no info do UCI e nos CSVs da validação
    struct SearchResult
    {
        std::string bestMove; // filho mais visitado
        double evalCp = 0.0;  // centipawns = 600*atanh(Q da raiz), com Q limitado para não dar infinito
        double qRoot = 0.0;   // Q da raiz, na visão de quem joga

        // o perfil instrumental guarda, em cada nó da PV, as visitas e move:n:P:Q:V de todos os filhos legais
        // ('-' quando Q ou V não existem); H, KL e somas são calculados fora do motor
        std::vector<InstrumentalProfileNode> instrumentalProfile;

        long long simulations = 0; // simulações concluídas
        long long collisions = 0;  // simulações descartadas porque chegaram a uma folha que já esperava a rede
        size_t treeNodes = 0;      // nós alocados na árvore
        int maxDepth = 0; // maior profundidade de uma simulação, em lances (seldepth no UCI)
        int avgDepth = 0; // profundidade média das simulações (depth no UCI, como no Lc0)
        long long elapsedMs = 0;
        bool treeFull = false; // a árvore encheu e a busca parou antes do orçamento

        std::vector<MoveStat> rootMoves; // ordenado por visitas, do maior para o menor
        std::vector<std::string> pv;     // variante principal (filho mais visitado em cada nível)
    };

    // função de progresso, chamada a cada params.infoIntervalMs com um resultado parcial
    // o lote anterior já foi todo aplicado na árvore
    using InfoCallback = std::function<void(const SearchResult &)>;

    // roda a busca MCTS (busca em árvore de Monte Carlo, guiada pela rede) a partir de rootBoard, a raiz da árvore
    // o coordenador (a thread que chama runSearch, a única que mexe na árvore) junta até params.gatherBatchSize folhas antes de chamar a rede
    // o avaliador vem de quem chama e pode ser compartilhado entre buscas
    // stopFlag é a parada externa (comando stop) e pode ser nullptr
    SearchResult runSearch(const chess::Board &rootBoard, const SearchParams &params, PositionEvaluator &evaluator, const std::atomic<bool> *stopFlag = nullptr, const InfoCallback &onInfo = nullptr);

} // namespace engine
