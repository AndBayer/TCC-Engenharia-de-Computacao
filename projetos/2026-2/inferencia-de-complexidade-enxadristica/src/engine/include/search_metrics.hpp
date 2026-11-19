#pragma once

#include "params.hpp"
#include "search.hpp"

namespace engine
{
    class Tree;
    // preenche os campos do resultado que vêm da árvore (no info periódico e no fim da busca)
    // contadores e tempo ficam com o search.cpp
    void populateSearchMetrics(Tree &tree, const SearchParams &params, SearchResult &result);

} // namespace engine
