#pragma once

#include "search.hpp"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace engine
{

    // perfil --> texto em blocos N{move:n:P:Q:V|...} separados por ';'
    inline std::string serializeInstrumentalProfile(const std::vector<InstrumentalProfileNode> &profile)
    {
        if (profile.empty())
            return "-";

        std::ostringstream serialized;
        serialized.imbue(std::locale::classic());
        for (size_t nodeIndex = 0; nodeIndex < profile.size(); ++nodeIndex)
        {
            if (nodeIndex > 0)
                serialized << ';';

            const auto &node = profile[nodeIndex];
            serialized << node.nodeVisits << '{';
            for (size_t childIndex = 0; childIndex < node.children.size(); ++childIndex)
            {
                if (childIndex > 0)
                    serialized << '|';
                const auto &child = node.children[childIndex];
                // max_digits10 --> a leitura do texto devolve o mesmo float/double
                serialized << child.move << ':' << child.visits << ':' << std::setprecision(std::numeric_limits<float>::max_digits10) << child.prior << ':';
                if (child.q)
                    serialized << std::setprecision(std::numeric_limits<double>::max_digits10) << *child.q;
                else
                    serialized << '-';
                serialized << ':';
                if (child.v)
                    serialized << std::setprecision(std::numeric_limits<float>::max_digits10) << *child.v;
                else
                    serialized << '-';
            }
            serialized << '}';
        }
        return serialized.str();
    }

    namespace detail
    {
        // texto --> inteiro sem sinal; recusa vazio ou sobra de caracteres
        inline uint32_t parseUint32(std::string_view text, const char *label)
        {
            if (text.empty())
                throw std::invalid_argument(std::string("empty ") + label);
            uint32_t value = 0;
            const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
            if (error != std::errc{} || end != text.data() + text.size())
                throw std::invalid_argument(std::string("invalid ") + label);
            return value;
        }

        // texto --> float/double finito; locale classic para o decimal ser sempre ponto
        template <typename Float>
        inline Float parseFinite(std::string_view text, const char *label)
        {
            if (text.empty())
                throw std::invalid_argument(std::string("empty ") + label);
            std::istringstream input{std::string(text)};
            input.imbue(std::locale::classic());
            Float value{};
            input >> value;
            if (!input || input.peek() != std::char_traits<char>::eof() || !std::isfinite(static_cast<double>(value)))
                throw std::invalid_argument(std::string("invalid ") + label);
            return value;
        }

        // lance UCI --> origem, destino e promoção opcional (e2e4, a7a8q)
        inline bool isUciMove(std::string_view move)
        {
            if (move.size() != 4 && move.size() != 5)
                return false;
            const auto file = [](char value)
            { return value >= 'a' && value <= 'h'; };
            const auto rank = [](char value)
            { return value >= '1' && value <= '8'; };
            if (!file(move[0]) || !rank(move[1]) ||
                !file(move[2]) || !rank(move[3]))
                return false;
            if (move.size() == 5 && move[4] != 'q' && move[4] != 'r' &&
                move[4] != 'b' && move[4] != 'n')
                return false;
            return true;
        }

        // corta o texto no separador; pedaços vazios ficam na lista
        inline std::vector<std::string_view> split(std::string_view text, char separator)
        {
            std::vector<std::string_view> parts;
            size_t begin = 0;
            for (;;)
            {
                const size_t end = text.find(separator, begin);
                parts.push_back(text.substr(begin, end == std::string_view::npos ? end : end - begin));
                if (end == std::string_view::npos)
                    return parts;
                begin = end + 1;
            }
        }
    } // namespace detail

    // texto --> perfil; recusa blocos inválidos
    inline std::vector<InstrumentalProfileNode>
    parseRawInstrumentalProfile(std::string_view serialized)
    {
        if (serialized == "-")
            return {};
        if (serialized.empty())
            throw std::invalid_argument("empty instrumental profile");

        std::vector<InstrumentalProfileNode> profile;
        for (const std::string_view nodeText : detail::split(serialized, ';'))
        {
            // bloco --> visitas{filhos}
            const size_t open = nodeText.find('{');
            if (open == std::string_view::npos || open == 0 || nodeText.back() != '}' || nodeText.find('{', open + 1) != std::string_view::npos)
                throw std::invalid_argument("invalid instrumental profile node");

            InstrumentalProfileNode node;
            node.nodeVisits = detail::parseUint32(nodeText.substr(0, open), "node visits");

            const std::string_view childrenText = nodeText.substr(open + 1, nodeText.size() - open - 2);
            if (childrenText.empty())
                throw std::invalid_argument("instrumental profile node has no children");

            uint64_t totalVisits = 0;
            std::unordered_set<std::string> moves;
            for (std::string_view childText : detail::split(childrenText, '|'))
            {
                // filho --> lance:visitas:prior:Q:V
                InstrumentalChildStats child;
                const auto fields = detail::split(childText, ':');
                if (fields.size() != 5 || !detail::isUciMove(fields[0]))
                    throw std::invalid_argument("invalid instrumental child");
                child.move = std::string(fields[0]);
                if (!moves.insert(child.move).second)
                    throw std::invalid_argument("duplicate instrumental child move");
                child.visits = detail::parseUint32(fields[1], "child visits");
                child.prior = detail::parseFinite<float>(fields[2], "child prior");
                if (child.prior < 0.0f)
                    throw std::invalid_argument("negative instrumental child prior");

                // filho visitado tem Q e filho sem visita não tem
                if (fields[3] == "-")
                {
                    if (child.visits != 0)
                        throw std::invalid_argument("visited child has no Q");
                }
                else
                {
                    if (child.visits == 0)
                        throw std::invalid_argument("unvisited child has Q");
                    const double q = detail::parseFinite<double>(fields[3], "child Q");
                    if (q < -1.000001 || q > 1.000001)
                        throw std::invalid_argument("instrumental child Q is outside [-1, 1]");
                    child.q = q;
                }

                // V é a avaliação da rede no filho; filho sem visita não tem V
                if (fields[4] != "-")
                {
                    if (child.visits == 0)
                        throw std::invalid_argument("unvisited child has V");
                    child.v = detail::parseFinite<float>(fields[4], "child V");
                }

                totalVisits += child.visits;
                node.children.push_back(std::move(child));
            }

            // o primeiro filho é o lance escolhido, então precisa ter visitas
            if (node.nodeVisits == 0 || totalVisits == 0 ||
                node.children.front().visits == 0)
                throw std::invalid_argument("invalid instrumental profile node semantics");

            profile.push_back(std::move(node));
        }
        return profile;
    }

} // namespace engine
