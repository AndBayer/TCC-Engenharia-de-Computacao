#include "uci.hpp"

#include <filesystem>
#include <iostream>
#include <string>

int main(int argc, char *argv[])
{
    // cin e cout sem sincronizar com o stdio do C, para ficarem mais rápidos
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // pasta do executável, onde o UCI também procura o modelo
    std::string exeDir;
    if (argc > 0 && argv[0] && *argv[0])
    {
        std::error_code ec;
        auto p = std::filesystem::absolute(std::filesystem::path(argv[0]), ec);
        if (!ec)
            exeDir = p.parent_path().string();
    }

    return engine::uciLoop(std::cin, std::cout, exeDir);
}
