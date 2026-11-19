#pragma once

#include <iosfwd>
#include <string>

namespace engine
{

    // loop UCI que lê de in, escreve em out e retorna no quit ou no fim da entrada
    // exeDir é a pasta do executável, onde o modelo é procurado se não estiver no diretório atual
    int uciLoop(std::istream &in, std::ostream &out, const std::string &exeDir);

} // namespace engine
