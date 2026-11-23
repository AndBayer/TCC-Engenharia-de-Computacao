Via Licitações

Sistema Web de Inteligência em Licitações Públicas com Automação RPA

Projeto desenvolvido como Trabalho de Conclusão de Curso (TCC) do curso de Engenharia de Computação da Pontifícia Universidade Católica do Paraná (PUCPR).

1. Sobre o projeto

O Via Licitações é uma solução computacional desenvolvida para apoiar empresas no acompanhamento, identificação e gerenciamento de oportunidades em processos de compras e contratações públicas.

A plataforma integra um sistema web de gestão com uma automação RPA (Robotic Process Automation), responsável pela coleta e processamento de informações de licitações públicas.

O sistema busca centralizar informações de processos licitatórios, permitindo que empresas acompanhem oportunidades relacionadas às suas atividades econômicas, regiões de interesse e palavras-chave cadastradas.

2. Objetivos
Automatizar a coleta de informações de licitações públicas.
Centralizar os dados coletados em um banco de dados relacional.
Disponibilizar uma interface web para consulta e gerenciamento de oportunidades.
Permitir o cadastro e gerenciamento de empresas e usuários.
Oferecer filtros por CNAE, palavras-chave e regiões geográficas.
Permitir o acompanhamento de licitações de interesse.
Disponibilizar ferramentas de organização comercial, comentários e tarefas.
Registrar operações, importações e execuções da automação.
3. Arquitetura da solução

A solução é dividida em dois componentes principais:

3.1 Sistema Web

Desenvolvido utilizando:

PHP.
HTML5.
CSS3.
JavaScript.
MySQL.
PDO para comunicação com o banco de dados.

O sistema é executado localmente por meio do servidor Apache disponibilizado pelo XAMPP.

Principais funcionalidades:

Autenticação e gerenciamento de usuários.
Painel administrativo.
Cadastro e gerenciamento de empresas.
Gerenciamento de equipes.
Consulta de licitações.
Filtros por CNAE, palavras-chave e regiões.
Gerenciamento de interesses em licitações.
Comentários e acompanhamento de oportunidades.
Gerenciamento de tarefas.
Notificações.
Registro de logs.
Importação e gerenciamento de dados.
3.2 Automação RPA

Desenvolvida em Python, utilizando:

Pandas: manipulação e processamento de dados.
BeautifulSoup: análise de conteúdo HTML.
Selenium: automação de navegação web.
WebDriver Manager: gerenciamento do ChromeDriver.

A automação é responsável pelo processo de coleta e processamento de informações de editais e licitações públicas, integrando os resultados ao ecossistema do Via Licitações.

3.3 Banco de dados

O sistema utiliza MySQL, com banco de dados denominado via_licitacoes.

O modelo de dados contempla 16 tabelas, organizadas nos seguintes domínios:

Domínio	Tabelas
Empresas e CNAEs	empresas, cnaes_catalogo, cnaes_interesse
Licitações	licitacoes, documentos_licitacao, fontes_coleta
Gestão de oportunidades	interesses_licitacoes, interesses_comentarios, interesses_tarefas
Inteligência comercial	leads
Preferências	palavras_interesse, regioes_interesse
Usuários e comunicação	usuarios, notificacoes
Auditoria e processamento	logs_execucao, importacoes

O relacionamento entre as entidades permite associar empresas, usuários, licitações e oportunidades comerciais, mantendo o histórico das operações realizadas.

4. Estrutura do projeto
text
via-licitacoes/
│
├── data/
│   ├── datasets/
│   └── exemplos/
│       └── via_licitacoes.sql
│
├── src/
│   ├── sistema/
│   │   ├── public/
│   │   │   ├── admin/
│   │   │   ├── api/
│   │   │   ├── app/
│   │   │   └── assets/
│   │   │       ├── css/
│   │   │       ├── img/
│   │   │       └── js/
│   │   │
│   │   ├── src/
│   │   │   ├── config/
│   │   │   ├── core/
│   │   │   ├── helpers/
│   │   │   └── models/
│   │   │
│   │   ├── .htaccess
│   │   └── README.md
│   │
│   └── automacao/
│       ├── main.py
│       ├── automacao.bat
│       └── requirements.txt
│
├── tests/
│   ├── integracao/
│   └── unitarios/
│
└── README.md

5. Requisitos para execução
Sistema Web
Windows ou sistema operacional compatível.
XAMPP.
Apache.
PHP com suporte a PDO MySQL.
MySQL ou MariaDB.
Navegador web moderno.
Automação RPA
Python 3.
Google Chrome.
Dependências Python descritas em src/automacao/requirements.txt.
Conexão com a internet para as operações de coleta.
6. Instalação e configuração
6.1 Preparação do ambiente

Instale o XAMPP e inicie os serviços:

Apache.
MySQL.
6.2 Instalação do sistema web

Copie o conteúdo da pasta src/sistema/ para:

text
C:\xampp\htdocs\via-licitacoes


A estrutura esperada será:

text
C:\xampp\htdocs\via-licitacoes\
├── public/
├── src/
├── .htaccess
└── README.md

6.3 Criação do banco de dados
Acesse o phpMyAdmin:

http://localhost/phpmyadmin

Crie um banco de dados chamado:

via_licitacoes

Selecione o banco criado.
Acesse a opção Importar.
Selecione o arquivo:

data/exemplos/via_licitacoes.sql

Execute a importação.

O arquivo SQL disponibilizado contém a estrutura do banco e os dados de demonstração que forem mantidos na versão publicada.

6.4 Configuração da conexão

A conexão com o banco de dados está definida em:

text
src/config/database.php


Verifique as seguintes configurações conforme o ambiente:

Host do MySQL.
Nome do banco.
Usuário.
Senha.
Charset.

A configuração local padrão do projeto utiliza MySQL em localhost, banco via_licitacoes e conexão PDO.

Ajuste as credenciais conforme a instalação do XAMPP utilizada.

6.5 Configuração da aplicação

O arquivo de configurações globais está localizado em:

text
src/config/config.php


Entre os parâmetros configuráveis estão:

Ambiente de execução.
Depuração.
URL base.
Caminhos físicos.
Fuso horário.

A URL base deve corresponder ao endereço em que a aplicação estiver instalada.

6.6 Acesso ao sistema

Após iniciar Apache e MySQL, acesse:

http://localhost/via-licitacoes/public

O acesso administrativo deve ser realizado com uma conta configurada no banco de dados. Não são disponibilizadas credenciais administrativas no repositório.

7. Instalação da automação RPA

A automação está localizada em:

text
src/automacao/


Recomenda-se utilizar um ambiente virtual Python.

Abra o PowerShell na pasta da automação:

powershell
cd C:\PUCPR-TCC\TCC-Engenharia-de-Computacao\projetos\2026-2\via-licitacoes\src\automacao


Crie o ambiente virtual:

powershell
python -m venv .venv


Ative o ambiente:

powershell
.\.venv\Scripts\Activate.ps1


Instale as dependências:

powershell
pip install -r requirements.txt


Execute a automação conforme os parâmetros implementados no script:

powershell
python main.py


Também está disponível o arquivo automacao.bat, destinado à inicialização da automação em ambiente Windows.

8. Integração entre os componentes

A solução integra o sistema web, o banco de dados MySQL e a automação Python.

O fluxo geral de funcionamento contempla:

Coleta de informações de licitações públicas.
Processamento e organização dos dados.
Integração das informações com o sistema.
Armazenamento dos registros no banco de dados.
Disponibilização das informações na interface web.
Gerenciamento das oportunidades identificadas pelos usuários.

A comunicação entre os componentes utiliza mecanismos de integração definidos na implementação do sistema.

9. Segurança

O projeto contempla mecanismos de autenticação, controle de acesso e comunicação entre componentes.

Para instalação e execução:

Não versionar senhas, tokens ou credenciais.
Não disponibilizar dados pessoais ou comerciais sem autorização.
Utilizar configurações externas para informações sensíveis.
Não utilizar contas administrativas de demonstração em ambientes de produção.
Desabilitar a exibição de erros em ambientes de produção.
Restringir o acesso aos diretórios internos e arquivos de configuração.
10. Testes e validação

Os diretórios destinados aos testes são:

text
tests/
├── integracao/
└── unitarios/


Os testes unitários são destinados à validação de funcionalidades isoladas.

Os testes de integração são destinados à verificação da comunicação entre os componentes da solução, incluindo sistema web, banco de dados e automação.

11. Tecnologias utilizadas
Tecnologia	Aplicação
PHP	Backend do sistema
HTML5	Estrutura das interfaces
CSS3	Estilização
JavaScript	Interatividade
MySQL	Banco de dados relacional
Python	Automação RPA
Pandas	Processamento de dados
BeautifulSoup	Análise de HTML
Selenium	Automação web
WebDriver Manager	Gerenciamento do driver
XAMPP	Ambiente local de execução
Git e GitHub	Versionamento do código
12. Organização acadêmica

Projeto desenvolvido no âmbito do Trabalho de Conclusão de Curso de Engenharia de Computação da Pontifícia Universidade Católica do Paraná (PUCPR).

Disciplina: Projeto Final de Curso – IOPT
Etapa: 4º Bimestre – Implementação Final
Marco de versionamento sugerido: v1.0.0-final

A documentação acadêmica formal, relatórios, apresentações e demais materiais de defesa devem ser submetidos exclusivamente pelo ambiente institucional Canvas, conforme as orientações da coordenação.

Este repositório destina-se ao versionamento e compartilhamento dos códigos-fonte, arquivos técnicos e materiais necessários à execução do software.

13. Licença e utilização

Consulte o arquivo LICENSE na raiz do repositório institucional para conhecer os termos de utilização e distribuição aplicáveis ao projeto.

Via Licitações — Trabalho de Conclusão de Curso
Engenharia de Computação | PUCPR
Implementação Final — 2026/2