# Controle de Ponto (MVP)

App simples para bater ponto com reconhecimento facial e geolocalização.
Feito para 3 colaboradores + 1 gestor, pronto para deploy no Render.

## Rodando localmente

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Variáveis opcionais (senão usa os valores padrão abaixo)
export GESTOR_EMAIL="gestor@empresa.com"
export GESTOR_SENHA="mude-esta-senha"
export SECRET_KEY="uma-chave-secreta-qualquer"

python app.py
```

Acesse http://localhost:5000 — faça login com o e-mail/senha do gestor
(padrão: gestor@empresa.com / mude-esta-senha) e cadastre os 3 colaboradores
com foto pelo Painel do Gestor.

## Deploy no Render

1. Suba este projeto para um repositório no GitHub.
2. No Render, clique em **New > Blueprint** e aponte para o repositório
   (o arquivo `render.yaml` já configura o Web Service com Docker e o banco
   Postgres gratuito automaticamente).
3. Nas variáveis de ambiente, defina `GESTOR_SENHA` (o `render.yaml` pede
   isso manualmente por segurança).
4. Aguarde o build (a primeira vez demora mais, pois compila o `dlib`).
5. Acesse a URL gerada pelo Render, faça login como gestor e cadastre os
   colaboradores.

### Por que Docker?
O reconhecimento facial usa a biblioteca `face_recognition`, que depende do
`dlib` — uma biblioteca C++ que precisa ser compilada com `cmake` e outras
dependências de sistema. O build padrão do Render (buildpack Python) não
inclui essas ferramentas, por isso o deploy é feito via Dockerfile.

## Limitações desta versão simples (V1)

- Banco de dados gratuito do Render tem limite de armazenamento e expira
  após 90 dias no plano free — para uso contínuo, migrar para um plano pago
  ou exportar os dados periodicamente (botão "Exportar CSV" no painel).
- Fotos não são salvas em disco, apenas a "assinatura" facial (encoding)
  usada para conferência — ou seja, elas nunca são realmente armazenadas.
- Reconhecimento facial local: qualquer foto suficientemente parecida com
  boa iluminação passa. Para maior rigor, ajuste `FACE_MATCH_TOLERANCE`
  em `app.py` (quanto menor, mais rígido).
- Plano free do Render "dorme" após 15 min de inatividade — o primeiro
  acesso do dia pode demorar ~30s para "acordar" o serviço.

## Próximos passos (quando for escalar)

- Migrar fotos/relatórios para armazenamento externo (S3, Cloudinary)
- Adicionar filtro por período e por colaborador no painel
- Notificação por e-mail/WhatsApp em caso de ponto fora do horário
- Múltiplos gestores e histórico de aprovação/contestação


## Paginação (10 registros por página)

Toda lista ou tabela do sistema mostra no máximo **10 registros por página** e
pagina o restante. Isso vale para: registros de ponto (gestor e colaborador),
resumo por dia, solicitações de ajuste (pendentes e histórico), jornadas
excepcionais, colaboradores (dashboard e cadastro) e o relatório de horas
extras. No chat do assistente, as tabelas de detalhe seguem a mesma regra.

Como foi feito:

- `ITENS_POR_PAGINA`, `paginar_query()` e `paginar_lista()` em `app.py` são o
  único ponto de verdade — antes cada rota tinha seu próprio limite (20, 25, 30)
  e algumas listas nem paginavam (usavam `.limit()` fixo e escondiam o resto).
- Cada bloco paginado tem seu **próprio parâmetro na URL** (`p_registros`,
  `p_resumo`, `p_pendentes`, `p_historico`, `p_colaboradores`...), então duas
  tabelas na mesma tela não andam mais juntas.
- Os links de paginação **preservam os outros parâmetros** da URL (ex.:
  `?mes=2026-08` na tela de horas extras) e levam de volta à altura da tabela.
- Página inválida (`?p_registros=abc`, `-5`, `99999`) não quebra mais nem
  devolve tabela vazia: cai na primeira/última página com conteúdo.
- Corrigido também o `contar_registros_hoje`, que olhava só as 10 últimas
  batidas do colaborador e errava o total de quem bate mais de 10 pontos no dia.

## Assistente inteligente (`/assistente`)

Chat em português que responde sobre os dados do ponto — **sem API externa,
sem chave paga e sem custo por pergunta**. O motor está em `assistente.py` e usa
exatamente as mesmas funções de cálculo das telas (`montar_resumo_diario`,
`calcular_horas_extras_dia`), então nunca responde um número diferente do que o
relatório mostra.

Exemplos de perguntas:

- "Quantas horas extras o colaborador João tem?"
- "Quantas horas a Maria trabalhou mês passado?"
- "Quem tem mais horas extras este mês?"
- "Quantos atrasos o Carlos teve nos últimos 30 dias?"
- "Quem ainda não bateu ponto hoje?"
- "Quais pendências estão esperando minha aprovação?"
- "Quantas horas extras de 01/07 a 15/07 a Helena tem?"

Entende nome escrito errado ou abreviado, e períodos em linguagem natural
("hoje", "ontem", "esta semana", "mês passado", "em julho", "nos últimos 15
dias", "de 01/07 a 15/07", "2026-07").

**Como ele aprende com o processo**

- Cada pergunta e resposta vira uma linha em `assistente_interacao`.
- 👍/👎 em uma resposta ajusta o peso daquele jeito de perguntar
  (`assistente_padrao`): perguntas futuras parecidas passam a cair na intenção
  certa mesmo sem bater nas regras fixas.
- Quando o assistente não reconhece o nome, ele pergunta de quem se trata; ao
  escolher a pessoa, o apelido usado é aprendido (`assistente_apelido`) e passa
  a funcionar sozinho ("jô" → Joana Ribeiro).
- As perguntas mais bem avaliadas viram as sugestões da tela inicial do chat.
- As respostas comparam o período pedido com a média dos meses anteriores do
  próprio colaborador, então a leitura fica mais rica conforme a base cresce.

**Permissões**

- Gestor: pergunta sobre qualquer colaborador e sobre a equipe.
- Colaborador: só os próprios dados. Perguntar sobre outra pessoa ou sobre a
  equipe recebe uma recusa educada, sem vazar número nenhum.

As três tabelas novas são criadas automaticamente no start (`db.create_all()`),
sem migração manual.
