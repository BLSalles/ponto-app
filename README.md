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
4. (Opcional) Defina `LOCATIONIQ_API_KEY` para a geocodificação dos endereços. Sem
   ela o app usa os servidores públicos do OpenStreetMap, que no IP compartilhado do
   Render costumam responder `429 Too many requests` — o ponto é salvo do mesmo jeito
   (com latitude/longitude), mas o endereço pode faltar até você clicar em
   **Preencher endereços faltando** no painel do gestor.
5. Aguarde o build (a primeira vez demora mais, pois compila o `dlib`).
6. Acesse a URL gerada pelo Render, faça login como gestor e cadastre os
   colaboradores.

### Por que Docker?
O reconhecimento facial usa a biblioteca `face_recognition`, que depende do
`dlib` — uma biblioteca C++ que precisa ser compilada com `cmake` e outras
dependências de sistema. O build padrão do Render (buildpack Python) não
inclui essas ferramentas, por isso o deploy é feito via Dockerfile.

## Cadastro facial por link (o colaborador se cadastra sozinho)

No Painel do Gestor → **Cadastro**, cada colaborador tem o botão **🔗 Link facial**. Ele gera
um link para a pessoa abrir no próprio celular, tirar a foto e ter o rosto cadastrado — sem o
gestor precisar ir até ela.

Limites do link, porque **quem abre cadastra o rosto que estiver na frente da câmera**:

- vale **uma vez só** (depois de usado, para de funcionar);
- **expira em 48h** (`CONVITE_FACIAL_HORAS` em `app.py`);
- gerar um link novo para a mesma pessoa **invalida o anterior na hora**, então link velho que
  ficou perdido num grupo de WhatsApp não serve mais;
- se o rosto enviado já estiver cadastrado para OUTRA pessoa, o cadastro é recusado (e o nome
  da outra pessoa não é revelado a quem está com o link — só aparece no log do servidor).

Mande o link direto para a pessoa certa (o botão de WhatsApp/e-mail já monta a mensagem).

## Ajustando o reconhecimento facial

A batida do ponto compara o rosto da câmera com o rosto cadastrado do colaborador que
**já entrou com e-mail e senha** — é uma checagem 1 contra 1, não uma identificação
"quem é essa pessoa". A distância entre os dois rostos precisa ficar abaixo de
`FACE_MATCH_TOLERANCE` (padrão `0.55`; a biblioteca `face_recognition` usa `0.6`).

- **Colaboradores apanhando pra bater o ponto?** Suba a tolerância de 0.05 em 0.05 na
  variável de ambiente `FACE_MATCH_TOLERANCE` (aceita de 0.30 a 0.70). Não precisa mexer
  no código nem refazer o build.
- **Como calibrar sem chutar:** cada tentativa imprime a distância real no log do Render:
  `[facial] Rejeitado: Ana (distância=0.5831, tolerância=0.55)`. Se as rejeições da pessoa
  certa estão na casa de 0.56–0.60, é a tolerância que está apertada.
- A detecção de rosto duplicado no cadastro usa um limiar próprio e mais estrito
  (`FACE_DUPLICADO_TOLERANCE`, padrão `0.45`), porque ali a comparação é contra todos os
  cadastrados e um falso positivo é bem mais caro.
- **A foto do cadastro manda no resultado.** Tire com boa luz, de frente, rosto centralizado
  e sem óculos escuros/boné. Uma foto ruim no cadastro faz TODA batida daquela pessoa ficar
  no limite.

## Jornada que vira a noite

Uma sessão (entrada → saída) que atravessa a meia-noite pertence ao dia em que a ENTRADA
aconteceu, mas as horas **param de ser creditadas nesse dia no horário de entrada padrão do dia
seguinte** — dali em diante a pessoa já está cumprindo o expediente normal do outro dia.

Exemplo com jornada padrão começando às 08:00: entrou 08:00 do dia 2 e saiu 14:00 do dia 3.

| Dia | Horas | Por quê |
|---|---|---|
| Dia 2 | 24h (08:00 do dia 2 → 08:00 do dia 3) | turno que virou — vira hora extra |
| Dia 3 | 6h (08:00 → 14:00) | expediente normal do dia 3 |

Turno da noite que termina ANTES do horário de entrada (ex.: 22:00 → 06:00) **não** é
fatiado: continua inteiro no dia em que começou.

O alerta de **jornada excepcional** continua olhando a sessão contínua inteira (as 30h do
exemplo), não a fatia — então o gestor ainda revisa o caso antes das horas extras entrarem
no total do mês.

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

## API externa (integração com o Lecom e outros sistemas)

Endpoints REST somente-leitura para outro sistema puxar os dados de ponto e
preencher a grid dele: **nome, data, hora de entrada, hora de saída e endereço**.

### 1. Ligar a API

A API só funciona com token configurado. Sem token no ambiente ela responde
`503` e não expõe nada.

```bash
# gera um token aleatório
flask gerar-token-api
```

Cole o valor gerado na variável de ambiente do servidor (no Render:
*Environment → Add Environment Variable*):

```
API_TOKENS=<token-gerado>
```

Para dar um token diferente para cada sistema que consome (e poder revogar um
sem derrubar o outro), separe por vírgula:

```
API_TOKENS=token-do-lecom,token-do-rh
```

### 2. Autenticação

Todo request precisa de um destes cabeçalhos:

```
X-API-Key: <token>
Authorization: Bearer <token>
```

Respostas de erro: `401` sem token, `403` token inválido, `503` API desligada.

### 3. Endpoints

| Endpoint | O que devolve |
|---|---|
| `GET /api/v1/ping` | Teste de credencial e conectividade |
| `GET /api/v1/colaboradores` | Lista de colaboradores (id, nome, e-mail) |
| `GET /api/v1/jornadas` | **Uma linha por jornada (entrada + saída) — é a da grid** |
| `GET /api/v1/batidas` | Uma linha por marcação, sem pareamento |

Filtros de `/jornadas` e `/batidas` (todos opcionais):

| Parâmetro | Valor |
|---|---|
| `inicio`, `fim` | `AAAA-MM-DD` ou `DD/MM/AAAA` (padrão: últimos 30 dias) |
| `colaborador_id` | id numérico |
| `email` | e-mail exato |
| `nome` | trecho do nome (busca parcial) |
| `pagina`, `por_pagina` | paginação (padrão 200 por página, máximo 1000) |
| `formato` | `json` (padrão) ou `csv` |

### 4. Exemplo

```bash
curl -H "X-API-Key: SEU_TOKEN" \
  "https://SEU-APP.onrender.com/api/v1/jornadas?inicio=01/08/2026&fim=31/08/2026"
```

```json
{
  "inicio": "2026-08-01",
  "fim": "2026-08-31",
  "fuso": "America/Sao_Paulo",
  "pagina": 1, "por_pagina": 200, "paginas": 1, "total": 2,
  "dados": [
    {
      "colaborador_id": 2,
      "nome": "Ana Souza",
      "email": "ana@empresa.com",
      "data": "2026-08-10",
      "data_br": "10/08/2026",
      "hora_entrada": "08:02:00",
      "hora_saida": "17:58:00",
      "data_hora_entrada": "2026-08-10T08:02:00-03:00",
      "data_hora_saida": "2026-08-10T17:58:00-03:00",
      "endereco_entrada": "Rua A, 100 - Centro, São Paulo",
      "endereco_saida": "Rua A, 100 - Centro, São Paulo",
      "latitude_entrada": -23.55, "longitude_entrada": -46.63,
      "latitude_saida": -23.55, "longitude_saida": -46.63,
      "total_horas": "09:56",
      "total_horas_decimal": 9.93,
      "completa": true,
      "registro_entrada_id": 1,
      "registro_saida_id": 2
    }
  ]
}
```

### 5. Detalhes que importam na integração

- **Fuso**: todos os horários já vêm no horário de Brasília. Os campos
  `data_hora_*` são ISO 8601 com o deslocamento explícito (`-03:00`).
- **Jornada que vira a noite**: fica no dia em que a **entrada** aconteceu —
  mesma convenção das telas do sistema. Entrou 11/08 às 22h e saiu 12/08 às 6h?
  A linha é do dia 11/08, com `data_hora_saida` no dia 12.
- **Jornada incompleta** (bateu entrada e ainda não bateu saída, ou esqueceu):
  `hora_saida: null` e `completa: false`. Vale tratar isso na grid.
- **Endereço**: vem da geocodificação reversa da coordenada da batida. Pode ser
  `null` em registros antigos ou se o serviço de mapas falhou na hora — o
  gestor pode preencher depois em *Consulta → Preencher endereços*.
- **Paginação**: quando `paginas > 1`, repita a chamada incrementando `pagina`
  até juntar o `total`.
- **CSV**: `&formato=csv` devolve o mesmo recorte separado por `;` e com BOM,
  já pronto para abrir no Excel com acentuação correta.
