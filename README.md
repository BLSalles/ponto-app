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

## Assistente do gestor (perguntas por voz ou texto)

O painel do gestor tem uma aba **Assistente**, onde ele pergunta com as próprias
palavras — falando ou digitando — coisas como:

- "Quantas horas extras o Alex tem esse mês?"
- "Quantos dias a Maria ficou sem bater ponto?"
- "A que horas o João entrou ontem?"
- "Quem chegou atrasado hoje?"
- "Tem alguma coisa pendente pra eu aprovar?"

### Como ligar

1. Crie uma chave em <https://console.anthropic.com> (Settings → API Keys) e
   coloque créditos na conta.
2. Defina a variável de ambiente no servidor:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export ANTHROPIC_MODEL="claude-haiku-4-5-20251001"   # opcional
```

   No Render: **Environment → Add Environment Variable** e reinicie o serviço.
3. Instale a dependência nova (`pip install -r requirements.txt`) e pronto — o
   item "Assistente" aparece sozinho no menu do gestor.

Sem a chave o app funciona exatamente como antes; a página do assistente só
avisa que está desligada.

### Como funciona por dentro (`assistente.py`)

A pergunta vai para a API da Anthropic junto com a descrição de **cinco
ferramentas** de leitura: `listar_colaboradores`, `horas_extras`, `ausencias`,
`registros_do_periodo` e `situacao_agora`. O modelo **não vê o banco e não
escreve SQL** — ele só escolhe a ferramenta e os argumentos. Quem consulta é o
próprio `assistente.py`, chamando as mesmas funções que já alimentam as telas
(`_calcular_relatorio_horas_extras`, `montar_resumo_diario`,
`calcular_horas_extras_dia`). Por isso o número que o assistente responde é, por
construção, igual ao da tela de Horas Extras.

A voz é do próprio navegador (Web Speech API): o áudio não passa pelo servidor,
chega aqui já como texto, e a resposta pode ser lida em voz alta. Funciona no
Chrome, Edge, Android e Safari recente; em navegador sem suporte, o microfone
some e o campo de texto continua funcionando.

Para conferir as ferramentas sem gastar chamada de API:

```python
from app import app
with app.app_context():
    exec_ = app.blueprints["assistente"].executores
    print(exec_["horas_extras"]({"nome": "Alex"}))
```

### Cuidados

- Só o gestor logado acessa (mesmo `gestor_required` do resto do painel).
- A chave da API fica no servidor; o navegador nunca a recebe.
- O assistente é **somente leitura** — não aprova ajuste nem altera registro.
- "Faltas" aqui significa *dia útil sem nenhuma batida*: o sistema não tem
  cadastro de férias, atestado nem feriado, e o assistente é instruído a
  deixar essa ressalva clara sempre que responder sobre ausências.
- Custo: com o modelo Haiku, cada pergunta sai por poucos centavos — a
  resposta é curta e o histórico enviado fica limitado às últimas mensagens.

## Limitações desta versão simples (V1)

- Banco de dados gratuito do Render tem limite de armazenamento e expira
  após 90 dias no plano free — para uso contínuo, migrar para um plano pago
  ou exportar os dados periodicamente (botão "Exportar CSV" no painel).
- Fotos não são salvas em disco, apenas a "assinatura" facial (encoding)
  usada para conferência — ou seja, elas nunca são realmente armazenadas.
- Sem paginação, sem separação por mês/dia. Simples de adicionar depois.
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
