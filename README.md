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
