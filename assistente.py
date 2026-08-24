"""
Assistente inteligente do controle de ponto.

O que é
-------
Um motor de perguntas e respostas em linguagem natural (português) que responde
o gestor (e, de forma restrita, o próprio colaborador) usando SÓ os dados reais
do processo de ponto: batidas, jornadas, ajustes, aprovações e configuração da
empresa. Não depende de nenhuma API externa, chave paga ou internet — roda
dentro do próprio Flask.

Como ele "entende" a pergunta
-----------------------------
1. Normaliza o texto (minúsculas, sem acentos, sem pontuação).
2. Descobre DE QUEM se fala: casa o texto com os nomes cadastrados usando
   tokens + similaridade aproximada (difflib), então "quantas hrs extras o
   joao tem" acha "João Pedro da Silva" mesmo escrito errado.
3. Descobre QUANDO: "hoje", "ontem", "essa semana", "mês passado", "em julho",
   "nos últimos 15 dias", "de 01/07 a 15/07", "2026-07"...
4. Descobre O QUE se pergunta (intenção): horas extras, horas trabalhadas,
   atrasos, faltas, pendências, quem está presente, ranking, último ponto etc.
5. Executa a consulta correspondente e escreve a resposta com os números reais.

Como ele APRENDE com o processo
-------------------------------
- Toda pergunta/resposta vira uma interação salva no banco.
- Quando o gestor marca uma resposta como útil (👍), o padrão da pergunta é
  guardado com peso positivo; quando marca como não útil (👎), o peso cai.
  Perguntas futuras que não batem com nenhuma regra fixa são resolvidas por
  similaridade com esses padrões já aprendidos.
- Apelidos aprendem sozinhos: se o gestor pergunta por "Jô" e depois confirma
  que era a "Joana Ribeiro", "jo" passa a apontar para ela.
- As respostas usam a base histórica como referência (média da equipe, média do
  próprio colaborador nos meses anteriores), então a leitura melhora conforme a
  empresa acumula dados.

Este módulo não importa nada do app.py — recebe os modelos e funções de que
precisa em um "contexto" (injeção de dependência). Isso evita import circular e
deixa o motor testável isoladamente.
"""

import re
import unicodedata
from datetime import datetime, date, timedelta
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# Utilidades de texto
# ---------------------------------------------------------------------------
MESES = {
    "janeiro": 1, "jan": 1, "fevereiro": 2, "fev": 2, "marco": 3, "mar": 3,
    "abril": 4, "abr": 4, "maio": 5, "mai": 5, "junho": 6, "jun": 6,
    "julho": 7, "jul": 7, "agosto": 8, "ago": 8, "setembro": 9, "set": 9,
    "outubro": 10, "out": 10, "novembro": 11, "nov": 11, "dezembro": 12, "dez": 12,
}

MESES_NOME = [
    "janeiro", "fevereiro", "março", "abril", "maio", "junho",
    "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
]

DIAS_SEMANA = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
               "sexta-feira", "sábado", "domingo"]

# Palavras que nunca ajudam a identificar uma pessoa (evita que "quantas horas
# extras a ana tem" case o "a" com alguém chamado "A. Silva").
PALAVRAS_IGNORADAS = {
    "quantas", "quanto", "quantos", "horas", "hora", "extra", "extras", "hrs", "hr", "h",
    "o", "a", "os", "as", "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "colaborador", "colaboradora", "funcionario", "funcionaria", "empregado", "pessoa",
    "tem", "teve", "fez", "fizeram", "esta", "estao", "e", "eh", "foi", "ficou",
    "qual", "quais", "quem", "quando", "onde", "porque", "por", "que", "com", "sem",
    "me", "mostra", "mostre", "diga", "fala", "falar", "quero", "saber", "ver",
    "hoje", "ontem", "mes", "mês", "semana", "ano", "dia", "dias", "periodo", "ultimo",
    "ultima", "ultimos", "ultimas", "passado", "passada", "esse", "essa", "este", "esta",
    "atual", "total", "trabalhou", "trabalhadas", "trabalhou", "banco", "saldo",
    "atrasos", "atraso", "faltas", "falta", "ponto", "pontos", "registro", "registros",
    "por", "favor", "pra", "para", "meu", "minha", "meus", "minhas", "eu",
    "equipe", "time", "empresa", "todos", "todas", "geral", "pessoal", "galera",
    "funcionarios", "colaboradores", "ninguem", "alguem", "cada", "algum", "alguma",
    "nao", "sim", "ainda", "ja", "acumulou", "acumuladas", "tenho", "tive", "fiz",
}


def normalizar(texto):
    """Minúsculas, sem acentos, sem pontuação — base de toda a comparação."""
    if not texto:
        return ""
    texto = unicodedata.normalize("NFKD", str(texto))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.lower()
    texto = re.sub(r"[^a-z0-9\s:/\-]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def tokens_uteis(texto_normalizado):
    return [t for t in texto_normalizado.split() if len(t) > 1 and t not in PALAVRAS_IGNORADAS]


def similaridade(a, b):
    return SequenceMatcher(None, a, b).ratio()


def formatar_data(d):
    return d.strftime("%d/%m/%Y")


def nome_mes(d):
    return f"{MESES_NOME[d.month - 1]} de {d.year}"


# ---------------------------------------------------------------------------
# Período
# ---------------------------------------------------------------------------
class Periodo:
    def __init__(self, inicio, fim, rotulo, explicito=True):
        self.inicio = inicio
        self.fim = fim
        self.rotulo = rotulo
        self.explicito = explicito  # False = período assumido por padrão

    def contem(self, d):
        return self.inicio <= d <= self.fim

    @property
    def frase(self):
        """Rótulo pronto para entrar no meio de uma frase, com a preposição certa
        ("em agosto de 2026", "nos últimos 30 dias", "hoje", "na semana passada")."""
        r = self.rotulo
        if r in ("hoje", "ontem", "anteontem"):
            return r
        if r.startswith("últimos") or r.startswith("últimas"):
            return f"nos {r}"
        if r in ("esta semana",):
            return "nesta semana"
        if r == "semana passada":
            return "na semana passada"
        if r == "todo o período":
            return "em todo o período"
        return f"em {r}"

    def __repr__(self):
        return f"<Periodo {self.inicio}..{self.fim} ({self.rotulo})>"


def _primeiro_dia_mes(d):
    return d.replace(day=1)


def _ultimo_dia_mes(d):
    proximo = date(d.year + 1, 1, 1) if d.month == 12 else date(d.year, d.month + 1, 1)
    return proximo - timedelta(days=1)


def detectar_periodo(texto, hoje):
    """Extrai o intervalo de datas citado na pergunta. Se nada for citado,
    assume o mês corrente (marcado como não-explícito, para a resposta poder
    avisar qual período foi considerado)."""
    t = normalizar(texto)

    # Intervalo explícito: "de 01/07 a 15/07", "entre 01/07/2026 e 20/07/2026"
    m = re.search(r"(?:de|entre)\s+(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\s+(?:a|ate|e)\s+(\d{1,2}/\d{1,2}(?:/\d{2,4})?)", t)
    if m:
        d1 = _parse_data_br(m.group(1), hoje)
        d2 = _parse_data_br(m.group(2), hoje)
        if d1 and d2:
            if d1 > d2:
                d1, d2 = d2, d1
            return Periodo(d1, d2, f"{formatar_data(d1)} a {formatar_data(d2)}")

    # Data única: "no dia 12/07", "em 12/07/2026"
    m = re.search(r"\b(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b", t)
    if m:
        d = _parse_data_br(m.group(1), hoje)
        if d:
            return Periodo(d, d, formatar_data(d))

    # AAAA-MM (formato do filtro de mês da tela de horas extras)
    m = re.search(r"\b(20\d{2})-(\d{1,2})\b", t)
    if m:
        ano, mes = int(m.group(1)), int(m.group(2))
        if 1 <= mes <= 12:
            inicio = date(ano, mes, 1)
            return Periodo(inicio, _ultimo_dia_mes(inicio), nome_mes(inicio))

    if re.search(r"\bhoje\b", t):
        return Periodo(hoje, hoje, "hoje")
    if re.search(r"\bontem\b", t):
        ontem = hoje - timedelta(days=1)
        return Periodo(ontem, ontem, "ontem")
    if re.search(r"\banteontem\b", t):
        d = hoje - timedelta(days=2)
        return Periodo(d, d, "anteontem")

    m = re.search(r"ultim[oa]s?\s+(\d{1,3})\s+dias?", t)
    if m:
        dias = max(1, min(365, int(m.group(1))))
        return Periodo(hoje - timedelta(days=dias - 1), hoje, f"últimos {dias} dias")

    m = re.search(r"ultim[oa]s?\s+(\d{1,2})\s+(?:meses|mes)", t)
    if m:
        meses = max(1, min(24, int(m.group(1))))
        inicio = _primeiro_dia_mes(hoje)
        for _ in range(meses - 1):
            inicio = _primeiro_dia_mes(inicio - timedelta(days=1))
        return Periodo(inicio, hoje, f"últimos {meses} meses")

    if re.search(r"semana passada|semana anterior", t):
        inicio_semana_atual = hoje - timedelta(days=hoje.weekday())
        inicio = inicio_semana_atual - timedelta(days=7)
        return Periodo(inicio, inicio + timedelta(days=6), "semana passada")

    if re.search(r"(esta|essa|este|esse|nesta|nessa)\s+semana|semana atual|na semana", t):
        inicio = hoje - timedelta(days=hoje.weekday())
        return Periodo(inicio, hoje, "esta semana")

    if re.search(r"mes passado|mes anterior", t):
        ultimo_dia_anterior = _primeiro_dia_mes(hoje) - timedelta(days=1)
        return Periodo(_primeiro_dia_mes(ultimo_dia_anterior), ultimo_dia_anterior,
                       nome_mes(ultimo_dia_anterior))

    if re.search(r"ano passado", t):
        return Periodo(date(hoje.year - 1, 1, 1), date(hoje.year - 1, 12, 31), f"{hoje.year - 1}")

    if re.search(r"(neste|nesse|este|esse)\s+ano|ano atual|no ano", t):
        return Periodo(date(hoje.year, 1, 1), hoje, f"{hoje.year}")

    # Nome de mês ("em julho", "julho de 2025")
    for nome, numero in MESES.items():
        if re.search(rf"\b{nome}\b", t):
            m_ano = re.search(rf"\b{nome}\b\s*(?:de\s*)?(20\d{{2}})", t)
            ano = int(m_ano.group(1)) if m_ano else hoje.year
            inicio = date(ano, numero, 1)
            if not m_ano and inicio > hoje:  # "em dezembro" no meio do ano = dezembro passado
                inicio = date(ano - 1, numero, 1)
            fim = min(_ultimo_dia_mes(inicio), hoje) if inicio.year == hoje.year and inicio.month == hoje.month else _ultimo_dia_mes(inicio)
            return Periodo(inicio, fim, nome_mes(inicio))

    if re.search(r"(este|esse|neste|nesse)\s+mes|mes atual|no mes", t):
        return Periodo(_primeiro_dia_mes(hoje), hoje, nome_mes(hoje))

    if re.search(r"sempre|desde o inicio|historico completo|todo o periodo|total geral", t):
        return Periodo(date(2000, 1, 1), hoje, "todo o período")

    # Padrão: mês corrente.
    return Periodo(_primeiro_dia_mes(hoje), hoje, nome_mes(hoje), explicito=False)


def _parse_data_br(texto, hoje):
    partes = texto.split("/")
    try:
        dia = int(partes[0])
        mes = int(partes[1])
        if len(partes) > 2:
            ano = int(partes[2])
            if ano < 100:
                ano += 2000
        else:
            ano = hoje.year
        return date(ano, mes, dia)
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Intenções
# ---------------------------------------------------------------------------
# Cada intenção tem frases-chave; quanto mais longa a frase que casar, mais forte
# o sinal (evita que "hora" solto vença "hora extra").
INTENCOES = {
    "horas_extras": [
        "horas extras", "hora extra", "horas extra", "hrs extras", "hr extra",
        "extras", "banco de horas", "saldo de horas", "he acumulada", "sobrejornada",
    ],
    "horas_trabalhadas": [
        "horas trabalhadas", "quantas horas trabalhou", "carga horaria", "total de horas",
        "horas no mes", "tempo trabalhado", "quantas horas fez", "jornada cumprida",
    ],
    "atrasos": [
        "atrasos", "atraso", "atrasou", "chegou tarde", "chegou atrasado", "quem se atrasa",
        "pontualidade",
    ],
    "faltas": [
        "faltas", "faltou", "ausencias", "ausencia", "ausente", "nao bateu ponto",
        "nao veio", "nao compareceu", "dias sem registro",
    ],
    "presentes_hoje": [
        "quem esta presente", "quem bateu ponto hoje", "presentes hoje", "quem trabalhou hoje",
        "quem ja bateu", "presenca hoje", "quem esta trabalhando",
    ],
    "ultimo_ponto": [
        "ultimo ponto", "ultima batida", "ultimo registro", "que horas chegou",
        "que horas saiu", "quando bateu", "horario de entrada de", "ja bateu ponto",
    ],
    "pendencias": [
        "pendencias", "pendencia", "solicitacoes", "solicitacao de ajuste", "ajustes pendentes",
        "aguardando aprovacao", "para aprovar", "jornada excepcional", "aprovacoes",
    ],
    "ranking": [
        "quem tem mais", "ranking", "top", "maior numero", "quem fez mais",
        "quem mais fez", "lista de horas extras", "todos os colaboradores",
        "resumo da equipe", "por colaborador",
    ],
    "registros_incompletos": [
        "incompleto", "incompletos", "esqueceu de bater", "sem saida", "sem entrada",
        "ponto aberto", "faltando batida",
    ],
    "resumo_dia": [
        "espelho de ponto", "resumo do dia", "batidas do dia", "registros do dia",
        "como foi o dia", "detalhe do dia", "marcacoes do dia",
    ],
    "configuracao": [
        "jornada padrao", "horario padrao", "configuracao", "qual o horario de entrada da empresa",
        "quantas horas por dia", "meta de horas",
    ],
    "custo_extras": [
        "custo", "quanto vou pagar", "valor das horas extras", "impacto na folha",
    ],
    "ajuda": [
        "o que voce faz", "o que voce sabe", "como usar", "ajuda", "me ajuda",
        "quais perguntas", "exemplos de pergunta", "voce consegue",
    ],
}


PALAVRAS_DE_INTENCAO = {
    palavra
    for frases in INTENCOES.values()
    for frase in frases
    for palavra in frase.split()
}


def termos_para_apelido(pergunta, nome_colaborador):
    """Quando o gestor corrige o assistente dizendo de quem era a pergunta, estes
    são os termos que passam a apontar para essa pessoa. Filtra tudo que é
    vocabulário do sistema (palavras de período, de intenção, números) para não
    aprender lixo como "extras" = Fulano."""
    nome_tokens = set(normalizar(nome_colaborador).split())
    termos = []
    for token in tokens_uteis(normalizar(pergunta)):
        if token in nome_tokens or token in PALAVRAS_DE_INTENCAO or token in MESES:
            continue
        if token.isdigit() or "/" in token or "-" in token or ":" in token:
            continue
        if len(token) < 2 or len(token) > 30:
            continue
        termos.append(token)
    return termos[:3]


# Regras por expressão regular: pegam variações que a comparação por frase
# inteira não pega — plural/singular, verbo conjugado ("trabalhei", "trabalhou"),
# abreviações ("hrs") e erros de digitação comuns ("exxtras", "estras").
REGRAS_REGEX = [
    ("horas_extras", r"\b(?:h|hs|hr|hrs|hora|horas)\s+e?x+t?ras?\b", 30),
    ("horas_extras", r"\bex+t?ras?\b", 24),
    ("horas_extras", r"banco de horas|saldo de horas|sobrejornada", 30),
    ("horas_trabalhadas", r"\btrabalh\w+", 20),
    ("horas_trabalhadas", r"\bquant[ao]s?\s+(?:h|hs|hr|hrs|hora|horas)\b", 18),
    ("horas_trabalhadas", r"carga hora|jornada cumprida|tempo trabalhado", 25),
    ("atrasos", r"\batras\w+|chegou (?:tarde|atrasad)|pontualidade", 25),
    ("faltas", r"\bfalt\w+|\bausen\w+|dias? sem registro|nao comparec\w+|nao veio", 25),
    ("presentes_hoje", r"quem\s+(?:ja\s+|ainda\s+)?(?:nao\s+)?bateu", 40),
    ("presentes_hoje", r"quem (?:esta|estao) (?:presente|trabalhando)|presentes hoje|presenca hoje", 40),
    ("registros_incompletos", r"\bincomplet\w+|esqueceu de bater|sem (?:saida|entrada)|ponto aberto", 30),
    ("pendencias", r"\bpendenc\w+|aguardando (?:aprovacao|revisao)|para aprovar|preciso aprovar", 30),
    ("pendencias", r"solicitac\w+|ajustes? pendente|jornada excepcional", 26),
    ("ranking", r"quem tem mais|quem fez mais|quem mais|ranking|top \d|maior numero", 32),
    ("ranking", r"resumo (?:da|do) (?:equipe|time)|por colaborador|de toda a equipe", 30),
    ("ultimo_ponto", r"ultim[oa] (?:ponto|batida|registro|marcacao)|que horas (?:chegou|saiu|entrou)", 32),
    ("resumo_dia", r"espelho de ponto|resumo do dia|batidas do dia|como foi o dia|marcacoes do dia", 32),
    ("configuracao", r"jornada padrao|horario padrao|meta de horas|horario da empresa", 30),
    ("ajuda", r"o que voce (?:faz|sabe|consegue)|como (?:te )?us[ao]|me ajuda|^ajuda$|exemplos? de pergunta", 30),
    ("resumo_colaborador", r"como (?:esta|vai|anda)\b|situacao (?:de|do|da)\b|resumo (?:de|do|da)\b", 16),
]


def _pontuar_intencoes(texto_normalizado):
    pontuacoes = {}
    for intencao, frases in INTENCOES.items():
        melhor = 0
        for frase in frases:
            if frase in texto_normalizado:
                melhor = max(melhor, len(frase))
        if melhor:
            pontuacoes[intencao] = melhor

    for intencao, padrao, peso in REGRAS_REGEX:
        if re.search(padrao, texto_normalizado):
            pontuacoes[intencao] = max(pontuacoes.get(intencao, 0), peso)

    # "quantas horas EXTRAS ..." é sobre extra, não sobre carga horária.
    if "horas_extras" in pontuacoes and "horas_trabalhadas" in pontuacoes:
        if re.search(r"e?x+t?ras?\b", texto_normalizado):
            pontuacoes["horas_trabalhadas"] = 0

    return {k: v for k, v in pontuacoes.items() if v > 0}


# ---------------------------------------------------------------------------
# Assistente
# ---------------------------------------------------------------------------
class Assistente:
    """Motor de perguntas e respostas. Recebe um `ctx` (SimpleNamespace) com os
    modelos e funções do app — ver `criar_assistente` em app.py."""

    LIMITE_LINHAS = 10  # nunca devolve mais de 10 linhas de detalhe (mesma regra das telas)

    # Intenções que fazem sentido tanto para uma pessoa quanto para a equipe —
    # nelas, um nome não reconhecido deve virar pergunta de volta, não relatório
    # geral.
    INTENCOES_PESSOAIS = {
        "horas_extras", "horas_trabalhadas", "atrasos", "faltas", "registros_incompletos",
        "resumo_dia", "resumo_colaborador", "ultimo_ponto", "custo_extras",
    }

    def __init__(self, ctx):
        self.ctx = ctx

    # -- acesso a dados -----------------------------------------------------
    def _colaboradores(self, incluir_inativos=True):
        q = self.ctx.Colaborador.query.filter_by(is_gestor=False)
        if not incluir_inativos:
            q = q.filter_by(ativo=True)
        return q.order_by(self.ctx.Colaborador.nome).all()

    def _registros_periodo(self, periodo, colaborador_id=None):
        RegistroPonto = self.ctx.RegistroPonto
        inicio_utc = datetime.combine(periodo.inicio, datetime.min.time()) - timedelta(hours=6)
        fim_utc = datetime.combine(periodo.fim, datetime.max.time()) + timedelta(hours=6)
        q = RegistroPonto.query.filter(
            RegistroPonto.data_hora >= inicio_utc, RegistroPonto.data_hora <= fim_utc
        )
        if colaborador_id:
            q = q.filter_by(colaborador_id=colaborador_id)
        registros = q.order_by(RegistroPonto.data_hora.asc()).all()
        return [r for r in registros if periodo.contem(self.ctx.para_brasilia(r.data_hora).date())]

    def _status_excepcional(self, colaborador_id, dia):
        """Lê (sem criar nada) o status da aprovação de uma jornada excepcional."""
        try:
            aprovacao = self.ctx.AprovacaoJornadaExcepcional.query.filter_by(
                colaborador_id=colaborador_id, data_referencia=dia
            ).first()
            return aprovacao.status if aprovacao else "pendente"
        except Exception:
            return "pendente"

    def _resumo(self, colaborador, periodo):
        """Números do colaborador no período: horas, extras, atrasos, faltas,
        dias incompletos e jornadas retidas para aprovação."""
        ctx = self.ctx
        config = ctx.obter_configuracao()
        hoje = ctx.agora_brasilia().date()

        registros = self._registros_periodo(periodo, colaborador.id)
        dias = [d for d in ctx.montar_resumo_diario(registros) if periodo.contem(d["data"])]

        total_horas = 0.0
        total_extra = 0.0
        extra_retida = 0.0
        atrasos = []
        incompletos = []
        retidos = []

        for dia in dias:
            total_horas += dia["total_horas_decimal"]
            extra = ctx.calcular_horas_extras_dia(
                dia["total_horas_decimal"], config.horario_entrada, config.horario_saida,
                eh_domingo_flag=ctx.eh_domingo(dia["data"]),
            )
            dia["horas_extras"] = extra
            if dia.get("excepcional"):
                status = self._status_excepcional(colaborador.id, dia["data"])
                dia["status_excepcional"] = status
                if status != "aprovado":
                    extra_retida += extra
                    dia["horas_extras"] = 0.0
                    retidos.append(dia)
                    extra = 0.0
            total_extra += extra

            if dia["incompleto"]:
                incompletos.append(dia)
            if dia["batidas"] and not ctx.eh_domingo(dia["data"]):
                primeira = min(dia["batidas"])
                if primeira.strftime("%H:%M") > config.horario_entrada:
                    atrasos.append((dia["data"], primeira))

        dias_com_registro = {d["data"] for d in dias if d["batidas"]}
        faltas = []
        cursor = periodo.inicio
        limite = min(periodo.fim, hoje)
        while cursor <= limite:
            if not ctx.eh_domingo(cursor) and cursor not in dias_com_registro:
                faltas.append(cursor)
            cursor += timedelta(days=1)

        return {
            "colaborador": colaborador,
            "dias": dias,
            "total_horas": round(total_horas, 2),
            "total_extra": round(total_extra, 2),
            "extra_retida": round(extra_retida, 2),
            "dias_trabalhados": len(dias_com_registro),
            "atrasos": atrasos,
            "faltas": faltas,
            "incompletos": incompletos,
            "retidos": retidos,
            "config": config,
        }

    # -- identificação de pessoas ------------------------------------------
    def _apelidos_aprendidos(self):
        try:
            return self.ctx.AssistenteApelido.query.all()
        except Exception:
            return []

    def detectar_colaboradores(self, texto, limite_confianca=0.72):
        """Devolve [(colaborador, confiança)] ordenado do mais provável para o
        menos. Casa nome completo, primeiro nome, e-mail, apelidos aprendidos e
        variações com erro de digitação."""
        t = normalizar(texto)
        palavras = tokens_uteis(t)
        colaboradores = self._colaboradores()

        apelidos = {}
        for a in self._apelidos_aprendidos():
            apelidos.setdefault(a.colaborador_id, []).append(normalizar(a.apelido))

        resultados = []
        for c in colaboradores:
            nome_norm = normalizar(c.nome)
            partes = [p for p in nome_norm.split() if len(p) > 1]
            email_local = normalizar((c.email or "").split("@")[0].replace(".", " "))
            melhor = 0.0

            if nome_norm and nome_norm in t:
                melhor = 1.0
            if email_local and email_local in t:
                melhor = max(melhor, 0.97)
            for apelido in apelidos.get(c.id, []):
                if apelido and re.search(rf"\b{re.escape(apelido)}\b", t):
                    melhor = max(melhor, 0.95)

            for parte in partes:
                if re.search(rf"\b{re.escape(parte)}\b", t):
                    # Nome próprio inteiro no texto: forte, ainda mais se for o primeiro nome.
                    melhor = max(melhor, 0.93 if parte == partes[0] else 0.85)
                for palavra in palavras:
                    r = similaridade(parte, palavra)
                    if r >= 0.8 and abs(len(parte) - len(palavra)) <= 3:
                        melhor = max(melhor, r * 0.9)

            if palavras:
                melhor = max(melhor, similaridade(nome_norm, " ".join(palavras)) * 0.8)

            if melhor >= limite_confianca:
                resultados.append((c, round(melhor, 3)))

        resultados.sort(key=lambda item: (-item[1], item[0].nome))
        return resultados

    # -- classificação ------------------------------------------------------
    def _padroes_aprendidos(self):
        try:
            return self.ctx.AssistentePadrao.query.filter(
                self.ctx.AssistentePadrao.peso > 0
            ).all()
        except Exception:
            return []

    def detectar_intencao(self, texto):
        """Devolve (intencao, confianca, origem)."""
        t = normalizar(texto)
        pontuacoes = _pontuar_intencoes(t)

        if pontuacoes:
            intencao = max(pontuacoes, key=pontuacoes.get)
            confianca = min(0.99, 0.6 + pontuacoes[intencao] / 40)
            return intencao, round(confianca, 2), "regra"

        # Nada bateu nas regras fixas: tenta o que já foi aprendido com o uso.
        melhor, melhor_r = None, 0.0
        for padrao in self._padroes_aprendidos():
            r = similaridade(t, padrao.padrao)
            if r > melhor_r:
                melhor, melhor_r = padrao, r
        if melhor and melhor_r >= 0.62:
            return melhor.intencao, round(melhor_r, 2), "aprendido"

        # Perguntas curtas com nome de gente e nada mais ("e a maria?") herdam a
        # intenção mais comum do histórico daquele usuário.
        if self.detectar_colaboradores(texto):
            return "resumo_colaborador", 0.5, "inferido"

        return "desconhecida", 0.0, "nenhuma"

    # -- ponto de entrada ---------------------------------------------------
    def responder(self, pergunta, usuario, colaborador_forcado=None):
        """Responde a pergunta respeitando o escopo do usuário.

        Gestor: pode perguntar sobre qualquer colaborador e sobre a equipe.
        Colaborador: só enxerga os próprios dados — qualquer pergunta sobre
        outra pessoa (ou sobre a equipe) é recusada de forma educada.

        `colaborador_forcado` é usado quando o gestor CORRIGE o assistente
        ("era sobre a Joana"): a pergunta é refeita já sabendo de quem se trata."""
        ctx = self.ctx
        hoje = ctx.agora_brasilia().date()
        pergunta = (pergunta or "").strip()
        if not pergunta:
            return self._resposta("Pode escrever sua pergunta que eu respondo com os dados do ponto.",
                                  intencao="vazia")

        eh_gestor = bool(getattr(usuario, "is_gestor", False))
        periodo = detectar_periodo(pergunta, hoje)
        intencao, confianca, origem = self.detectar_intencao(pergunta)
        candidatos = self.detectar_colaboradores(pergunta)

        if not eh_gestor:
            # Colaborador citando OUTRA pessoa: recusa antes de qualquer cálculo.
            outros = [c for c, _ in candidatos if c.id != usuario.id]
            if outros:
                return self._resposta(
                    f"Só posso mostrar os seus próprios dados de ponto — informações de "
                    f"{outros[0].nome} são visíveis apenas para o gestor.",
                    intencao="fora_do_escopo",
                    sugestoes=["Quantas horas extras eu tenho este mês?",
                               "Quantas horas trabalhei este mês?"],
                )

        if colaborador_forcado is not None and eh_gestor:
            candidatos = [(colaborador_forcado, 1.0)]
            if intencao in ("desconhecida", "sem_colaborador", "ambiguo"):
                intencao, confianca, origem = "resumo_colaborador", 0.9, "correcao"

        if not eh_gestor:
            # Escopo do colaborador: sempre ele mesmo, e sem intenções de equipe.
            if intencao in ("ranking", "presentes_hoje", "pendencias", "custo_extras"):
                if intencao == "pendencias":
                    return self._minhas_solicitacoes(usuario, periodo)
                return self._resposta(
                    "Só consigo mostrar os seus próprios dados de ponto. Para informações da "
                    "equipe, fale com o gestor.",
                    intencao="fora_do_escopo",
                )
            alvo = usuario
            candidatos = [(usuario, 1.0)]
        else:
            alvo = candidatos[0][0] if candidatos else None

        # A pergunta cita alguém que o sistema não conhece? (ex.: um apelido novo).
        # Nesse caso é melhor perguntar de quem se trata do que responder sobre a
        # equipe inteira como se a pessoa não tivesse sido citada.
        if eh_gestor and alvo is None and intencao in self.INTENCOES_PESSOAIS:
            if termos_para_apelido(pergunta, ""):
                return self._exige_colaborador(None, periodo, "Quantas horas extras FULANO tem este mês?")

        # Ambiguidade: dois nomes parecidos com confiança quase igual.
        if eh_gestor and len(candidatos) > 1 and candidatos[0][1] - candidatos[1][1] < 0.06:
            return self._resposta(
                "Encontrei mais de um colaborador que combina com esse nome. De quem você quer saber?",
                intencao="ambiguo",
                opcoes=[c for c, _ in candidatos[:5]],
                periodo=periodo,
            )

        despachante = {
            "horas_extras": self._horas_extras,
            "horas_trabalhadas": self._horas_trabalhadas,
            "atrasos": self._atrasos,
            "faltas": self._faltas,
            "registros_incompletos": self._incompletos,
            "resumo_dia": self._resumo_dia,
            "resumo_colaborador": self._resumo_colaborador,
            "ultimo_ponto": self._ultimo_ponto,
            "presentes_hoje": self._presentes_hoje,
            "pendencias": self._pendencias,
            "ranking": self._ranking,
            "configuracao": self._configuracao,
            "custo_extras": self._custo_extras,
            "ajuda": self._ajuda,
        }

        funcao = despachante.get(intencao)
        if funcao is None:
            return self._nao_entendi(pergunta, eh_gestor)

        try:
            resposta = funcao(alvo=alvo, periodo=periodo, usuario=usuario, eh_gestor=eh_gestor,
                              pergunta=pergunta)
        except Exception as ex:  # nunca derruba o chat por causa de um dado estranho
            print(f"[assistente] falha ao responder '{pergunta}': {ex}")
            return self._resposta(
                "Tive um problema para calcular essa resposta agora. Tente reformular a pergunta "
                "ou consultar a tela correspondente.",
                intencao="erro",
            )

        resposta["intencao"] = intencao
        resposta["confianca"] = confianca
        resposta["origem_intencao"] = origem
        resposta.setdefault("periodo", periodo.rotulo)
        return resposta

    # -- montagem de resposta ----------------------------------------------
    def _resposta(self, texto, intencao=None, detalhes=None, colunas=None, sugestoes=None,
                  periodo=None, destaque=None, opcoes=None):
        return {
            "resposta": texto,
            "intencao": intencao,
            # A regra de "no máximo 10 registros por vez" vale também aqui: o chat
            # nunca despeja uma tabela gigante — mostra 10 e diz quantos existem.
            "detalhes": (detalhes or [])[: self.LIMITE_LINHAS],
            "detalhes_total": len(detalhes or []),
            "colunas": colunas or [],
            "sugestoes": sugestoes or [],
            "periodo": periodo.rotulo if isinstance(periodo, Periodo) else periodo,
            "destaque": destaque,
            # Botões "era sobre quem?" — clicar neles ensina o assistente.
            "opcoes": [{"id": c.id, "nome": c.nome} for c in (opcoes or [])],
        }

    def _eh_proprio(self, alvo, usuario):
        """A pergunta é do colaborador sobre ele mesmo? (muda o tratamento para
        a 2ª pessoa: "Você tem 3h de hora extra" em vez de "Fulano tem...")."""
        return (alvo is not None and usuario is not None
                and getattr(usuario, "id", None) == alvo.id
                and not getattr(usuario, "is_gestor", False))

    def _tratar(self, alvo, usuario):
        return "Você" if self._eh_proprio(alvo, usuario) else f"**{alvo.nome}**"

    def _exige_colaborador(self, alvo, periodo, exemplo):
        if alvo is not None:
            return None
        colaboradores = self._colaboradores(incluir_inativos=False)
        nomes = [c.nome.split()[0] for c in colaboradores[:3]]
        sugestoes = [exemplo.replace("FULANO", n) for n in nomes] or [exemplo.replace("FULANO", "Maria")]
        return self._resposta(
            "Não identifiquei de qual colaborador você está falando. Clique no nome abaixo — eu "
            "aprendo esse apelido e acerto da próxima vez. Ou pergunte sobre a equipe toda "
            "(ex.: \"quem tem mais horas extras este mês?\").",
            intencao="sem_colaborador",
            sugestoes=sugestoes,
            opcoes=colaboradores[:8],
            periodo=periodo,
        )

    # -- respostas por intenção --------------------------------------------
    def _horas_extras(self, alvo, periodo, eh_gestor, usuario=None, **kwargs):
        ctx = self.ctx
        if alvo is None:
            return self._ranking(alvo=None, periodo=periodo, eh_gestor=eh_gestor, usuario=usuario, **kwargs)

        r = self._resumo(alvo, periodo)
        extra_txt = ctx.formatar_horas(r["total_extra"])

        linhas = []
        for dia in sorted([d for d in r["dias"] if d["horas_extras"] > 0 or d in r["retidos"]],
                          key=lambda d: d["data"], reverse=True):
            linhas.append([
                formatar_data(dia["data"]),
                dia["total_horas"],
                ctx.formatar_horas(dia["horas_extras"]) if dia["horas_extras"] else "retida",
            ])

        texto = (f"{self._tratar(alvo, usuario)} tem **{extra_txt}** de hora extra {periodo.frase}"
                 f" ({r['dias_trabalhados']} dia(s) com registro, {ctx.formatar_horas(r['total_horas'])} trabalhadas).")

        if r["extra_retida"] > 0:
            texto += (f"\n\n⚠️ Além disso, **{ctx.formatar_horas(r['extra_retida'])}** estão retidas: "
                      f"{len(r['retidos'])} jornada(s) acima do limite aguardando sua aprovação em "
                      f"*Ajustes*. Só entram no total depois de aprovadas.")
        if r["incompletos"]:
            texto += (f"\n\n📌 {len(r['incompletos'])} dia(s) estão incompletos (falta bater um ponto), "
                      f"então a hora extra deles ainda não foi contada.")
        if not periodo.explicito:
            texto += f"\n\n_Considerei {periodo.rotulo}. Dá para pedir outro período, ex.: \"mês passado\"._"

        # Comparação com o histórico do próprio colaborador — leitura que só faz
        # sentido depois que a empresa acumula dados.
        media = self._media_extra_meses_anteriores(alvo, periodo)
        if media is not None and r["total_extra"] > 0:
            if r["total_extra"] > media * 1.25:
                texto += f"\n\n📈 Está acima da média dos meses anteriores desse colaborador ({ctx.formatar_horas(media)})."
            elif r["total_extra"] < media * 0.75:
                texto += f"\n\n📉 Está abaixo da média dos meses anteriores desse colaborador ({ctx.formatar_horas(media)})."

        return self._resposta(
            texto,
            detalhes=linhas,
            colunas=["Dia", "Trabalhado", "Hora extra"],
            periodo=periodo,
            destaque=extra_txt,
            sugestoes=[
                f"Quantas horas {alvo.nome.split()[0]} trabalhou {periodo.frase}?",
                f"{alvo.nome.split()[0]} teve atrasos {periodo.frase}?",
                "Quem tem mais horas extras este mês?",
            ],
        )

    def _media_extra_meses_anteriores(self, colaborador, periodo, meses=3):
        """Média de hora extra do colaborador nos meses anteriores ao período
        perguntado (None se não houver histórico suficiente)."""
        try:
            referencia = _primeiro_dia_mes(periodo.inicio)
            totais = []
            for _ in range(meses):
                fim = referencia - timedelta(days=1)
                inicio = _primeiro_dia_mes(fim)
                r = self._resumo(colaborador, Periodo(inicio, fim, nome_mes(inicio)))
                if r["dias_trabalhados"]:
                    totais.append(r["total_extra"])
                referencia = inicio
            return round(sum(totais) / len(totais), 2) if totais else None
        except Exception:
            return None

    def _horas_trabalhadas(self, alvo, periodo, eh_gestor, usuario=None, **kwargs):
        ctx = self.ctx
        faltou = self._exige_colaborador(alvo, periodo, "Quantas horas FULANO trabalhou este mês?")
        if faltou:
            return faltou

        r = self._resumo(alvo, periodo)
        media_dia = (r["total_horas"] / r["dias_trabalhados"]) if r["dias_trabalhados"] else 0
        texto = (f"{self._tratar(alvo, usuario)} trabalhou **{ctx.formatar_horas(r['total_horas'])}** "
                 f"{periodo.frase}, em {r['dias_trabalhados']} dia(s) com registro "
                 f"(média de {ctx.formatar_horas(media_dia)} por dia). "
                 f"Hora extra no período: {ctx.formatar_horas(r['total_extra'])}.")
        if r["incompletos"]:
            texto += f"\n\n📌 {len(r['incompletos'])} dia(s) incompletos podem estar reduzindo esse total."

        linhas = [[formatar_data(d["data"]), d["total_horas"],
                   ctx.formatar_horas(d["horas_extras"]) if d["horas_extras"] else "—"]
                  for d in sorted(r["dias"], key=lambda d: d["data"], reverse=True) if d["batidas"]]
        return self._resposta(texto, detalhes=linhas, colunas=["Dia", "Trabalhado", "Extra"],
                              periodo=periodo, destaque=ctx.formatar_horas(r["total_horas"]))

    def _atrasos(self, alvo, periodo, eh_gestor, usuario=None, **kwargs):
        ctx = self.ctx
        config = ctx.obter_configuracao()
        if alvo is None:
            linhas = []
            for c in self._colaboradores(incluir_inativos=False):
                r = self._resumo(c, periodo)
                if r["atrasos"]:
                    linhas.append([c.nome, len(r["atrasos"]),
                                   formatar_data(max(a[0] for a in r["atrasos"]))])
            linhas.sort(key=lambda l: -l[1])
            if not linhas:
                return self._resposta(f"Ninguém chegou depois de {config.horario_entrada} {periodo.frase}. 👏",
                                      periodo=periodo)
            texto = (f"Atrasos {periodo.frase} (entrada padrão {config.horario_entrada}) — "
                     f"{len(linhas)} colaborador(es):")
            return self._resposta(texto, detalhes=linhas,
                                  colunas=["Colaborador", "Dias com atraso", "Último"], periodo=periodo)

        r = self._resumo(alvo, periodo)
        if not r["atrasos"]:
            return self._resposta(
                f"{self._tratar(alvo, usuario)} não teve atrasos {periodo.frase} (entrada padrão {config.horario_entrada}).",
                periodo=periodo, destaque="0 atrasos")
        linhas = [[formatar_data(d), h.strftime("%H:%M"), config.horario_entrada]
                  for d, h in sorted(r["atrasos"], reverse=True)]
        return self._resposta(
            f"{self._tratar(alvo, usuario)} teve **{len(r['atrasos'])} atraso(s)** {periodo.frase} "
            f"(primeira batida depois de {config.horario_entrada}).",
            detalhes=linhas, colunas=["Dia", "Chegou", "Previsto"], periodo=periodo,
            destaque=f"{len(r['atrasos'])} atraso(s)")

    def _faltas(self, alvo, periodo, eh_gestor, usuario=None, **kwargs):
        if alvo is None:
            linhas = []
            for c in self._colaboradores(incluir_inativos=False):
                r = self._resumo(c, periodo)
                if r["faltas"]:
                    linhas.append([c.nome, len(r["faltas"]), formatar_data(max(r["faltas"]))])
            linhas.sort(key=lambda l: -l[1])
            if not linhas:
                return self._resposta(f"Nenhum dia útil sem registro {periodo.frase}.", periodo=periodo)
            return self._resposta(
                f"Dias úteis sem nenhuma batida {periodo.frase} (domingos não contam):",
                detalhes=linhas, colunas=["Colaborador", "Dias sem registro", "Mais recente"],
                periodo=periodo)

        r = self._resumo(alvo, periodo)
        if not r["faltas"]:
            return self._resposta(f"{self._tratar(alvo, usuario)} bateu ponto em todos os dias úteis {periodo.frase}.",
                                  periodo=periodo, destaque="0 dias sem registro")
        linhas = [[formatar_data(d), DIAS_SEMANA[d.weekday()]] for d in sorted(r["faltas"], reverse=True)]
        return self._resposta(
            f"{self._tratar(alvo, usuario)} ficou **{len(r['faltas'])} dia(s) úteis sem nenhuma batida** "
            f"{periodo.frase}. Isso pode ser falta, férias ou atestado — o sistema só sabe que não houve registro.",
            detalhes=linhas, colunas=["Dia", "Dia da semana"], periodo=periodo,
            destaque=f"{len(r['faltas'])} dia(s)")

    def _incompletos(self, alvo, periodo, eh_gestor, usuario=None, **kwargs):
        if alvo is None:
            linhas = []
            for c in self._colaboradores(incluir_inativos=False):
                r = self._resumo(c, periodo)
                if r["incompletos"]:
                    linhas.append([c.nome, len(r["incompletos"]),
                                   formatar_data(max(d["data"] for d in r["incompletos"]))])
            linhas.sort(key=lambda l: -l[1])
            if not linhas:
                return self._resposta(f"Nenhum dia incompleto {periodo.frase}. Todos os pares entrada/saída fecharam.",
                                      periodo=periodo)
            return self._resposta(
                f"Dias com batida faltando {periodo.frase} (entrada sem saída, ou o contrário):",
                detalhes=linhas, colunas=["Colaborador", "Dias incompletos", "Mais recente"],
                periodo=periodo)

        r = self._resumo(alvo, periodo)
        if not r["incompletos"]:
            return self._resposta(f"{self._tratar(alvo, usuario)} não tem dias incompletos {periodo.frase}.",
                                  periodo=periodo, destaque="tudo fechado")
        linhas = [[formatar_data(d["data"]), len(d["batidas"]), d["total_horas"]]
                  for d in sorted(r["incompletos"], key=lambda d: d["data"], reverse=True)]
        return self._resposta(
            f"{self._tratar(alvo, usuario)} tem **{len(r['incompletos'])} dia(s) incompletos** {periodo.frase}. "
            f"Esses dias precisam de uma solicitação de ajuste para o cálculo ficar correto.",
            detalhes=linhas, colunas=["Dia", "Batidas", "Total apurado"], periodo=periodo)

    def _resumo_dia(self, alvo, periodo, eh_gestor, usuario=None, **kwargs):
        ctx = self.ctx
        faltou = self._exige_colaborador(alvo, periodo, "Como foi o dia de FULANO ontem?")
        if faltou:
            return faltou
        dia = periodo.fim
        registros = ctx.registros_do_dia(alvo.id, dia)
        if not registros:
            return self._resposta(f"{self._tratar(alvo, usuario)} não tem nenhuma batida em {formatar_data(dia)}.",
                                  periodo=periodo)
        linhas = [[ctx.para_brasilia(r.data_hora).strftime("%H:%M:%S"),
                   "Entrada" if (r.tipo or "") == "entrada" else "Saída",
                   (r.endereco or "—")[:60]] for r in registros]
        resumo = ctx.montar_resumo_diario(registros)
        total = resumo[0]["total_horas"] if resumo else "00:00"
        return self._resposta(
            f"Batidas de **{alvo.nome}** em {formatar_data(dia)} — {len(registros)} marcação(ões), "
            f"total apurado de {total}.",
            detalhes=linhas, colunas=["Hora", "Tipo", "Local"], periodo=periodo, destaque=total)

    def _resumo_colaborador(self, alvo, periodo, eh_gestor, usuario=None, **kwargs):
        ctx = self.ctx
        faltou = self._exige_colaborador(alvo, periodo, "Como está FULANO este mês?")
        if faltou:
            return faltou
        r = self._resumo(alvo, periodo)
        texto = (f"{self._tratar(alvo, usuario)} {periodo.frase}:\n\n"
                 f"- Horas trabalhadas: **{ctx.formatar_horas(r['total_horas'])}**\n"
                 f"- Horas extras: **{ctx.formatar_horas(r['total_extra'])}**"
                 + (f" (+{ctx.formatar_horas(r['extra_retida'])} aguardando aprovação)" if r["extra_retida"] else "") + "\n"
                 f"- Dias com registro: **{r['dias_trabalhados']}**\n"
                 f"- Atrasos: **{len(r['atrasos'])}** · Dias úteis sem registro: **{len(r['faltas'])}** · "
                 f"Dias incompletos: **{len(r['incompletos'])}**")
        linhas = [[formatar_data(d["data"]), d["total_horas"],
                   ctx.formatar_horas(d["horas_extras"]) if d["horas_extras"] else "—"]
                  for d in sorted(r["dias"], key=lambda d: d["data"], reverse=True) if d["batidas"]]
        return self._resposta(texto, detalhes=linhas, colunas=["Dia", "Trabalhado", "Extra"],
                              periodo=periodo, destaque=ctx.formatar_horas(r["total_extra"]))

    def _ultimo_ponto(self, alvo, periodo, eh_gestor, usuario=None, **kwargs):
        ctx = self.ctx
        RegistroPonto = ctx.RegistroPonto
        if alvo is None and eh_gestor:
            registros = (RegistroPonto.query.order_by(RegistroPonto.data_hora.desc())
                         .limit(self.LIMITE_LINHAS).all())
            if not registros:
                return self._resposta("Ainda não há nenhuma batida registrada no sistema.")
            linhas = [[r.colaborador.nome, ctx.para_brasilia(r.data_hora).strftime("%d/%m %H:%M"),
                       "Entrada" if (r.tipo or "") == "entrada" else "Saída"] for r in registros]
            return self._resposta("Últimas batidas registradas:", detalhes=linhas,
                                  colunas=["Colaborador", "Quando", "Tipo"])

        faltou = self._exige_colaborador(alvo, periodo, "Qual o último ponto de FULANO?")
        if faltou:
            return faltou
        ultimo = (RegistroPonto.query.filter_by(colaborador_id=alvo.id)
                  .order_by(RegistroPonto.data_hora.desc()).first())
        if not ultimo:
            return self._resposta(f"{self._tratar(alvo, usuario)} ainda não tem nenhuma batida registrada.")
        quando = ctx.para_brasilia(ultimo.data_hora)
        tipo = "entrada" if (ultimo.tipo or "") == "entrada" else "saída"
        local = f" em {ultimo.endereco}" if ultimo.endereco else ""
        proximo = "saída" if tipo == "entrada" else "entrada"
        eh_proprio = self._eh_proprio(alvo, usuario)
        sujeito = "Sua última batida" if eh_proprio else f"A última batida de **{alvo.nome}**"
        fecho = ("A sua próxima marcação será uma " if eh_proprio
                 else "A próxima marcação dessa pessoa será uma ")
        return self._resposta(
            f"{sujeito} foi uma **{tipo}** em "
            f"{quando.strftime('%d/%m/%Y às %H:%M:%S')}{local}. {fecho}{proximo}.",
            periodo=periodo, destaque=quando.strftime("%d/%m %H:%M"))

    def _presentes_hoje(self, periodo, **kwargs):
        ctx = self.ctx
        hoje = ctx.agora_brasilia().date()
        config = ctx.obter_configuracao()
        colaboradores = self._colaboradores(incluir_inativos=False)

        presentes, ausentes = [], []
        for c in colaboradores:
            registros = ctx.registros_do_dia(c.id, hoje)
            if registros:
                primeira = ctx.para_brasilia(registros[0].data_hora)
                ultimo_tipo = (registros[-1].tipo or "").lower()
                presentes.append([c.nome, primeira.strftime("%H:%M"),
                                  "trabalhando" if ultimo_tipo == "entrada" else "já saiu"])
            else:
                ausentes.append(c.nome)

        if ctx.eh_domingo(hoje):
            cabecalho = "Hoje é domingo — não há expectativa de comparecimento. "
        else:
            cabecalho = ""
        texto = (f"{cabecalho}**{len(presentes)} de {len(colaboradores)}** colaboradores bateram ponto hoje "
                 f"({formatar_data(hoje)}).")
        if ausentes and not ctx.eh_domingo(hoje):
            texto += f"\n\nAinda sem registro: {', '.join(ausentes[:10])}"
            if len(ausentes) > 10:
                texto += f" e mais {len(ausentes) - 10}."
        texto += f"\n\n_Entrada padrão da empresa: {config.horario_entrada}._"
        return self._resposta(texto, detalhes=presentes,
                              colunas=["Colaborador", "Primeira batida", "Situação"],
                              destaque=f"{len(presentes)}/{len(colaboradores)}")

    def _pendencias(self, periodo, **kwargs):
        ctx = self.ctx
        ajustes = (ctx.SolicitacaoAjuste.query.filter_by(status="pendente")
                   .order_by(ctx.SolicitacaoAjuste.criado_em.asc()).all())
        try:
            excepcionais = (ctx.AprovacaoJornadaExcepcional.query.filter_by(status="pendente")
                            .order_by(ctx.AprovacaoJornadaExcepcional.criado_em.asc()).all())
        except Exception:
            excepcionais = []

        if not ajustes and not excepcionais:
            return self._resposta("Não há nada aguardando sua aprovação. Tudo em dia. ✅",
                                  destaque="0 pendências")

        linhas = []
        for s in ajustes:
            linhas.append([s.colaborador.nome, "Ajuste de ponto",
                           f"{formatar_data(s.data_referencia)} {s.tipo} {s.horario_solicitado}"])
        for a in excepcionais:
            linhas.append([a.colaborador.nome, "Jornada excepcional",
                           f"{formatar_data(a.data_referencia)} · {a.horas_total:.1f}h seguidas"])

        texto = (f"Você tem **{len(ajustes)} solicitação(ões) de ajuste** e "
                 f"**{len(excepcionais)} jornada(s) excepcional(is)** aguardando revisão. "
                 f"Todas ficam na tela *Ajustes*.")
        return self._resposta(texto, detalhes=linhas, colunas=["Colaborador", "Tipo", "Referência"],
                              destaque=f"{len(ajustes) + len(excepcionais)} pendências")

    def _minhas_solicitacoes(self, usuario, periodo):
        ctx = self.ctx
        solicitacoes = (ctx.SolicitacaoAjuste.query.filter_by(colaborador_id=usuario.id)
                        .order_by(ctx.SolicitacaoAjuste.criado_em.desc()).limit(self.LIMITE_LINHAS).all())
        if not solicitacoes:
            return self._resposta("Você não tem nenhuma solicitação de ajuste registrada.",
                                  intencao="pendencias")
        linhas = [[formatar_data(s.data_referencia), s.tipo, s.horario_solicitado, s.status]
                  for s in solicitacoes]
        pendentes = sum(1 for s in solicitacoes if s.status == "pendente")
        return self._resposta(
            f"Você tem {len(solicitacoes)} solicitação(ões) registrada(s), sendo {pendentes} aguardando o gestor.",
            intencao="pendencias", detalhes=linhas, colunas=["Dia", "Tipo", "Horário", "Status"])

    def _ranking(self, periodo, eh_gestor, **kwargs):
        ctx = self.ctx
        colaboradores = self._colaboradores(incluir_inativos=False)
        if not colaboradores:
            return self._resposta("Nenhum colaborador cadastrado ainda.", periodo=periodo)

        dados = []
        for c in colaboradores:
            r = self._resumo(c, periodo)
            dados.append((c, r))
        dados.sort(key=lambda item: -item[1]["total_extra"])

        total_empresa = sum(r["total_extra"] for _, r in dados)
        com_extra = [d for d in dados if d[1]["total_extra"] > 0]
        linhas = [[c.nome, ctx.formatar_horas(r["total_extra"]), ctx.formatar_horas(r["total_horas"])]
                  for c, r in dados if r["total_extra"] > 0 or r["total_horas"] > 0]

        if not com_extra:
            texto = f"Ninguém acumulou hora extra {periodo.frase}."
        else:
            primeiro, r1 = com_extra[0]
            texto = (f"{periodo.frase.capitalize()}, a equipe acumulou **{ctx.formatar_horas(total_empresa)}** "
                     f"de hora extra, distribuídas entre {len(com_extra)} colaborador(es). "
                     f"Quem tem mais é **{primeiro.nome}**, com {ctx.formatar_horas(r1['total_extra'])}.")
            retida = sum(r["extra_retida"] for _, r in dados)
            if retida:
                texto += f"\n\n⚠️ Há ainda {ctx.formatar_horas(retida)} aguardando aprovação de jornadas excepcionais."
        return self._resposta(texto, detalhes=linhas,
                              colunas=["Colaborador", "Hora extra", "Trabalhadas"],
                              periodo=periodo, destaque=ctx.formatar_horas(total_empresa))

    def _configuracao(self, periodo, **kwargs):
        ctx = self.ctx
        config = ctx.obter_configuracao()
        duracao = (int(config.horario_saida.split(":")[0]) + int(config.horario_saida.split(":")[1]) / 60) - \
                  (int(config.horario_entrada.split(":")[0]) + int(config.horario_entrada.split(":")[1]) / 60)
        texto = (f"A jornada padrão configurada é **{config.horario_entrada} às {config.horario_saida}** "
                 f"({ctx.formatar_horas(duracao)} por dia, intervalo incluído no intervalo entre batidas). "
                 f"A meta diária é de {ctx.formatar_horas(config.meta_horas_diarias)} e o lembrete de ponto "
                 f"é reenviado a cada {config.intervalo_lembrete_minutos} minutos.\n\n"
                 f"Hora extra = tempo trabalhado no dia − duração padrão da jornada. "
                 f"Domingo é exceção: o dia inteiro conta como extra.")
        return self._resposta(texto, periodo=periodo)

    def _custo_extras(self, periodo, alvo, eh_gestor, **kwargs):
        ctx = self.ctx
        colaboradores = [alvo] if alvo else self._colaboradores(incluir_inativos=False)
        total = sum(self._resumo(c, periodo)["total_extra"] for c in colaboradores)
        texto = (f"O sistema não guarda salários, então não consigo calcular o valor em reais. "
                 f"O que posso dizer é o volume: **{ctx.formatar_horas(total)}** de hora extra em "
                 f"{periodo.rotulo}"
                 + (f" para {alvo.nome}." if alvo else " na equipe toda.")
                 + "\n\nCom o valor da hora de cada colaborador em mãos, basta multiplicar por esse total "
                   "(mais o adicional previsto em convenção).")
        return self._resposta(texto, periodo=periodo, destaque=ctx.formatar_horas(total))

    def _ajuda(self, eh_gestor, **kwargs):
        if eh_gestor:
            texto = ("Eu leio os dados de ponto da empresa e respondo em linguagem natural. "
                     "Posso falar sobre horas extras, horas trabalhadas, atrasos, dias sem registro, "
                     "dias incompletos, últimas batidas, quem está presente hoje, pendências de "
                     "aprovação e comparações entre colaboradores.\n\n"
                     "Você pode citar o nome do colaborador e o período livremente — entendo coisas "
                     "como \"mês passado\", \"nos últimos 15 dias\" ou \"de 01/07 a 15/07\".")
            sugestoes = [
                "Quantas horas extras a Maria tem este mês?",
                "Quem tem mais horas extras este mês?",
                "Quem ainda não bateu ponto hoje?",
                "Quais pendências estão esperando minha aprovação?",
                "Quantos atrasos o João teve mês passado?",
            ]
        else:
            texto = ("Eu respondo sobre o SEU ponto: horas trabalhadas, horas extras, atrasos, "
                     "dias incompletos e o andamento das suas solicitações de ajuste. "
                     "Dados de outros colaboradores só o gestor consulta.")
            sugestoes = [
                "Quantas horas extras eu tenho este mês?",
                "Quantas horas trabalhei na semana passada?",
                "Tenho algum dia incompleto?",
                "Como está minha solicitação de ajuste?",
            ]
        return self._resposta(texto, sugestoes=sugestoes)

    def _nao_entendi(self, pergunta, eh_gestor):
        return self._resposta(
            "Ainda não entendi essa pergunta. Tente ser mais direto sobre o que você quer saber "
            "(horas extras, horas trabalhadas, atrasos, faltas, pendências, presença) e, se for "
            "sobre alguém específico, cite o nome cadastrado.\n\n"
            "_Guardei sua pergunta: quando você me ensinar a resposta certa (marcando 👍 numa "
            "pergunta parecida), eu passo a acertar essa também._",
            intencao="desconhecida",
            sugestoes=(["Quantas horas extras a Maria tem este mês?",
                        "Quem tem mais horas extras este mês?",
                        "Quem ainda não bateu ponto hoje?"] if eh_gestor else
                       ["Quantas horas extras eu tenho este mês?",
                        "Quantas horas trabalhei este mês?"]),
        )
