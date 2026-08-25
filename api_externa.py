"""
API externa (REST) do controle de ponto.

Para que serve
--------------
Expor, para sistemas de terceiros (ex.: o Lecom), os dados que eles precisam
para preencher a grid de apuração: NOME do colaborador, DATA, HORA DE ENTRADA,
HORA DE SAÍDA e ENDEREÇO onde a batida aconteceu.

Como se autentica
-----------------
Por token fixo, enviado em um destes cabeçalhos:

    X-API-Key: <token>
    Authorization: Bearer <token>

Os tokens válidos vêm da variável de ambiente API_TOKENS (vários separados por
vírgula) ou API_TOKEN (um só). Se nenhuma estiver configurada, a API responde
503 e NÃO devolve dado nenhum — nunca fica aberta por esquecimento.

Endpoints (todos GET, todos exigem token)
-----------------------------------------
    GET /api/v1/ping             -> testa credencial e conectividade
    GET /api/v1/colaboradores    -> lista de colaboradores
    GET /api/v1/jornadas         -> UMA LINHA POR JORNADA (entrada + saída) <- é a da grid
    GET /api/v1/batidas          -> batidas cruas, uma linha por marcação

Filtros aceitos por /jornadas e /batidas
----------------------------------------
    inicio, fim      datas em AAAA-MM-DD ou DD/MM/AAAA (padrão: últimos 30 dias)
    colaborador_id   id numérico
    email            e-mail exato do colaborador
    nome             trecho do nome (busca parcial, sem diferenciar maiúsculas)
    pagina           1, 2, 3... (padrão 1)
    por_pagina       até 1000 (padrão 200)
    formato          json (padrão) ou csv

Este módulo não importa nada do app.py: recebe modelos e funções por injeção de
dependência (mesmo padrão do assistente.py), então usa exatamente o mesmo fuso,
o mesmo pareamento entrada/saída e os mesmos números que as telas mostram.
"""

import csv
import io
import os
from datetime import datetime, timedelta
from functools import wraps
from hmac import compare_digest

from flask import Blueprint, Response, jsonify, request

# ---------------------------------------------------------------------------
# Contexto injetado pelo app.py (ver criar_blueprint_api)
# ---------------------------------------------------------------------------
ctx = None

LIMITE_POR_PAGINA = 1000
PADRAO_POR_PAGINA = 200
PADRAO_DIAS = 30

api = Blueprint("api_externa", __name__, url_prefix="/api/v1")


# ---------------------------------------------------------------------------
# Autenticação
# ---------------------------------------------------------------------------
def tokens_configurados():
    """Tokens aceitos, lidos do ambiente a cada requisição (permite trocar o
    token no painel do Render sem alterar código)."""
    bruto = os.environ.get("API_TOKENS") or os.environ.get("API_TOKEN") or ""
    return [t.strip() for t in bruto.split(",") if t.strip()]


def token_da_requisicao():
    token = request.headers.get("X-API-Key", "").strip()
    if token:
        return token
    autorizacao = request.headers.get("Authorization", "").strip()
    if autorizacao.lower().startswith("bearer "):
        return autorizacao[7:].strip()
    return ""


def token_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        validos = tokens_configurados()
        if not validos:
            return jsonify({
                "erro": "api_nao_configurada",
                "mensagem": "A API externa está desligada: defina a variável de ambiente API_TOKENS no servidor.",
            }), 503
        enviado = token_da_requisicao()
        if not enviado:
            return jsonify({
                "erro": "token_ausente",
                "mensagem": "Envie o token no cabeçalho 'X-API-Key' ou 'Authorization: Bearer <token>'.",
            }), 401
        # comparação em tempo constante: evita descobrir o token por medição de tempo
        if not any(compare_digest(enviado, valido) for valido in validos):
            return jsonify({"erro": "token_invalido", "mensagem": "Token inválido."}), 403
        return f(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Leitura e validação dos filtros
# ---------------------------------------------------------------------------
class ErroDeFiltro(Exception):
    pass


def _ler_data(valor, rotulo):
    """Aceita AAAA-MM-DD e DD/MM/AAAA."""
    for formato in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(valor, formato).date()
        except ValueError:
            continue
    raise ErroDeFiltro("Parametro '%s' invalido: use AAAA-MM-DD ou DD/MM/AAAA." % rotulo)


def ler_filtros():
    hoje = ctx.agora_brasilia().date()

    fim_txt = request.args.get("fim", "").strip()
    inicio_txt = request.args.get("inicio", "").strip()
    fim = _ler_data(fim_txt, "fim") if fim_txt else hoje
    inicio = _ler_data(inicio_txt, "inicio") if inicio_txt else fim - timedelta(days=PADRAO_DIAS - 1)
    if inicio > fim:
        raise ErroDeFiltro("O parametro 'inicio' nao pode ser posterior a 'fim'.")

    colaborador_id = request.args.get("colaborador_id", "").strip()
    if colaborador_id:
        if not colaborador_id.isdigit():
            raise ErroDeFiltro("Parametro 'colaborador_id' deve ser um numero.")
        colaborador_id = int(colaborador_id)
    else:
        colaborador_id = None

    try:
        pagina = max(1, int(request.args.get("pagina", 1)))
    except ValueError:
        raise ErroDeFiltro("Parametro 'pagina' deve ser um numero.")
    try:
        por_pagina = int(request.args.get("por_pagina", PADRAO_POR_PAGINA))
    except ValueError:
        raise ErroDeFiltro("Parametro 'por_pagina' deve ser um numero.")
    por_pagina = max(1, min(por_pagina, LIMITE_POR_PAGINA))

    formato = request.args.get("formato", "json").strip().lower()
    if formato not in ("json", "csv"):
        raise ErroDeFiltro("Parametro 'formato' deve ser 'json' ou 'csv'.")

    return {
        "inicio": inicio,
        "fim": fim,
        "colaborador_id": colaborador_id,
        "email": request.args.get("email", "").strip() or None,
        "nome": request.args.get("nome", "").strip() or None,
        "pagina": pagina,
        "por_pagina": por_pagina,
        "formato": formato,
    }


def buscar_registros(filtros, dias_extras_antes=1):
    """Traz as batidas do período pedido, já filtradas por DATA DE BRASÍLIA.

    A coluna data_hora é gravada em UTC, então a consulta usa uma folga de horas
    no banco e o recorte fino é feito depois de converter para Brasília — mesma
    técnica de registros_do_dia(), pra não perder batida de início/fim do dia.

    `dias_extras_antes` amplia a busca para trás: uma jornada que começou no dia
    anterior e terminou dentro do período precisa da sua batida de ENTRADA para
    ser pareada corretamente.
    """
    inicio_busca = filtros["inicio"] - timedelta(days=dias_extras_antes)
    inicio_utc = datetime.combine(inicio_busca, datetime.min.time()) - timedelta(hours=6)
    fim_utc = datetime.combine(filtros["fim"], datetime.max.time()) + timedelta(hours=30)

    query = (
        ctx.RegistroPonto.query
        .join(ctx.Colaborador, ctx.RegistroPonto.colaborador_id == ctx.Colaborador.id)
        .filter(ctx.RegistroPonto.data_hora >= inicio_utc)
        .filter(ctx.RegistroPonto.data_hora <= fim_utc)
    )
    if filtros["colaborador_id"]:
        query = query.filter(ctx.RegistroPonto.colaborador_id == filtros["colaborador_id"])
    if filtros["email"]:
        query = query.filter(ctx.Colaborador.email == filtros["email"])
    if filtros["nome"]:
        query = query.filter(ctx.Colaborador.nome.ilike("%" + filtros["nome"] + "%"))

    registros = query.order_by(ctx.RegistroPonto.data_hora.asc()).all()

    return [
        r for r in registros
        if inicio_busca <= ctx.para_brasilia(r.data_hora).date() <= filtros["fim"]
    ]


# ---------------------------------------------------------------------------
# Montagem das jornadas (entrada -> saída)
# ---------------------------------------------------------------------------
def montar_jornadas(registros):
    """Pareia cada ENTRADA com a SAÍDA seguinte do mesmo colaborador, em ordem
    cronológica, e devolve uma linha por jornada.

    Segue a mesma convenção de montar_resumo_diario() no app: uma jornada que
    atravessa a meia-noite pertence ao dia em que a ENTRADA aconteceu.
    """
    from collections import defaultdict

    por_colaborador = defaultdict(list)
    for r in sorted(registros, key=lambda x: x.data_hora):
        por_colaborador[r.colaborador_id].append(r)

    jornadas = []
    for batidas_do_colaborador in por_colaborador.values():
        entrada_aberta = None
        for r in batidas_do_colaborador:
            tipo = (r.tipo or "").lower()
            if tipo == "entrada":
                if entrada_aberta is not None:
                    # duas entradas seguidas: a primeira ficou sem saída
                    jornadas.append(_linha_jornada(entrada_aberta, None))
                entrada_aberta = r
            elif tipo == "saida":
                if entrada_aberta is not None:
                    jornadas.append(_linha_jornada(entrada_aberta, r))
                    entrada_aberta = None
                else:
                    # saída sem entrada correspondente antes dela
                    jornadas.append(_linha_jornada(None, r))
            else:
                # tipo legado/desconhecido: não dá pra parear com segurança
                jornadas.append(_linha_jornada(r, None))
        if entrada_aberta is not None:
            # jornada em andamento (ou pendência real de batida de saída)
            jornadas.append(_linha_jornada(entrada_aberta, None))

    jornadas.sort(key=lambda j: (j["data"], j["nome"], j["hora_entrada"] or ""), reverse=True)
    return jornadas


def _linha_jornada(entrada, saida):
    base = entrada or saida
    dt_entrada = ctx.para_brasilia(entrada.data_hora) if entrada else None
    dt_saida = ctx.para_brasilia(saida.data_hora) if saida else None
    dia = (dt_entrada or dt_saida).date()

    total_horas = None
    total_decimal = None
    if dt_entrada and dt_saida:
        segundos = (dt_saida - dt_entrada).total_seconds()
        if segundos > 0:
            total_decimal = round(segundos / 3600, 2)
            total_horas = "%02d:%02d" % (segundos // 3600, (segundos % 3600) // 60)

    return {
        "colaborador_id": base.colaborador_id,
        "nome": base.colaborador.nome,
        "email": base.colaborador.email,
        "data": dia.isoformat(),
        "data_br": dia.strftime("%d/%m/%Y"),
        "hora_entrada": dt_entrada.strftime("%H:%M:%S") if dt_entrada else None,
        "hora_saida": dt_saida.strftime("%H:%M:%S") if dt_saida else None,
        "data_hora_entrada": dt_entrada.isoformat() if dt_entrada else None,
        "data_hora_saida": dt_saida.isoformat() if dt_saida else None,
        "endereco_entrada": (entrada.endereco if entrada else None) or None,
        "endereco_saida": (saida.endereco if saida else None) or None,
        "latitude_entrada": entrada.latitude if entrada else None,
        "longitude_entrada": entrada.longitude if entrada else None,
        "latitude_saida": saida.latitude if saida else None,
        "longitude_saida": saida.longitude if saida else None,
        "total_horas": total_horas,
        "total_horas_decimal": total_decimal,
        "completa": bool(dt_entrada and dt_saida),
        "registro_entrada_id": entrada.id if entrada else None,
        "registro_saida_id": saida.id if saida else None,
    }


def _linha_batida(r):
    dt = ctx.para_brasilia(r.data_hora)
    return {
        "id": r.id,
        "colaborador_id": r.colaborador_id,
        "nome": r.colaborador.nome,
        "email": r.colaborador.email,
        "data": dt.date().isoformat(),
        "data_br": dt.strftime("%d/%m/%Y"),
        "hora": dt.strftime("%H:%M:%S"),
        "data_hora": dt.isoformat(),
        "tipo": (r.tipo or "").lower() or None,
        "endereco": r.endereco or None,
        "latitude": r.latitude,
        "longitude": r.longitude,
        "origem": r.origem,
    }


# ---------------------------------------------------------------------------
# Resposta (paginação + json/csv)
# ---------------------------------------------------------------------------
def responder(linhas, filtros, nome_csv, colunas_csv):
    total = len(linhas)
    corte = (filtros["pagina"] - 1) * filtros["por_pagina"]
    pagina_atual = linhas[corte:corte + filtros["por_pagina"]]
    paginas = max(1, -(-total // filtros["por_pagina"]))  # divisão inteira arredondando pra cima

    if filtros["formato"] == "csv":
        saida = io.StringIO()
        campos = [c[0] for c in colunas_csv]
        writer = csv.DictWriter(saida, fieldnames=campos, extrasaction="ignore", delimiter=";")
        writer.writerow(dict(colunas_csv))  # cabeçalho com nomes amigáveis
        for linha in pagina_atual:
            writer.writerow({campo: ("" if linha.get(campo) is None else linha[campo]) for campo in campos})
        # BOM na frente para o Excel abrir a acentuação corretamente
        return Response(
            "﻿" + saida.getvalue(),
            mimetype="text/csv; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=" + nome_csv},
        )

    return jsonify({
        "inicio": filtros["inicio"].isoformat(),
        "fim": filtros["fim"].isoformat(),
        "fuso": "America/Sao_Paulo",
        "pagina": filtros["pagina"],
        "por_pagina": filtros["por_pagina"],
        "paginas": paginas,
        "total": total,
        "dados": pagina_atual,
    })


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@api.route("/ping", methods=["GET"])
@token_required
def ping():
    """Teste de credencial: se responder ok=true, o token e a URL estão certos."""
    return jsonify({
        "ok": True,
        "servico": "SmartPoint - API de ponto",
        "versao": "1.0",
        "agora": ctx.agora_brasilia().isoformat(),
        "fuso": "America/Sao_Paulo",
    })


@api.route("/colaboradores", methods=["GET"])
@token_required
def colaboradores():
    query = ctx.Colaborador.query
    if request.args.get("incluir_inativos", "").lower() not in ("1", "true", "sim"):
        query = query.filter(ctx.Colaborador.ativo.is_(True))
    lista = [
        {"id": c.id, "nome": c.nome, "email": c.email, "gestor": bool(c.is_gestor), "ativo": bool(c.ativo)}
        for c in query.order_by(ctx.Colaborador.nome).all()
    ]
    return jsonify({"total": len(lista), "dados": lista})


@api.route("/jornadas", methods=["GET"])
@token_required
def jornadas():
    """Uma linha por jornada: nome, data, entrada, saída e endereço.
    É o endpoint pensado para preencher a grid do Lecom."""
    try:
        filtros = ler_filtros()
    except ErroDeFiltro as ex:
        return jsonify({"erro": "parametro_invalido", "mensagem": str(ex)}), 400

    registros = buscar_registros(filtros)
    inicio_iso = filtros["inicio"].isoformat()
    fim_iso = filtros["fim"].isoformat()
    # o dia extra buscado antes serve só para fechar o pareamento; ele não entra no resultado
    linhas = [j for j in montar_jornadas(registros) if inicio_iso <= j["data"] <= fim_iso]

    return responder(linhas, filtros, "jornadas.csv", [
        ("nome", "Colaborador"),
        ("email", "E-mail"),
        ("data_br", "Data"),
        ("hora_entrada", "Entrada"),
        ("hora_saida", "Saida"),
        ("endereco_entrada", "Endereco Entrada"),
        ("endereco_saida", "Endereco Saida"),
        ("total_horas", "Total"),
    ])


@api.route("/batidas", methods=["GET"])
@token_required
def batidas():
    """Uma linha por marcação, sem pareamento — para quem prefere montar a
    jornada do próprio lado."""
    try:
        filtros = ler_filtros()
    except ErroDeFiltro as ex:
        return jsonify({"erro": "parametro_invalido", "mensagem": str(ex)}), 400

    registros = buscar_registros(filtros, dias_extras_antes=0)
    linhas = [_linha_batida(r) for r in reversed(registros)]

    return responder(linhas, filtros, "batidas.csv", [
        ("nome", "Colaborador"),
        ("email", "E-mail"),
        ("data_br", "Data"),
        ("hora", "Hora"),
        ("tipo", "Tipo"),
        ("endereco", "Endereco"),
        ("latitude", "Latitude"),
        ("longitude", "Longitude"),
    ])


# ---------------------------------------------------------------------------
# Registro no app
# ---------------------------------------------------------------------------
def criar_blueprint_api(contexto):
    """Recebe o contexto (modelos + helpers do app.py) e devolve o blueprint
    pronto para app.register_blueprint()."""
    global ctx
    ctx = contexto
    return api
