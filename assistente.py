"""
Assistente do Gestor — perguntas em linguagem natural (voz ou texto) sobre os
dados do ponto.

Como funciona
-------------
O gestor pergunta "quantas horas extras o Alex tem esse mês?". A pergunta vai
para a API da Anthropic (Messages API) junto com a descrição de um conjunto
pequeno e fechado de FERRAMENTAS. O modelo não vê o banco e não escreve SQL:
ele apenas escolhe qual ferramenta chamar e com quais argumentos. Quem executa
a consulta é este arquivo, reaproveitando exatamente as mesmas funções que já
alimentam as telas do painel (`_calcular_relatorio_horas_extras`,
`montar_resumo_diario`, `calcular_horas_extras_dia`, ...). Assim o número que o
assistente fala é, por construção, o mesmo número que aparece na tela.

Segurança
---------
- Só o gestor logado acessa (decorador `gestor_required` do app).
- A chave da API fica no servidor (variável de ambiente ANTHROPIC_API_KEY).
  Nunca é enviada ao navegador.
- O modelo só consegue ler o que as ferramentas abaixo devolvem. Não há acesso
  livre ao banco, nem escrita: o assistente é somente leitura.

Configuração (variáveis de ambiente)
------------------------------------
ANTHROPIC_API_KEY   obrigatória para ligar o assistente (sem ela o app roda
                    normalmente, só com o assistente desativado).
ANTHROPIC_MODEL     opcional. Padrão: claude-haiku-4-5-20251001 (mais barato e
                    rápido; dá conta bem desse tipo de pergunta). Para respostas
                    mais elaboradas: claude-sonnet-5.
"""

import os
import json
import unicodedata
from datetime import datetime, timedelta, date

from flask import Blueprint, render_template, request, jsonify, session

# Modelo padrão: o mais barato da família. Trocável por variável de ambiente.
MODELO_PADRAO = "claude-haiku-4-5-20251001"

MAX_ITERACOES_FERRAMENTA = 8   # trava de segurança contra loop infinito de tool use
MAX_CARACTERES_PERGUNTA = 500
MAX_MENSAGENS_HISTORICO = 8    # mantém o contexto curto (e a conta baixa)


def _normalizar(texto):
    """'Aléx  Silva ' -> 'alex silva' (sem acento, minúsculo, sem espaço sobrando).

    Serve para casar o nome falado pelo gestor com o nome cadastrado mesmo com
    acento faltando, letra trocada ou transcrição imperfeita da voz.
    """
    if not texto:
        return ""
    sem_acento = unicodedata.normalize("NFKD", str(texto))
    sem_acento = "".join(c for c in sem_acento if not unicodedata.combining(c))
    return " ".join(sem_acento.lower().split())


def init_assistente(app, ctx):
    """Registra as rotas do assistente.

    `ctx` é o próprio módulo app.py — passamos ele inteiro para reaproveitar
    modelos e funções de cálculo sem criar import circular.
    """

    bp = Blueprint("assistente", __name__)

    # ------------------------------------------------------------------
    # Acesso aos dados — cada função abaixo vira uma "ferramenta" do modelo
    # ------------------------------------------------------------------

    def _mes_referencia(mes_texto):
        """'2026-08' -> (2026, 8). Vazio/inválido -> mês atual em Brasília."""
        try:
            ano, mes = (int(p) for p in str(mes_texto).split("-")[:2])
            if 1 <= mes <= 12 and 2000 <= ano <= 2100:
                return ano, mes
        except Exception:
            pass
        hoje = ctx.agora_brasilia().date()
        return hoje.year, hoje.month

    def _data_iso(texto, padrao=None):
        try:
            return date.fromisoformat(str(texto)[:10])
        except Exception:
            return padrao

    DIAS_SEMANA_PT = [
        "segunda-feira", "terça-feira", "quarta-feira",
        "quinta-feira", "sexta-feira", "sábado", "domingo",
    ]

    def _dia_semana(d):
        return DIAS_SEMANA_PT[d.weekday()]

    def _colaboradores_ativos():
        return (
            ctx.Colaborador.query
            .filter_by(is_gestor=False)
            .order_by(ctx.Colaborador.nome)
            .all()
        )

    def _resolver_colaborador(nome_procurado):
        """Encontra o colaborador pelo nome falado/digitado.

        Devolve (colaborador, erro). Tenta, em ordem: nome exato, primeiro nome,
        nome contido, e por último semelhança (tolera 'Alexx' -> 'Alex').
        """
        alvo = _normalizar(nome_procurado)
        if not alvo:
            return None, "Nenhum nome foi informado."

        colaboradores = _colaboradores_ativos()
        if not colaboradores:
            return None, "Não há colaboradores cadastrados."

        pares = [(c, _normalizar(c.nome)) for c in colaboradores]

        exatos = [c for c, n in pares if n == alvo]
        if len(exatos) == 1:
            return exatos[0], None

        primeiro_nome = [c for c, n in pares if n.split()[0] == alvo.split()[0]]
        if len(primeiro_nome) == 1:
            return primeiro_nome[0], None

        contidos = [c for c, n in pares if alvo in n or n in alvo]
        if len(contidos) == 1:
            return contidos[0], None

        candidatos = exatos or primeiro_nome or contidos
        if len(candidatos) > 1:
            nomes = ", ".join(c.nome for c in candidatos)
            return None, f"Mais de um colaborador combina com '{nome_procurado}': {nomes}. Pergunte ao gestor qual deles."

        # Última tentativa: semelhança de texto (erro de digitação/transcrição)
        import difflib
        nomes_normalizados = [n for _, n in pares]
        parecidos = difflib.get_close_matches(alvo, nomes_normalizados, n=2, cutoff=0.6)
        if not parecidos:
            # tenta só pelo primeiro nome de cada um
            primeiros = [n.split()[0] for n in nomes_normalizados]
            parecidos_primeiro = difflib.get_close_matches(alvo.split()[0], primeiros, n=2, cutoff=0.6)
            if len(parecidos_primeiro) == 1:
                idx = primeiros.index(parecidos_primeiro[0])
                return pares[idx][0], None
        if len(parecidos) == 1:
            idx = nomes_normalizados.index(parecidos[0])
            return pares[idx][0], None

        disponiveis = ", ".join(c.nome for c in colaboradores)
        return None, (
            f"Não encontrei ninguém chamado '{nome_procurado}'. "
            f"Colaboradores cadastrados: {disponiveis}."
        )

    def _registros_no_periodo(colaborador_id, dia_inicial, dia_final):
        """Batidas do colaborador entre dois dias (inclusive), no fuso de Brasília.

        A janela em UTC é folgada de propósito (±6h) e o dia exato é filtrado
        depois em Brasília — mesma estratégia usada no restante do app.
        """
        inicio_utc = datetime.combine(dia_inicial, datetime.min.time()) - timedelta(hours=6)
        fim_utc = datetime.combine(dia_final + timedelta(days=1), datetime.min.time()) + timedelta(hours=6)
        candidatos = (
            ctx.RegistroPonto.query
            .filter(
                ctx.RegistroPonto.colaborador_id == colaborador_id,
                ctx.RegistroPonto.data_hora >= inicio_utc,
                ctx.RegistroPonto.data_hora < fim_utc,
            )
            .order_by(ctx.RegistroPonto.data_hora.asc())
            .all()
        )
        return [r for r in candidatos if dia_inicial <= ctx.para_brasilia(r.data_hora).date() <= dia_final]

    # --- ferramenta: listar_colaboradores ------------------------------

    def ferramenta_listar_colaboradores(_args):
        colaboradores = _colaboradores_ativos()
        return {
            "total": len(colaboradores),
            "colaboradores": [
                {
                    "nome": c.nome,
                    "email": c.email,
                    "ativo": bool(c.ativo),
                    "cadastro_facial": bool(c.face_encoding),
                }
                for c in colaboradores
            ],
        }

    # --- ferramenta: horas_extras --------------------------------------

    def ferramenta_horas_extras(args):
        ano, mes = _mes_referencia(args.get("mes"))
        resumo, primeiro_dia, config = ctx._calcular_relatorio_horas_extras(ano, mes)

        nome = (args.get("nome") or "").strip()
        if nome:
            colaborador, erro = _resolver_colaborador(nome)
            if erro:
                return {"erro": erro}
            resumo = [r for r in resumo if r["colaborador"].id == colaborador.id]
            if not resumo:
                return {
                    "mes": f"{ano:04d}-{mes:02d}",
                    "colaborador": colaborador.nome,
                    "horas_extras_decimal": 0.0,
                    "horas_extras": "0min",
                    "observacao": "Nenhum registro de ponto para esse colaborador no mês.",
                }

        def formatar(r):
            return {
                "colaborador": r["colaborador"].nome,
                "horas_extras_decimal": r["total_extra"],
                "horas_extras": ctx.formatar_horas(r["total_extra"]),
                "horas_trabalhadas_no_mes": ctx.formatar_horas(r["total_horas_mes"]),
                "dias_com_extra": [
                    {
                        "data": d["data"].isoformat(),
                        "dia_semana": _dia_semana(d["data"]),
                        "horas_no_dia": d["total_horas"],
                        "horas_extras": ctx.formatar_horas(d["horas_extras"]),
                        "dia_incompleto": d["incompleto"],
                        "jornada_excepcional": d.get("excepcional", False),
                        "status_aprovacao": d.get("status_excepcional"),
                    }
                    for d in r["dias_com_extra"][:31]
                ],
            }

        return {
            "mes": f"{ano:04d}-{mes:02d}",
            "jornada_padrao": f"{config.horario_entrada} às {config.horario_saida}",
            "regra": (
                "Hora extra = horas trabalhadas no dia menos a duração da jornada padrão. "
                "Domingo trabalhado conta integralmente como hora extra. "
                "Jornada contínua acima de 14h fica retida (não soma) até o gestor aprovar."
            ),
            "resultado": [formatar(r) for r in resumo],
        }

    # --- ferramenta: ausencias -----------------------------------------

    def ferramenta_ausencias(args):
        ano, mes = _mes_referencia(args.get("mes"))
        hoje = ctx.agora_brasilia().date()
        primeiro_dia = date(ano, mes, 1)
        ultimo_dia = (date(ano + 1, 1, 1) if mes == 12 else date(ano, mes + 1, 1)) - timedelta(days=1)
        if ultimo_dia > hoje:
            ultimo_dia = hoje  # não conta falta em dia que ainda não aconteceu
        if primeiro_dia > hoje:
            return {"erro": "Esse mês ainda não começou."}

        contar_sabado = bool(args.get("contar_sabado", True))

        nome = (args.get("nome") or "").strip()
        if nome:
            colaborador, erro = _resolver_colaborador(nome)
            if erro:
                return {"erro": erro}
            colaboradores = [colaborador]
        else:
            colaboradores = _colaboradores_ativos()

        dias_uteis = []
        dia = primeiro_dia
        while dia <= ultimo_dia:
            if not ctx.eh_domingo(dia) and (contar_sabado or dia.weekday() != 5):
                dias_uteis.append(dia)
            dia += timedelta(days=1)

        resultado = []
        for c in colaboradores:
            registros = _registros_no_periodo(c.id, primeiro_dia, ultimo_dia)
            dias_com_batida = {ctx.para_brasilia(r.data_hora).date() for r in registros}
            ausentes = [d for d in dias_uteis if d not in dias_com_batida]
            resultado.append({
                "colaborador": c.nome,
                "dias_sem_nenhuma_batida": len(ausentes),
                "datas": [d.isoformat() for d in ausentes],
                "dias_uteis_considerados": len(dias_uteis),
            })

        return {
            "mes": f"{ano:04d}-{mes:02d}",
            "periodo_apurado": f"{primeiro_dia.isoformat()} a {ultimo_dia.isoformat()}",
            "criterio": (
                "IMPORTANTE: o sistema não tem cadastro de faltas, férias, atestado ou feriado. "
                "O que está abaixo é 'dia útil sem nenhuma batida de ponto' — domingos ficam de fora, "
                f"{'sábados contam como dia útil' if contar_sabado else 'sábados foram excluídos'}, e feriados "
                "aparecem como ausência. Trate como indício a conferir, não como falta confirmada."
            ),
            "resultado": resultado,
        }

    # --- ferramenta: registros_do_periodo ------------------------------

    def ferramenta_registros(args):
        nome = (args.get("nome") or "").strip()
        colaborador, erro = _resolver_colaborador(nome)
        if erro:
            return {"erro": erro}

        hoje = ctx.agora_brasilia().date()
        dia_final = _data_iso(args.get("data_fim"), hoje)
        dia_inicial = _data_iso(args.get("data_inicio"), dia_final - timedelta(days=7))
        if dia_inicial > dia_final:
            dia_inicial, dia_final = dia_final, dia_inicial
        if (dia_final - dia_inicial).days > 92:
            dia_inicial = dia_final - timedelta(days=92)  # evita resposta gigante

        config = ctx.obter_configuracao()
        registros = _registros_no_periodo(colaborador.id, dia_inicial, dia_final)
        dias = ctx.montar_resumo_diario(registros)

        detalhes = []
        for d in dias:
            extra = ctx.calcular_horas_extras_dia(
                d["total_horas_decimal"], config.horario_entrada, config.horario_saida,
                eh_domingo_flag=ctx.eh_domingo(d["data"]),
            )
            d["horas_extras"] = extra
            ctx.aplicar_status_excepcional(d, colaborador.id)
            batidas = [b.strftime("%H:%M") for b in d["batidas"]]
            primeira = batidas[0] if batidas else None
            detalhes.append({
                "data": d["data"].isoformat(),
                "dia_semana": _dia_semana(d["data"]),
                "batidas": batidas,
                "primeira_batida": primeira,
                "atrasado": bool(primeira and not ctx.eh_domingo(d["data"]) and primeira > config.horario_entrada),
                "horas_trabalhadas": d["total_horas"],
                "horas_extras": ctx.formatar_horas(d["horas_extras"]),
                "dia_incompleto": d["incompleto"],
                "jornada_excepcional": d.get("excepcional", False),
                "status_aprovacao": d.get("status_excepcional"),
            })

        return {
            "colaborador": colaborador.nome,
            "periodo": f"{dia_inicial.isoformat()} a {dia_final.isoformat()}",
            "horario_padrao": f"{config.horario_entrada} às {config.horario_saida}",
            "dias": detalhes,
            "observacao": "Sem nenhuma batida no período." if not detalhes else None,
        }

    # --- ferramenta: situacao_agora ------------------------------------

    def ferramenta_situacao_agora(_args):
        from collections import defaultdict

        config = ctx.obter_configuracao()
        agora = ctx.agora_brasilia()
        hoje = agora.date()
        colaboradores = _colaboradores_ativos()

        limite_utc = datetime.combine(hoje, datetime.min.time()) - timedelta(hours=6)
        candidatos = (
            ctx.RegistroPonto.query
            .filter(ctx.RegistroPonto.data_hora >= limite_utc)
            .order_by(ctx.RegistroPonto.data_hora.asc())
            .all()
        )
        por_colaborador = defaultdict(list)
        for r in candidatos:
            if ctx.para_brasilia(r.data_hora).date() == hoje:
                por_colaborador[r.colaborador_id].append(r)

        domingo = ctx.eh_domingo(hoje)
        presentes, atrasados, sem_registro = [], [], []
        for c in colaboradores:
            regs = por_colaborador.get(c.id)
            if not regs:
                if not domingo:
                    sem_registro.append(c.nome)
                continue
            primeira = ctx.para_brasilia(regs[0].data_hora)
            ultima = regs[-1]
            presentes.append({
                "colaborador": c.nome,
                "primeira_batida": primeira.strftime("%H:%M"),
                "ultima_batida": ctx.para_brasilia(ultima.data_hora).strftime("%H:%M"),
                "ultimo_tipo": ultima.tipo,
                "esta_trabalhando_agora": (ultima.tipo or "").lower() == "entrada",
            })
            if not domingo and primeira.strftime("%H:%M") > config.horario_entrada:
                atrasados.append({"colaborador": c.nome, "entrou_as": primeira.strftime("%H:%M")})

        ajustes = (
            ctx.SolicitacaoAjuste.query
            .filter_by(status="pendente")
            .order_by(ctx.SolicitacaoAjuste.criado_em.desc())
            .limit(20)
            .all()
        )
        excepcionais = (
            ctx.AprovacaoJornadaExcepcional.query
            .filter_by(status="pendente")
            .order_by(ctx.AprovacaoJornadaExcepcional.data_referencia.desc())
            .limit(20)
            .all()
        )
        nomes = {c.id: c.nome for c in ctx.Colaborador.query.all()}

        return {
            "agora": agora.strftime("%d/%m/%Y %H:%M"),
            "hoje_e_domingo": domingo,
            "horario_padrao": f"{config.horario_entrada} às {config.horario_saida}",
            "total_colaboradores": len(colaboradores),
            "bateram_ponto_hoje": presentes,
            "atrasados_hoje": atrasados,
            "sem_nenhuma_batida_hoje": sem_registro,
            "ajustes_pendentes": [
                {
                    "colaborador": nomes.get(a.colaborador_id, "?"),
                    "data": a.data_referencia.isoformat(),
                    "tipo": a.tipo,
                    "horario_solicitado": a.horario_solicitado,
                    "motivo": a.motivo,
                }
                for a in ajustes
            ],
            "jornadas_excepcionais_pendentes": [
                {
                    "colaborador": nomes.get(j.colaborador_id, "?"),
                    "data": j.data_referencia.isoformat(),
                    "horas": ctx.formatar_horas(j.horas_total),
                }
                for j in excepcionais
            ],
        }

    # ------------------------------------------------------------------
    # Catálogo de ferramentas enviado ao modelo
    # ------------------------------------------------------------------

    FERRAMENTAS = [
        {
            "name": "listar_colaboradores",
            "description": (
                "Lista todos os colaboradores cadastrados (nome, e-mail, se está ativo e se tem "
                "cadastro facial). Use quando o gestor perguntar quem trabalha na empresa, quantas "
                "pessoas existem, ou quando precisar confirmar o nome exato de alguém."
            ),
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "horas_extras",
            "description": (
                "Horas extras de um colaborador ou de toda a equipe em um mês, com o detalhe dia a dia. "
                "Usa exatamente o mesmo cálculo da tela 'Horas Extras' do painel."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "nome": {
                        "type": "string",
                        "description": "Nome do colaborador. Omita para trazer a equipe inteira (ranking do maior para o menor).",
                    },
                    "mes": {
                        "type": "string",
                        "description": "Mês no formato AAAA-MM. Omita para o mês atual.",
                    },
                },
            },
        },
        {
            "name": "ausencias",
            "description": (
                "Dias úteis em que o colaborador não bateu ponto nenhuma vez no mês. ATENÇÃO: o sistema "
                "não registra faltas, férias, atestados nem feriados — isso é um indício de ausência, "
                "não uma falta confirmada. Sempre deixe essa ressalva clara na resposta."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "nome": {"type": "string", "description": "Nome do colaborador. Omita para a equipe inteira."},
                    "mes": {"type": "string", "description": "Mês no formato AAAA-MM. Omita para o mês atual."},
                    "contar_sabado": {
                        "type": "boolean",
                        "description": "Se sábado conta como dia útil. Padrão: true (regra atual do sistema).",
                    },
                },
            },
        },
        {
            "name": "registros_do_periodo",
            "description": (
                "Batidas de ponto de um colaborador em um período: horários de entrada e saída de cada dia, "
                "total trabalhado, se chegou atrasado e se o dia ficou incompleto. Use para perguntas como "
                "'a que horas o João entrou ontem?', 'quantos atrasos a Maria teve na semana passada?'."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "nome": {"type": "string", "description": "Nome do colaborador."},
                    "data_inicio": {"type": "string", "description": "Data inicial AAAA-MM-DD. Padrão: 7 dias antes da data final."},
                    "data_fim": {"type": "string", "description": "Data final AAAA-MM-DD. Padrão: hoje."},
                },
                "required": ["nome"],
            },
        },
        {
            "name": "situacao_agora",
            "description": (
                "Fotografia do dia de hoje: quem já bateu ponto, quem está trabalhando neste momento, "
                "quem chegou atrasado, quem ainda não apareceu, e o que está pendente de aprovação "
                "(solicitações de ajuste e jornadas excepcionais)."
            ),
            "input_schema": {"type": "object", "properties": {}},
        },
    ]

    EXECUTORES = {
        "listar_colaboradores": ferramenta_listar_colaboradores,
        "horas_extras": ferramenta_horas_extras,
        "ausencias": ferramenta_ausencias,
        "registros_do_periodo": ferramenta_registros,
        "situacao_agora": ferramenta_situacao_agora,
    }

    def _prompt_sistema():
        agora = ctx.agora_brasilia()
        marca = ctx.obter_marca()
        return (
            f"Você é o assistente do gestor do {marca.nome_empresa}, um sistema de controle de ponto "
            "com reconhecimento facial. O gestor conversa com você por voz ou por texto, muitas vezes "
            "pelo celular, no meio do expediente.\n\n"
            f"Data e hora agora: {agora.strftime('%d/%m/%Y %H:%M')} ({_dia_semana(agora)}), fuso de Brasília.\n\n"
            "REGRAS:\n"
            "1. Nunca invente números. Todo dado vem das ferramentas. Se a ferramenta não trouxer, "
            "diga que o sistema não tem essa informação.\n"
            "2. A pergunta pode vir de transcrição de voz, com nome errado ou palavra faltando. "
            "Interprete a intenção; a busca por nome já tolera erro de grafia. Se ficar ambíguo entre "
            "duas pessoas, pergunte qual delas.\n"
            "3. Responda em português do Brasil, curto e falado — a resposta pode ser lida em voz alta. "
            "Comece pelo número que ele pediu, depois um detalhe útil se houver. "
            "Nada de tabelas, markdown ou listas longas: no máximo 3 ou 4 frases. "
            "Se o gestor pedir o detalhe dia a dia, aí sim liste os dias.\n"
            "4. Horas sempre no formato falado: '12h30min', '45min', '3h'.\n"
            "5. Datas em português: 'dia 12 de agosto', 'ontem', 'na terça'.\n"
            "6. Sobre faltas: o sistema NÃO tem cadastro de falta, férias, atestado ou feriado. "
            "O que existe é 'dia útil sem nenhuma batida'. Sempre avise isso ao responder sobre faltas.\n"
            "7. Você é somente leitura: não aprova ajustes, não altera nada. Se pedirem uma ação, "
            "explique em que tela do painel isso é feito (Ajustes, Horas Extras, Cadastro, Configurações).\n"
            "8. Se a pergunta não tiver nada a ver com ponto, jornada ou equipe, diga educadamente "
            "que você só responde sobre os dados do controle de ponto."
        )

    # ------------------------------------------------------------------
    # Rotas
    # ------------------------------------------------------------------

    def assistente_habilitado():
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    @bp.route("/gestor/assistente", methods=["GET"])
    @ctx.gestor_required
    def gestor_assistente():
        return render_template(
            "gestor_assistente.html",
            habilitado=assistente_habilitado(),
            nome_gestor=session.get("nome", ""),
        )

    @bp.route("/api/gestor/perguntar", methods=["POST"])
    @ctx.gestor_required
    def api_perguntar():
        if not assistente_habilitado():
            return jsonify({
                "erro": "O assistente está desligado: falta a variável de ambiente ANTHROPIC_API_KEY."
            }), 503

        try:
            from anthropic import Anthropic
        except ImportError:
            return jsonify({
                "erro": "Biblioteca 'anthropic' não instalada. Rode: pip install anthropic"
            }), 503

        dados = request.get_json(silent=True) or {}
        pergunta = (dados.get("pergunta") or "").strip()[:MAX_CARACTERES_PERGUNTA]
        if not pergunta:
            return jsonify({"erro": "Pergunta vazia."}), 400

        # Histórico vindo do navegador (só do próprio gestor logado), limitado
        # para manter o custo previsível.
        mensagens = []
        for m in (dados.get("historico") or [])[-MAX_MENSAGENS_HISTORICO:]:
            papel = m.get("role")
            conteudo = str(m.get("content") or "")[:2000]
            if papel in ("user", "assistant") and conteudo:
                mensagens.append({"role": papel, "content": conteudo})
        mensagens.append({"role": "user", "content": pergunta})

        cliente = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        modelo = os.environ.get("ANTHROPIC_MODEL", MODELO_PADRAO)

        ferramentas_usadas = []
        try:
            for _ in range(MAX_ITERACOES_FERRAMENTA):
                resposta = cliente.messages.create(
                    model=modelo,
                    max_tokens=1024,
                    system=_prompt_sistema(),
                    tools=FERRAMENTAS,
                    messages=mensagens,
                )

                if resposta.stop_reason != "tool_use":
                    texto = "".join(b.text for b in resposta.content if b.type == "text").strip()
                    return jsonify({
                        "resposta": texto or "Não consegui responder isso.",
                        "ferramentas_usadas": ferramentas_usadas,
                    })

                # O modelo pediu uma ou mais ferramentas: executa e devolve o resultado.
                mensagens.append({"role": "assistant", "content": resposta.content})
                resultados = []
                for bloco in resposta.content:
                    if bloco.type != "tool_use":
                        continue
                    executor = EXECUTORES.get(bloco.name)
                    ferramentas_usadas.append(bloco.name)
                    try:
                        saida = executor(bloco.input or {}) if executor else {"erro": "Ferramenta desconhecida."}
                    except Exception as ex:
                        ctx.db.session.rollback()
                        print(f"[assistente] erro na ferramenta {bloco.name}: {ex}")
                        saida = {"erro": f"Falha ao consultar os dados: {ex}"}
                    resultados.append({
                        "type": "tool_result",
                        "tool_use_id": bloco.id,
                        "content": json.dumps(saida, ensure_ascii=False, default=str),
                    })
                mensagens.append({"role": "user", "content": resultados})

            return jsonify({
                "resposta": "Essa pergunta ficou complexa demais para eu responder de uma vez. "
                            "Pode quebrar em partes menores?",
                "ferramentas_usadas": ferramentas_usadas,
            })

        except Exception as ex:
            print(f"[assistente] erro: {ex}")
            return jsonify({"erro": f"Não consegui falar com o assistente agora. ({type(ex).__name__})"}), 502

    # Exposto para teste manual sem gastar chamada de API:
    #   from app import app
    #   app.blueprints["assistente"].executores["horas_extras"]({"nome": "Alex"})
    bp.executores = EXECUTORES

    app.register_blueprint(bp)
    return bp
