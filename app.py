import os
import io
import csv
import base64
import json
import numpy as np
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from functools import wraps
from pywebpush import webpush, WebPushException
from apscheduler.schedulers.background import BackgroundScheduler

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, Response, jsonify, send_from_directory
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import face_recognition

# ---------------------------------------------------------------------------
# Configuração básica
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "troque-esta-chave-em-producao")

# Usa Postgres se DATABASE_URL estiver definida (Render), senão SQLite local.
db_url = os.environ.get("DATABASE_URL", "sqlite:///ponto.db")
if db_url.startswith("postgres://"):
    # SQLAlchemy exige "postgresql://"
    db_url = db_url.replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# Distância máxima aceita entre encodings faciais para considerar "é a mesma pessoa"
# Quanto menor, mais rígido. 0.6 é o padrão da biblioteca face_recognition.
FACE_MATCH_TOLERANCE = 0.55

# ---------------------------------------------------------------------------
# Fuso horário - Brasília
# ---------------------------------------------------------------------------
TZ_UTC = ZoneInfo("UTC")
TZ_BRASILIA = ZoneInfo("America/Sao_Paulo")


def agora_brasilia():
    """Retorna o instante atual já ciente do fuso horário de Brasília."""
    return datetime.now(TZ_BRASILIA)


def para_brasilia(dt):
    """Converte um datetime (armazenado em UTC, naive ou aware) para o horário de Brasília."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ_UTC)
    return dt.astimezone(TZ_BRASILIA)


@app.template_filter("brasilia")
def filtro_brasilia(dt, fmt="%d/%m/%Y %H:%M:%S"):
    """Filtro Jinja: {{ registro.data_hora | brasilia }}"""
    dt_local = para_brasilia(dt)
    return dt_local.strftime(fmt) if dt_local else "—"


@app.context_processor
def injetar_contexto_navegacao():
    """Disponibiliza o número de ajustes pendentes para o badge do menu do gestor em qualquer página."""
    if session.get("is_gestor"):
        return {"ajustes_pendentes_nav": SolicitacaoAjuste.query.filter_by(status="pendente").count()}
    return {}


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
class Colaborador(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nome = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    senha_hash = db.Column(db.String(255), nullable=False)
    is_gestor = db.Column(db.Boolean, default=False)
    face_encoding = db.Column(db.PickleType, nullable=True)  # vetor de 128 posições
    ativo = db.Column(db.Boolean, default=True)

    def set_senha(self, senha):
        self.senha_hash = generate_password_hash(senha)

    def check_senha(self, senha):
        return check_password_hash(self.senha_hash, senha)


class RegistroPonto(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    colaborador_id = db.Column(db.Integer, db.ForeignKey("colaborador.id"), nullable=False)
    data_hora = db.Column(db.DateTime, default=datetime.utcnow)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    endereco = db.Column(db.String(255), nullable=True)  # endereço legível obtido por geocodificação reversa
    distancia_facial = db.Column(db.Float, nullable=True)  # quão próximo do rosto cadastrado
    tipo = db.Column(db.String(20), default="entrada")  # entrada / saida (opcional)
    origem = db.Column(db.String(20), default="facial")  # facial | ajuste_manual

    colaborador = db.relationship("Colaborador", backref="registros")


class SolicitacaoAjuste(db.Model):
    """Solicitação do colaborador para corrigir um ponto esquecido, sujeita à aprovação do gestor."""
    id = db.Column(db.Integer, primary_key=True)
    colaborador_id = db.Column(db.Integer, db.ForeignKey("colaborador.id"), nullable=False)
    data_referencia = db.Column(db.Date, nullable=False)  # dia que precisa do ajuste
    tipo = db.Column(db.String(20), nullable=False)  # entrada / saida
    horario_solicitado = db.Column(db.String(5), nullable=False)  # "08:00"
    motivo = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default="pendente")  # pendente | aprovado | recusado
    criado_em = db.Column(db.DateTime, default=datetime.utcnow)
    respondido_em = db.Column(db.DateTime, nullable=True)
    resposta_gestor = db.Column(db.Text, nullable=True)

    colaborador = db.relationship("Colaborador", backref="solicitacoes_ajuste")


class ConfiguracaoJornada(db.Model):
    """Configuração única (definida pelo gestor) da jornada padrão e do lembrete de ponto."""
    id = db.Column(db.Integer, primary_key=True)
    horario_entrada = db.Column(db.String(5), default="08:00")
    horario_saida = db.Column(db.String(5), default="18:00")
    intervalo_lembrete_minutos = db.Column(db.Integer, default=60)


def obter_configuracao():
    """Devolve a configuração de jornada (cria uma com valores padrão se ainda não existir)."""
    config = ConfiguracaoJornada.query.first()
    if not config:
        config = ConfiguracaoJornada()
        db.session.add(config)
        db.session.commit()
    return config


class PushSubscription(db.Model):
    """Inscrição de notificação push (Web Push) de um colaborador em um dispositivo/navegador."""
    id = db.Column(db.Integer, primary_key=True)
    colaborador_id = db.Column(db.Integer, db.ForeignKey("colaborador.id"), nullable=False)
    endpoint = db.Column(db.Text, nullable=False, unique=True)
    p256dh = db.Column(db.String(255), nullable=False)
    auth = db.Column(db.String(255), nullable=False)
    criado_em = db.Column(db.DateTime, default=datetime.utcnow)

    colaborador = db.relationship("Colaborador", backref="push_subscriptions")


class UltimoLembretePush(db.Model):
    """Controla quando foi o último push de lembrete enviado por colaborador/tipo,
    para respeitar o intervalo configurado sem duplicar envios."""
    id = db.Column(db.Integer, primary_key=True)
    colaborador_id = db.Column(db.Integer, db.ForeignKey("colaborador.id"), nullable=False)
    tipo = db.Column(db.String(20), nullable=False)  # entrada | saida
    enviado_em = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (db.UniqueConstraint("colaborador_id", "tipo", name="uniq_colaborador_tipo_lembrete"),)


# ---------------------------------------------------------------------------
# Helpers de autenticação
# ---------------------------------------------------------------------------
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


def gestor_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        user = Colaborador.query.get(session["user_id"])
        if not user or not user.is_gestor:
            flash("Acesso restrito ao gestor.")
            return redirect(url_for("ponto"))
        return f(*args, **kwargs)
    return wrapper


def imagem_base64_para_encoding(imagem_base64):
    """Recebe uma string base64 (data URL) e devolve o encoding facial (128-d) ou None."""
    if "," in imagem_base64:
        imagem_base64 = imagem_base64.split(",", 1)[1]
    dados = base64.b64decode(imagem_base64)
    imagem = face_recognition.load_image_file(io.BytesIO(dados))
    encodings = face_recognition.face_encodings(imagem)
    if not encodings:
        return None
    return encodings[0]


def obter_endereco(latitude, longitude):
    """Converte latitude/longitude em um endereço legível via geocodificação reversa (Nominatim/OpenStreetMap).
    Retorna None se as coordenadas não vierem ou se a consulta falhar (sem quebrar o registro do ponto)."""
    if latitude is None or longitude is None:
        return None
    try:
        resposta = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={
                "format": "jsonv2",
                "lat": latitude,
                "lon": longitude,
                "zoom": 18,
                "addressdetails": 1,
            },
            headers={"User-Agent": "controle-ponto-app/1.0"},
            timeout=5,
        )
        resposta.raise_for_status()
        dados = resposta.json()
        return dados.get("display_name")
    except Exception:
        return None


# A janela e o intervalo do lembrete agora são configuráveis pelo gestor em
# /gestor/configuracoes (tabela ConfiguracaoJornada), não mais por variável de ambiente.


def colaborador_tem_registro_hoje(colaborador_id):
    """Verifica se o colaborador já bateu ao menos um ponto hoje (horário de Brasília)."""
    return contar_registros_hoje(colaborador_id) > 0


def contar_registros_hoje(colaborador_id):
    """Conta quantas batidas de ponto o colaborador já fez hoje (horário de Brasília)."""
    hoje = agora_brasilia().date()
    registros_recentes = (
        RegistroPonto.query
        .filter_by(colaborador_id=colaborador_id)
        .order_by(RegistroPonto.data_hora.desc())
        .limit(10)
        .all()
    )
    return sum(1 for r in registros_recentes if para_brasilia(r.data_hora).date() == hoje)


def montar_resumo_diario(registros):
    """Agrupa registros por dia (Brasília) e estima horas trabalhadas pareando os horários
    em ordem (1º=entrada, 2º=saída, 3º=entrada...). Sinaliza dias com número ímpar de batidas."""
    from collections import defaultdict
    import datetime as _dt

    por_dia = defaultdict(list)
    for r in registros:
        dt_local = para_brasilia(r.data_hora)
        por_dia[dt_local.date()].append(dt_local)

    resumo = []
    for dia, horarios in sorted(por_dia.items(), reverse=True):
        horarios = sorted(horarios)
        total_segundos = 0
        for i in range(0, len(horarios) - 1, 2):
            total_segundos += (horarios[i + 1] - horarios[i]).total_seconds()
        horas = int(total_segundos // 3600)
        minutos = int((total_segundos % 3600) // 60)
        resumo.append({
            "data": dia,
            "batidas": horarios,
            "total_horas": f"{horas:02d}:{minutos:02d}",
            "incompleto": len(horarios) % 2 != 0,
        })
    return resumo


# ---------------------------------------------------------------------------
# Web Push (notificações reais, mesmo com o app fechado)
# ---------------------------------------------------------------------------
# As chaves são geradas uma única vez com "flask generate-vapid-keys" e salvas
# como variáveis de ambiente no Render. Sem elas, o push fica desativado (o app
# continua funcionando normalmente, só sem o lembrete de fundo).
VAPID_PRIVATE_KEY = os.environ.get("VAPID_PRIVATE_KEY")
VAPID_PUBLIC_KEY = os.environ.get("VAPID_PUBLIC_KEY")
VAPID_CONTACT_EMAIL = os.environ.get("VAPID_CONTACT_EMAIL", "contato@empresa.com")
PUSH_HABILITADO = bool(VAPID_PRIVATE_KEY and VAPID_PUBLIC_KEY)


def gerar_par_chaves_vapid():
    """Gera um par de chaves VAPID novo (privada, pública) já em formato urlsafe-base64,
    prontas para virarem variáveis de ambiente VAPID_PRIVATE_KEY / VAPID_PUBLIC_KEY."""
    from py_vapid import Vapid02
    from cryptography.hazmat.primitives import serialization

    def _b64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    v = Vapid02()
    v.generate_keys()

    priv_value = v.private_key.private_numbers().private_value.to_bytes(32, "big")
    pub_raw = v.public_key.public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    )
    return _b64url(priv_value), _b64url(pub_raw)


def enviar_push_colaborador(colaborador, titulo, corpo):
    """Envia uma notificação Web Push para todos os dispositivos inscritos do colaborador.
    Remove automaticamente inscrições que o navegador já invalidou (410/404)."""
    if not PUSH_HABILITADO:
        return

    payload = json.dumps({"title": titulo, "body": corpo})
    for sub in list(colaborador.push_subscriptions):
        try:
            webpush(
                subscription_info={
                    "endpoint": sub.endpoint,
                    "keys": {"p256dh": sub.p256dh, "auth": sub.auth},
                },
                data=payload,
                vapid_private_key=VAPID_PRIVATE_KEY,
                vapid_claims={"sub": f"mailto:{VAPID_CONTACT_EMAIL}"},
            )
        except WebPushException as ex:
            status = getattr(ex.response, "status_code", None)
            if status in (404, 410):
                db.session.delete(sub)
                db.session.commit()
        except Exception:
            # Nunca deixa uma falha de push derrubar o job do agendador.
            pass


def verificar_e_enviar_lembretes_push():
    """Roda periodicamente (APScheduler): verifica quem está devendo bater ponto
    e envia push respeitando o intervalo configurado pelo gestor."""
    if not PUSH_HABILITADO:
        return

    with app.app_context():
        config = obter_configuracao()
        agora_str = agora_brasilia().strftime("%H:%M")
        colaboradores = Colaborador.query.filter_by(is_gestor=False).all()

        for colaborador in colaboradores:
            if not colaborador.push_subscriptions:
                continue

            count = contar_registros_hoje(colaborador.id)
            pendencia = None
            if count == 0 and agora_str >= config.horario_entrada:
                pendencia = "entrada"
            elif count % 2 == 1 and agora_str >= config.horario_saida:
                pendencia = "saida"

            if not pendencia:
                continue

            ultimo = UltimoLembretePush.query.filter_by(
                colaborador_id=colaborador.id, tipo=pendencia
            ).first()

            if ultimo:
                minutos_desde_ultimo = (datetime.utcnow() - ultimo.enviado_em).total_seconds() / 60
                if minutos_desde_ultimo < config.intervalo_lembrete_minutos:
                    continue

            texto = (
                "Você ainda não bateu a ENTRADA hoje!"
                if pendencia == "entrada"
                else "Você ainda não bateu a SAÍDA hoje!"
            )
            enviar_push_colaborador(colaborador, "Lembrete de ponto", texto)

            if ultimo:
                ultimo.enviado_em = datetime.utcnow()
            else:
                db.session.add(UltimoLembretePush(
                    colaborador_id=colaborador.id, tipo=pendencia, enviado_em=datetime.utcnow()
                ))
            db.session.commit()


# ---------------------------------------------------------------------------
# Rotas - autenticação
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return redirect(url_for("login"))


@app.route("/api/push-subscribe", methods=["POST"])
@login_required
def push_subscribe():
    """Recebe a inscrição de Web Push do navegador do colaborador e salva no banco."""
    dados = request.get_json(silent=True) or {}
    endpoint = dados.get("endpoint")
    keys = dados.get("keys", {})
    p256dh = keys.get("p256dh")
    auth = keys.get("auth")

    if not (endpoint and p256dh and auth):
        return jsonify({"ok": False, "mensagem": "Dados de inscrição incompletos."}), 400

    existente = PushSubscription.query.filter_by(endpoint=endpoint).first()
    if existente:
        existente.colaborador_id = session["user_id"]
        existente.p256dh = p256dh
        existente.auth = auth
    else:
        db.session.add(PushSubscription(
            colaborador_id=session["user_id"], endpoint=endpoint, p256dh=p256dh, auth=auth
        ))
    db.session.commit()
    return jsonify({"ok": True})


@app.route("/healthz")
def healthz():
    """Endpoint leve para um pinger externo (ex.: cron-job.org, UptimeRobot) manter o
    serviço acordado no plano gratuito do Render, o que é necessário para o agendador
    de lembretes (APScheduler) continuar rodando."""
    return jsonify({"status": "ok"})


@app.route("/service-worker.js")
def service_worker():
    """Serve o service worker a partir da raiz do site (não de /static/), para que
    o escopo dele cubra o app inteiro — essencial para notificações no iOS."""
    resposta = send_from_directory(
        os.path.join(app.root_path, "static"), "service-worker.js", mimetype="application/javascript"
    )
    resposta.headers["Service-Worker-Allowed"] = "/"
    return resposta


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        senha = request.form["senha"]
        user = Colaborador.query.filter_by(email=email).first()
        if user and user.check_senha(senha):
            session["user_id"] = user.id
            session["nome"] = user.nome
            session["is_gestor"] = user.is_gestor
            if user.is_gestor:
                return redirect(url_for("gestor_consulta"))
            return redirect(url_for("ponto"))
        flash("E-mail ou senha inválidos.")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Rotas - colaborador
# ---------------------------------------------------------------------------
@app.route("/ponto", methods=["GET"])
@login_required
def ponto():
    config = obter_configuracao()
    registros_hoje_count = contar_registros_hoje(session["user_id"])
    return render_template(
        "ponto.html",
        nome=session.get("nome"),
        registros_hoje_count=registros_hoje_count,
        horario_entrada=config.horario_entrada,
        horario_saida=config.horario_saida,
        intervalo_lembrete_minutos=config.intervalo_lembrete_minutos,
        vapid_public_key=VAPID_PUBLIC_KEY or "",
        push_habilitado=PUSH_HABILITADO,
    )


@app.route("/api/registrar-ponto", methods=["POST"])
@login_required
def registrar_ponto():
    dados = request.get_json()
    foto_base64 = dados.get("foto")
    latitude = dados.get("latitude")
    longitude = dados.get("longitude")

    if not foto_base64:
        return jsonify({"ok": False, "mensagem": "Nenhuma foto recebida."}), 400

    user = Colaborador.query.get(session["user_id"])
    if user.face_encoding is None:
        return jsonify({
            "ok": False,
            "mensagem": "Seu rosto ainda não foi cadastrado. Procure o gestor."
        }), 400

    encoding_atual = imagem_base64_para_encoding(foto_base64)
    if encoding_atual is None:
        return jsonify({"ok": False, "mensagem": "Não foi possível identificar um rosto na imagem. Tente novamente com boa iluminação."}), 400

    distancia = np.linalg.norm(np.array(user.face_encoding) - np.array(encoding_atual))

    if distancia > FACE_MATCH_TOLERANCE:
        return jsonify({
            "ok": False,
            "mensagem": "O rosto não confere com o cadastrado. Registro não realizado."
        }), 403

    endereco = obter_endereco(latitude, longitude)

    registros_hoje_count = contar_registros_hoje(user.id)
    tipo_registro = "entrada" if registros_hoje_count % 2 == 0 else "saida"

    registro = RegistroPonto(
        colaborador_id=user.id,
        latitude=latitude,
        longitude=longitude,
        endereco=endereco,
        distancia_facial=float(distancia),
        origem="facial",
        tipo=tipo_registro,
    )
    db.session.add(registro)
    db.session.commit()

    horario_local = para_brasilia(registro.data_hora)
    rotulo_tipo = "Entrada" if tipo_registro == "entrada" else "Saída"
    mensagem = f"{rotulo_tipo} registrada às {horario_local.strftime('%d/%m/%Y %H:%M:%S')} (horário de Brasília)."
    if endereco:
        mensagem += f" Local: {endereco}."
    return jsonify({
        "ok": True,
        "mensagem": mensagem,
        "horario": horario_local.strftime("%H:%M:%S"),
        "data": horario_local.strftime("%d/%m/%Y"),
        "endereco": endereco,
        "tipo": tipo_registro,
        "registros_hoje_count": registros_hoje_count + 1,
    })


@app.route("/meus-registros", methods=["GET"])
@login_required
def meus_registros():
    user = Colaborador.query.get(session["user_id"])
    registros = (
        RegistroPonto.query
        .filter_by(colaborador_id=user.id)
        .order_by(RegistroPonto.data_hora.desc())
        .limit(200)
        .all()
    )
    resumo_diario = montar_resumo_diario(registros)

    solicitacoes = (
        SolicitacaoAjuste.query
        .filter_by(colaborador_id=user.id)
        .order_by(SolicitacaoAjuste.criado_em.desc())
        .limit(30)
        .all()
    )

    return render_template(
        "meus_registros.html",
        registros=registros,
        resumo_diario=resumo_diario,
        solicitacoes=solicitacoes,
        hoje=agora_brasilia().date(),
    )


@app.route("/meus-registros/solicitar-ajuste", methods=["POST"])
@login_required
def solicitar_ajuste():
    data_referencia_str = request.form.get("data_referencia")
    tipo = request.form.get("tipo")
    horario_solicitado = request.form.get("horario_solicitado")
    motivo = request.form.get("motivo", "").strip()

    if not (data_referencia_str and tipo and horario_solicitado and motivo):
        flash("Preencha todos os campos da solicitação de ajuste.")
        return redirect(url_for("meus_registros"))

    try:
        data_referencia = datetime.strptime(data_referencia_str, "%Y-%m-%d").date()
    except ValueError:
        flash("Data inválida.")
        return redirect(url_for("meus_registros"))

    solicitacao = SolicitacaoAjuste(
        colaborador_id=session["user_id"],
        data_referencia=data_referencia,
        tipo=tipo,
        horario_solicitado=horario_solicitado,
        motivo=motivo,
    )
    db.session.add(solicitacao)
    db.session.commit()
    flash("Solicitação de ajuste enviada. Aguarde a aprovação do gestor.")
    return redirect(url_for("meus_registros"))


# ---------------------------------------------------------------------------
# Rotas - gestor
# ---------------------------------------------------------------------------
@app.route("/gestor", methods=["GET"])
@gestor_required
def gestor_dashboard():
    return redirect(url_for("gestor_consulta"))


@app.route("/gestor/consulta", methods=["GET"])
@gestor_required
def gestor_consulta():
    registros = (
        RegistroPonto.query.order_by(RegistroPonto.data_hora.desc()).limit(200).all()
    )
    colaboradores = Colaborador.query.filter_by(is_gestor=False).order_by(Colaborador.nome).all()

    hoje_brasilia = agora_brasilia().date()
    registros_hoje = sum(
        1 for r in registros if para_brasilia(r.data_hora).date() == hoje_brasilia
    )
    total_colaboradores = len(colaboradores)
    sem_cadastro_facial = sum(1 for c in colaboradores if not c.face_encoding)

    colaboradores_sem_registro_hoje = [
        c for c in colaboradores if not colaborador_tem_registro_hoje(c.id)
    ]

    ajustes_pendentes = SolicitacaoAjuste.query.filter_by(status="pendente").count()

    return render_template(
        "gestor_consulta.html",
        registros=registros,
        colaboradores=colaboradores,
        registros_hoje=registros_hoje,
        total_colaboradores=total_colaboradores,
        sem_cadastro_facial=sem_cadastro_facial,
        colaboradores_sem_registro_hoje=colaboradores_sem_registro_hoje,
        ajustes_pendentes=ajustes_pendentes,
    )


@app.route("/gestor/cadastro", methods=["GET"])
@gestor_required
def gestor_cadastro():
    colaboradores = Colaborador.query.filter_by(is_gestor=False).order_by(Colaborador.nome).all()
    return render_template("gestor_cadastro.html", colaboradores=colaboradores)


@app.route("/gestor/configuracoes", methods=["GET"])
@gestor_required
def gestor_configuracoes():
    config = obter_configuracao()
    return render_template("gestor_configuracoes.html", config=config, push_habilitado=PUSH_HABILITADO)


@app.route("/gestor/configuracoes/gerar-chaves-push", methods=["POST"])
@gestor_required
def gerar_chaves_push():
    """Gera um novo par de chaves VAPID direto pela interface, sem precisar de terminal/shell.
    As chaves são mostradas UMA VEZ na tela para o gestor copiar e colar no Render."""
    chave_privada, chave_publica = gerar_par_chaves_vapid()
    config = obter_configuracao()
    return render_template(
        "gestor_configuracoes.html",
        config=config,
        push_habilitado=PUSH_HABILITADO,
        chave_privada_gerada=chave_privada,
        chave_publica_gerada=chave_publica,
    )


@app.route("/gestor/configuracoes/salvar", methods=["POST"])
@gestor_required
def salvar_configuracoes():
    config = obter_configuracao()

    horario_entrada = request.form.get("horario_entrada", "").strip()
    horario_saida = request.form.get("horario_saida", "").strip()
    intervalo = request.form.get("intervalo_lembrete_minutos", "").strip()

    def _valido_hhmm(valor):
        try:
            h, m = valor.split(":")
            return 0 <= int(h) <= 23 and 0 <= int(m) <= 59
        except Exception:
            return False

    if not (_valido_hhmm(horario_entrada) and _valido_hhmm(horario_saida)):
        flash("Informe horários válidos no formato HH:MM.")
        return redirect(url_for("gestor_configuracoes"))

    try:
        intervalo_int = int(intervalo)
        if intervalo_int < 5 or intervalo_int > 240:
            raise ValueError
    except Exception:
        flash("O intervalo do lembrete deve ser um número entre 5 e 240 minutos.")
        return redirect(url_for("gestor_configuracoes"))

    config.horario_entrada = horario_entrada
    config.horario_saida = horario_saida
    config.intervalo_lembrete_minutos = intervalo_int
    db.session.commit()

    flash("Configurações de jornada e lembretes atualizadas com sucesso.")
    return redirect(url_for("gestor_configuracoes"))


@app.route("/gestor/cadastrar", methods=["POST"])
@gestor_required
def cadastrar_colaborador():
    nome = request.form["nome"].strip()
    email = request.form["email"].strip().lower()
    senha = request.form["senha"]
    foto_base64 = request.form.get("foto")

    if Colaborador.query.filter_by(email=email).first():
        flash("Já existe um colaborador com esse e-mail.")
        return redirect(url_for("gestor_cadastro"))

    novo = Colaborador(nome=nome, email=email, is_gestor=False)
    novo.set_senha(senha)

    if foto_base64:
        encoding = imagem_base64_para_encoding(foto_base64)
        if encoding is None:
            flash("Não foi possível reconhecer um rosto na foto enviada. Colaborador criado sem cadastro facial.")
        else:
            novo.face_encoding = encoding.tolist()

    db.session.add(novo)
    db.session.commit()
    flash(f"Colaborador {nome} cadastrado com sucesso.")
    return redirect(url_for("gestor_cadastro"))


@app.route("/gestor/exportar-csv")
@gestor_required
def exportar_csv():
    registros = RegistroPonto.query.order_by(RegistroPonto.data_hora.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Colaborador", "Data/Hora (Horário de Brasília)", "Endereço", "Latitude", "Longitude", "Distância Facial"])
    for r in registros:
        horario_local = para_brasilia(r.data_hora)
        writer.writerow([
            r.colaborador.nome,
            horario_local.strftime("%d/%m/%Y %H:%M:%S"),
            r.endereco or "",
            r.latitude,
            r.longitude,
            round(r.distancia_facial, 4) if r.distancia_facial is not None else "",
        ])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=relatorio_ponto.csv"},
    )


# ---------------------------------------------------------------------------
# Rotas - gestor: solicitações de ajuste de ponto
# ---------------------------------------------------------------------------
@app.route("/gestor/ajustes", methods=["GET"])
@gestor_required
def gestor_ajustes():
    pendentes = (
        SolicitacaoAjuste.query
        .filter_by(status="pendente")
        .order_by(SolicitacaoAjuste.criado_em.asc())
        .all()
    )
    historico = (
        SolicitacaoAjuste.query
        .filter(SolicitacaoAjuste.status != "pendente")
        .order_by(SolicitacaoAjuste.respondido_em.desc())
        .limit(50)
        .all()
    )
    return render_template("gestor_ajustes.html", pendentes=pendentes, historico=historico)


@app.route("/gestor/ajustes/<int:ajuste_id>/aprovar", methods=["POST"])
@gestor_required
def aprovar_ajuste(ajuste_id):
    solicitacao = SolicitacaoAjuste.query.get_or_404(ajuste_id)

    if solicitacao.status != "pendente":
        flash("Esta solicitação já foi respondida.")
        return redirect(url_for("gestor_ajustes"))

    try:
        hora, minuto = (int(p) for p in solicitacao.horario_solicitado.split(":"))
        dt_local = datetime.combine(solicitacao.data_referencia, datetime.min.time()).replace(
            hour=hora, minute=minuto, tzinfo=TZ_BRASILIA
        )
        dt_utc_naive = dt_local.astimezone(TZ_UTC).replace(tzinfo=None)
    except Exception:
        flash("Não foi possível interpretar o horário solicitado.")
        return redirect(url_for("gestor_ajustes"))

    registro = RegistroPonto(
        colaborador_id=solicitacao.colaborador_id,
        data_hora=dt_utc_naive,
        tipo=solicitacao.tipo,
        endereco="Ajuste manual aprovado pelo gestor",
        origem="ajuste_manual",
    )
    db.session.add(registro)

    solicitacao.status = "aprovado"
    solicitacao.respondido_em = datetime.utcnow()
    solicitacao.resposta_gestor = request.form.get("resposta", "").strip()
    db.session.commit()

    flash(f"Ajuste aprovado para {solicitacao.colaborador.nome}.")
    return redirect(url_for("gestor_ajustes"))


@app.route("/gestor/ajustes/<int:ajuste_id>/recusar", methods=["POST"])
@gestor_required
def recusar_ajuste(ajuste_id):
    solicitacao = SolicitacaoAjuste.query.get_or_404(ajuste_id)

    if solicitacao.status != "pendente":
        flash("Esta solicitação já foi respondida.")
        return redirect(url_for("gestor_ajustes"))

    solicitacao.status = "recusado"
    solicitacao.respondido_em = datetime.utcnow()
    solicitacao.resposta_gestor = request.form.get("resposta", "").strip()
    db.session.commit()

    flash(f"Ajuste recusado para {solicitacao.colaborador.nome}.")
    return redirect(url_for("gestor_ajustes"))


# ---------------------------------------------------------------------------
# Inicialização / seed do primeiro gestor
# ---------------------------------------------------------------------------
@app.cli.command("init-db")
def init_db():
    """Cria as tabelas e um usuário gestor inicial. Rodar com: flask init-db"""
    db.create_all()
    _garantir_colunas_novas()
    email_gestor = os.environ.get("GESTOR_EMAIL", "gestor@empresa.com")
    if not Colaborador.query.filter_by(email=email_gestor).first():
        gestor = Colaborador(nome="Gestor", email=email_gestor, is_gestor=True)
        gestor.set_senha(os.environ.get("GESTOR_SENHA", "mude-esta-senha"))
        db.session.add(gestor)
        db.session.commit()
        print(f"Gestor criado: {email_gestor}")
    else:
        print("Gestor já existe.")


@app.cli.command("generate-vapid-keys")
def generate_vapid_keys():
    """Gera um par de chaves VAPID para o Web Push. Rodar UMA VEZ com: flask generate-vapid-keys
    e copiar a saída para as variáveis de ambiente VAPID_PUBLIC_KEY e VAPID_PRIVATE_KEY no Render."""
    chave_privada, chave_publica = gerar_par_chaves_vapid()

    print("\nCopie estas duas variáveis para o Render (Settings → Environment):\n")
    print(f"VAPID_PRIVATE_KEY={chave_privada}")
    print(f"VAPID_PUBLIC_KEY={chave_publica}")
    print("\nOpcional (aparece na mensagem enviada ao navegador como contato do remetente):")
    print("VAPID_CONTACT_EMAIL=seu-email@empresa.com\n")
    print("Depois de configurar, reinicie o serviço no Render para o push ser ativado.")


# ---------------------------------------------------------------------------
# Migração leve (sem Flask-Migrate): garante que colunas novas existam
# em bancos já criados anteriormente (ex.: banco Postgres já em produção).
# ---------------------------------------------------------------------------
def _garantir_colunas_novas():
    from sqlalchemy import inspect, text

    inspector = inspect(db.engine)
    colunas = [col["name"] for col in inspector.get_columns("registro_ponto")]
    comandos = []
    if "endereco" not in colunas:
        comandos.append("ALTER TABLE registro_ponto ADD COLUMN endereco VARCHAR(255)")
    if "origem" not in colunas:
        comandos.append("ALTER TABLE registro_ponto ADD COLUMN origem VARCHAR(20)")

    if comandos:
        with db.engine.connect() as conn:
            for comando in comandos:
                conn.execute(text(comando))
            conn.commit()


with app.app_context():
    db.create_all()
    _garantir_colunas_novas()
    email_gestor = os.environ.get("GESTOR_EMAIL", "gestor@empresa.com")
    if not Colaborador.query.filter_by(email=email_gestor).first():
        gestor = Colaborador(nome="Gestor", email=email_gestor, is_gestor=True)
        gestor.set_senha(os.environ.get("GESTOR_SENHA", "mude-esta-senha"))
        db.session.add(gestor)
        db.session.commit()


# ---------------------------------------------------------------------------
# Agendador de lembretes (Web Push) — roda dentro do próprio processo Flask.
#
# IMPORTANTE (plano gratuito do Render): o serviço "dorme" após ~15 min sem
# receber requisições, e junto com ele esse agendador também para. Configure
# um pinger externo gratuito (ex.: cron-job.org, UptimeRobot) para acessar
# GET /healthz a cada 5-10 minutos e manter o serviço acordado.
# ---------------------------------------------------------------------------
if PUSH_HABILITADO and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
    scheduler = BackgroundScheduler(timezone="America/Sao_Paulo")
    scheduler.add_job(verificar_e_enviar_lembretes_push, "interval", minutes=5, id="lembretes_push")
    scheduler.start()
elif not PUSH_HABILITADO:
    print(
        "[push] VAPID_PUBLIC_KEY/VAPID_PRIVATE_KEY não configuradas — lembretes por "
        "notificação push desativados. Rode 'flask generate-vapid-keys' para gerar as chaves."
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

