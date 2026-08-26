import os
import io
import csv
import base64
import json
import time
import threading
import numpy as np
import requests
from datetime import datetime, timedelta, date
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

# Evita os erros "SSL SYSCALL error: EOF detected" / "SSL error: decryption failed"
# que aparecem quando o Postgres do Render fecha conexões ociosas (ex.: depois do
# app "dormir" no plano gratuito) e o pool tenta reaproveitar uma conexão morta.
# pool_pre_ping testa a conexão antes de usar (reconecta sozinho se necessário) e
# pool_recycle descarta conexões antigas antes que o servidor as feche primeiro.
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_pre_ping": True,
    "pool_recycle": 280,
}

db = SQLAlchemy(app)

# Distância máxima aceita entre encodings faciais para considerar "é a mesma pessoa".
# Quanto menor, mais rígido. 0.6 é o padrão "solto" da biblioteca face_recognition;
# usamos um valor mais estrito porque é melhor pedir pra tentar de novo do que
# aceitar o rosto de outra pessoa por engano.
FACE_MATCH_TOLERANCE = 0.45

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
    """Disponibiliza o número de ajustes pendentes e a configuração de marca (logo/cores)
    para qualquer página, sem precisar passar manualmente em cada rota."""
    contexto = {"marca": obter_marca()}
    if session.get("is_gestor"):
        contexto["ajustes_pendentes_nav"] = SolicitacaoAjuste.query.filter_by(status="pendente").count()
    return contexto


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


class AprovacaoJornadaExcepcional(db.Model):
    """Registro de aprovação para dias em que a sessão trabalhada (entrada->saída) ultrapassou
    LIMITE_HORAS_EXCEPCIONAL horas seguidas (ex.: virada de turno, esquecimento de bater saída
    seguido de nova entrada 24h depois etc).

    Enquanto não houver aprovação do gestor aqui, essas horas NÃO entram no total de hora extra
    automaticamente — ficam visíveis (o colaborador e o gestor veem o total real trabalhado), mas
    só contam pra folha depois de uma revisão humana. Isso evita que uma jornada fora do padrão
    vire hora extra sozinha sem ninguém checar se foi real ou um esquecimento de bater o ponto."""
    id = db.Column(db.Integer, primary_key=True)
    colaborador_id = db.Column(db.Integer, db.ForeignKey("colaborador.id"), nullable=False)
    data_referencia = db.Column(db.Date, nullable=False)  # dia da ENTRADA da sessão excepcional
    horas_total = db.Column(db.Float, nullable=False)  # horas trabalhadas nessa sessão, calculadas
    status = db.Column(db.String(20), default="pendente")  # pendente | aprovado | recusado
    criado_em = db.Column(db.DateTime, default=datetime.utcnow)
    respondido_em = db.Column(db.DateTime, nullable=True)
    resposta_gestor = db.Column(db.Text, nullable=True)
    registro_entrada_id = db.Column(db.Integer, db.ForeignKey("registro_ponto.id"), nullable=True)
    registro_saida_id = db.Column(db.Integer, db.ForeignKey("registro_ponto.id"), nullable=True)

    colaborador = db.relationship("Colaborador", backref="aprovacoes_jornada_excepcional")

    __table_args__ = (
        db.UniqueConstraint("colaborador_id", "data_referencia", name="uq_aprovacao_excepcional_colaborador_dia"),
    )


class ConfiguracaoJornada(db.Model):
    """Configuração única (definida pelo gestor) da jornada padrão e do lembrete de ponto."""
    id = db.Column(db.Integer, primary_key=True)
    horario_entrada = db.Column(db.String(5), default="08:00")
    horario_saida = db.Column(db.String(5), default="18:00")
    intervalo_lembrete_minutos = db.Column(db.Integer, default=60)
    meta_horas_diarias = db.Column(db.Float, default=8.0)


def obter_configuracao():
    """Devolve a configuração de jornada (cria uma com valores padrão se ainda não existir)."""
    config = ConfiguracaoJornada.query.first()
    if not config:
        config = ConfiguracaoJornada()
        db.session.add(config)
        db.session.commit()
    return config


class ConfiguracaoMarca(db.Model):
    """Configuração de marca (white-label): permite que cada gestor personalize o
    nome da empresa, as logos e as cores do sistema — feito para o app ser vendido
    e reaproveitado por diferentes clientes/empresas."""
    id = db.Column(db.Integer, primary_key=True)
    nome_empresa = db.Column(db.String(60), default="SmartPoint")
    logo_login_base64 = db.Column(db.Text, nullable=True)    # logo grande, só na tela de login
    logo_topbar_base64 = db.Column(db.Text, nullable=True)   # logo pequena, no cabeçalho do app
    icone_192_base64 = db.Column(db.Text, nullable=True)     # ícone PWA 192x192 (gerado a partir da logo do cabeçalho)
    icone_512_base64 = db.Column(db.Text, nullable=True)     # ícone PWA 512x512
    cor_primaria = db.Column(db.String(7), default="#0f5fa8")
    cor_secundaria = db.Column(db.String(7), default="#17b26a")

    # Nome de coluna antigo (mantido só para compatibilidade com bancos já existentes,
    # onde a logo enviada antes virava tanto login quanto cabeçalho ao mesmo tempo).
    logo_header_base64 = db.Column(db.Text, nullable=True)


def obter_marca():
    """Devolve a configuração de marca (cria uma com valores padrão se ainda não existir)."""
    marca = ConfiguracaoMarca.query.first()
    if not marca:
        marca = ConfiguracaoMarca()
        db.session.add(marca)
        db.session.commit()
    elif marca.logo_header_base64 and not marca.logo_login_base64:
        # Compatibilidade: em versões antigas, a única logo enviada virava a do login.
        marca.logo_login_base64 = marca.logo_header_base64
        db.session.commit()
    return marca


def redimensionar_para_base64(arquivo_bytes, altura_max):
    """Redimensiona uma imagem (mantendo a proporção) para uma altura máxima em px
    e devolve um data-URI PNG pronto para salvar no banco."""
    from PIL import Image
    import io as _io

    img = Image.open(_io.BytesIO(arquivo_bytes)).convert("RGBA")
    if img.height > altura_max:
        proporcao = altura_max / img.height
        img = img.resize((max(1, int(img.width * proporcao)), altura_max), Image.LANCZOS)
    buf = _io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def gerar_icones_pwa(arquivo_bytes):
    """Gera as duas variantes quadradas (192px/512px, fundo branco) usadas como
    ícone do PWA / Tela de Início, a partir da imagem enviada para o cabeçalho."""
    from PIL import Image
    import io as _io

    img = Image.open(_io.BytesIO(arquivo_bytes)).convert("RGBA")

    def _icone_quadrado(tamanho):
        lado = max(img.width, img.height)
        fundo = Image.new("RGBA", (lado, lado), (255, 255, 255, 255))
        offset = ((lado - img.width) // 2, (lado - img.height) // 2)
        fundo.paste(img, offset, img)
        fundo = fundo.resize((tamanho, tamanho), Image.LANCZOS).convert("RGB")
        buf = _io.BytesIO()
        fundo.save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    return _icone_quadrado(192), _icone_quadrado(512)



@app.template_filter("escurecer")
def filtro_escurecer(cor_hex, fator=0.3):
    """Escurece uma cor hex (#rrggbb) em X% — usado para gerar tons de hover/gradiente
    automaticamente a partir da cor primária escolhida pelo gestor."""
    try:
        cor_hex = (cor_hex or "").lstrip("#")
        r, g, b = int(cor_hex[0:2], 16), int(cor_hex[2:4], 16), int(cor_hex[4:6], 16)
        r, g, b = (max(0, int(c * (1 - fator))) for c in (r, g, b))
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return cor_hex


@app.template_filter("clarear")
def filtro_clarear(cor_hex, fator=0.85):
    """Clareia uma cor hex misturando com branco (fator = proporção de branco, 0 a 1) —
    usado para gerar o tom clarinho de fundo (ex.: badges, hover leve) a partir da cor
    primária escolhida pelo gestor."""
    try:
        cor_hex = (cor_hex or "").lstrip("#")
        r, g, b = int(cor_hex[0:2], 16), int(cor_hex[2:4], 16), int(cor_hex[4:6], 16)
        r, g, b = (min(255, int(c + (255 - c) * fator)) for c in (r, g, b))
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return cor_hex


class AssistenteInteracao(db.Model):
    """Cada pergunta feita ao assistente e a resposta dada. É a base do
    aprendizado: guarda o que foi perguntado, o que o motor entendeu e se a
    resposta foi útil."""
    id = db.Column(db.Integer, primary_key=True)
    colaborador_id = db.Column(db.Integer, db.ForeignKey("colaborador.id"), nullable=True)
    pergunta = db.Column(db.Text, nullable=False)
    pergunta_normalizada = db.Column(db.Text, nullable=False)
    intencao = db.Column(db.String(40), nullable=True)
    confianca = db.Column(db.Float, default=0.0)
    colaborador_alvo_id = db.Column(db.Integer, db.ForeignKey("colaborador.id"), nullable=True)
    resposta = db.Column(db.Text, nullable=True)
    util = db.Column(db.Boolean, nullable=True)  # None = ainda sem avaliação
    criado_em = db.Column(db.DateTime, default=datetime.utcnow)

    colaborador = db.relationship("Colaborador", foreign_keys=[colaborador_id])
    colaborador_alvo = db.relationship("Colaborador", foreign_keys=[colaborador_alvo_id])


class AssistentePadrao(db.Model):
    """Padrão de pergunta aprendido com o uso: quando o gestor marca uma
    resposta como útil, a forma como ele escreveu a pergunta passa a valer como
    exemplo daquela intenção. Perguntas futuras parecidas caem na mesma resposta
    mesmo sem bater nas regras fixas."""
    id = db.Column(db.Integer, primary_key=True)
    padrao = db.Column(db.Text, nullable=False)
    intencao = db.Column(db.String(40), nullable=False)
    peso = db.Column(db.Float, default=1.0)
    usos = db.Column(db.Integer, default=1)
    atualizado_em = db.Column(db.DateTime, default=datetime.utcnow)


class AssistenteApelido(db.Model):
    """Apelido/forma de escrever aprendida para um colaborador ('jô' -> Joana
    Ribeiro), ensinada quando o gestor corrige o assistente."""
    id = db.Column(db.Integer, primary_key=True)
    apelido = db.Column(db.String(60), nullable=False, unique=True)
    colaborador_id = db.Column(db.Integer, db.ForeignKey("colaborador.id"), nullable=False)
    acertos = db.Column(db.Integer, default=1)
    criado_em = db.Column(db.DateTime, default=datetime.utcnow)

    colaborador = db.relationship("Colaborador", backref="apelidos_assistente")


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
        user = Colaborador.query.get(session["user_id"])
        if not user or not user.ativo:
            session.clear()
            flash("Sua conta foi desativada. Procure o gestor.")
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


MENSAGENS_POSICIONAMENTO = {
    "sem_rosto": "Não conseguimos identificar um rosto. Centralize seu rosto na câmera com boa iluminação.",
    "muito_longe": "Aproxime um pouco mais o rosto da câmera.",
    "muito_perto": "Afaste um pouco o rosto da câmera.",
}


def imagem_base64_para_encoding(imagem_base64, num_jitters=2):
    """Recebe uma string base64 (data URL) e devolve (encoding, motivo).

    - Sucesso: (encoding_128d, None)
    - Falha: (None, motivo) — motivo é uma chave de MENSAGENS_POSICIONAMENTO,
      para dar um retorno específico ("aproxime-se"/"afaste-se") em vez de um
      genérico "não deu certo".

    Cuidados importantes para evitar falso-positivo entre pessoas diferentes:
    - Se houver mais de um rosto na foto (ex.: alguém passando atrás), usa o MAIOR
      rosto detectado (mais próximo da câmera) em vez de simplesmente o primeiro
      da lista, que não tem ordem garantida.
    - num_jitters > 1 faz o dlib reamostrar o rosto algumas vezes e tirar a média,
      gerando um encoding mais estável/confiável (mais lento, mas o ponto não é
      uma operação de alta frequência)."""
    if "," in imagem_base64:
        imagem_base64 = imagem_base64.split(",", 1)[1]
    dados = base64.b64decode(imagem_base64)
    imagem = face_recognition.load_image_file(io.BytesIO(dados))

    localizacoes = face_recognition.face_locations(imagem)
    if not localizacoes:
        return None, "sem_rosto"

    if len(localizacoes) > 1:
        # (top, right, bottom, left) -> escolhe a maior área, isto é, o rosto mais em destaque.
        localizacoes = [max(localizacoes, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))]

    # Verifica se o rosto está num tamanho razoável dentro do quadro — muito pequeno
    # (pessoa longe) prejudica a precisão do reconhecimento tanto quanto muito
    # grande (pessoa colada na câmera, cortando parte do rosto).
    altura_img = len(imagem)
    largura_img = len(imagem[0]) if altura_img else 0
    top, right, bottom, left = localizacoes[0]
    area_rosto = max(0, bottom - top) * max(0, right - left)
    area_imagem = altura_img * largura_img
    proporcao = (area_rosto / area_imagem) if area_imagem else 0

    if proporcao < 0.035:
        return None, "muito_longe"
    if proporcao > 0.45:
        return None, "muito_perto"

    encodings = face_recognition.face_encodings(imagem, known_face_locations=localizacoes, num_jitters=num_jitters)
    if not encodings:
        return None, "sem_rosto"
    return encodings[0], None


# ---------------------------------------------------------------------------
# Geocodificação reversa (coordenadas -> endereço legível)
# ---------------------------------------------------------------------------
# O Nominatim público limita 1 requisição/segundo POR IP e devolve 429 ("Too many
# requests") para o IP inteiro. Em hospedagem compartilhada (Render) esse IP é dividido
# com outros apps, então o 429 aparece mesmo quando NÓS mandamos pouca coisa. Defesas:
#   1) cache: a mesma coordenada (o pátio da empresa) é resolvida uma única vez;
#   2) throttle global + backoff: no máximo 1 chamada/segundo saindo daqui por provedor;
#   3) provedores alternativos: se o Nominatim recusar, tenta Photon e BigDataCloud;
#   4) cache negativo curto: uma coordenada que acabou de falhar não é remartelada.
GEOCODE_INTERVALO_MINIMO = 1.2   # segundos entre duas chamadas ao mesmo provedor
GEOCODE_TTL_FALHA = 300          # segundos ignorando uma coordenada que acabou de falhar
GEOCODE_TTL_BLOQUEIO = 900       # segundos pulando um provedor que respondeu 429/503
GEOCODE_CACHE_MAX = 2000
# Opcional: com uma chave do LocationIQ (plano gratuito, 5k req/dia) o app deixa de
# depender do servidor público e o 429 some. Basta definir LOCATIONIQ_API_KEY no Render.
LOCATIONIQ_API_KEY = os.environ.get("LOCATIONIQ_API_KEY", "").strip()

_cache_enderecos = {}
_cache_falhas = {}
_provedor_bloqueado_ate = {}
_geocode_lock = threading.Lock()
_geocode_ultima_chamada = {}


def _chave_geocode(latitude, longitude):
    """Arredonda para ~1 metro: batidas no mesmo local caem todas na mesma chave de cache."""
    return (round(float(latitude), 5), round(float(longitude), 5))


def _aguardar_vez(provedor):
    """Serializa as chamadas e garante o intervalo mínimo exigido pelo provedor."""
    with _geocode_lock:
        espera = GEOCODE_INTERVALO_MINIMO - (time.monotonic() - _geocode_ultima_chamada.get(provedor, 0.0))
        if espera > 0:
            time.sleep(espera)
        _geocode_ultima_chamada[provedor] = time.monotonic()


def _provedor_indisponivel(provedor):
    """Um 429 não é passageiro: no IP compartilhado do Render o Nominatim recusa tudo por
    um bom tempo. Depois do primeiro 429 paramos de bater nele por alguns minutos e vamos
    direto no provedor alternativo — assim a batida do ponto não perde tempo tentando."""
    ate = _provedor_bloqueado_ate.get(provedor)
    return bool(ate) and time.monotonic() < ate


def _bloquear_provedor(provedor, motivo):
    _provedor_bloqueado_ate[provedor] = time.monotonic() + GEOCODE_TTL_BLOQUEIO
    print(f"[endereco] {provedor} em cooldown por {GEOCODE_TTL_BLOQUEIO}s ({motivo})")


def _guardar_no_cache(chave, endereco):
    if len(_cache_enderecos) >= GEOCODE_CACHE_MAX:
        _cache_enderecos.clear()
    _cache_enderecos[chave] = endereco


def _endereco_ja_conhecido(chave):
    """Procura no banco um registro na MESMA coordenada que já tenha endereço resolvido.
    É um cache que sobrevive ao restart do processo (o de memória não) e evita ir na rede
    para o endereço da empresa, que se repete em praticamente todas as batidas."""
    latitude, longitude = chave
    delta = 0.00005
    try:
        registro = (
            RegistroPonto.query
            .filter(RegistroPonto.endereco.isnot(None))
            .filter(RegistroPonto.latitude.between(latitude - delta, latitude + delta))
            .filter(RegistroPonto.longitude.between(longitude - delta, longitude + delta))
            .first()
        )
        return registro.endereco if registro else None
    except Exception:
        # Sem contexto de app / banco indisponível: segue para a rede normalmente.
        return None


def _montar_endereco(partes):
    """Junta as partes ignorando vazios e repetições (ex.: cidade == município)."""
    resultado = []
    for parte in partes:
        parte = (parte or "").strip()
        if parte and parte not in resultado:
            resultado.append(parte)
    return ", ".join(resultado) or None


def _provedor_locationiq(latitude, longitude):
    resposta = requests.get(
        "https://us1.locationiq.com/v1/reverse",
        params={
            "key": LOCATIONIQ_API_KEY,
            "lat": latitude,
            "lon": longitude,
            "format": "json",
            "zoom": 18,
            "accept-language": "pt-BR",
        },
        timeout=8,
    )
    if resposta.status_code != 200:
        return None, f"status {resposta.status_code}"
    return (resposta.json() or {}).get("display_name"), "sem display_name"


def _provedor_nominatim(latitude, longitude):
    resposta = requests.get(
        "https://nominatim.openstreetmap.org/reverse",
        params={
            "format": "jsonv2",
            "lat": latitude,
            "lon": longitude,
            "zoom": 18,
            "addressdetails": 1,
        },
        headers={
            # O Nominatim exige um User-Agent que identifique a aplicação (política de uso
            # deles); sem isso as requisições podem ser bloqueadas silenciosamente.
            "User-Agent": f"SmartPoint-ControleDePonto/1.0 (contato: {VAPID_CONTACT_EMAIL})",
            "Accept-Language": "pt-BR",
        },
        timeout=8,
    )
    if resposta.status_code != 200:
        return None, f"status {resposta.status_code}"
    return (resposta.json() or {}).get("display_name"), "sem display_name"


def _provedor_photon(latitude, longitude):
    """Photon (komoot) — mesma base do OpenStreetMap, sem chave e sem o limite agressivo."""
    resposta = requests.get(
        "https://photon.komoot.io/reverse",
        params={"lat": latitude, "lon": longitude},
        headers={"User-Agent": f"SmartPoint-ControleDePonto/1.0 (contato: {VAPID_CONTACT_EMAIL})"},
        timeout=8,
    )
    if resposta.status_code != 200:
        return None, f"status {resposta.status_code}"
    feicoes = (resposta.json() or {}).get("features") or []
    if not feicoes:
        return None, "sem resultados"
    p = feicoes[0].get("properties") or {}
    rua = " ".join(x for x in [p.get("street") or p.get("name"), p.get("housenumber")] if x)
    return _montar_endereco([
        rua, p.get("district"), p.get("city") or p.get("county"),
        p.get("state"), p.get("postcode"), p.get("country"),
    ]), "sem endereço utilizável"


def _provedor_bigdatacloud(latitude, longitude):
    """Último recurso: sem chave e sem limite prático, mas resolve só até o nível de bairro."""
    resposta = requests.get(
        "https://api.bigdatacloud.net/data/reverse-geocode-client",
        params={"latitude": latitude, "longitude": longitude, "localityLanguage": "pt"},
        timeout=8,
    )
    if resposta.status_code != 200:
        return None, f"status {resposta.status_code}"
    d = resposta.json() or {}
    return _montar_endereco([
        d.get("locality"), d.get("city"), d.get("principalSubdivision"), d.get("countryName"),
    ]), "sem endereço utilizável"


# Rótulo exibido para cada tipo de marcação (o banco guarda "entrada"/"saida" e registros
# antigos podem não ter tipo nenhum).
ROTULOS_TIPO_REGISTRO = {"entrada": "Entrada", "saida": "Saída"}


def obter_endereco(latitude, longitude, tentativas=2, orcamento=12.0):
    """Converte latitude/longitude em um endereço legível.

    Retorna None se as coordenadas não vierem ou se todos os provedores falharem (sem
    quebrar o registro do ponto) — mas SEMPRE imprime o motivo no log, para dar pra
    diagnosticar pelo Render.

    `orcamento` é o tempo máximo (em segundos) gasto tentando os provedores. O registro
    do ponto espera por essa função, então é melhor salvar a batida só com lat/long e
    preencher o endereço depois (botão do gestor) do que deixar o colaborador travado."""
    if latitude is None or longitude is None:
        return None

    try:
        chave = _chave_geocode(latitude, longitude)
    except (TypeError, ValueError):
        return None

    if chave in _cache_enderecos:
        return _cache_enderecos[chave]

    conhecido = _endereco_ja_conhecido(chave)
    if conhecido:
        _guardar_no_cache(chave, conhecido)
        return conhecido

    falhou_em = _cache_falhas.get(chave)
    if falhou_em and (time.monotonic() - falhou_em) < GEOCODE_TTL_FALHA:
        # Já tentamos essa coordenada há pouco e todos os provedores recusaram: insistir
        # agora só piora o 429. Volta a tentar depois do TTL.
        return None

    provedores = [("nominatim", _provedor_nominatim), ("photon", _provedor_photon),
                  ("bigdatacloud", _provedor_bigdatacloud)]
    if LOCATIONIQ_API_KEY:
        provedores.insert(0, ("locationiq", _provedor_locationiq))

    limite = time.monotonic() + max(orcamento, 0)
    ultimo_erro = None
    for nome, consultar in provedores:
        if _provedor_indisponivel(nome):
            continue
        for tentativa in range(1, tentativas + 1):
            if time.monotonic() > limite:
                ultimo_erro = f"{ultimo_erro} (orçamento de {orcamento:.0f}s esgotado)"
                print(f"[endereco] parando em ({latitude}, {longitude}): {ultimo_erro}")
                _cache_falhas[chave] = time.monotonic()
                return None
            try:
                _aguardar_vez(nome)
                endereco, motivo = consultar(latitude, longitude)
                if endereco:
                    _guardar_no_cache(chave, endereco)
                    _cache_falhas.pop(chave, None)
                    return endereco
                ultimo_erro = f"{nome}: {motivo}"
            except Exception as ex:
                motivo = str(ex)
                ultimo_erro = f"{nome}: {ex}"
            print(f"[endereco] {nome} tentativa {tentativa}/{tentativas} falhou para ({latitude}, {longitude}): {ultimo_erro}")
            if motivo in ("status 429", "status 403", "status 503"):
                _bloquear_provedor(nome, motivo)
                break  # não adianta insistir: parte pro próximo provedor
            if tentativa < tentativas:
                time.sleep(2 ** tentativa)  # backoff: 2s, 4s...

    _cache_falhas[chave] = time.monotonic()
    print(f"[endereco] desistindo de ({latitude}, {longitude}) por {GEOCODE_TTL_FALHA}s — último erro: {ultimo_erro}")
    return None


# A janela e o intervalo do lembrete agora são configuráveis pelo gestor em
# /gestor/configuracoes (tabela ConfiguracaoJornada), não mais por variável de ambiente.


def colaborador_tem_registro_hoje(colaborador_id):
    """Verifica se o colaborador já bateu ao menos um ponto hoje (horário de Brasília)."""
    return contar_registros_hoje(colaborador_id) > 0


def eh_domingo(data):
    """Domingo não é considerado dia útil padrão em nenhum lugar do sistema:
    não gera lembrete, não conta como ausência/atraso, e se alguém trabalhar
    mesmo assim, o dia inteiro vira hora extra (não só o que passar do horário
    padrão) — é exceção à jornada normal, não parte dela."""
    return data.weekday() == 6  # 0=segunda ... 6=domingo


def registros_do_dia(colaborador_id, dia=None):
    """Devolve TODAS as batidas do colaborador em um dia (horário de Brasília),
    em ordem cronológica.

    Antes isso era feito olhando só as 10 últimas batidas do colaborador: quem
    batesse mais de 10 pontos no dia (ou tivesse batidas de madrugada no meio)
    tinha o dia contado errado. Agora a janela é buscada por data no banco
    (com folga de fuso), sem limite artificial de linhas."""
    dia = dia or agora_brasilia().date()
    inicio_utc = datetime.combine(dia, datetime.min.time()) - timedelta(hours=6)
    fim_utc = datetime.combine(dia, datetime.max.time()) + timedelta(hours=6)
    candidatos = (
        RegistroPonto.query
        .filter_by(colaborador_id=colaborador_id)
        .filter(RegistroPonto.data_hora >= inicio_utc, RegistroPonto.data_hora <= fim_utc)
        .order_by(RegistroPonto.data_hora.asc())
        .all()
    )
    return [r for r in candidatos if para_brasilia(r.data_hora).date() == dia]


def contar_registros_hoje(colaborador_id):
    """Conta quantas batidas de ponto o colaborador já fez hoje (horário de Brasília)."""
    return len(registros_do_dia(colaborador_id))


def determinar_proximo_tipo(colaborador_id):
    """Decide se a PRÓXIMA batida desse colaborador é 'entrada' ou 'saida'.

    Antes isso era decidido contando quantas batidas ele já tinha HOJE (par=entrada, ímpar=saída).
    Isso quebra assim que uma sessão vira a noite: o contador reseta à meia-noite, então mesmo que
    a última batida real (de madrugada) tenha sido uma "saida", o sistema via 1 batida "hoje" e
    concluía (errado) que a próxima também seria "saida".

    A regra certa não depende do dia: olha só o `tipo` da ÚLTIMA batida em qualquer dia. Se foi
    "saida" (ou não há nenhuma batida ainda), a próxima é "entrada". Se foi "entrada", a próxima é
    "saida" — não importa se isso aconteceu ontem, hoje, ou faz uma semana."""
    ultimo = (
        RegistroPonto.query
        .filter_by(colaborador_id=colaborador_id)
        .order_by(RegistroPonto.data_hora.desc())
        .first()
    )
    if ultimo is None or (ultimo.tipo or "").lower() == "saida":
        return "entrada"
    return "saida"


LIMITE_HORAS_EXCEPCIONAL = 14.0  # sessão contínua (entrada->saída) acima disso não vira hora extra sozinha; fica pendente de aprovação do gestor


def montar_resumo_diario(registros):
    """Agrupa os registros em sessões trabalhadas (entrada -> saída) e soma por dia (Brasília).

    Antes, o pareamento era só por POSIÇÃO dentro do dia civil (1º=entrada, 2º=saída, 3º=entrada...),
    ignorando o campo `tipo` que cada batida já guarda. Isso quebrava jornadas que viram a noite:
    ex. entrou 08:00 de um dia e só saiu 07:00 do dia seguinte -> a "saída de virada" caía no balde
    do dia seguinte e era confundida com a entrada normal daquele dia, deslocando todos os pares
    seguintes e podendo gerar hora extra errada (às vezes bem grande) justamente no dia seguinte.

    Agora o pareamento usa o `tipo` real de cada batida (entrada/saida) em ordem cronológica, e uma
    sessão que atravessa a meia-noite é contada inteira no dia em que a ENTRADA aconteceu (convenção
    comum de apuração de ponto: o dia de trabalho é o dia em que o turno começou)."""
    from collections import defaultdict

    registros_ordenados = sorted(registros, key=lambda r: r.data_hora)

    por_dia = defaultdict(lambda: {"segundos": 0.0, "batidas": [], "incompleto": False, "excepcional": False, "excepcional_entrada_id": None, "excepcional_saida_id": None})
    entrada_aberta = None  # datetime local da entrada aguardando a saída correspondente
    entrada_aberta_id = None  # id do RegistroPonto dessa entrada, pra poder corrigi-lo depois se preciso

    for r in registros_ordenados:
        dt_local = para_brasilia(r.data_hora)
        por_dia[dt_local.date()]["batidas"].append(dt_local)

        tipo = (r.tipo or "").lower()
        if tipo == "entrada":
            if entrada_aberta is not None:
                # entrada seguida de outra entrada, sem saída no meio: a anterior fica incompleta
                por_dia[entrada_aberta.date()]["incompleto"] = True
            entrada_aberta = dt_local
            entrada_aberta_id = r.id
        elif tipo == "saida":
            if entrada_aberta is not None:
                segundos = (dt_local - entrada_aberta).total_seconds()
                if segundos > 0:
                    por_dia[entrada_aberta.date()]["segundos"] += segundos
                    if segundos / 3600 > LIMITE_HORAS_EXCEPCIONAL:
                        por_dia[entrada_aberta.date()]["excepcional"] = True
                        # guarda os dois registros que formaram essa sessão longa: se o gestor
                        # recusar (era esquecimento, não virada real), é a batida de SAÍDA que
                        # precisa virar ENTRADA (ver recusar_jornada_excepcional).
                        por_dia[entrada_aberta.date()]["excepcional_entrada_id"] = entrada_aberta_id
                        por_dia[entrada_aberta.date()]["excepcional_saida_id"] = r.id
                else:
                    por_dia[entrada_aberta.date()]["incompleto"] = True
                entrada_aberta = None
                entrada_aberta_id = None
            else:
                # saída sem entrada correspondente antes dela
                por_dia[dt_local.date()]["incompleto"] = True
        else:
            # tipo desconhecido/legado: não dá pra parear com segurança
            por_dia[dt_local.date()]["incompleto"] = True

    if entrada_aberta is not None:
        # última entrada do período ainda não teve saída (jornada em andamento, ou pendência real)
        por_dia[entrada_aberta.date()]["incompleto"] = True

    resumo = []
    for dia, info in sorted(por_dia.items(), reverse=True):
        total_segundos = info["segundos"]
        horas = int(total_segundos // 3600)
        minutos = int((total_segundos % 3600) // 60)
        resumo.append({
            "data": dia,
            "batidas": sorted(info["batidas"]),
            "total_horas": f"{horas:02d}:{minutos:02d}",
            "total_horas_decimal": round(total_segundos / 3600, 2),
            "incompleto": info["incompleto"],
            "excepcional": info["excepcional"],
            "excepcional_entrada_id": info["excepcional_entrada_id"],
            "excepcional_saida_id": info["excepcional_saida_id"],
        })
    return resumo


def _horario_para_decimal(hhmm):
    """Converte 'HH:MM' em horas decimais (ex.: '08:30' -> 8.5)."""
    h, m = hhmm.split(":")
    return int(h) + int(m) / 60


def calcular_horas_extras_dia(total_horas_decimal, horario_entrada_padrao, horario_saida_padrao, eh_domingo_flag=False):
    """Calcula a hora extra do dia.

    Regra: hora extra = (horas realmente trabalhadas no dia) - (duração padrão da
    jornada, ou seja, horário de saída padrão menos horário de entrada padrão).

    Isso já cobre naturalmente os dois cenários que a jornada padrão (8h às 18h,
    10h de duração) costuma gerar:
    - Entrou 8h, saiu 19h -> trabalhou 11h -> 11h - 10h = 1h de hora extra.
    - Entrou 8h30, saiu 19h -> trabalhou 10h30 -> 10h30 - 10h = 30min de hora extra
      (o atraso na entrada "consome" parte da hora extra feita na saída).

    Se o resultado for negativo (trabalhou menos que a jornada padrão), a hora
    extra é zero — não vira "hora extra negativa" aqui, é só ausência de extra.

    Domingo é EXCEÇÃO: como não é dia útil padrão, se o colaborador trabalhar mesmo
    assim, o dia INTEIRO conta como hora extra (não só o que passar da jornada
    padrão) — ele não tinha expectativa nenhuma de trabalhar naquele dia.
    """
    if eh_domingo_flag:
        return round(max(0.0, total_horas_decimal), 2)

    duracao_padrao = _horario_para_decimal(horario_saida_padrao) - _horario_para_decimal(horario_entrada_padrao)
    if duracao_padrao <= 0:
        duracao_padrao = 8.0  # salvaguarda, não deveria acontecer com horários válidos
    extra = total_horas_decimal - duracao_padrao
    return max(0.0, round(extra, 2))


def aplicar_status_excepcional(dia_resumo, colaborador_id):
    """Se o dia foi marcado como 'excepcional' (sessão contínua acima de LIMITE_HORAS_EXCEPCIONAL),
    garante que exista uma pendência de aprovação (cria se ainda não existir, idempotente pela
    constraint única colaborador+data) e decide se a hora extra desse dia entra no total ou fica
    zerada até o gestor aprovar.

    O total de horas TRABALHADAS (`total_horas`/`total_horas_decimal`) nunca é escondido — o
    colaborador e o gestor sempre veem o valor real. O que fica retido é só a hora extra calculada
    a partir dele, pra não pagar/computar algo que ninguém revisou ainda.

    Esta função NUNCA pode derrubar a tela de registros/ajustes com um 500: qualquer erro de banco
    aqui (tabela ainda não migrada, corrida de duas requisições tentando criar a mesma pendência ao
    mesmo tempo, etc.) é capturado, logado, e tratado de forma segura — o dia fica com hora extra
    retida (mesmo comportamento de 'pendente'), mas a página continua funcionando normalmente."""
    if not dia_resumo.get("excepcional"):
        dia_resumo["status_excepcional"] = None
        return dia_resumo

    status = "pendente"
    try:
        aprovacao = AprovacaoJornadaExcepcional.query.filter_by(
            colaborador_id=colaborador_id, data_referencia=dia_resumo["data"]
        ).first()
        if aprovacao is None:
            aprovacao = AprovacaoJornadaExcepcional(
                colaborador_id=colaborador_id,
                data_referencia=dia_resumo["data"],
                horas_total=dia_resumo["total_horas_decimal"],
                status="pendente",
                registro_entrada_id=dia_resumo.get("excepcional_entrada_id"),
                registro_saida_id=dia_resumo.get("excepcional_saida_id"),
            )
            db.session.add(aprovacao)
            try:
                db.session.commit()
            except Exception:
                # duas requisições tentaram criar a mesma pendência ao mesmo tempo (corrida) —
                # desfaz e busca a que já foi criada pela outra requisição.
                db.session.rollback()
                aprovacao = AprovacaoJornadaExcepcional.query.filter_by(
                    colaborador_id=colaborador_id, data_referencia=dia_resumo["data"]
                ).first()
        if aprovacao is not None:
            status = aprovacao.status
    except Exception as ex:
        db.session.rollback()
        print(f"[jornada-excepcional] falha ao registrar/consultar pendência (colaborador {colaborador_id}, dia {dia_resumo.get('data')}): {ex}")
        # segue com status "pendente" (fail-safe): retém a hora extra sem quebrar a página

    dia_resumo["status_excepcional"] = status
    if status != "aprovado":
        # hora extra fica retida até aprovação; o total de horas trabalhadas continua visível
        dia_resumo["horas_extras_retidas"] = dia_resumo.get("horas_extras", 0.0)
        dia_resumo["horas_extras"] = 0.0
    return dia_resumo


@app.template_filter("horas")
def formatar_horas(horas_decimal):
    """Formata horas decimais de um jeito fácil de ler: '1h30min', '45min', '2h'."""
    total_minutos = int(round(horas_decimal * 60))
    if total_minutos <= 0:
        return "0min"
    h, m = divmod(total_minutos, 60)
    if h and m:
        return f"{h}h{m:02d}min"
    if h:
        return f"{h}h"
    return f"{m}min"


# ---------------------------------------------------------------------------
# Paginação (usada por TODAS as listas e tabelas do sistema)
#
# Regra do sistema: nenhuma tela mostra mais de ITENS_POR_PAGINA registros de
# uma vez — passou disso, pagina.
#
# Antes, cada rota fazia seu próprio `paginate(page=request.args.get("page"))`
# com um limite diferente (20, 25, 30) e algumas listas nem paginavam (vinham
# com .limit() fixo, escondendo silenciosamente o restante). Além disso, como
# todas usavam o MESMO parâmetro "page", duas tabelas na mesma tela (ex.: a de
# ajustes) andavam juntas, e clicar em "Próxima" apagava os outros filtros da
# URL (ex.: ?mes=2026-08).
#
# Agora existe um só ponto de verdade: `paginar_query` (para consultas ao banco)
# e `paginar_lista` (para listas já calculadas em memória, como o resumo diário),
# ambas devolvendo o mesmo objeto `Paginacao`, com um parâmetro de URL próprio
# por tabela.
# ---------------------------------------------------------------------------
ITENS_POR_PAGINA = 10


class Paginacao:
    """Resultado paginado, com a mesma interface para consultas do banco e para
    listas em memória (o template não precisa saber a diferença)."""

    def __init__(self, itens, pagina, por_pagina, total, param="page", ancora=None):
        self.items = itens
        self.per_page = por_pagina
        self.total = total
        self.param = param
        self.ancora = ancora
        self.pages = (total + por_pagina - 1) // por_pagina if total else 0
        # Nunca deixa a página pedida ficar fora do intervalo válido: pedir
        # ?p=999 (ou ?p=-3, ou ?p=abc) devolve a última/primeira página com
        # conteúdo, em vez de uma tabela vazia sem explicação.
        self.page = min(max(1, pagina), self.pages) if self.pages else 1

    @property
    def has_prev(self):
        return self.page > 1

    @property
    def has_next(self):
        return self.page < self.pages

    @property
    def prev_num(self):
        return self.page - 1 if self.has_prev else None

    @property
    def next_num(self):
        return self.page + 1 if self.has_next else None

    @property
    def primeiro_item(self):
        return 0 if not self.total else (self.page - 1) * self.per_page + 1

    @property
    def ultimo_item(self):
        return min(self.page * self.per_page, self.total)

    @property
    def numeros(self):
        """Páginas a exibir na barra: sempre a primeira, a última, e as vizinhas
        da atual. `None` marca onde entra o '…'."""
        if not self.pages:
            return []
        visiveis = set()
        for n in (1, 2, self.pages - 1, self.pages):
            if 1 <= n <= self.pages:
                visiveis.add(n)
        for n in range(self.page - 2, self.page + 3):
            if 1 <= n <= self.pages:
                visiveis.add(n)

        resultado = []
        anterior = 0
        for n in sorted(visiveis):
            if anterior and n - anterior > 1:
                resultado.append(None)
            resultado.append(n)
            anterior = n
        return resultado


def _pagina_pedida(param):
    """Lê o número da página da URL de forma tolerante (valor ausente, vazio,
    negativo ou não numérico vira 1, sem estourar 400/500)."""
    try:
        return max(1, int(request.args.get(param, 1)))
    except (TypeError, ValueError):
        return 1


def paginar_query(query, param="page", ancora=None, por_pagina=None):
    """Pagina uma query do SQLAlchemy contando o total no banco (não carrega
    tudo na memória)."""
    por_pagina = por_pagina or ITENS_POR_PAGINA
    total = query.order_by(None).count()
    pagina = _pagina_pedida(param)
    paginacao = Paginacao([], pagina, por_pagina, total, param=param, ancora=ancora)
    if total:
        paginacao.items = query.limit(por_pagina).offset((paginacao.page - 1) * por_pagina).all()
    return paginacao


def paginar_lista(lista, param="page", ancora=None, por_pagina=None):
    """Pagina uma lista já pronta em memória (ex.: resumo diário calculado a
    partir das batidas)."""
    por_pagina = por_pagina or ITENS_POR_PAGINA
    lista = list(lista or [])
    pagina = _pagina_pedida(param)
    paginacao = Paginacao([], pagina, por_pagina, len(lista), param=param, ancora=ancora)
    inicio = (paginacao.page - 1) * por_pagina
    paginacao.items = lista[inicio:inicio + por_pagina]
    return paginacao


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
            print(f"[push] enviado para {colaborador.nome} (endpoint ...{sub.endpoint[-12:]})")
        except WebPushException as ex:
            status = getattr(ex.response, "status_code", None)
            print(f"[push] falhou para {colaborador.nome}: status={status}")
            if status in (404, 410):
                db.session.delete(sub)
                db.session.commit()
                print(f"[push] inscrição expirada removida ({colaborador.nome})")
        except Exception as ex:
            # Nunca deixa uma falha de push derrubar o job do agendador.
            print(f"[push] erro inesperado ao enviar para {colaborador.nome}: {ex}")


def verificar_e_enviar_lembretes_push():
    """Roda periodicamente (APScheduler): verifica quem está devendo bater ponto
    e envia push respeitando o intervalo configurado pelo gestor."""
    if not PUSH_HABILITADO:
        return

    try:
        with app.app_context():
            if eh_domingo(agora_brasilia().date()):
                return  # domingo não é dia útil padrão — sem lembrete de ponto

            config = obter_configuracao()
            agora_str = agora_brasilia().strftime("%H:%M")
            colaboradores = Colaborador.query.filter_by(is_gestor=False).all()
            enviados = 0

            for colaborador in colaboradores:
                if not colaborador.push_subscriptions:
                    continue

                # Mesma correção de determinar_proximo_tipo: não conta batidas "de hoje" (isso
                # reseta à meia-noite), olha o tipo da última batida real do colaborador.
                proximo_tipo = determinar_proximo_tipo(colaborador.id)
                bateu_hoje = contar_registros_hoje(colaborador.id) > 0
                pendencia = None
                if proximo_tipo == "entrada" and not bateu_hoje and agora_str >= config.horario_entrada:
                    pendencia = "entrada"
                elif proximo_tipo == "saida" and agora_str >= config.horario_saida:
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
                enviados += 1

                if ultimo:
                    ultimo.enviado_em = datetime.utcnow()
                else:
                    db.session.add(UltimoLembretePush(
                        colaborador_id=colaborador.id, tipo=pendencia, enviado_em=datetime.utcnow()
                    ))
                db.session.commit()

            print(f"[push] verificação às {agora_str} (Brasília) — {enviados} lembrete(s) enviado(s).")
    except Exception as ex:
        # Nunca deixa um erro transitório (ex.: banco temporariamente indisponível)
        # cancelar o agendador — na próxima execução (5 min depois) tenta de novo.
        print(f"[push] falha ao verificar/enviar lembretes: {ex}")


# ---------------------------------------------------------------------------
# Rotas - autenticação
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    if "user_id" in session:
        if session.get("is_gestor"):
            return redirect(url_for("gestor_consulta"))
        return redirect(url_for("ponto"))
    return redirect(url_for("login"))


@app.route("/api/endereco-preview", methods=["POST"])
@login_required
def endereco_preview():
    """Devolve o endereço aproximado a partir de lat/lon, para mostrar no mapa
    ANTES do colaborador confirmar o ponto (o registro em si roda a mesma
    geocodificação de novo no momento de salvar, de forma independente)."""
    dados = request.get_json(silent=True) or {}
    latitude = dados.get("latitude")
    longitude = dados.get("longitude")
    endereco = obter_endereco(latitude, longitude)
    return jsonify({"endereco": endereco})


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
            if not user.ativo:
                flash("Este colaborador está desativado. Procure o gestor.")
                return render_template("login.html")
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
    if session.get("is_gestor"):
        return redirect(url_for("gestor_consulta"))
    config = obter_configuracao()

    hoje = agora_brasilia().date()
    registros_hoje_lista = registros_do_dia(session["user_id"], hoje)
    registros_hoje_count = len(registros_hoje_lista)

    # Para o resumo do dia, precisamos também da última batida ANTES de hoje: uma
    # jornada que virou a noite só fecha corretamente se a entrada da véspera
    # entrar no cálculo.
    registros_recentes = (
        RegistroPonto.query
        .filter_by(colaborador_id=session["user_id"])
        .order_by(RegistroPonto.data_hora.desc())
        .limit(max(20, registros_hoje_count + 10))
        .all()
    )
    resumo_lista = montar_resumo_diario(registros_recentes)
    resumo_hoje = next((r for r in resumo_lista if r["data"] == hoje), None)

    hora_atual = agora_brasilia().hour
    if hora_atual < 12:
        saudacao = "Bom dia"
    elif hora_atual < 18:
        saudacao = "Boa tarde"
    else:
        saudacao = "Boa noite"

    return render_template(
        "ponto.html",
        nome=session.get("nome"),
        saudacao=saudacao,
        registros_hoje_count=registros_hoje_count,
        registros_hoje_lista=registros_hoje_lista,
        resumo_hoje=resumo_hoje,
        meta_horas_diarias=config.meta_horas_diarias,
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

    encoding_atual, motivo_falha = imagem_base64_para_encoding(foto_base64)
    if encoding_atual is None:
        mensagem = MENSAGENS_POSICIONAMENTO.get(motivo_falha, "Não foi possível identificar um rosto na imagem.")
        return jsonify({"ok": False, "mensagem": mensagem, "reposicionar": True}), 400

    distancia = np.linalg.norm(np.array(user.face_encoding) - np.array(encoding_atual))

    if distancia > FACE_MATCH_TOLERANCE:
        print(f"[facial] Rejeitado: {user.nome} (distância={distancia:.4f}, tolerância={FACE_MATCH_TOLERANCE})")
        return jsonify({
            "ok": False,
            "mensagem": "O rosto não confere com o cadastrado.",
            "reposicionar": True,
        }), 403

    endereco = obter_endereco(latitude, longitude)

    registros_hoje_count = contar_registros_hoje(user.id)
    tipo_registro = determinar_proximo_tipo(user.id)

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

    # Janela usada para montar o resumo por dia. Antes era "as 200 últimas batidas",
    # um corte arbitrário que escondia dias inteiros de quem bate muito ponto; agora
    # é uma janela por DATA (últimos 6 meses), e o resumo resultante é paginado.
    limite_resumo_utc = datetime.utcnow() - timedelta(days=185)
    registros_para_resumo = (
        RegistroPonto.query
        .filter_by(colaborador_id=user.id)
        .filter(RegistroPonto.data_hora >= limite_resumo_utc)
        .order_by(RegistroPonto.data_hora.desc())
        .all()
    )
    resumo_diario = montar_resumo_diario(registros_para_resumo)

    config = obter_configuracao()
    hoje = agora_brasilia().date()
    total_extra_mes = 0.0
    for dia in resumo_diario:
        dia["horas_extras"] = calcular_horas_extras_dia(
            dia["total_horas_decimal"], config.horario_entrada, config.horario_saida,
            eh_domingo_flag=eh_domingo(dia["data"]),
        )
        aplicar_status_excepcional(dia, user.id)
        if dia["data"].year == hoje.year and dia["data"].month == hoje.month:
            total_extra_mes += dia["horas_extras"]

    # Cada bloco da tela tem seu PRÓPRIO parâmetro de página, para que navegar no
    # histórico não mexa no resumo por dia nem nas solicitações (antes os três
    # dividiriam o mesmo "page").
    resumo_paginado = paginar_lista(resumo_diario, param="p_resumo", ancora="resumo-por-dia")

    registros = paginar_query(
        RegistroPonto.query
        .filter_by(colaborador_id=user.id)
        .order_by(RegistroPonto.data_hora.desc()),
        param="p_registros",
        ancora="historico-detalhado",
    )

    solicitacoes = paginar_query(
        SolicitacaoAjuste.query
        .filter_by(colaborador_id=user.id)
        .order_by(SolicitacaoAjuste.criado_em.desc()),
        param="p_solicitacoes",
        ancora="minhas-solicitacoes",
    )

    return render_template(
        "meus_registros.html",
        registros=registros,
        resumo_diario=resumo_paginado,
        solicitacoes=solicitacoes,
        hoje=hoje,
        total_extra_mes=round(total_extra_mes, 2),
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
    from collections import defaultdict

    config = obter_configuracao()
    registros_paginados = paginar_query(
        RegistroPonto.query.order_by(RegistroPonto.data_hora.desc()),
        param="p_registros",
        ancora="registros-ponto",
    )
    colaboradores = Colaborador.query.filter_by(is_gestor=False).order_by(Colaborador.nome).all()
    total_colaboradores = len(colaboradores)
    sem_cadastro_facial = sum(1 for c in colaboradores if not c.face_encoding)

    hoje_brasilia = agora_brasilia().date()

    # Consulta dedicada dos registros de HOJE (não depende do limite de exibição
    # da tabela abaixo, então continua correta mesmo com empresas grandes).
    limite_inferior_utc = datetime.combine(hoje_brasilia, datetime.min.time()) - timedelta(hours=6)
    candidatos_hoje = (
        RegistroPonto.query
        .filter(RegistroPonto.data_hora >= limite_inferior_utc)
        .order_by(RegistroPonto.data_hora.asc())
        .all()
    )
    registros_hoje_todos = [r for r in candidatos_hoje if para_brasilia(r.data_hora).date() == hoje_brasilia]

    por_colaborador_hoje = defaultdict(list)
    for r in registros_hoje_todos:
        por_colaborador_hoje[r.colaborador_id].append(r)

    presentes = sum(1 for c in colaboradores if por_colaborador_hoje.get(c.id))
    hoje_eh_domingo = eh_domingo(hoje_brasilia)
    colaboradores_sem_registro_hoje = [] if hoje_eh_domingo else [
        c for c in colaboradores if not por_colaborador_hoje.get(c.id)
    ]

    atrasados = 0
    entradas_por_hora = {}
    for c in colaboradores:
        regs = por_colaborador_hoje.get(c.id)
        if not regs:
            continue
        primeira_hoje = para_brasilia(regs[0].data_hora)  # já vem ordenado asc
        if not hoje_eh_domingo and primeira_hoje.strftime("%H:%M") > config.horario_entrada:
            atrasados += 1
        entradas_por_hora[primeira_hoje.hour] = entradas_por_hora.get(primeira_hoje.hour, 0) + 1

    ajustes_pendentes = SolicitacaoAjuste.query.filter_by(status="pendente").count()

    horas_grafico = [f"{h:02d}:00" for h in range(6, 21)]
    contagem_grafico = [entradas_por_hora.get(h, 0) for h in range(6, 21)]

    registros_sem_endereco = RegistroPonto.query.filter(
        RegistroPonto.endereco.is_(None), RegistroPonto.latitude.isnot(None)
    ).count()

    return render_template(
        "gestor_consulta.html",
        registros=registros_paginados,
        colaboradores=paginar_lista(colaboradores, param="p_colaboradores", ancora="lista-colaboradores"),
        ausentes_paginados=paginar_lista(
            colaboradores_sem_registro_hoje, param="p_ausentes", ancora="sem-ponto-hoje"
        ),
        registros_hoje=len(registros_hoje_todos),
        total_colaboradores=total_colaboradores,
        sem_cadastro_facial=sem_cadastro_facial,
        colaboradores_sem_registro_hoje=colaboradores_sem_registro_hoje,
        ajustes_pendentes=ajustes_pendentes,
        presentes=presentes,
        atrasados=atrasados,
        horas_grafico=horas_grafico,
        contagem_grafico=contagem_grafico,
        registros_sem_endereco=registros_sem_endereco,
        hoje_eh_domingo=hoje_eh_domingo,
    )


@app.route("/gestor/cadastro", methods=["GET"])
@gestor_required
def gestor_cadastro():
    colaboradores = paginar_query(
        Colaborador.query.filter_by(is_gestor=False).order_by(Colaborador.nome),
        param="p_colaboradores",
        ancora="colaboradores-cadastrados",
    )
    return render_template("gestor_cadastro.html", colaboradores=colaboradores)


@app.route("/gestor/configuracoes", methods=["GET"])
@gestor_required
def gestor_configuracoes():
    config = obter_configuracao()
    return render_template("gestor_configuracoes.html", config=config, push_habilitado=PUSH_HABILITADO)


@app.route("/gestor/configuracoes/marca", methods=["POST"])
@gestor_required
def salvar_marca():
    """Permite ao gestor personalizar o nome da empresa, as logos (login e cabeçalho,
    de forma independente) e as cores do sistema (white-label)."""
    import re

    marca = obter_marca()

    nome_empresa = request.form.get("nome_empresa", "").strip()
    cor_primaria = request.form.get("cor_primaria", "").strip()
    cor_secundaria = request.form.get("cor_secundaria", "").strip()
    arquivo_login = request.files.get("logo_login")
    arquivo_topbar = request.files.get("logo_topbar")

    def _valida_hex(cor):
        return bool(re.fullmatch(r"#[0-9a-fA-F]{6}", cor or ""))

    if nome_empresa:
        marca.nome_empresa = nome_empresa[:60]
    if _valida_hex(cor_primaria):
        marca.cor_primaria = cor_primaria
    if _valida_hex(cor_secundaria):
        marca.cor_secundaria = cor_secundaria

    if arquivo_login and arquivo_login.filename:
        try:
            dados = arquivo_login.read()
            if len(dados) > 3 * 1024 * 1024:
                flash("A logo da tela de login é muito grande (máximo 3MB).")
            else:
                marca.logo_login_base64 = redimensionar_para_base64(dados, altura_max=160)
        except Exception:
            flash("Não foi possível processar a logo da tela de login. Tente um arquivo PNG ou JPG.")

    if arquivo_topbar and arquivo_topbar.filename:
        try:
            dados = arquivo_topbar.read()
            if len(dados) > 3 * 1024 * 1024:
                flash("A logo do cabeçalho é muito grande (máximo 3MB).")
            else:
                marca.logo_topbar_base64 = redimensionar_para_base64(dados, altura_max=64)
                marca.icone_192_base64, marca.icone_512_base64 = gerar_icones_pwa(dados)
        except Exception:
            flash("Não foi possível processar a logo do cabeçalho. Tente um arquivo PNG ou JPG.")

    db.session.commit()
    flash("Marca do sistema atualizada com sucesso.")
    return redirect(url_for("gestor_configuracoes"))


@app.route("/gestor/configuracoes/marca/remover-logo-login", methods=["POST"])
@gestor_required
def remover_logo_login():
    marca = obter_marca()
    marca.logo_login_base64 = None
    marca.logo_header_base64 = None  # compatibilidade com registros antigos
    db.session.commit()
    flash("Logo da tela de login removida.")
    return redirect(url_for("gestor_configuracoes"))


@app.route("/gestor/configuracoes/marca/remover-logo-topbar", methods=["POST"])
@gestor_required
def remover_logo_topbar():
    marca = obter_marca()
    marca.logo_topbar_base64 = None
    marca.icone_192_base64 = None
    marca.icone_512_base64 = None
    db.session.commit()
    flash("Logo do cabeçalho removida — voltando para o ícone padrão de relógio.")
    return redirect(url_for("gestor_configuracoes"))


@app.route("/manifest.json")
def manifest_json():
    """Manifesto do PWA gerado dinamicamente com o nome/cores/ícone da marca configurada
    pelo gestor, para que o app instalado na Tela de Início reflita a marca certa."""
    marca = obter_marca()
    icone_192 = marca.icone_192_base64 or (request.url_root.rstrip("/") + url_for("static", filename="icons/icon-192.png"))
    icone_512 = marca.icone_512_base64 or (request.url_root.rstrip("/") + url_for("static", filename="icons/icon-512.png"))
    nome_empresa = marca.nome_empresa or "SmartPoint"
    return jsonify({
        "name": f"{nome_empresa} — Controle de Ponto",
        "short_name": nome_empresa[:12],
        "start_url": "/ponto",
        "display": "standalone",
        "background_color": marca.cor_primaria,
        "theme_color": marca.cor_primaria,
        "icons": [
            {"src": icone_192, "sizes": "192x192", "type": "image/png"},
            {"src": icone_512, "sizes": "512x512", "type": "image/png"},
        ],
    })


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
    meta_horas = request.form.get("meta_horas_diarias", "").strip()

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

    try:
        meta_horas_float = float(meta_horas.replace(",", "."))
        if meta_horas_float <= 0 or meta_horas_float > 24:
            raise ValueError
    except Exception:
        flash("A meta de horas diárias deve ser um número entre 0 e 24.")
        return redirect(url_for("gestor_configuracoes"))

    config.horario_entrada = horario_entrada
    config.horario_saida = horario_saida
    config.intervalo_lembrete_minutos = intervalo_int
    config.meta_horas_diarias = meta_horas_float
    db.session.commit()

    flash("Configurações de jornada e lembretes atualizadas com sucesso.")
    return redirect(url_for("gestor_configuracoes"))


def encontrar_colaborador_por_rosto(encoding_novo, ignorar_id=None):
    """Verifica se esse rosto já está cadastrado em OUTRO colaborador — evita que duas
    pessoas fiquem com o mesmo rosto por engano (o que quebraria o reconhecimento pros
    dois, já que o sistema não saberia mais diferenciá-los)."""
    candidatos = Colaborador.query.filter(Colaborador.face_encoding.isnot(None)).all()
    for c in candidatos:
        if ignorar_id and c.id == ignorar_id:
            continue
        distancia = np.linalg.norm(np.array(c.face_encoding) - np.array(encoding_novo))
        if distancia <= FACE_MATCH_TOLERANCE:
            return c
    return None


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
        encoding, motivo_falha = imagem_base64_para_encoding(foto_base64)
        if encoding is None:
            mensagem_motivo = MENSAGENS_POSICIONAMENTO.get(motivo_falha, "Não foi possível reconhecer um rosto na foto enviada.")
            flash(f"{mensagem_motivo} Colaborador criado sem cadastro facial — edite o cadastro para tentar de novo.")
        else:
            duplicado = encontrar_colaborador_por_rosto(encoding)
            if duplicado:
                flash(f"Esse rosto já parece estar cadastrado para {duplicado.nome}. Colaborador criado sem cadastro facial — confira se não é duplicidade antes de tentar de novo.")
            else:
                novo.face_encoding = encoding.tolist()

    db.session.add(novo)
    db.session.commit()
    flash(f"Colaborador {nome} cadastrado com sucesso.")
    return redirect(url_for("gestor_cadastro"))


@app.route("/gestor/colaborador/<int:colaborador_id>/editar", methods=["GET"])
@gestor_required
def editar_colaborador(colaborador_id):
    colaborador = Colaborador.query.get_or_404(colaborador_id)
    if colaborador.is_gestor:
        flash("Não é possível editar a conta do gestor por aqui.")
        return redirect(url_for("gestor_cadastro"))
    return render_template("gestor_editar_colaborador.html", colaborador=colaborador)


@app.route("/gestor/colaborador/<int:colaborador_id>/editar", methods=["POST"])
@gestor_required
def salvar_edicao_colaborador(colaborador_id):
    colaborador = Colaborador.query.get_or_404(colaborador_id)
    if colaborador.is_gestor:
        flash("Não é possível editar a conta do gestor por aqui.")
        return redirect(url_for("gestor_cadastro"))

    nome = request.form.get("nome", "").strip()
    email = request.form.get("email", "").strip().lower()
    senha = request.form.get("senha", "").strip()
    ativo = request.form.get("ativo") == "on"
    foto_base64 = request.form.get("foto")

    if not (nome and email):
        flash("Nome e e-mail são obrigatórios.")
        return redirect(url_for("editar_colaborador", colaborador_id=colaborador_id))

    email_em_uso = Colaborador.query.filter(
        Colaborador.email == email, Colaborador.id != colaborador_id
    ).first()
    if email_em_uso:
        flash("Já existe outro colaborador com esse e-mail.")
        return redirect(url_for("editar_colaborador", colaborador_id=colaborador_id))

    colaborador.nome = nome
    colaborador.email = email
    colaborador.ativo = ativo

    if senha:
        colaborador.set_senha(senha)

    mensagem = f"Dados de {nome} atualizados com sucesso."
    if foto_base64:
        encoding, motivo_falha = imagem_base64_para_encoding(foto_base64)
        if encoding is None:
            mensagem_motivo = MENSAGENS_POSICIONAMENTO.get(motivo_falha, "Não foi possível reconhecer um rosto na nova foto.")
            flash(f"{mensagem_motivo} O rosto cadastrado anteriormente foi mantido.")
        else:
            duplicado = encontrar_colaborador_por_rosto(encoding, ignorar_id=colaborador.id)
            if duplicado:
                flash(f"Esse rosto já parece estar cadastrado para {duplicado.nome}. O rosto anterior foi mantido — confira se não é duplicidade.")
            else:
                colaborador.face_encoding = encoding.tolist()
                mensagem += " Rosto atualizado."

    db.session.commit()
    flash(mensagem)
    return redirect(url_for("gestor_cadastro"))


@app.route("/gestor/exportar-csv")
@gestor_required
def exportar_csv():
    registros = RegistroPonto.query.order_by(RegistroPonto.data_hora.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Colaborador", "Data/Hora (Horário de Brasília)", "Tipo", "Endereço", "Latitude", "Longitude", "Distância Facial"])
    for r in registros:
        horario_local = para_brasilia(r.data_hora)
        writer.writerow([
            r.colaborador.nome,
            horario_local.strftime("%d/%m/%Y %H:%M:%S"),
            ROTULOS_TIPO_REGISTRO.get((r.tipo or "").lower(), ""),
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


@app.route("/gestor/preencher-enderecos", methods=["POST"])
@gestor_required
def preencher_enderecos():
    """Recupera o endereço de registros antigos que ficaram sem (ex.: por falha temporária
    na geocodificação). Processa em lotes pequenos para respeitar o limite de 1 req/s dos
    serviços de geocodificação — clique de novo para continuar preenchendo o restante."""
    pendentes = (
        RegistroPonto.query
        .filter(RegistroPonto.endereco.is_(None))
        .filter(RegistroPonto.latitude.isnot(None))
        .order_by(RegistroPonto.data_hora.desc())
        .limit(30)
        .all()
    )

    preenchidos = 0
    for r in pendentes:
        # Aqui ninguém está esperando na tela, então vale insistir mais que na batida.
        endereco = obter_endereco(r.latitude, r.longitude, tentativas=3, orcamento=30.0)
        if endereco:
            r.endereco = endereco
            preenchidos += 1
        # Sem sleep aqui: obter_endereco já cuida do intervalo entre chamadas e resolve
        # coordenadas repetidas pelo cache, sem ir na rede de novo.
    db.session.commit()

    restantes = RegistroPonto.query.filter(
        RegistroPonto.endereco.is_(None), RegistroPonto.latitude.isnot(None)
    ).count()

    if restantes:
        flash(f"{preenchidos} endereço(s) preenchido(s). Ainda restam {restantes} — clique novamente para continuar.")
    else:
        flash(f"{preenchidos} endereço(s) preenchido(s). Todos os registros já têm endereço.")

    return redirect(url_for("gestor_consulta"))


# ---------------------------------------------------------------------------
# Rotas - gestor: relatório de horas extras
# ---------------------------------------------------------------------------
def _calcular_relatorio_horas_extras(ano, mes):
    """Monta o relatório de horas extras do mês para todos os colaboradores."""
    config = obter_configuracao()

    primeiro_dia = date(ano, mes, 1)
    proximo_mes = date(ano + 1, 1, 1) if mes == 12 else date(ano, mes + 1, 1)

    colaboradores = Colaborador.query.filter_by(is_gestor=False).order_by(Colaborador.nome).all()

    # Janela generosa em UTC (cobre qualquer fuso) — filtramos o dia exato em Brasília depois.
    inicio_utc = datetime.combine(primeiro_dia, datetime.min.time()) - timedelta(hours=6)
    fim_utc = datetime.combine(proximo_mes, datetime.min.time()) + timedelta(hours=6)
    candidatos = (
        RegistroPonto.query
        .filter(RegistroPonto.data_hora >= inicio_utc, RegistroPonto.data_hora < fim_utc)
        .order_by(RegistroPonto.data_hora.asc())
        .all()
    )

    from collections import defaultdict
    por_colaborador = defaultdict(list)
    for r in candidatos:
        if primeiro_dia <= para_brasilia(r.data_hora).date() < proximo_mes:
            por_colaborador[r.colaborador_id].append(r)

    resumo_colaboradores = []
    for c in colaboradores:
        dias = montar_resumo_diario(por_colaborador.get(c.id, []))
        total_extra = 0.0
        total_horas_mes = 0.0
        dias_com_extra = []
        for dia in dias:
            extra = calcular_horas_extras_dia(
                dia["total_horas_decimal"], config.horario_entrada, config.horario_saida,
                eh_domingo_flag=eh_domingo(dia["data"]),
            )
            dia["horas_extras"] = extra
            aplicar_status_excepcional(dia, c.id)
            extra = dia["horas_extras"]  # pode ter sido zerada por aplicar_status_excepcional
            total_extra += extra
            total_horas_mes += dia["total_horas_decimal"]
            if extra > 0 or dia.get("excepcional"):
                dias_com_extra.append({
                    "data": dia["data"],
                    "total_horas": dia["total_horas"],
                    "horas_extras": extra,
                    "incompleto": dia["incompleto"],
                    "excepcional": dia.get("excepcional", False),
                    "status_excepcional": dia.get("status_excepcional"),
                })
        resumo_colaboradores.append({
            "colaborador": c,
            "total_extra": round(total_extra, 2),
            "total_horas_mes": round(total_horas_mes, 2),
            "dias_com_extra": sorted(dias_com_extra, key=lambda d: d["data"], reverse=True),
        })

    resumo_colaboradores.sort(key=lambda r: r["total_extra"], reverse=True)
    return resumo_colaboradores, primeiro_dia, config


@app.route("/gestor/horas-extras", methods=["GET"])
@gestor_required
def gestor_horas_extras():
    mes_param = request.args.get("mes", "")
    try:
        ano, mes = (int(p) for p in mes_param.split("-"))
    except Exception:
        hoje = agora_brasilia()
        ano, mes = hoje.year, hoje.month

    resumo_colaboradores, primeiro_dia, config = _calcular_relatorio_horas_extras(ano, mes)
    total_extra_empresa = round(sum(r["total_extra"] for r in resumo_colaboradores), 2)
    colaboradores_com_extra = sum(1 for r in resumo_colaboradores if r["total_extra"] > 0)

    return render_template(
        "gestor_horas_extras.html",
        resumo_colaboradores=paginar_lista(
            resumo_colaboradores, param="p_colaboradores", ancora="por-colaborador"
        ),
        colaboradores_com_extra=colaboradores_com_extra,
        mes_selecionado=f"{ano:04d}-{mes:02d}",
        nome_mes=primeiro_dia.strftime("%B de %Y"),
        total_extra_empresa=total_extra_empresa,
        horario_entrada=config.horario_entrada,
        horario_saida=config.horario_saida,
    )


@app.route("/gestor/horas-extras/exportar-csv", methods=["GET"])
@gestor_required
def exportar_csv_horas_extras():
    mes_param = request.args.get("mes", "")
    try:
        ano, mes = (int(p) for p in mes_param.split("-"))
    except Exception:
        hoje = agora_brasilia()
        ano, mes = hoje.year, hoje.month

    resumo_colaboradores, primeiro_dia, _config = _calcular_relatorio_horas_extras(ano, mes)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Colaborador", "E-mail", "Total de horas trabalhadas no mês", "Total de horas extras no mês"])
    for r in resumo_colaboradores:
        writer.writerow([
            r["colaborador"].nome,
            r["colaborador"].email,
            formatar_horas(r["total_horas_mes"]),
            formatar_horas(r["total_extra"]),
        ])

    nome_arquivo = f"horas_extras_{ano:04d}-{mes:02d}.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={nome_arquivo}"},
    )
def _listar_pendentes_excepcionais():
    """Busca as jornadas excepcionais pendentes de aprovação. Nunca deixa a tabela ainda não
    existir (banco recém-migrado) derrubar a tela — se a consulta falhar, loga e retorna lista
    vazia (a seção correspondente simplesmente não aparece, em vez de dar 500)."""
    try:
        return (
            AprovacaoJornadaExcepcional.query
            .filter_by(status="pendente")
            .order_by(AprovacaoJornadaExcepcional.criado_em.asc())
            .all()
        )
    except Exception as ex:
        db.session.rollback()
        print(f"[jornada-excepcional] falha ao listar pendências: {ex}")
        return []


@app.route("/gestor/ajustes", methods=["GET"])
@gestor_required
def gestor_ajustes():
    # Três blocos independentes na mesma tela -> três parâmetros de página
    # distintos (antes, o único "page" fazia o histórico "pular" junto com
    # qualquer outra navegação).
    pendentes = paginar_query(
        SolicitacaoAjuste.query
        .filter_by(status="pendente")
        .order_by(SolicitacaoAjuste.criado_em.asc()),
        param="p_pendentes",
        ancora="solicitacoes-pendentes",
    )
    historico = paginar_query(
        SolicitacaoAjuste.query
        .filter(SolicitacaoAjuste.status != "pendente")
        .order_by(SolicitacaoAjuste.respondido_em.desc()),
        param="p_historico",
        ancora="historico-ajustes",
    )

    # Jornadas excepcionais (sessão contínua > LIMITE_HORAS_EXCEPCIONAL) aguardando revisão —
    # ver comentário em aplicar_status_excepcional(). Ficam nesta mesma tela de ajustes porque,
    # do ponto de vista do gestor, é o mesmo tipo de decisão: revisar uma pendência de ponto.
    # Protegido contra a tabela ainda não existir no banco (ver _listar_pendentes_excepcionais).
    pendentes_excepcionais = paginar_lista(
        _listar_pendentes_excepcionais(), param="p_excepcionais", ancora="jornadas-excepcionais"
    )

    return render_template(
        "gestor_ajustes.html",
        pendentes=pendentes,
        historico=historico,
        pendentes_excepcionais=pendentes_excepcionais,
        limite_horas_excepcional=LIMITE_HORAS_EXCEPCIONAL,
    )


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


@app.route("/gestor/jornadas-excepcionais/<int:aprovacao_id>/aprovar", methods=["POST"])
@gestor_required
def aprovar_jornada_excepcional(aprovacao_id):
    aprovacao = AprovacaoJornadaExcepcional.query.get_or_404(aprovacao_id)

    if aprovacao.status != "pendente":
        flash("Esta jornada já foi revisada.")
        return redirect(url_for("gestor_ajustes"))

    aprovacao.status = "aprovado"
    aprovacao.respondido_em = datetime.utcnow()
    aprovacao.resposta_gestor = request.form.get("resposta", "").strip()
    db.session.commit()

    flash(f"Jornada excepcional de {aprovacao.colaborador.nome} em {aprovacao.data_referencia.strftime('%d/%m/%Y')} aprovada — a hora extra já entra no total do mês.")
    return redirect(url_for("gestor_ajustes"))


@app.route("/gestor/jornadas-excepcionais/<int:aprovacao_id>/recusar", methods=["POST"])
@gestor_required
def recusar_jornada_excepcional(aprovacao_id):
    aprovacao = AprovacaoJornadaExcepcional.query.get_or_404(aprovacao_id)

    if aprovacao.status != "pendente":
        flash("Esta jornada já foi revisada.")
        return redirect(url_for("gestor_ajustes"))

    aprovacao.status = "recusado"
    aprovacao.respondido_em = datetime.utcnow()
    aprovacao.resposta_gestor = request.form.get("resposta", "").strip()

    # Recusar aqui normalmente significa "isso não foi uma virada de turno real, foi um
    # esquecimento de bater a saída". Nesse caso, a batida que HOJE está marcada como "saida"
    # (porque fechou a sessão gigante) na verdade era a entrada normal do dia seguinte — o
    # colaborador só chegou pro trabalho e bateu o ponto, e o sistema confundiu isso com o
    # fechamento de um turno de madrugada. Corrigimos o tipo dela pra "entrada", que é o que
    # realmente aconteceu. A entrada original (do dia anterior) continua sem saída registrada —
    # isso fica sinalizado como "incompleto" e precisa de uma SolicitacaoAjuste normal pra
    # preencher o horário real em que a pessoa foi embora naquele dia.
    corrigir = request.form.get("corrigir_saida_para_entrada", "0") == "1"
    if corrigir and aprovacao.registro_saida_id:
        registro_saida = RegistroPonto.query.get(aprovacao.registro_saida_id)
        if registro_saida is not None and (registro_saida.tipo or "").lower() == "saida":
            registro_saida.tipo = "entrada"
            registro_saida.origem = "ajuste_manual"

    db.session.commit()

    flash(f"Jornada excepcional de {aprovacao.colaborador.nome} em {aprovacao.data_referencia.strftime('%d/%m/%Y')} recusada — nenhuma hora extra será computada. A batida que fechava a sessão foi corrigida para 'entrada'; falta um ajuste manual com o horário real de saída do dia {aprovacao.data_referencia.strftime('%d/%m/%Y')}.")
    return redirect(url_for("gestor_ajustes"))


# ---------------------------------------------------------------------------
# Rotas - Assistente inteligente
#
# O motor fica em assistente.py e não conhece o app: recebe aqui, por injeção,
# os modelos e as funções de cálculo que já existem (mesma regra de hora extra,
# mesmo pareamento entrada/saída, mesmo fuso). Assim o assistente nunca responde
# um número diferente do que as telas mostram.
# ---------------------------------------------------------------------------
import assistente as motor_assistente


def _contexto_assistente():
    from types import SimpleNamespace
    return SimpleNamespace(
        Colaborador=Colaborador,
        RegistroPonto=RegistroPonto,
        SolicitacaoAjuste=SolicitacaoAjuste,
        AprovacaoJornadaExcepcional=AprovacaoJornadaExcepcional,
        AssistenteApelido=AssistenteApelido,
        AssistentePadrao=AssistentePadrao,
        obter_configuracao=obter_configuracao,
        montar_resumo_diario=montar_resumo_diario,
        calcular_horas_extras_dia=calcular_horas_extras_dia,
        registros_do_dia=registros_do_dia,
        para_brasilia=para_brasilia,
        agora_brasilia=agora_brasilia,
        eh_domingo=eh_domingo,
        formatar_horas=formatar_horas,
    )


def obter_assistente():
    return motor_assistente.Assistente(_contexto_assistente())


def _sugestoes_iniciais(usuario):
    """Sugestões da tela inicial do chat: começa com exemplos e, conforme a
    empresa usa o assistente, dá lugar às perguntas que mais deram certo
    (aprendizado de uso real, não lista fixa)."""
    if usuario.is_gestor:
        padrao = [
            "Quantas horas extras a equipe tem este mês?",
            "Quem ainda não bateu ponto hoje?",
            "Quais pendências estão esperando minha aprovação?",
            "Quem tem mais horas extras este mês?",
        ]
    else:
        padrao = [
            "Quantas horas extras eu tenho este mês?",
            "Quantas horas trabalhei este mês?",
            "Tenho algum dia incompleto?",
            "Como está minha solicitação de ajuste?",
        ]
    try:
        aprovadas = (
            AssistenteInteracao.query
            .filter_by(util=True)
            .order_by(AssistenteInteracao.criado_em.desc())
            .limit(30)
            .all()
        )
        vistas, sugeridas = set(), []
        for i in aprovadas:
            chave = i.pergunta_normalizada
            if chave in vistas:
                continue
            vistas.add(chave)
            sugeridas.append(i.pergunta)
            if len(sugeridas) >= 4:
                break
        if sugeridas:
            return sugeridas + padrao[: max(0, 4 - len(sugeridas))]
    except Exception as ex:
        print(f"[assistente] não foi possível montar sugestões aprendidas: {ex}")
    return padrao


@app.route("/assistente", methods=["GET"])
@login_required
def assistente_chat():
    usuario = Colaborador.query.get(session["user_id"])
    return render_template(
        "assistente.html",
        sugestoes=_sugestoes_iniciais(usuario),
        eh_gestor=bool(usuario.is_gestor),
    )


@app.route("/api/assistente/perguntar", methods=["POST"])
@login_required
def assistente_perguntar():
    usuario = Colaborador.query.get(session["user_id"])
    dados = request.get_json(silent=True) or {}
    pergunta = (dados.get("pergunta") or "").strip()[:500]
    colaborador_id = dados.get("colaborador_id")

    if not pergunta:
        return jsonify({"ok": False, "mensagem": "Escreva uma pergunta."}), 400

    forcado = None
    if colaborador_id and usuario.is_gestor:
        forcado = Colaborador.query.get(colaborador_id)

    resultado = obter_assistente().responder(pergunta, usuario, colaborador_forcado=forcado)

    # Registra a interação (base do aprendizado). Uma falha aqui não pode
    # impedir a resposta de chegar ao usuário.
    interacao_id = None
    try:
        interacao = AssistenteInteracao(
            colaborador_id=usuario.id,
            pergunta=pergunta,
            pergunta_normalizada=motor_assistente.normalizar(pergunta),
            intencao=resultado.get("intencao"),
            confianca=resultado.get("confianca") or 0.0,
            colaborador_alvo_id=forcado.id if forcado else None,
            resposta=(resultado.get("resposta") or "")[:4000],
        )
        db.session.add(interacao)
        db.session.commit()
        interacao_id = interacao.id
    except Exception as ex:
        db.session.rollback()
        print(f"[assistente] falha ao registrar interação: {ex}")

    resultado["interacao_id"] = interacao_id
    resultado["ok"] = True
    return jsonify(resultado)


@app.route("/api/assistente/feedback", methods=["POST"])
@login_required
def assistente_feedback():
    """👍/👎 numa resposta. É assim que o assistente aprende a entender o jeito
    que ESTA empresa escreve: a pergunta aprovada vira exemplo da intenção
    detectada; a reprovada perde peso e deixa de ser usada como referência."""
    dados = request.get_json(silent=True) or {}
    interacao_id = dados.get("interacao_id")
    util = bool(dados.get("util"))

    interacao = AssistenteInteracao.query.get(interacao_id) if interacao_id else None
    if interacao is None:
        return jsonify({"ok": False, "mensagem": "Interação não encontrada."}), 404
    if interacao.colaborador_id != session["user_id"]:
        return jsonify({"ok": False, "mensagem": "Sem permissão."}), 403

    interacao.util = util

    intencao = interacao.intencao
    if intencao and intencao not in ("desconhecida", "erro", "vazia", "ambiguo", "sem_colaborador"):
        padrao = AssistentePadrao.query.filter_by(
            padrao=interacao.pergunta_normalizada, intencao=intencao
        ).first()
        if padrao is None:
            padrao = AssistentePadrao(
                padrao=interacao.pergunta_normalizada, intencao=intencao,
                peso=1.0 if util else 0.0, usos=1,
            )
            db.session.add(padrao)
        else:
            padrao.peso = max(0.0, min(5.0, (padrao.peso or 0) + (0.5 if util else -1.0)))
            padrao.usos = (padrao.usos or 0) + 1
            padrao.atualizado_em = datetime.utcnow()

    try:
        db.session.commit()
    except Exception as ex:
        db.session.rollback()
        print(f"[assistente] falha ao salvar feedback: {ex}")
        return jsonify({"ok": False, "mensagem": "Não foi possível salvar o feedback."}), 500

    return jsonify({
        "ok": True,
        "mensagem": "Obrigado! Vou usar isso para melhorar." if util
                    else "Anotado — vou evitar responder assim de novo.",
    })


@app.route("/api/assistente/ensinar", methods=["POST"])
@gestor_required
def assistente_ensinar():
    """O gestor diz de quem era a pergunta que o assistente não identificou.
    Os termos usados por ele viram apelidos daquele colaborador, e a pergunta é
    respondida na hora, já corrigida."""
    dados = request.get_json(silent=True) or {}
    interacao_id = dados.get("interacao_id")
    colaborador_id = dados.get("colaborador_id")

    colaborador = Colaborador.query.get(colaborador_id) if colaborador_id else None
    interacao = AssistenteInteracao.query.get(interacao_id) if interacao_id else None
    if colaborador is None or interacao is None:
        return jsonify({"ok": False, "mensagem": "Dados incompletos."}), 400

    aprendidos = []
    for termo in motor_assistente.termos_para_apelido(interacao.pergunta, colaborador.nome):
        existente = AssistenteApelido.query.filter_by(apelido=termo).first()
        if existente and existente.colaborador_id != colaborador.id:
            continue  # apelido já pertence a outra pessoa — não sobrescreve
        if existente:
            existente.acertos = (existente.acertos or 0) + 1
        else:
            db.session.add(AssistenteApelido(apelido=termo, colaborador_id=colaborador.id))
            aprendidos.append(termo)

    interacao.colaborador_alvo_id = colaborador.id
    try:
        db.session.commit()
    except Exception as ex:
        db.session.rollback()
        print(f"[assistente] falha ao aprender apelido: {ex}")

    usuario = Colaborador.query.get(session["user_id"])
    resultado = obter_assistente().responder(
        interacao.pergunta, usuario, colaborador_forcado=colaborador
    )
    resultado["ok"] = True
    resultado["interacao_id"] = interacao_id
    resultado["aprendido"] = aprendidos
    return jsonify(resultado)


# ---------------------------------------------------------------------------
# API externa (REST) para integração com outros sistemas (ex.: Lecom)
#
# Mesma ideia do assistente: o módulo api_externa.py não conhece o app, recebe
# os modelos e helpers por injeção. Assim a API devolve exatamente os mesmos
# números e o mesmo pareamento entrada/saída que as telas mostram.
#
# Autenticação por token na variável de ambiente API_TOKENS (ou API_TOKEN).
# Sem ela configurada, a API responde 503 e não expõe dado nenhum.
# ---------------------------------------------------------------------------
import api_externa


def _contexto_api():
    from types import SimpleNamespace
    return SimpleNamespace(
        Colaborador=Colaborador,
        RegistroPonto=RegistroPonto,
        para_brasilia=para_brasilia,
        agora_brasilia=agora_brasilia,
    )


app.register_blueprint(api_externa.criar_blueprint_api(_contexto_api()))


@app.cli.command("gerar-token-api")
def gerar_token_api():
    """Gera um token aleatório para a API externa. Rodar com: flask gerar-token-api

    Copie o valor impresso para a variável de ambiente API_TOKENS do servidor
    (no Render: Environment -> Add Environment Variable) e entregue o mesmo
    valor para quem vai consumir a API."""
    import secrets
    print(secrets.token_urlsafe(32))


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
    comandos = []

    colunas_registro = [col["name"] for col in inspector.get_columns("registro_ponto")]
    if "endereco" not in colunas_registro:
        comandos.append("ALTER TABLE registro_ponto ADD COLUMN endereco VARCHAR(255)")
    if "origem" not in colunas_registro:
        comandos.append("ALTER TABLE registro_ponto ADD COLUMN origem VARCHAR(20)")

    if "configuracao_jornada" in inspector.get_table_names():
        colunas_jornada = [col["name"] for col in inspector.get_columns("configuracao_jornada")]
        if "meta_horas_diarias" not in colunas_jornada:
            comandos.append("ALTER TABLE configuracao_jornada ADD COLUMN meta_horas_diarias FLOAT DEFAULT 8.0")

    if "configuracao_marca" in inspector.get_table_names():
        colunas_marca = [col["name"] for col in inspector.get_columns("configuracao_marca")]
        if "logo_login_base64" not in colunas_marca:
            comandos.append("ALTER TABLE configuracao_marca ADD COLUMN logo_login_base64 TEXT")
        if "logo_topbar_base64" not in colunas_marca:
            comandos.append("ALTER TABLE configuracao_marca ADD COLUMN logo_topbar_base64 TEXT")

    # aprovacao_jornada_excepcional foi criada num deploy anterior ao das colunas
    # registro_entrada_id/registro_saida_id — como db.create_all() só cria tabelas que ainda não
    # existem (nunca altera uma tabela já existente), essas colunas ficaram faltando em produção
    # e quebravam a tela de ajustes com "UndefinedColumn". Adiciona aqui se faltarem.
    if "aprovacao_jornada_excepcional" in inspector.get_table_names():
        colunas_aprovacao = [col["name"] for col in inspector.get_columns("aprovacao_jornada_excepcional")]
        if "registro_entrada_id" not in colunas_aprovacao:
            comandos.append("ALTER TABLE aprovacao_jornada_excepcional ADD COLUMN registro_entrada_id INTEGER REFERENCES registro_ponto(id)")
        if "registro_saida_id" not in colunas_aprovacao:
            comandos.append("ALTER TABLE aprovacao_jornada_excepcional ADD COLUMN registro_saida_id INTEGER REFERENCES registro_ponto(id)")

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

