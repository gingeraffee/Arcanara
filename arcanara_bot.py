# -*- coding: utf-8 -*-
import discord
from discord.ext import commands, tasks
from discord import app_commands
import re
import random
from discord.errors import NotFound as DiscordNotFound
import time
import json
from pathlib import Path
import os
import psycopg
from datetime import datetime, date
import traceback
from zoneinfo import ZoneInfo
from psycopg.types.json import Json
from psycopg.rows import dict_row
from typing import Dict, Any, List, Optional
from card_images import make_image_attachment  # uses assets/cards/rws_stx/ etc.

# Optional top.gg integration
try:
    import topgg
    TOPGG_AVAILABLE = True
except ImportError:
    TOPGG_AVAILABLE = False
    print("⚠️ topgg not installed - stats posting disabled")

print("✅ Arcanara boot: VERSION 2025-12-21-TopGG-2")

MYSTERY_STATE: Dict[int, Dict[str, Any]] = {}

# ==============================
# CONFIGURATION
# ==============================
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("❌ BOT_TOKEN environment variable not found. Please set it in your host environment settings.")

TOPGG_TOKEN = os.getenv("TOPGG_TOKEN")
# Top.gg token is optional - bot will run without it but won't post stats

# ==============================
# DATABASE (Render Postgres)
# ==============================
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("❌ DATABASE_URL environment variable not found. Add your Render Postgres DATABASE_URL to this service.")

_DB_READY = False  # prevents re-creating tables multiple times


def db_connect():
    return psycopg.connect(
        DATABASE_URL,
        row_factory=dict_row,
        connect_timeout=10,
    )


def ensure_tables():
    """Create tables if they don't exist (safe to run on startup)."""
    with db_connect() as conn:
        with conn.cursor() as cur:
            # Existing table: user tone preference
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS tarot_user_prefs (
                    user_id BIGINT PRIMARY KEY,
                    tone TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            # ---- MIGRATION: older schema used "mode" instead of "tone"
            cur.execute("""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'tarot_user_prefs'
                      AND column_name = 'mode'
                )
                AND NOT EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'tarot_user_prefs'
                      AND column_name = 'tone'
                )
                THEN
                    ALTER TABLE tarot_user_prefs RENAME COLUMN mode TO tone;
                END IF;
            END $$;
            """)

            # New table: user settings (opt-in history + images toggle)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS tarot_user_settings (
                    user_id BIGINT PRIMARY KEY,
                    history_opt_in BOOLEAN NOT NULL DEFAULT FALSE,
                    images_enabled BOOLEAN NOT NULL DEFAULT TRUE,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )

            # New table: reading history (only used if opt-in)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS tarot_reading_history (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    command TEXT NOT NULL,
                    tone TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            
            # Daily Card (persist per user per day)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS tarot_daily_card (
                    user_id BIGINT NOT NULL,
                    day DATE NOT NULL,
                    card_name TEXT NOT NULL,
                    orientation TEXT NOT NULL,  -- 'Upright' or 'Reversed'
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (user_id, day)
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tarot_daily_card_day
                ON tarot_daily_card (day);
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tarot_history_user_time
                ON tarot_reading_history (user_id, created_at DESC);
                """
            )

        conn.commit()


# ==============================
# TAROT TONES (DB-backed)
# ==============================
DEFAULT_TONE = "poetic"

TONE_SPECS = {
    "quick": ["voice_pulse", "call_to_action"],
    "poetic": ["voice_lead", "meaning", "voice_pulse", "mantra", "voice_turn", "call_to_action"],


    "direct": ["reader_voice", "tell", "do_dont", "prescription", "watch_for", "pitfall", "questions", "next_24h", "call_to_action"],
    "shadow": ["reader_voice", "tell", "shadow", "watch_for", "pitfall", "questions", "call_to_action"],

    "love":   ["reader_voice", "tell", "relationships", "green_red", "pitfall", "questions", "call_to_action"],
    "work":   ["reader_voice", "tell", "work", "prescription", "watch_for", "next_24h", "call_to_action"],
    "money":  ["reader_voice", "tell", "money", "prescription", "watch_for", "next_24h", "call_to_action"],

    "full": ["voice_lead", "reader_voice", "tell", "meaning", "voice_pulse", "mantra", "do_dont",
         "prescription", "watch_for", "pitfall", "shadow", "green_red", "questions",
         "next_24h", "voice_turn", "call_to_action"],

}

TONE_LABELS = {
    "full":   "Full Spectrum (deep + practical)",
    "direct": "Direct (straight talk, no fluff)",
    "shadow": "Shadow Work (truth + integration)",
    "poetic": "Poetic (symbolic, soft edges)",
    "quick":  "Quick Hit (one clear message)",
    "love":   "Love Lens (people + patterns)",
    "work":   "Work Lens (purpose + friction)",
    "money":  "Money Lens (resources + decisions)",
}

_original_autocomplete = discord.InteractionResponse.autocomplete

async def _safe_autocomplete(self, choices):
    try:
        return await _original_autocomplete(self, choices)
    except DiscordNotFound as e:
        # 10062 = Unknown interaction (common when user types fast / interaction expires)
        if getattr(e, "code", None) == 10062:
            return
        raise

discord.InteractionResponse.autocomplete = _safe_autocomplete

def normalize_tone(tone: str) -> str:
    t = (tone or "").lower().strip()
    return t if t in TONE_SPECS else DEFAULT_TONE

def tone_label(tone: str) -> str:
    t = normalize_tone(tone)
    return TONE_LABELS.get(t, TONE_LABELS[DEFAULT_TONE])

def get_effective_tone(user_id: int, tone_override: Optional[str] = None) -> str:
    return normalize_tone(tone_override) if tone_override else get_user_tone(user_id)

def get_user_tone(user_id: int) -> str:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT tone FROM tarot_user_prefs WHERE user_id=%s", (user_id,))
            row = cur.fetchone()
    return normalize_tone(row["tone"]) if row else DEFAULT_TONE

def set_user_tone(user_id: int, tone: str) -> str:
    t = normalize_tone(tone)
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO tarot_user_prefs (user_id, tone)
                VALUES (%s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    tone = EXCLUDED.tone,
                    updated_at = NOW()
            """, (user_id, t))
        conn.commit()
    return t

def reset_user_tone(user_id: int) -> str:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM tarot_user_prefs WHERE user_id=%s", (user_id,))
        conn.commit()
    return DEFAULT_TONE


def _clip(text: str, max_len: int = 3800) -> str:
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"

def _orientation_key(orientation: str) -> str:
    o = (orientation or "").strip().lower()
    return "upright" if o.startswith("u") else "reversed"


def _get_orientation_data(card: dict, orientation: str) -> dict:
    """
    Always returns a dict for the selected orientation.
    Supports old decks where upright/reversed might be a string or list.
    """
    okey = _orientation_key(orientation)
    val = card.get(okey)

    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        return {"meaning": val}
    if isinstance(val, list):
        joined = "\n".join(str(x) for x in val if str(x).strip())
        return {"meaning": joined}
    return {}


def render_card_text(card: Dict[str, Any], orientation: str, tone: str) -> str:
    tone = normalize_tone(tone)
    spec = TONE_SPECS.get(tone, TONE_SPECS[DEFAULT_TONE])

    is_rev = orientation.strip().lower().startswith("r")
    okey = "reversed" if is_rev else "upright"

    # ✅ NEW: normalize orientation data to a dict
    odata_raw = card.get(okey, {})
    if isinstance(odata_raw, dict):
        odata = odata_raw
    elif isinstance(odata_raw, str):
        odata = {"meaning": odata_raw}
    elif isinstance(odata_raw, list):
        odata = {"meaning": "\n".join(str(x) for x in odata_raw if str(x).strip())}
    else:
        odata = {}

    # ✅ NEW: meaning must be a string
    meaning = (odata.get("meaning") or "—")
    if not isinstance(meaning, str):
        meaning = str(meaning)
    # ✅ OPTIONAL UPGRADE: new voice fields (safe, no behavior change unless used)
    voice = odata.get("voice", {})
    if not isinstance(voice, dict):
        voice = {}

    v_lead = (voice.get("lead_in") or "").strip()
    v_pulse = (voice.get("pulse") or "").strip()
    v_turn = (voice.get("turn") or "").strip()
    dg = card.get("direct_guidance", {}) or {}
    lenses = dg.get("lenses", {}) or {}

    def do_dont():
        do = dg.get("do", "")
        dont = dg.get("dont", "")
        if do and dont:
            return f"**Do:** {do}\n**Don't:** {dont}"
        return do or dont

    def questions():
        qs = dg.get("questions", []) or []
        qs = [q for q in qs if isinstance(q, str) and q.strip()]
        return "**Ask:** " + " | ".join(qs[:3]) if qs else ""

    blocks = []
    for token in spec:
        if token == "meaning":
            blocks.append(meaning)

        elif token == "mantra":
            m = dg.get("mantra", "")
            if m:
                blocks.append(f"**Mantra:** {m}")

        elif token == "quick":
            q = dg.get("quick", "")
            if q:
                blocks.append(q)

        elif token == "do":
            d = dg.get("do", "")
            if d:
                blocks.append(f"**Do:** {d}")

        elif token == "do_dont":
            dd = do_dont()
            if dd:
                blocks.append(dd)

        elif token == "watch_for":
            w = dg.get("watch_for", "")
            if w:
                blocks.append(f"**Watch for:** {w}")

        elif token == "shadow":
            s = dg.get("shadow", "")
            if s:
                blocks.append(f"**Shadow:** {s}")

        elif token == "questions":
            qs = questions()
            if qs:
                blocks.append(qs)

        elif token == "next_24h":
            n = dg.get("next_24h", "")
            if n:
                blocks.append(f"**Next 24h:** {n}")

        elif token == "relationships":
            txt = lenses.get("relationships") or dg.get("relationships", "")
            if txt:
                blocks.append(f"**Love/People:** {txt}")

        elif token == "work":
            txt = lenses.get("work") or dg.get("work", "")
            if txt:
                blocks.append(f"**Work:** {txt}")

        elif token == "money":
            txt = lenses.get("money") or dg.get("money", "")
            if txt:
                blocks.append(f"**Money:** {txt}")

        # ---- v2 fields ----
        elif token == "tell":
            t = dg.get("tell", "")
            if t:
                blocks.append(f"**The truth:** {t}")

        elif token == "prescription":
            p = dg.get("prescription", "")
            if p:
                blocks.append(f"**Do this:** {p}")

        elif token == "pitfall":
            p = dg.get("pitfall", "")
            if p:
                blocks.append(f"**Pitfall:** {p}")

        elif token == "green_red":
            gf = dg.get("green_flag", "")
            rf = dg.get("red_flag", "")
            if gf or rf:
                line = []
                if gf:
                    line.append(f"**Green flag:** {gf}")
                if rf:
                    line.append(f"**Red flag:** {rf}")
                blocks.append("\n".join(line))

        elif token == "reader_voice":
            rv = dg.get("reader_voice", "")
            if rv:
                blocks.append(f"*{rv}*")

        elif token == "poetic_hint":
            ph = dg.get("poetic_hint", "")
            if ph and not (v_lead or v_pulse or v_turn):
                blocks.append(f"*{ph}*")

        elif token == "voice_lead":
            if v_lead:
                blocks.append(f"*{v_lead}*")

        elif token == "voice_pulse":
            if v_pulse:
                blocks.append(f"*{v_pulse}*")

        elif token == "voice_turn":
            if v_turn:
                blocks.append(f"*{v_turn}*")

        elif token == "call_to_action":
            a = card.get("call_to_action", "")
            if a:
                blocks.append(f"**Action:** {a}")

    return _clip("\n\n".join(blocks))


# ==============================
# USER SETTINGS + HISTORY (DB-backed)
# ==============================
def get_user_settings(user_id: int) -> dict:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT history_opt_in, images_enabled
                FROM tarot_user_settings
                WHERE user_id=%s
                """,
                (user_id,),
            )
            row = cur.fetchone()
    return row or {"history_opt_in": False, "images_enabled": True}


def set_user_settings(
    user_id: int,
    *,
    history_opt_in: Optional[bool] = None,
    images_enabled: Optional[bool] = None,
) -> dict:
    current = get_user_settings(user_id)
    if history_opt_in is None:
        history_opt_in = current["history_opt_in"]
    if images_enabled is None:
        images_enabled = current["images_enabled"]

    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tarot_user_settings (user_id, history_opt_in, images_enabled)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    history_opt_in = EXCLUDED.history_opt_in,
                    images_enabled = EXCLUDED.images_enabled,
                    updated_at = NOW()
                """,
                (user_id, history_opt_in, images_enabled),
            )
        conn.commit()

    return {"history_opt_in": history_opt_in, "images_enabled": images_enabled}

def fetch_history(user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT command, tone, payload, created_at
                FROM tarot_reading_history
                WHERE user_id=%s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (user_id, limit),
            )
            rows = cur.fetchall() or []
    return rows


def summarize_history_row(command: str, payload: Dict[str, Any]) -> str:
    """Turn stored payload into a short human-readable line."""
    try:
        if command == "cardoftheday":
            card = payload.get("card", "Unknown")
            orientation = payload.get("orientation", "")
            intention = payload.get("intention")
            base = f"**{card}** ({orientation})"
            if intention:
                base += f" — *{intention}*"
            return base

        if command in ("read", "threecard", "celtic"):
            cards = payload.get("cards", []) or []
            # cards elements look like: {"position": "...", "name": "...", "orientation": "..."}
            parts = []
            for c in cards[:10]:
                pos = c.get("position", "—")
                name = c.get("name", "Unknown")
                ori = c.get("orientation", "")
                parts.append(f"{pos}: {name} ({ori})")
            return "; ".join(parts) if parts else "Spread saved (no card details)."

        if command == "meaning":
            q = payload.get("query", "—")
            matched = payload.get("matched", "—")
            return f"Meaning lookup — **{matched}** (query: *{q}*)"

        if command == "clarify":
            card = (payload.get("card") or {}).get("name", "Unknown")
            ori = (payload.get("card") or {}).get("orientation", "")
            intention = payload.get("intention")
            base = f"Clarifier — **{card}** ({ori})"
            if intention:
                base += f" — *{intention}*"
            return base

        if command == "reveal":
            card = (payload.get("card") or {}).get("name", "Unknown")
            ori = (payload.get("card") or {}).get("orientation", "")
            return f"Mystery reveal — **{card}** ({ori})"

        # fallback
        return "Saved reading."
    except Exception:
        return "Saved reading."


def log_history_if_opted_in(
    user_id: int,
    command: str,
    tone: str,
    payload: dict,
    *,
    settings: Optional[Dict[str, Any]] = None,
) -> None:
    """
    If settings are provided, uses them (no extra DB read).
    If not provided, fetches settings from DB.
    Never crashes a command if logging fails.
    """
    try:
        if settings is None:
            settings = get_user_settings(user_id)

        if not settings.get("history_opt_in", False):
            return

        with db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO tarot_reading_history (user_id, command, tone, payload)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (user_id, command, tone, Json(payload)),
                )
            conn.commit()

    except Exception as e:
        print(f"⚠️ history log failed: {type(e).__name__}: {e}")


# ==============================
# LOAD TAROT JSON
# ==============================
def load_tarot_json():
    base_dir = Path(__file__).resolve().parent
    json_path = base_dir / "Tarot_Official.JSON"
    if not json_path.exists():
        raise FileNotFoundError(
            f"❌ Tarot JSON not found at {json_path}. Make sure 'Tarot_Official.JSON' is in the same directory."
        )
    with json_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


tarot_cards = load_tarot_json()
print(f"✅ Loaded {len(tarot_cards)} tarot cards successfully!")

# ==============================
# AUTOCOMPLETE: CARD NAMES
# ==============================
CARD_NAMES: List[str] = sorted({c.get("name", "") for c in tarot_cards if c.get("name")})

def _rank_card_matches(query: str, names: List[str], limit: int = 25) -> List[str]:
    q = (query or "").strip().lower()
    if not q:
        return names[:limit]

    starts = []
    contains = []
    for n in names:
        nl = n.lower()
        if nl.startswith(q):
            starts.append(n)
        elif q in nl:
            contains.append(n)

    # Startswith matches first, then contains matches
    results = starts + contains
    return results[:limit]

async def card_name_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> List[app_commands.Choice[str]]:
    try:
        matches = _rank_card_matches(current, CARD_NAMES, limit=25)
        return [app_commands.Choice(name=m, value=m) for m in matches]
    except Exception as e:
        print(f"⚠️ autocomplete failed: {type(e).__name__}: {e}")
        return []


# ==============================
# SEEKER MEMORY SYSTEM
# ==============================
BASE_DIR = Path(__file__).resolve().parent
KNOWN_SEEKERS_FILE = BASE_DIR / "known_seekers.json"


def load_known_seekers() -> Dict[str, Any]:
    if KNOWN_SEEKERS_FILE.exists():
        try:
            with KNOWN_SEEKERS_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ could not load known_seekers: {type(e).__name__}: {e}")
            return {}
    return {}


def save_known_seekers(data: Dict[str, Any]) -> None:
    try:
        with KNOWN_SEEKERS_FILE.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ could not save known_seekers: {type(e).__name__}: {e}")


known_seekers: Dict[str, Any] = load_known_seekers()
user_intentions: Dict[int, str] = {}


# ==============================
# BOT SETUP
# ==============================
intents = discord.Intents.default()
intents.guilds = True
intents.message_content = False
bot = commands.Bot(command_prefix="!", intents=intents)


# ==============================
# EMOJIS
# ==============================
E = {
    "sun": "☀️",
    "moon": "🌙",
    "crystal": "🔮",
    "light": "💡",
    "clock": "🕰️",
    "star": "🌟",
    "book": "📖",
    "spark": "✨",
    "warn": "⚠️",
    "fire": "🔥",
    "water": "💧",
    "sword": "⚔️",
    "leaf": "🌿",
    "arcana": "🌌",
    "shuffle": "🔁",
}


# ==============================
# NAME NORMALIZATION
# ==============================
NUM_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
NUM_WORDS_RE = re.compile(r"\b(" + "|".join(NUM_WORDS.keys()) + r")\b")


def normalize_card_name(name: str) -> str:
    s = name.lower().strip()
    s = NUM_WORDS_RE.sub(lambda m: NUM_WORDS[m.group(1)], s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


# ==============================
# HELPERS
# ==============================
DEFAULT_TZ = ZoneInfo("America/Chicago")

def _today_local_date() -> date:
    return datetime.now(DEFAULT_TZ).date()

def get_daily_card_row(user_id: int, day) -> Optional[Dict[str, Any]]:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT card_name, orientation, created_at
                FROM tarot_daily_card
                WHERE user_id=%s AND day=%s
                """,
                (user_id, day),
            )
            return cur.fetchone()

def set_daily_card_row(user_id: int, day, card_name: str, orientation: str) -> None:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tarot_daily_card (user_id, day, card_name, orientation)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, day) DO NOTHING
                """,
                (user_id, day, card_name, orientation),
            )
        conn.commit()

def find_card_by_name(name: str) -> Optional[Dict[str, Any]]:
    return next((c for c in tarot_cards if c.get("name") == name), None)

def draw_card():
    card = random.choice(tarot_cards)
    orientation = random.choice(["Upright", "Reversed"])
    return card, orientation


def draw_unique_cards(num_cards: int):
    deck = tarot_cards.copy()
    random.shuffle(deck)
    drawn = []
    for _ in range(min(num_cards, len(deck))):
        card = deck.pop()
        orientation = random.choice(["Upright", "Reversed"])
        drawn.append((card, orientation))
    return drawn


def clip_field(text: str, limit: int = 1024) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def suit_color(suit):
    return {
        "Wands": 0xE25822,
        "Cups": 0x0077BE,
        "Swords": 0xB0B0B0,
        "Pentacles": 0x2E8B57,
        "Major Arcana": 0xA020F0,
    }.get(suit, 0x9370DB)


def suit_emoji(suit):
    return {
        "Wands": E["fire"],
        "Cups": E["water"],
        "Swords": E["sword"],
        "Pentacles": E["leaf"],
        "Major Arcana": E["arcana"],
    }.get(suit, E["crystal"])


def _chunk_lines(lines: List[str], max_len: int = 950) -> List[str]:
    """Chunk lines into strings that fit comfortably in an embed field."""
    chunks: List[str] = []
    buf: List[str] = []
    size = 0
    for line in lines:
        add = len(line) + 1
        if buf and size + add > max_len:
            chunks.append("\n".join(buf))
            buf = [line]
            size = add
        else:
            buf.append(line)
            size += add
    if buf:
        chunks.append("\n".join(buf))
    return chunks
    
async def safe_defer(interaction: discord.Interaction, *, ephemeral: bool = True) -> bool:
    """Defer safely. Returns False if the interaction is no longer valid."""
    # Autocomplete interactions must NOT be deferred
    if interaction.type == discord.InteractionType.autocomplete:
        return True

    try:
        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=ephemeral)
        return True
    except DiscordNotFound:
        # 10062 Unknown interaction
        return False
    except discord.HTTPException as e:
        # 40060 already acknowledged (not fatal)
        if getattr(e, "code", None) == 40060:
            return True
        raise

# ==============================
# ONBOARDING (patched: /tone + /shuffle language, no /tone or /reset)
# ==============================
def _chunk_text(text: str, max_len: int = 1900) -> List[str]:
    """
    Chunk a long text into multiple messages safely under Discord 2000-char limit.
    Tries to split on double newlines, then single newlines, then hard-splits.
    """
    text = (text or "").strip()
    if len(text) <= max_len:
        return [text] if text else []

    parts: List[str] = []
    buf = ""

    # Prefer paragraph breaks
    for para in text.split("\n\n"):
        candidate = (buf + ("\n\n" if buf else "") + para).strip()
        if len(candidate) <= max_len:
            buf = candidate
            continue

        if buf:
            parts.append(buf)
            buf = ""

        # If a single paragraph is still too big, split by lines
        if len(para) > max_len:
            line_buf = ""
            for line in para.split("\n"):
                cand2 = (line_buf + ("\n" if line_buf else "") + line).strip()
                if len(cand2) <= max_len:
                    line_buf = cand2
                else:
                    if line_buf:
                        parts.append(line_buf)
                        line_buf = ""
                    # hard split line if needed
                    while len(line) > max_len:
                        parts.append(line[:max_len])
                        line = line[max_len:]
                    if line:
                        line_buf = line
            if line_buf:
                parts.append(line_buf)
        else:
            parts.append(para)

    if buf:
        parts.append(buf)

    return [p for p in parts if p.strip()]


def build_onboarding_messages(guild: discord.Guild) -> List[str]:
    # One message, Chronobot-style. Keep it under 2000 chars.
    msg = (
        f"🔮 **Arcanara has crossed the threshold**\n"
        f"I’ve anchored to **{guild.name}**.\n"
        "I don’t read messages. I don’t rummage through DMs.\n"
        "I *do* translate symbols into clean choices — with a little shimmer on the edges.\n\n"

        "🧭 **Quick Setup**\n"
        "1) **/tone** — choose how I speak (full, direct, poetic, shadow, love, work, money)\n"
        "2) **/intent** — set your intention (your focus / question)\n"
        "3) **/settings** — images on/off + history opt-in (off by default)\n"
        "4) **/shuffle** — reset intention + tone (fresh slate)\n\n"

        "✨ **Start Here**\n"
        "• **/cardoftheday** — one clear message for today\n"
        "• **/read** — Situation • Obstacle • Guidance (you provide an intention)\n"
        "• **/threecard** — Past • Present • Future\n"
        "• **/celtic** — full 10-card Celtic Cross\n"
        "• **/clarify** — one extra card for your current intention\n"
        "• **/meaning** — look up any card (upright + reversed)\n"
        "• **/history** — reflect on past readings\n"
        "• **/mystery** → **/reveal** — dramatic pause included\n\n"

        "🔒 **Privacy**\n"
        "History is **opt-in** only. Use **/forgetme** to delete stored data.\n\n"

        "🛡️ **Permissions (so I can speak)**\n"
        "• **Send Messages** (required)\n"
        "• **Attach Files** (recommended for card images)\n"
        "• **Embed Links** (optional)\n\n"

        "Need the full guided help at any time? Use **/insight**.\n"
        "Admins: **/resendwelcome** re-sends this welcome."
    )

    return [msg]



async def find_bot_inviter(guild: discord.Guild, bot_user: discord.ClientUser) -> Optional[discord.User]:
    """Attempts to find who added the bot by checking the guild audit log. Requires 'View Audit Log' permission."""
    try:
        async for entry in guild.audit_logs(limit=10, action=discord.AuditLogAction.bot_add):
            target = getattr(entry, "target", None)
            if target and target.id == bot_user.id:
                return entry.user
    except (discord.Forbidden, discord.HTTPException):
        return None
    return None


async def send_onboarding_message(guild: discord.Guild):
    messages = build_onboarding_messages(guild)

    # 1) Prefer inviter (audit log), else owner
    recipient = await find_bot_inviter(guild, bot.user)
    if recipient is None:
        recipient = guild.owner

    # Try DM recipient
    if recipient:
        try:
            for msg in messages:
                await recipient.send(content=msg)
            return
        except (discord.Forbidden, discord.HTTPException):
            pass

    # Fallback: post in system channel / first available text channel
    me = guild.me
    channel = guild.system_channel
    if channel and me and channel.permissions_for(me).send_messages:
        try:
            for msg in messages:
                await channel.send(content=msg)
            return
        except discord.HTTPException:
            pass

    for ch in guild.text_channels:
        if me and ch.permissions_for(me).send_messages:
            try:
                for msg in messages:
                    await ch.send(content=msg)
                return
            except discord.HTTPException:
                continue

@bot.event
async def on_guild_join(guild: discord.Guild):
    try:
        await send_onboarding_message(guild)
        print(f"✅ Onboarding sent for guild: {guild.name} ({guild.id})")
    except Exception as e:
        print(f"⚠️ Onboarding failed for guild {guild.id}: {type(e).__name__}: {e}")


# ==============================
# IN-CHARACTER RESPONSES
# ==============================
in_character_lines = {
    "shuffle": [
        "The deck hums with fresh energy once more.",
        "All is reset. The cards breathe again.",
        "Order dissolves into possibility — the deck is ready.",
    ],
    "daily": [
        "Here is the energy that threads through your day...",
        "This card has stepped forward to guide you.",
        "Its message hums softly — take it with you into the light.",
    ],
    "spread": [
        "The weave of time unfolds — past, present, and future speak.",
        "Let us see how the threads intertwine for your path.",
        "Each card now reveals its whisper in the larger story.",
    ],
    "deep": [
        "This spread carries depth — breathe as you read its symbols.",
        "A more ancient current flows beneath these cards.",
        "The deck speaks slowly now; listen beyond the words.",
    ],
    "general": [
        "The veil lifts and a message takes shape...",
        "Listen closely — the cards are patient but precise.",
        "A single spark of insight is about to emerge...",
    ],
}


# ==============================
# EPHEMERAL SENDER (in-character, attachment-safe, ack-safe)
# ==============================
def _prepend_in_character(embed: discord.Embed, mood: str) -> discord.Embed:
    line = random.choice(in_character_lines.get(mood, in_character_lines["general"]))
    if embed.description:
        embed.description = f"*{line}*\n\n{embed.description}"
    else:
        embed.description = f"*{line}*"
    return embed


async def send_ephemeral(
    interaction: discord.Interaction,
    *,
    embed: Optional[discord.Embed] = None,
    embeds: Optional[List[discord.Embed]] = None,
    content: Optional[str] = None,
    mood: str = "general",
    file_obj: Optional[discord.File] = None,
):
    def _send_kwargs(**kw):
        # Only include file if it's real (discord.py chokes on file=None)
        if file_obj is not None:
            kw["file"] = file_obj
        return kw

    try:
        # If already deferred/answered, use followup
        if not interaction.response.is_done():
            send_fn = interaction.response.send_message
        else:
            send_fn = interaction.followup.send

        if embed is not None:
            embed = _prepend_in_character(embed, mood)
            await send_fn(**_send_kwargs(content=content, embed=embed, ephemeral=True))
            return

        if embeds:
            embeds = list(embeds)
            embeds[0] = _prepend_in_character(embeds[0], mood)
            await send_fn(**_send_kwargs(content=content, embeds=embeds, ephemeral=True))
            return

        await send_fn(**_send_kwargs(content=content or "—", ephemeral=True))
        
    except DiscordNotFound:
        # Interaction expired / unknown; nothing we can do
        return

    except discord.HTTPException as e:
        # If Discord says “already acknowledged”, fall back to followup
        if getattr(e, "code", None) == 40060:
            try:
                if embed is not None:
                    embed = _prepend_in_character(embed, mood)
                    await interaction.followup.send(**_send_kwargs(content=content, embed=embed, ephemeral=True))
                    return
                if embeds:
                    embeds = list(embeds)
                    embeds[0] = _prepend_in_character(embeds[0], mood)
                    await interaction.followup.send(**_send_kwargs(content=content, embeds=embeds, ephemeral=True))
                    return
                await interaction.followup.send(**_send_kwargs(content=content or "—", ephemeral=True))
                return
            except Exception:
                pass
        raise


# ==============================
# EVENTS
# ==============================
# ==============================
# TOP.GG STATS POSTING
# ==============================
@tasks.loop(minutes=30)
async def post_topgg_stats():
    """Post server and user count to top.gg every 30 minutes"""
    if not TOPGG_AVAILABLE or not TOPGG_TOKEN:
        return
    
    try:
        topgg_client = topgg.DBLClient(bot, TOPGG_TOKEN, autopost=False)
        await topgg_client.post_guild_count(
            guild_count=len(bot.guilds),
            shard_count=bot.shard_count or 1
        )
        print(f"✅ Posted to top.gg: {len(bot.guilds)} servers")
    except Exception as e:
        print(f"⚠️ top.gg post failed: {type(e).__name__}: {e}")

@bot.event
async def on_ready():
    global _DB_READY
    if not _DB_READY:
        try:
            ensure_tables()
            _DB_READY = True
            print("✅ DB ready.")
        except Exception as e:
            print(f"❌ DB init failed: {type(e).__name__}: {e}")
            return

    try:
        await bot.tree.sync()
        print("✅ Slash commands synced.")
    except Exception as e:
        print(f"⚠️ Slash sync failed: {type(e).__name__}: {e}")

    # Start top.gg stats posting
    if TOPGG_AVAILABLE and TOPGG_TOKEN and not post_topgg_stats.is_running():
        post_topgg_stats.start()
        print("✅ top.gg stats task started.")
        # Post immediately on startup
        try:
            topgg_client = topgg.DBLClient(bot, TOPGG_TOKEN, autopost=False)
            await topgg_client.post_guild_count(
                guild_count=len(bot.guilds),
                shard_count=bot.shard_count or 1
            )
            print(f"✅ Initial post to top.gg: {len(bot.guilds)} servers")
        except Exception as e:
            print(f"⚠️ top.gg initial post failed: {type(e).__name__}: {e}")
    elif not TOPGG_AVAILABLE:
        print("⚠️ topgg library not installed - stats will not be posted.")
    elif not TOPGG_TOKEN:
        print("⚠️ TOPGG_TOKEN not set - stats will not be posted.")

    print(f"{E['crystal']} Arcanara is awake and shimmering as {bot.user}")


@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    # Never try to respond to autocomplete interactions
    if interaction.type == discord.InteractionType.autocomplete:
        return

    orig = getattr(error, "original", error)
    print(f"⚠️ Slash command error: {type(error).__name__}: {error}")
    print(f"⚠️ Original: {type(orig).__name__}: {orig}")
    traceback.print_exception(type(orig), orig, orig.__traceback__)

    try:
        await send_ephemeral(
            interaction,
            content="⚠️ A thread snagged in the weave. Try again in a moment.",
            mood="general",
        )
    except Exception as e:
        print(f"⚠️ Failed to send error message: {type(e).__name__}: {e}")


# ==============================
# SLASH COMMANDS (EPHEMERAL)
# ==============================
@bot.tree.command(name="shuffle", description="Cleanse the deck and reset your intention + tone.")
async def shuffle_slash(interaction: discord.Interaction):
    if not await safe_defer(interaction, ephemeral=True):
        return
           
    # Reset user state
    user_intentions.pop(interaction.user.id, None)
    MYSTERY_STATE.pop(interaction.user.id, None)
    reset_user_tone(interaction.user.id)  # resets stored tone/mode to default
    random.shuffle(tarot_cards)

    embed = discord.Embed(
        title=f"{E['shuffle']} Cleanse Complete {E['shuffle']}",
        description=(
            "The deck is cleared.\n\n"
            f"• **Intention**: reset\n"
            f"• **Tone**: reset to **{DEFAULT_TONE}**\n\n"
            "Set a fresh intention with `/intent`, then draw with `/cardoftheday` or `/read`."
        ),
        color=0x9370DB
    )

    await send_ephemeral(interaction, embeds=[embed], mood="shuffle")

@bot.tree.command(name="history", description="View your recent Arcanara readings (opt-in only).")
@app_commands.describe(limit="How many entries to show (max 20)")
async def history_slash(interaction: discord.Interaction, limit: Optional[int] = 10):
    if not await safe_defer(interaction, ephemeral=True):
        return

    limit = 10 if limit is None else max(1, min(int(limit), 20))

    settings = get_user_settings(interaction.user.id)
    if not settings.get("history_opt_in", False):
        await send_ephemeral(
            interaction,
            content=(
                f"{E['warn']} Your history is currently **off**.\n\n"
                "Turn it on with `/settings history:on` if you want Arcanara to remember your readings.\n"
                "You can delete it any time with `/forgetme`."
            ),
            mood="general",
        )
        return

    rows = fetch_history(interaction.user.id, limit=limit)
    if not rows:
        await send_ephemeral(
            interaction,
            content="No saved readings yet. Once history is on, I’ll remember your pulls here.",
            mood="general",
        )
        return

    lines: List[str] = []
    for r in rows:
        cmd = r.get("command", "—")
        tone = r.get("tone", "full")
        payload = r.get("payload", {}) or {}
        created_at = r.get("created_at")

        # Discord relative time formatting: <t:UNIX:R>
        stamp = ""
        if hasattr(created_at, "timestamp"):
            stamp = f"<t:{int(created_at.timestamp())}:R>"

        summary = summarize_history_row(cmd, payload)
        lines.append(f"• {stamp} /{cmd} ({tone}) — {summary}")

    text = _clip("\n".join(lines), max_len=3800)

    embed = discord.Embed(
        title=f"{E['book']} Your Recent Readings",
        description=text,
        color=0x6A5ACD,
    )
    embed.set_footer(text="History is opt-in • Use /forgetme to delete stored data.")

    await send_ephemeral(interaction, embed=embed, mood="general")

@bot.tree.command(name="cardoftheday", description="Reveal the card that guides your day.")
async def cardoftheday_slash(interaction: discord.Interaction):
    if not await safe_defer(interaction, ephemeral=True):
        return

    day = _today_local_date()
    row = get_daily_card_row(interaction.user.id, day)

    if row:
        orientation = row["orientation"]
        card = find_card_by_name(row["card_name"])
        if card is None:
            # If your deck JSON changed and the name doesn't match anymore, fall back gracefully
            card, orientation = draw_card()
            set_daily_card_row(interaction.user.id, day, card["name"], orientation)
    else:
        card, orientation = draw_card()
        set_daily_card_row(interaction.user.id, day, card["name"], orientation)

    tone = get_effective_tone(interaction.user.id)
    meaning = render_card_text(card, orientation, tone)

    settings = get_user_settings(interaction.user.id)

    is_reversed = (orientation == "Reversed")
    file_obj, attach_url = None, None

    if settings.get("images_enabled", True):
        try:
            file_obj, attach_url = make_image_attachment(card["name"], is_reversed)
            if not attach_url and file_obj is not None:
                attach_url = f"attachment://{file_obj.filename}"
        except Exception as e:
            print(f"⚠️ make_image_attachment failed: {type(e).__name__}: {e}")
            file_obj, attach_url = None, None

    tone_emoji = E["sun"] if orientation == "Upright" else E["moon"]
    intent_text = user_intentions.get(interaction.user.id)

    desc = f"**{card['name']} ({orientation} {tone_emoji}) • {tone_label(tone)}**\n\n{meaning}"
    if intent_text:
        desc += f"\n\n{E['light']} **Intention:** *{intent_text}*"

    log_history_if_opted_in(
        interaction.user.id,
        command="cardoftheday",
        tone=tone,
        payload={
            "card": card["name"],
            "orientation": orientation,
            "intention": intent_text,
            "images_enabled": bool(settings.get("images_enabled", True)),
            "day": str(day),
        },
        settings=settings,
    )

    embed = discord.Embed(
        title=f"{E['crystal']} Card of the Day",
        description=desc,
        color=suit_color(card["suit"]),
    )

    if attach_url:
        embed.set_image(url=attach_url)

    await send_ephemeral(interaction, embed=embed, mood="daily", file_obj=file_obj)


@bot.tree.command(name="read", description="Three-card reading: Situation • Obstacle • Guidance.")
@app_commands.describe(intention="Your question or intention (example: my career path)")
async def read_slash(interaction: discord.Interaction, intention: str):
    if not await safe_defer(interaction, ephemeral=True):
        return

    user_intentions[interaction.user.id] = intention
    tone = get_effective_tone(interaction.user.id)

    cards = draw_unique_cards(3)
    positions = ["Situation", "Obstacle", "Guidance"]

    log_history_if_opted_in(
        interaction.user.id,
        command="read",
        tone=tone,
        payload={
            "intention": intention,
            "spread": "situation_obstacle_guidance",
            "cards": [
                {"position": pos, "name": card["name"], "orientation": orientation}
                for pos, (card, orientation) in zip(positions, cards)
            ],
        },
    )

    embed = discord.Embed(
        title=f"{E['crystal']} Intuitive Reading {E['crystal']}",
        description=f"{E['light']} **Intention:** *{intention}*\n\n**How I’ll read this:** {tone_label(tone)}",
        color=0x9370DB,
    )

    pretty_positions = [f"Situation {E['sun']}", f"Obstacle {E['sword']}", f"Guidance {E['star']}"]
    for pos, (card, orientation) in zip(pretty_positions, cards):
        meaning = render_card_text(card, orientation, tone)
        embed.add_field(
            name=f"{pos}: {card['name']} ({orientation})",
            value=meaning if len(meaning) < 1000 else meaning[:997] + "...",
            inline=False,
        )

    embed.set_footer(text=f"{E['spark']} Let these cards guide your awareness, not dictate your choices.")
    await send_ephemeral(interaction, embed=embed, mood="spread")


@bot.tree.command(name="threecard", description="Past • Present • Future spread.")
async def threecard_slash(interaction: discord.Interaction):
    if not await safe_defer(interaction, ephemeral=True):
        return

    positions = ["Past", "Present", "Future"]
    cards = draw_unique_cards(3)

    tone = get_effective_tone(interaction.user.id)
    intent_text = user_intentions.get(interaction.user.id)

    log_history_if_opted_in(
        interaction.user.id,
        command="threecard",
        tone=tone,
        payload={
            "intention": intent_text,
            "spread": "past_present_future",
            "cards": [
                {"position": pos, "name": card["name"], "orientation": orientation}
                for pos, (card, orientation) in zip(positions, cards)
            ],
        },
    )

    desc = "Past • Present • Future"
    if intent_text:
        desc += f"\n\n{E['light']} **Intention:** *{intent_text}*"
    desc += f"\n\n**How I’ll read this:** {tone_label(tone)}"

    embed = discord.Embed(
        title=f"{E['crystal']} Three-Card Spread",
        description=desc,
        color=0xA020F0,
    )

    pretty_positions = [f"Past {E['clock']}", f"Present {E['moon']}", f"Future {E['star']}"]
    for pos, (card, orientation) in zip(pretty_positions, cards):
        meaning = render_card_text(card, orientation, tone)
        embed.add_field(
            name=f"{pos}: {card['name']} ({orientation})",
            value=meaning if len(meaning) < 1000 else meaning[:997] + "...",
            inline=False,
        )

    await send_ephemeral(interaction, embed=embed, mood="spread")

@bot.tree.command(name="celtic", description="Full 10-card Celtic Cross spread.")
@app_commands.checks.cooldown(1, 120.0)
async def celtic_slash(interaction: discord.Interaction):
    if not await safe_defer(interaction, ephemeral=True):
        return

    positions = [
        "Present Situation", "Challenge", "Root Cause", "Past", "Conscious Goal",
        "Near Future", "Self", "External Influence", "Hopes & Fears", "Outcome",
    ]
    cards = draw_unique_cards(10)
    tone = get_effective_tone(interaction.user.id)

    log_history_if_opted_in(
        interaction.user.id,
        command="celtic",
        tone=tone,
        payload={
            "spread": "celtic_cross",
            "cards": [
                {"position": pos, "name": card["name"], "orientation": orientation}
                for pos, (card, orientation) in zip(positions, cards)
            ],
        },
    )

    embeds_to_send: List[discord.Embed] = []
    embed = discord.Embed(
        title=f"{E['crystal']} Celtic Cross Spread {E['crystal']}",
        description=f"A deep, archetypal exploration of your path.\n\n**How I’ll read this:** {tone_label(tone)}",
        color=0xA020F0,
    )
    total_length = len(embed.title) + len(embed.description)

    pretty_positions = [
        "1️⃣ Present Situation", "2️⃣ Challenge", "3️⃣ Root Cause", "4️⃣ Past", "5️⃣ Conscious Goal",
        "6️⃣ Near Future", "7️⃣ Self", "8️⃣ External Influence", "9️⃣ Hopes & Fears", "🔟 Outcome",
    ]

    for pos, (card, orientation) in zip(pretty_positions, cards):
        meaning = render_card_text(card, orientation, tone)
        field_name = f"{pos}: {card['name']} ({orientation})"
        field_value = meaning if len(meaning) < 1000 else meaning[:997] + "..."
        field_length = len(field_name) + len(field_value)

        if total_length + field_length > 5800:
            embeds_to_send.append(embed)
            embed = discord.Embed(
                title=f"{E['crystal']} Celtic Cross (Continued)",
                description=f"**How I’ll read this:** {tone_label(tone)}",
                color=0xA020F0,
            )
            total_length = len(embed.title) + len(embed.description)

        embed.add_field(name=field_name, value=field_value, inline=False)
        total_length += field_length

    embeds_to_send.append(embed)

    # First embed via send_ephemeral
    await send_ephemeral(interaction, embed=embeds_to_send[0], mood="deep")

    # Remaining embeds must be followups (interaction already acknowledged)
    for e in embeds_to_send[1:]:
        await interaction.followup.send(embeds=[e], ephemeral=True)

@bot.tree.command(name="tone", description="Choose Arcanara’s reading tone (your default lens).")
@app_commands.choices(
    tone=[
        app_commands.Choice(name="full", value="full"),
        app_commands.Choice(name="direct", value="direct"),
        app_commands.Choice(name="shadow", value="shadow"),
        app_commands.Choice(name="poetic", value="poetic"),
        app_commands.Choice(name="quick", value="quick"),
        app_commands.Choice(name="love", value="love"),
        app_commands.Choice(name="work", value="work"),
        app_commands.Choice(name="money", value="money"),
    ]
)
async def tone_slash(interaction: discord.Interaction, tone: app_commands.Choice[str]):
    if not await safe_defer(interaction, ephemeral=True):
        return
        
    chosen = set_user_tone(interaction.user.id, tone.value)
    await send_ephemeral(
        interaction,
        content=f"✅ Tone set to **{chosen}**.\n\nTip: Pair it with an intention using `/intent`.",
        mood="general",
    )


@bot.tree.command(name="resendwelcome", description="Resend Arcanara’s onboarding message (admin).")
@app_commands.checks.has_permissions(manage_guild=True)
@app_commands.guild_only()
@app_commands.choices(
    where=[
        app_commands.Choice(name="dm (owner/inviter)", value="dm"),
        app_commands.Choice(name="post here", value="here"),
    ]
)

async def resendwelcome_slash(interaction: discord.Interaction, where: app_commands.Choice[str]):
    if not await safe_defer(interaction, ephemeral=True):
        return

    guild = interaction.guild
    if guild is None:
        await interaction.followup.send("⚠️ This command can only be used in a server.", ephemeral=True)
        return

    messages = build_onboarding_messages(guild)

    try:
        if where.value == "here":
            ch = interaction.channel
            if isinstance(ch, (discord.TextChannel, discord.Thread)):
                for msg in messages:
                    await ch.send(content=msg)
                await interaction.followup.send("✅ Welcome message posted here.", ephemeral=True)
            else:
                await interaction.followup.send("⚠️ I can’t post in this channel type.", ephemeral=True)
        else:
            await send_onboarding_message(guild)
            await interaction.followup.send("✅ Welcome message sent (DM owner/inviter, with channel fallback).", ephemeral=True)


    except Exception as e:
        print(f"⚠️ resendwelcome failed: {type(e).__name__}: {e}")
        await interaction.followup.send(
            "⚠️ A thread snagged while sending the welcome. Check permissions/logs.",
            ephemeral=True,
        )

@bot.tree.command(name="meaning", description="Show upright and reversed meanings for a card (with card photo).")
@app_commands.describe(card="Card name (example: The Lovers)")
@app_commands.autocomplete(card=card_name_autocomplete)
async def meaning_slash(interaction: discord.Interaction, card: str):
    if not await safe_defer(interaction, ephemeral=True):
        return

    norm_query = normalize_card_name(card)

    matches = [
        c for c in tarot_cards
        if normalize_card_name(c.get("name", "")) == norm_query
        or norm_query in normalize_card_name(c.get("name", ""))
    ]

    if not matches:
        await send_ephemeral(
            interaction,
            content=f"{E['warn']} I searched the deck but found no card named **{card}**.",
            mood="general",
        )
        return

    chosen = matches[0]
    chosen_name = chosen.get("name", "").strip()

    tone = get_effective_tone(interaction.user.id)
    settings = get_user_settings(interaction.user.id)

    suit = chosen.get("suit") or "Major Arcana"
    color = suit_color(suit)

    # Log lookup (only if opted in)
    log_history_if_opted_in(
        interaction.user.id,
        command="meaning",
        tone=tone,
        payload={"query": card, "matched": chosen_name, "shown": ["Upright", "Reversed"]},
        settings=settings,
    )

    # Build ONE embed (single embed + image, like cardoftheday)
    embed = discord.Embed(
        title=f"{E['book']} {chosen_name or '(unknown)'} • {tone_label(tone)}",
        description="",
        color=color,
    )

    upright_text = clip_field(render_card_text(chosen, "Upright", tone), 1024)
    reversed_text = clip_field(render_card_text(chosen, "Reversed", tone), 1024)

    embed.add_field(name=f"Upright {E['sun']} • {tone}", value=upright_text or "—", inline=False)
    embed.add_field(name=f"Reversed {E['moon']} • {tone}", value=reversed_text or "—", inline=False)
    embed.set_footer(text=f"{E['light']} Interpreting symbols through Arcanara • Tarot Bot")

    # --- Image: same attachment style as cardoftheday ---
    file_obj, attach_url = make_image_attachment(chosen_name, False)

    if file_obj:
        embed.set_image(url=f"attachment://{file_obj.filename}")
        await send_ephemeral(interaction, embed=embed, mood="general", file_obj=file_obj)
    else:
        # No image found; still send the meaning
        await send_ephemeral(interaction, embed=embed, mood="general")


@bot.tree.command(name="clarify", description="Draw a clarifier card for your current intention.")
async def clarify_slash(interaction: discord.Interaction):
    if not await safe_defer(interaction, ephemeral=True):
        return

    card, orientation = draw_card()
    tone_emoji = E["sun"] if orientation == "Upright" else E["moon"]
    intent_text = user_intentions.get(interaction.user.id)

    tone = get_effective_tone(interaction.user.id)
    meaning = render_card_text(card, orientation, tone)

    log_history_if_opted_in(
        interaction.user.id,
        command="clarify",
        tone=tone,
        payload={
            "intention": intent_text,
            "card": {"name": card["name"], "orientation": orientation},
        },
    )

    desc = f"**{card['name']} ({orientation} {tone_emoji}) • {tone_label(tone)}**\n\n{meaning}"
    if intent_text:
        desc += f"\n\n{E['light']} **Clarifying Intention:** *{intent_text}*"

    embed = discord.Embed(
        title=f"{E['light']} Clarifier Card {E['light']}",
        description=desc,
        color=suit_color(card["suit"]),
    )
    embed.set_footer(text=f"{E['spark']} A clarifier shines a smaller light within your larger spread.")
    await send_ephemeral(interaction, embed=embed, mood="general")

@bot.tree.command(name="intent", description="Set (or view) your current intention.")
@app_commands.describe(intention="Leave blank to view your current intention.")
async def intent_slash(interaction: discord.Interaction, intention: Optional[str] = None):
    if not await safe_defer(interaction, ephemeral=True):
        return

    if not intention:
        current = user_intentions.get(interaction.user.id)
        if current:
            await send_ephemeral(interaction, content=f"{E['light']} Your current intention is: *{current}*")
        else:
            await send_ephemeral(interaction, content=f"{E['warn']} You haven’t set an intention yet. Use `/intent intention: ...`")
        return

    user_intentions[interaction.user.id] = intention
    await send_ephemeral(interaction, content=f"{E['spark']} Intention set to: *{intention}*")

@bot.tree.command(name="mystery", description="Pull a mystery card (image only). Use /reveal to see the meaning.")
async def mystery_slash(interaction: discord.Interaction):
    if not await safe_defer(interaction, ephemeral=True):
        return

    card = random.choice(tarot_cards)
    is_reversed = random.random() < 0.5

    MYSTERY_STATE[interaction.user.id] = {
        "name": card["name"],
        "is_reversed": is_reversed,
        "ts": time.time(),
    }

    settings = get_user_settings(interaction.user.id)

    embed_top = discord.Embed(
        title=f"{E['crystal']} {card['name']}" + (" — Reversed" if is_reversed else ""),
        description="Type **/reveal** to see the meaning.",
        color=suit_color(card["suit"]),
    )

    file_obj, attach_url = None, None

    if settings.get("images_enabled", True):
        try:
            file_obj, attach_url = make_image_attachment(card["name"], is_reversed)

            # If make_image_attachment returns a File but no URL, use attachment://
            if not attach_url and file_obj is not None:
                attach_url = f"attachment://{file_obj.filename}"

            if attach_url:
                embed_top.set_image(url=attach_url)
            else:
                # No attachment produced (but command should still succeed)
                embed_top.description = (
                    "I drew a mystery card, but the image didn’t manifest.\n"
                    "Type **/reveal** to see the meaning."
                )

        except Exception as e:
            print(f"⚠️ make_image_attachment failed in /mystery: {type(e).__name__}: {e}")
            file_obj, attach_url = None, None
            embed_top.description = (
                "I drew a mystery card, but the image thread snapped.\n"
                "Type **/reveal** to see the meaning."
            )

    else:
        embed_top.description = (
            "Images are currently **off**.\n"
            "Turn them on with `/settings images:on`, or type **/reveal** to see the meaning."
        )

    await send_ephemeral(interaction, embed=embed_top, mood="general", file_obj=file_obj)



@bot.tree.command(name="reveal", description="Reveal the meaning of your last mystery card.")
async def reveal_slash(interaction: discord.Interaction):
    if not await safe_defer(interaction, ephemeral=True):
        return

    state = MYSTERY_STATE.get(interaction.user.id)
    if not state:
        # IMPORTANT FIX: after defer, use followup (send_ephemeral will do that)
        await send_ephemeral(
            interaction,
            content=f"{E['warn']} No mystery card on file. Use **/mystery** first.",
            mood="general",
        )
        return

    try:
        name = state["name"]
        is_reversed = state["is_reversed"]

        card = next((c for c in tarot_cards if c["name"] == name), None)
        if not card:
            await send_ephemeral(
                interaction,
                content=f"{E['warn']} I lost track of that card. Try **/mystery** again.",
                mood="general",
            )
            return

        tone = get_effective_tone(interaction.user.id)
        orientation = "Reversed" if is_reversed else "Upright"
        meaning = render_card_text(card, orientation, tone)

        settings = get_user_settings(interaction.user.id)
        log_history_if_opted_in(
            interaction.user.id,
            command="reveal",
            tone=tone,
            payload={
                "source": "mystery",
                "card": {"name": card["name"], "orientation": orientation},
            },
            settings=settings,
        )

        embed = discord.Embed(
            title=f"{E['book']} Reveal: {card['name']} ({orientation}) • {tone_label(tone)}",
            description=meaning,
            color=suit_color(card["suit"]),
        )
        embed.set_footer(text=f"{E['light']} Interpreting symbols through Arcanara • Tarot Bot")

        await send_ephemeral(interaction, embed=embed, mood="general")

    finally:
        MYSTERY_STATE.pop(interaction.user.id, None)


@bot.tree.command(name="insight", description="A guided intro to Arcanara (and a full list of commands).")
async def insight_slash(interaction: discord.Interaction):
    if not await safe_defer(interaction, ephemeral=True):
        return
    user_id_str = str(interaction.user.id)
    user_name = interaction.user.display_name

    first_time = user_id_str not in known_seekers
    if first_time:
        known_seekers[user_id_str] = {"name": user_name}
        save_known_seekers(known_seekers)

    current_tone = get_effective_tone(interaction.user.id)
    current_intent = user_intentions.get(interaction.user.id, None)

    greetings_first = [
        f"Come closer, {user_name} — let’s see what wants to be known.",
        f"{user_name}… I felt you arrive before you spoke.",
        f"Alright, {user_name}. No theatrics — just clarity.",
        f"Welcome, {user_name}. The deck likes honest questions.",
    ]
    greetings_returning = [
        f"Back again, {user_name}? Good. The story didn’t end without you.",
        f"There you are, {user_name}. Same you — new chapter.",
        f"Welcome back, {user_name}. Let’s pick up the thread.",
        f"{user_name}… the deck remembers your rhythm.",
    ]
    opener = random.choice(greetings_first if first_time else greetings_returning)

    intent_line = f"**Your intention:** *{current_intent}*" if current_intent else "**Your intention:** *unspoken… for now.*"
    tone_line = f"**How I’ll speak:** {tone_label(current_tone)}"

    guided = (
        f"{intent_line}\n"
        f"{tone_line}\n\n"
        "Here’s how we do this:\n"
        "• Want a single clean message for today? Try **/cardoftheday**.\n"
        "• Got a situation with teeth? Use **/read** and give me your question.\n"
        "• Want the timeline vibe? **/threecard** (past • present • future).\n"
        "• Need the *deep* dive? **/celtic** — it pulls the whole pattern.\n"
        "• Not sure what a card means? Ask **/meaning**.\n"
        "• Feeling uncertain? **/clarify** will pull one more lantern from the dark.\n\n"
        "And if you’re in the mood for a little mischief:\n"
        "• **/mystery** (image only) … then **/reveal** when you’re ready.\n\n"
        "If you want to wipe the slate clean: **/shuffle** resets intention + tone."
    )

    cmds = [c for c in bot.tree.get_commands() if isinstance(c, app_commands.Command)]
    cmds = sorted(cmds, key=lambda c: c.name)

    lines = []
    for c in cmds:
        desc = (c.description or "").strip()
        lines.append(f"• `/{c.name}` — {desc}" if desc else f"• `/{c.name}`")

    chunks: List[str] = []
    buf: List[str] = []
    size = 0
    for line in lines:
        if size + len(line) + 1 > 900:
            chunks.append("\n".join(buf))
            buf = [line]
            size = len(line) + 1
        else:
            buf.append(line)
            size += len(line) + 1
    if buf:
        chunks.append("\n".join(buf))

    embed = discord.Embed(
        title=f"{E['crystal']} Arcanara",
        description=f"*{opener}*\n\n{guided}",
        color=0xB28DFF,
    )

    embed.add_field(name="What I can do for you", value=chunks[0] if chunks else "—", inline=False)
    for i, part in enumerate(chunks[1:], start=2):
        embed.add_field(name=f"What I can do for you (cont. {i})", value=part, inline=False)

    embed.set_footer(text="A tarot reading is a mirror, not a cage. You steer.")
    await send_ephemeral(interaction, embed=embed, mood="general")


@bot.tree.command(name="privacy", description="What Arcanara stores and how to delete it.")
async def privacy_slash(interaction: discord.Interaction):
    embed = discord.Embed(
        title="🔒 Arcanara Privacy",
        description=(
            "**Stored data (optional / minimal):**\n"
            "• Your chosen `/tone`\n"
            "• Your `/settings` (images on/off, history opt-in)\n"
            "• Reading history **only if you opt in**\n\n"
            "**Delete everything:** use `/forgetme`.\n"
            "Arcanara does not read server messages or DMs."
        ),
        color=0x6A5ACD,
    )
    await send_ephemeral(interaction, embed=embed, mood="general")


@bot.tree.command(name="forgetme", description="Delete your stored Arcanara data.")
async def forgetme_slash(interaction: discord.Interaction):
    if not await safe_defer(interaction, ephemeral=True):
        return
        
    uid = interaction.user.id

    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM tarot_user_prefs WHERE user_id=%s", (uid,))
            cur.execute("DELETE FROM tarot_user_settings WHERE user_id=%s", (uid,))
            cur.execute("DELETE FROM tarot_reading_history WHERE user_id=%s", (uid,))
        conn.commit()

    user_intentions.pop(uid, None)
    MYSTERY_STATE.pop(uid, None)

    await send_ephemeral(interaction, content="✅ Your thread has been cut clean. Stored data deleted.", mood="general")

@bot.tree.command(name="settings", description="Control history + images for your readings.")
@app_commands.choices(
    history=[app_commands.Choice(name="on", value="on"), app_commands.Choice(name="off", value="off")],
    images=[app_commands.Choice(name="on", value="on"), app_commands.Choice(name="off", value="off")],
)
async def settings_slash(
    interaction: discord.Interaction,
    history: Optional[app_commands.Choice[str]] = None,
    images: Optional[app_commands.Choice[str]] = None,
):
    if not await safe_defer(interaction, ephemeral=True):
        return

    h = None if history is None else (history.value == "on")
    i = None if images is None else (images.value == "on")

    set_user_settings(interaction.user.id, history_opt_in=h, images_enabled=i)
    s = get_user_settings(interaction.user.id)

    await send_ephemeral(
        interaction,
        content=(
            "✅ Settings saved.\n"
            f"• History: **{'on' if s['history_opt_in'] else 'off'}**\n"
            f"• Images: **{'on' if s['images_enabled'] else 'off'}**"
        ),
        mood="general",
    )


# ==============================
# RUN BOT
# ==============================
bot.run(BOT_TOKEN)
