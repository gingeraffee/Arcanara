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
from datetime import datetime, date, timedelta
import traceback
from zoneinfo import ZoneInfo
from psycopg.types.json import Json
from psycopg.rows import dict_row
from typing import Dict, Any, List, Optional
import asyncio  # For reading ceremonies
from card_images import make_image_attachment  # uses assets/cards/rws_stx/ etc.
import aiohttp  # For top.gg API calls

# Top.gg will use direct HTTP API instead of a library
TOPGG_AVAILABLE = True

print("✅ Arcanara boot: VERSION 2025-01-15-TopGG-HTTP")

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
    """Premium formatted card text - clean, readable, and beautiful"""
    tone = normalize_tone(tone)
    spec = TONE_SPECS.get(tone, TONE_SPECS[DEFAULT_TONE])

    is_rev = orientation.strip().lower().startswith("r")
    okey = "reversed" if is_rev else "upright"

    # Normalize orientation data to a dict
    odata_raw = card.get(okey, {})
    if isinstance(odata_raw, dict):
        odata = odata_raw
    elif isinstance(odata_raw, str):
        odata = {"meaning": odata_raw}
    elif isinstance(odata_raw, list):
        odata = {"meaning": "\n".join(str(x) for x in odata_raw if str(x).strip())}
    else:
        odata = {}

    # Meaning must be a string
    meaning = (odata.get("meaning") or "—")
    if not isinstance(meaning, str):
        meaning = str(meaning)
    
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
        return "**Ask yourself:** " + " • ".join(qs[:3]) if qs else ""

    # Collect blocks with premium formatting
    blocks = []
    shared_items = []  # Items that are same for upright/reversed (like mantra)
    
    for token in spec:
        if token == "meaning":
            blocks.append(meaning)

        elif token == "mantra":
            m = dg.get("mantra", "")
            if m:
                shared_items.append(f"✧ *{m}*")

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
                blocks.append(f"**Shadow side:** {s}")

        elif token == "questions":
            qs = questions()
            if qs:
                blocks.append(qs)

        elif token == "next_24h":
            n = dg.get("next_24h", "")
            if n:
                blocks.append(f"**Next 24 hours:** {n}")

        elif token == "relationships":
            txt = lenses.get("relationships") or dg.get("relationships", "")
            if txt:
                blocks.append(f"**In relationships:** {txt}")

        elif token == "work":
            txt = lenses.get("work") or dg.get("work", "")
            if txt:
                blocks.append(f"**At work:** {txt}")

        elif token == "money":
            txt = lenses.get("money") or dg.get("money", "")
            if txt:
                blocks.append(f"**With money:** {txt}")

        # v2 fields
        elif token == "tell":
            t = dg.get("tell", "")
            if t:
                blocks.append(t)  # No label, this IS the message

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
                    line.append(f"✓ {gf}")
                if rf:
                    line.append(f"✗ {rf}")
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

    # Assemble final text with premium spacing
    result_parts = []
    
    # Main content
    if blocks:
        result_parts.append("\n\n".join(blocks))
    
    # Shared/universal items at the end (mantras, etc)
    if shared_items:
        result_parts.append(DIVIDERS["thin"])
        result_parts.append("\n".join(shared_items))
    
    return _clip("\n\n".join(result_parts))


def render_meaning_both_sides(card: Dict[str, Any], tone: str) -> str:
    """
    Special formatter for /meaning command - shows both orientations beautifully
    with shared content (like mantras) only once at the end
    """
    tone = normalize_tone(tone)
    
    # Get card guidance (shared between both orientations)
    dg = card.get("direct_guidance", {}) or {}
    mantra = dg.get("mantra", "").strip()
    action = card.get("call_to_action", "").strip()
    
    # Helper to get clean core meaning only (no mantras/actions)
    def get_core_meaning(orientation: str) -> str:
        is_rev = orientation.lower().startswith("r")
        okey = "reversed" if is_rev else "upright"
        
        odata_raw = card.get(okey, {})
        if isinstance(odata_raw, dict):
            odata = odata_raw
        elif isinstance(odata_raw, str):
            odata = {"meaning": odata_raw}
        else:
            odata = {}
        
        # Get just the core narrative meaning
        meaning = (odata.get("meaning") or "—")
        if not isinstance(meaning, str):
            meaning = str(meaning)
        
        # Get reader voice if available
        voice_text = ""
        voice = odata.get("voice", {})
        if isinstance(voice, dict):
            lead = voice.get("lead_in", "").strip()
            pulse = voice.get("pulse", "").strip()
            if lead or pulse:
                voice_text = f"*{lead}*\n\n" if lead else ""
                voice_text += f"*{pulse}*" if pulse else ""
        
        # Get specific guidance for this orientation
        reader_voice = dg.get("reader_voice", "").strip()
        tell = dg.get("tell", "").strip()
        
        parts = []
        if voice_text:
            parts.append(voice_text)
        if meaning:
            parts.append(meaning)
        if reader_voice:
            parts.append(f"*{reader_voice}*")
        if tell:
            parts.append(tell)
        
        return "\n\n".join(parts) if parts else "—"
    
    # Build upright and reversed sections
    upright = get_core_meaning("Upright")
    reversed = get_core_meaning("Reversed")
    
    # Build the combined output with shared wisdom at bottom
    result_parts = []
    
    # Upright section
    result_parts.append(f"**☀️ Upright**\n{upright}")
    
    # Reversed section
    result_parts.append(f"**🌙 Reversed**\n{reversed}")
    
    # Shared universal wisdom at the end
    shared = []
    if mantra:
        shared.append(f"✧ *{mantra}*")
    if action:
        shared.append(f"**Take action:** {action}")
    
    if shared:
        result_parts.append(DIVIDERS["thin"])
        result_parts.append("\n".join(shared))
    
    return _clip("\n\n".join(result_parts))


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


# ==============================
# DAILY CARD STREAK TRACKING & TIME-AWARE GREETINGS
# ==============================
def get_daily_card_streak(user_id: int) -> int:
    """
    Calculate current streak of consecutive days user has drawn daily cards.
    Returns number of consecutive days (including today if drawn).
    """
    with db_connect() as conn:
        with conn.cursor() as cur:
            # Get last 30 days of daily cards for this user
            cur.execute(
                """
                SELECT day
                FROM tarot_daily_card
                WHERE user_id = %s
                ORDER BY day DESC
                LIMIT 30
                """,
                (user_id,)
            )
            rows = cur.fetchall()
    
    if not rows:
        return 0
    
    # Check consecutive days backwards from today
    today = _today_local_date()
    streak = 0
    expected_day = today
    
    for row in rows:
        day = row["day"]
        if day == expected_day:
            streak += 1
            expected_day = expected_day - timedelta(days=1)
        else:
            break
    
    return streak


def get_time_of_day_greeting() -> str:
    """Get a time-appropriate greeting based on current hour."""
    now = datetime.now(ZoneInfo("America/Chicago"))  # Use central time as default
    hour = now.hour
    
    if 5 <= hour < 12:
        return random.choice([
            "Good morning",
            "Morning light arrives",
            "As the day begins",
            "Dawn breaks",
        ])
    elif 12 <= hour < 17:
        return random.choice([
            "Good afternoon",
            "Midday pause",
            "As the sun climbs high",
            "In the afternoon glow",
        ])
    elif 17 <= hour < 21:
        return random.choice([
            "Good evening",
            "As daylight fades",
            "Evening arrives",
            "Dusk settles in",
        ])
    else:
        return random.choice([
            "Under the night sky",
            "In the quiet hours",
            "As darkness holds space",
            "Late night wisdom",
        ])


def get_mystical_timestamp() -> str:
    """Get a mystical description of current time for footers."""
    now = datetime.now(ZoneInfo("America/Chicago"))
    hour = now.hour
    
    if 5 <= hour < 9:
        return "drawn at dawn"
    elif 9 <= hour < 12:
        return "drawn in morning light"
    elif 12 <= hour < 15:
        return "drawn at midday"
    elif 15 <= hour < 18:
        return "drawn in afternoon glow"
    elif 18 <= hour < 21:
        return "drawn as daylight fades"
    elif 21 <= hour < 24:
        return "drawn under the evening sky"
    else:
        return "drawn in the quiet hours"


# ==============================
# POST-READING SUGGESTIONS (Smart Contextual Guidance)
# ==============================
def get_post_reading_suggestion(card_name: str, command: str = "cardoftheday") -> Optional[str]:
    """
    Analyze the reading and offer contextual suggestions.
    Returns a suggestion message or None.
    """
    # Heavy/challenging cards that might benefit from clarification
    HEAVY_CARDS = [
        "The Tower", "Death", "The Devil", "Ten of Swords",
        "Three of Swords", "Five of Pentacles", "Nine of Swords",
        "The Hanged Man", "Five of Cups"
    ]
    
    # Cards about choices/decisions
    DECISION_CARDS = [
        "The Lovers", "Two of Swords", "Seven of Cups",
        "The Chariot", "Justice", "Temperance"
    ]
    
    # Cards about new beginnings
    NEW_BEGINNING_CARDS = [
        "The Fool", "Ace of Wands", "Ace of Cups",
        "Ace of Swords", "Ace of Pentacles", "The Magician"
    ]
    
    # Cards about completion/endings
    COMPLETION_CARDS = [
        "The World", "Ten of Pentacles", "Ten of Cups",
        "Ten of Wands", "Death", "The Tower"
    ]
    
    # Generate suggestions based on card type
    suggestions = []
    
    if card_name in HEAVY_CARDS:
        suggestions.append("This one carries weight. **/clarify** can offer additional perspective.")
        suggestions.append("Heavy energy here. Want to **/clarify** for more insight?")
    
    if card_name in DECISION_CARDS:
        suggestions.append("Standing at a crossroads? **/threecard** can map the paths ahead.")
        suggestions.append("For deeper clarity on this decision, try **/celtic** for the full picture.")
    
    if card_name in NEW_BEGINNING_CARDS:
        suggestions.append("New beginnings ask for intention. Set yours with **/intent**.")
        suggestions.append("Starting fresh? **/intent** helps you name what you're moving toward.")
    
    if card_name in COMPLETION_CARDS:
        suggestions.append("Something's completing. **/threecard** can show what comes next.")
    
    # Return random suggestion if any apply
    if suggestions:
        return random.choice(suggestions)
    
    return None


def analyze_recent_pattern(user_id: int) -> Optional[str]:
    """
    Analyze user's recent readings for patterns.
    Returns a pattern observation message or None.
    """
    try:
        settings = get_user_settings(user_id)
        if not settings.get("history_opt_in", False):
            return None
        
        with db_connect() as conn:
            with conn.cursor() as cur:
                # Get last 7 readings
                cur.execute(
                    """
                    SELECT payload
                    FROM tarot_reading_history
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT 7
                    """,
                    (user_id,)
                )
                rows = cur.fetchall()
        
        if len(rows) < 5:
            return None  # Not enough data
        
        # Extract cards from readings
        cards = []
        for row in rows:
            payload = row.get("payload", {})
            if isinstance(payload, dict):
                # Single card readings
                if "card" in payload:
                    cards.append(payload["card"])
                # Multi-card readings
                elif "cards" in payload:
                    for c in payload["cards"]:
                        if isinstance(c, dict) and "name" in c:
                            cards.append(c["name"])
        
        if len(cards) < 5:
            return None
        
        # Analyze patterns
        # Count suit frequency
        suit_counts = {"Cups": 0, "Wands": 0, "Swords": 0, "Pentacles": 0, "Major": 0}
        reversed_count = 0
        
        for card_info in cards:
            card_name = card_info if isinstance(card_info, str) else card_info.get("name", "")
            
            # Find the card in deck
            card_obj = next((c for c in tarot_cards if c.get("name") == card_name), None)
            if card_obj:
                suit = card_obj.get("suit", "")
                if "Major" in suit:
                    suit_counts["Major"] += 1
                elif suit in suit_counts:
                    suit_counts[suit] += 1
            
            # Count reversals
            if isinstance(card_info, dict):
                if card_info.get("orientation", "").lower().startswith("r"):
                    reversed_count += 1
        
        # Generate pattern observations
        total_cards = len(cards)
        
        # Suit dominance
        max_suit = max(suit_counts, key=suit_counts.get)
        if suit_counts[max_suit] >= total_cards * 0.4:  # 40%+ of one suit
            messages = {
                "Cups": "You're deep in emotional territory lately. Notice the pattern?",
                "Wands": "Lots of fire energy this week. Creative sparks or conflicts?",
                "Swords": "Mental overload? Your readings lean heavily toward Swords lately.",
                "Pentacles": "Grounded in the material world this week. Pentacles dominate your draws.",
                "Major": "Big archetypal energy. Major Arcana keeps appearing for you.",
            }
            return messages.get(max_suit)
        
        # High reversal rate
        if reversed_count >= total_cards * 0.6:  # 60%+ reversed
            return "Most of your cards lately pull reversed. What's asking to be integrated?"
        
    except Exception as e:
        print(f"⚠️ Pattern analysis failed: {e}")
        return None
    
    return None


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


# ==============================
# PREMIUM COLOR PALETTE - Mystical Evening
# ==============================
# Philosophy: Colors evoke candlelight, velvet, old books, and sacred spaces
# Think: intimate tarot reading in a dim room with warm lighting

PREMIUM_COLORS = {
    # Suits - each tells an elemental story
    "Wands": 0xC84B31,        # Deep ember (fire, passion, creative spark)
    "Cups": 0x2C5F8D,         # Deep water blue (emotion, intuition, depth)
    "Swords": 0x5D6D7E,       # Storm gray (intellect, clarity through clouds)
    "Pentacles": 0x3D5A3D,    # Forest green (earth, abundance, grounding)
    "Major Arcana": 0x6B4E71, # Deep amethyst (spiritual, mysterious, regal)
    
    # Special reading types
    "Daily": 0x8B6F47,        # Antique gold (sacred morning ritual)
    "Mystery": 0x2D2D3A,      # Almost black (the unknown)
    "Celtic": 0x4A3145,       # Deep plum (complex readings)
    "Clarify": 0x7E6B8F,      # Soft purple (gentle guidance)
    "ThreeCard": 0x5B4E6C,    # Dusky purple (past-present-future)
    
    # UI/System
    "Welcome": 0x5B4E6C,      # Dusky purple (greeting)
    "Settings": 0x4A5568,     # Slate (neutral but warm)
    "Error": 0x8B4545,        # Deep burgundy (not harsh red)
    "General": 0x6B5B7E,      # Soft twilight purple
}

def suit_color(suit):
    """Get premium color for a suit"""
    return PREMIUM_COLORS.get(suit, PREMIUM_COLORS["General"])


# ==============================
# MYSTICAL DIVIDERS & DECORATIONS
# ==============================
DIVIDERS = {
    "thin": "· · · · ·",
    "medium": "━━━━━━━",
    "ornate": "✧ · · · ✧",
    "mystical": "⋆｡˚ ☁︎ ˚｡⋆",
    "dots": "◦ ◦ ◦",
}

# ==============================
# READING CEREMONY MESSAGES
# ==============================
CEREMONY_MESSAGES = {
    "shuffle": [
        "Shuffling the deck... ✧",
        "The cards settle into new order...",
        "⋆｡˚ Clearing the energy... ⋆｡˚",
        "Reshuffling... breathe...",
    ],
    "daily_draw": [
        "⋆｡˚ Drawing your daily card...",
        "One card steps forward for today...",
        "Pulling your companion for the day...",
        "The deck offers its morning message...",
    ],
    "single_draw": [
        "⋆｡˚ Drawing a card...",
        "One card rises to meet your question...",
        "Pulling from the deck...",
        "A card emerges...",
    ],
    "spread_layout": [
        "Laying out the spread... card by card...",
        "⋆｡˚ The pattern forms... ⋆｡˚",
        "Drawing the cards... one at a time...",
        "Arranging the spread before you...",
    ],
    "celtic_layout": [
        "Laying out the Celtic Cross... this will take a moment...",
        "⋆｡˚ Ten cards... unfolding the pattern... ⋆｡˚",
        "The full spread emerges... breathe as it forms...",
        "Card by card, the Celtic Cross reveals itself...",
    ],
    "mystery_draw": [
        "⋆｡˚ Drawing a mystery card...",
        "One card, face down... what do you sense?",
        "A card waits in shadow...",
        "Pulling a card... the image remains hidden...",
    ],
}

async def show_ceremony(interaction: discord.Interaction, ceremony_type: str = "shuffle", pause_seconds: float = 1.8):
    """
    Display a ceremony message with a pause before the actual reading.
    Creates anticipation and ritual feeling.
    """
    messages = CEREMONY_MESSAGES.get(ceremony_type, CEREMONY_MESSAGES["shuffle"])
    message = random.choice(messages)
    
    try:
        # Send the ceremony message
        if not interaction.response.is_done():
            await interaction.response.send_message(message, ephemeral=True)
        else:
            await interaction.followup.send(message, ephemeral=True)
        
        # Pause for the ceremony moment
        await asyncio.sleep(pause_seconds)
        
    except Exception as e:
        # If ceremony fails, just continue silently
        print(f"⚠️ Ceremony display failed: {e}")
        pass

# ==============================
# PREMIUM FOOTER MESSAGES
# ==============================
FOOTER_MESSAGES = {
    "daily": [
        "Carry this energy softly through your day.",
        "A gentle companion for the hours ahead.",
        "Today's thread has been woven — walk mindfully.",
        "Let this card whisper its wisdom as the day unfolds.",
        "One message, offered with care. The rest is yours to write.",
    ],
    "deep_reading": [
        "Trust your inner knowing as you integrate this wisdom.",
        "The cards reflect, but you hold the compass.",
        "Let this settle into your awareness at its own pace.",
        "Sit with what resonates. Release what doesn't.",
        "A map is not the journey — you still choose the path.",
    ],
    "single": [
        "One card, infinite interpretations — what resonates?",
        "The mirror has shown its image — what do you see?",
        "Take what serves, leave what doesn't.",
        "A single spark can light the way forward.",
        "Sometimes one card is all you need to hear.",
    ],
    "mystery": [
        "The veil lifts only when you're ready.",
        "Sometimes not knowing is part of the medicine.",
        "Trust what you felt before the reveal.",
        "Your intuition spoke first — remember what it said.",
    ],
    "spread": [
        "Each card speaks to the others — notice the threads.",
        "The full story lives between the cards, not just in them.",
        "Trust the pattern that emerges.",
        "Let the larger picture form at its own pace.",
    ],
    "clarify": [
        "A smaller light to illuminate what was unclear.",
        "Sometimes we need to look twice to truly see.",
        "This card offers a softer angle on the same truth.",
        "The clarification arrives — hold both perspectives.",
    ],
    "general": [
        "A tarot reading is a mirror, not a cage. You steer.",
        "The cards offer reflection — your wisdom makes meaning.",
        "Listen with your intuition, not just your mind.",
        "What the cards reveal, you already knew somewhere deep.",
    ],
}

def get_footer(reading_type: str = "general") -> str:
    """Get a random premium footer message for the reading type"""
    messages = FOOTER_MESSAGES.get(reading_type, FOOTER_MESSAGES["general"])
    return random.choice(messages)


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
        "Slate wiped clean. The cards await your next question.",
        "A fresh shuffle, a fresh start. What wants to be known?",
    ],
    "daily": [
        "Here is the energy that threads through your day...",
        "This card has stepped forward to guide you.",
        "Its message hums softly — take it with you into the light.",
        "The dawn pulls a single card forward — your companion for today.",
        "One card steps into the light. This is what asks for your attention.",
        "Before the day unfolds, a message arrives...",
    ],
    "daily_repeat": [
        "The same card returns — its message isn't finished with you yet.",
        "This energy persists. Perhaps today you'll hear it differently.",
        "Again, this one. There's something here you need to sit with longer.",
    ],
    "spread": [
        "The weave of time unfolds — past, present, and future speak.",
        "Let us see how the threads intertwine for your path.",
        "Each card now reveals its whisper in the larger story.",
        "Three cards fall into place. Each one speaks to the others.",
        "The timeline spreads before you — what patterns do you see?",
    ],
    "deep": [
        "This spread carries depth — breathe as you read its symbols.",
        "A more ancient current flows beneath these cards.",
        "The deck speaks slowly now; listen beyond the words.",
        "Ten cards. A full constellation. This will take time — breathe.",
        "We're about to unfold something layered. Make space for complexity.",
        "The Celtic Cross reveals itself — trust the larger pattern.",
    ],
    "question": [
        "You've brought a question with weight. Let's see what wants to speak.",
        "The deck quiets when honest questions arrive. Here is what surfaces...",
        "I feel the gravity of your question. The cards are listening.",
        "Your question hangs in the air. A single card rises to meet it.",
    ],
    "mystery": [
        "Close your eyes... what does this card want you to feel first?",
        "Face down, waiting. What's your intuition saying?",
        "The card rests in shadow. What do you sense before the light?",
        "Sometimes we need to sit with the unknown before the reveal.",
    ],
    "clarify": [
        "A second card steps forward — let's see what it illuminates.",
        "You asked for clarity. Here is another angle on the same truth.",
        "The clarifier arrives — notice what shifts in your understanding.",
    ],
    "general": [
        "The veil lifts and a message takes shape...",
        "Listen closely — the cards are patient but precise.",
        "A single spark of insight is about to emerge...",
        "The cards have something to say. Are you ready to hear it?",
        "Stillness first, then the message reveals itself.",
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
    """Post server count to top.gg every 30 minutes using HTTP API"""
    if not TOPGG_TOKEN:
        return
    
    try:
        server_count = len(bot.guilds)
        url = f"https://top.gg/api/bots/{bot.user.id}/stats"
        headers = {
            "Authorization": TOPGG_TOKEN,
            "Content-Type": "application/json"
        }
        data = {
            "server_count": server_count
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as resp:
                if resp.status == 200:
                    print(f"✅ Posted to top.gg: {server_count} servers")
                else:
                    print(f"⚠️ top.gg returned status {resp.status}: {await resp.text()}")
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
    if TOPGG_TOKEN and not post_topgg_stats.is_running():
        post_topgg_stats.start()
        print("✅ top.gg stats task started.")
        # Post immediately on startup
        try:
            server_count = len(bot.guilds)
            url = f"https://top.gg/api/bots/{bot.user.id}/stats"
            headers = {
                "Authorization": TOPGG_TOKEN,
                "Content-Type": "application/json"
            }
            data = {
                "server_count": server_count
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, headers=headers) as resp:
                    if resp.status == 200:
                        print(f"✅ Initial post to top.gg: {server_count} servers")
                    else:
                        print(f"⚠️ top.gg initial post status {resp.status}")
        except Exception as e:
            print(f"⚠️ top.gg initial post failed: {type(e).__name__}: {e}")
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
        color=PREMIUM_COLORS["General"]
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
        color=PREMIUM_COLORS["Settings"],
    )
    embed.set_footer(text="History is opt-in • Use /forgetme to delete stored data.")

    await send_ephemeral(interaction, embed=embed, mood="general")

@bot.tree.command(name="cardoftheday", description="Reveal the card that guides your day.")
async def cardoftheday_slash(interaction: discord.Interaction):
    day = _today_local_date()
    row = get_daily_card_row(interaction.user.id, day)
    
    is_first_draw_today = (row is None)
    
    # If drawing a new card today, show ceremony with time-aware greeting
    if is_first_draw_today:
        greeting = get_time_of_day_greeting()
        ceremony_messages = [
            f"{greeting}. Let's see what today asks of you...",
            f"{greeting}. One card steps forward...",
            f"{greeting}. The deck offers its guidance...",
        ]
        custom_message = random.choice(ceremony_messages)
        
        # Show custom ceremony message
        try:
            await interaction.response.send_message(custom_message, ephemeral=True)
            await asyncio.sleep(2.0)
        except Exception:
            pass
    else:
        # Card already drawn - return message
        if not await safe_defer(interaction, ephemeral=True):
            return

    if row:
        orientation = row["orientation"]
        card = find_card_by_name(row["card_name"])
        if card is None:
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

    # Premium clean header
    desc = f"**{card['name']}** {tone_emoji} *{orientation}*\n{DIVIDERS['dots']}\n\n{meaning}"
    
    # Add intention if set
    if intent_text:
        desc += f"\n\n✧ *Your intention: {intent_text}*"
    
    # If returning to check card again, add gentle reminder
    if not is_first_draw_today:
        desc = f"*Your card for today remains unchanged. The message holds...*\n\n{DIVIDERS['thin']}\n\n{desc}"

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
            "is_first_draw": is_first_draw_today,
        },
        settings=settings,
    )

    embed = discord.Embed(
        title=f"✧ Your Daily Card ✧",
        description=desc,
        color=PREMIUM_COLORS["Daily"],
    )

    if attach_url:
        embed.set_image(url=attach_url)
    
    # Calculate streak for footer
    streak = get_daily_card_streak(interaction.user.id)
    timestamp = get_mystical_timestamp()
    
    # Dynamic footer based on streak
    if is_first_draw_today and streak >= 3:
        footer_messages = [
            f"{streak} days in a row. The ritual deepens.",
            f"Day {streak} of your practice. Well done.",
            f"{streak} days of consistency. The pattern holds.",
        ]
        footer = random.choice(footer_messages)
    elif is_first_draw_today:
        footer = f"{get_footer('daily')} • {timestamp}"
    else:
        footer = "The same card speaks again. What new meaning emerges?"
    
    embed.set_footer(text=footer)

    await send_ephemeral(interaction, embed=embed, mood="daily", file_obj=file_obj)
    
    # Post-reading suggestions (only for first draw, not returns)
    if is_first_draw_today:
        # Wait a moment before suggesting
        await asyncio.sleep(1.5)
        
        # Check for card-specific suggestion
        suggestion = get_post_reading_suggestion(card["name"], "cardoftheday")
        
        # Check for pattern observation
        if not suggestion:
            suggestion = analyze_recent_pattern(interaction.user.id)
        
        # Send suggestion if we have one
        if suggestion:
            try:
                await interaction.followup.send(
                    f"*{suggestion}*",
                    ephemeral=True
                )
            except Exception:
                pass  # Silently fail if suggestion can't be sent


@bot.tree.command(name="read", description="Three-card reading: Situation • Obstacle • Guidance.")
@app_commands.describe(intention="Your question or intention (example: my career path)")
async def read_slash(interaction: discord.Interaction, intention: str):
    # Show ceremony first
    await show_ceremony(interaction, "spread_layout", pause_seconds=2.0)

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
        title=f"◇ Intuitive Reading ◇",
        description=f"✧ **Intention:** *{intention}*\n\n**How I’ll read this:** {tone_label(tone)}",
        color=PREMIUM_COLORS["General"],
    )

    pretty_positions = [f"Situation {E['sun']}", f"Obstacle {E['sword']}", f"Guidance {E['star']}"]
    for pos, (card, orientation) in zip(pretty_positions, cards):
        meaning = render_card_text(card, orientation, tone)
        embed.add_field(
            name=f"{pos}: {card['name']} ({orientation})",
            value=meaning if len(meaning) < 1000 else meaning[:997] + "...",
            inline=False,
        )

    embed.set_footer(text=get_footer("spread"))
    await send_ephemeral(interaction, embed=embed, mood="question")
    
    # Post-reading suggestion
    await asyncio.sleep(1.5)
    
    # Check if any heavy cards appeared
    card_names = [card["name"] for card, _ in cards]
    suggestions = [get_post_reading_suggestion(name, "read") for name in card_names]
    suggestions = [s for s in suggestions if s]
    
    if suggestions:
        try:
            await interaction.followup.send(
                f"*{random.choice(suggestions)}*",
                ephemeral=True
            )
        except Exception:
            pass


@bot.tree.command(name="threecard", description="Past • Present • Future spread.")
async def threecard_slash(interaction: discord.Interaction):
    # Show ceremony
    await show_ceremony(interaction, "spread_layout", pause_seconds=2.0)

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
        color=PREMIUM_COLORS["ThreeCard"],
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
    # Longer ceremony for the big spread
    await show_ceremony(interaction, "celtic_layout", pause_seconds=2.5)

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
        color=PREMIUM_COLORS["Celtic"],
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
                color=PREMIUM_COLORS["Celtic"],
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

    # Build clean embed with both orientations
    embed = discord.Embed(
        title=f"◇ {chosen_name} ◇",
        description=render_meaning_both_sides(chosen, tone),
        color=color,
    )
    
    embed.set_footer(text=get_footer("single"))

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
    # Clarify ceremony
    await show_ceremony(interaction, "single_draw", pause_seconds=1.5)

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
    embed.set_footer(text=get_footer("clarify"))
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
    # Special mystery ceremony
    await show_ceremony(interaction, "mystery_draw", pause_seconds=2.0)

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
        embed.set_footer(text=get_footer("mystery"))

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
        color=PREMIUM_COLORS["Welcome"],
    )

    embed.add_field(name="What I can do for you", value=chunks[0] if chunks else "—", inline=False)
    for i, part in enumerate(chunks[1:], start=2):
        embed.add_field(name=f"What I can do for you (cont. {i})", value=part, inline=False)

    embed.set_footer(text=get_footer("general"))
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
        color=PREMIUM_COLORS["Settings"],
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
