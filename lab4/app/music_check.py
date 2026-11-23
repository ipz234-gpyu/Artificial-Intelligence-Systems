import json
import re

SYSTEM_PROMPT_V1 = (
    "Ти — музичний помічник. Відповідай ЛИШЕ одним JSON-об'єктом, без пояснень і без markdown.\n"
    'Формат: {"key": "A minor", "bars": [["Am"], ["F"], ["C"], ["G"]]}\n'
    "bars — список тактів; кожен такт — список акордів (рядків).\n"
    "Назви акордів — англійською нотацією: C, Dm, F#m, Bb, G7, Cmaj7, Bdim.\n"
    "Точно дотримуйся вказаної кількості тактів і тональності. "
    "Усі акорди мають належати до тональності, якщо тільки не просять блюз або джаз.\n"
    'Якщо запит не стосується акордових прогресій, поверни {"error": "коротка причина"}.'
)

SYSTEM_PROMPT_V2 = (
    "Ти — музичний помічник. Відповідай ЛИШЕ одним JSON-об'єктом, без пояснень і без markdown.\n"
    'Формат: {"key": "<тональність>", "bars": [[<акорди 1-го такту>], [<акорди 2-го такту>], ...]}\n'
    "Правила:\n"
    "- key — тональність із запиту; якщо її не вказано, обери сам.\n"
    "- Кількість тактів і кількість акордів у такті — рівно як у запиті (за замовчуванням по одному).\n"
    "- Назви акордів — англійською нотацією: C, Dm, F#m, Bb, G7, Cmaj7, Bdim.\n"
    "- Усі акорди мають бути зі ступенів цієї тональності, окрім блюзу й джазу.\n"
    "Приклади:\n"
    'Запит: прогресія в G minor, 3 такти → {"key": "G minor", "bars": [["Gm"], ["Eb"], ["F"]]}\n'
    'Запит: весела прогресія в Eb major, 2 такти, по 2 акорди в такті → '
    '{"key": "Eb major", "bars": [["Eb", "Bb"], ["Cm", "Ab"]]}\n'
    'Запит: розкажи про погоду → {"error": "запит не про акордові прогресії"}'
)
SYSTEM_PROMPT = SYSTEM_PROMPT_V2  # промпт застосунку за замовчуванням

# українські назви нот -> літерні (ре, мі моделі розуміють гірше за інші)
SOLFEGE = {"до": "C", "ре": "D", "мі": "E", "фа": "F", "соль": "G", "ля": "A", "сі": "B"}
_KEY_PHRASE = re.compile(
    r"\b(?:в|у)\s+(до|ре|мі|фа|соль|ля|сі)(?:[-\s]?(бемоль|дієз))?\b(?:\s+(мажор|мінор)\w*)?",
    re.IGNORECASE,
)


def normalize_request(text):
    """'в сі-бемоль мажорі' -> 'в тональності Bb major'; решту тексту не чіпає."""
    def repl(m):
        note = SOLFEGE[m.group(1).lower()]
        acc = {"бемоль": "b", "дієз": "#"}.get((m.group(2) or "").lower(), "")
        mode = {"мажор": " major", "мінор": " minor"}.get((m.group(3) or "").lower(), "")
        return f"в тональності {note}{acc}{mode}"
    return _KEY_PHRASE.sub(repl, text)


NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
ACCIDENTAL = {"#": 1, "b": -1, "": 0}

# суфікс акорда -> інтервали від тоніки (півтони)
QUALITIES = {
    "": (0, 4, 7), "m": (0, 3, 7), "min": (0, 3, 7),
    "dim": (0, 3, 6), "aug": (0, 4, 8),
    "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "M7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10), "m7b5": (0, 3, 6, 10), "dim7": (0, 3, 6, 9),
    "sus2": (0, 2, 7), "sus4": (0, 5, 7), "6": (0, 4, 7, 9),
    "9": (0, 4, 7, 10, 2),
}

MAJOR = {0, 2, 4, 5, 7, 9, 11}
# мінор: натуральний + підвищений VII ступінь (гармонічний мінор), щоб E/E7 в ля мінорі вважалися своїми
MINOR = {0, 2, 3, 5, 7, 8, 10, 11}

CHORD_RE = re.compile(r"^([A-G])([#b]?)(.*)$")
KEY_RE = re.compile(r"^([A-G])([#b]?)\s*(major|minor|Major|Minor|blues|Blues|maj|min|m|M)?$")


def parse_chord(name):
    """'F#m7' -> (корінь 0..11, інтервали). None, якщо назва невідома."""
    m = CHORD_RE.match(name.strip())
    if not m:
        return None
    letter, acc, suffix = m.groups()
    if suffix not in QUALITIES:
        return None
    return (NOTE_PC[letter] + ACCIDENTAL[acc]) % 12, QUALITIES[suffix]


def chord_tones(chord):
    root, intervals = chord
    return {(root + i) % 12 for i in intervals}


def parse_key(text):
    """'A minor' -> (9, 'minor'). None, якщо не вдалося розібрати."""
    m = KEY_RE.match(text.strip())
    if not m:
        return None
    letter, acc, word = m.groups()
    tonic = (NOTE_PC[letter] + ACCIDENTAL[acc]) % 12
    minor = word in ("minor", "Minor", "min", "m")
    return tonic, "minor" if minor else "major"


def scale_pcs(key):
    tonic, mode = key
    base = MINOR if mode == "minor" else MAJOR
    return {(tonic + i) % 12 for i in base}


def extract_json(text):
    """Дістає JSON-об'єкт з відповіді.

    Прибирає блок міркувань (<think>...</think>, а також випадок, коли є лише
    закривний </think>) і ```-огорожі. Відповідь мусить бути самим JSON-ом,
    як вимагає системний промпт: JSON, знайдений усередині міркувань або
    після вступного тексту, не зараховується.
    """
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    if not text.startswith("{"):
        return None
    end = text.rfind("}")
    if end == -1:
        return None
    try:
        return json.loads(text[:end + 1])
    except json.JSONDecodeError:
        return None


def schema_ok(obj):
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("key"), str)
        and isinstance(obj.get("bars"), list) and obj["bars"]
        and all(isinstance(b, list) and b and all(isinstance(c, str) for c in b) for b in obj["bars"])
    )


def evaluate(text, case):
    """Перевіряє відповідь за описом тесту. Повертає прапорці й список зауважень."""
    res = {"json_ok": False, "bars_ok": False, "chords_ok": False, "notes": []}
    obj = extract_json(text)

    if case.get("expect_error"):
        ok = isinstance(obj, dict) and isinstance(obj.get("error"), str) and "bars" not in obj
        res.update(json_ok=ok, bars_ok=ok, chords_ok=ok)
        if not ok:
            res["notes"].append("очікувався {\"error\": ...}")
        return res

    if not schema_ok(obj):
        res["notes"].append("некоректний JSON або схема")
        return res
    res["json_ok"] = True
    bars = obj["bars"]

    # кількість тактів (і акордів у такті, якщо задано)
    bars_ok = len(bars) == case["bars"]
    if not bars_ok:
        res["notes"].append(f"тактів {len(bars)}, потрібно {case['bars']}")
    if "per_bar" in case and not all(len(b) == case["per_bar"] for b in bars):
        bars_ok = False
        res["notes"].append(f"має бути по {case['per_bar']} акорди в такті")
    res["bars_ok"] = bars_ok

    # тональність
    model_key = parse_key(obj["key"])
    if model_key is None:
        res["notes"].append(f"тональність не розібрано: {obj['key']!r}")
        return res
    if "key" in case and model_key != parse_key(case["key"]):
        res["notes"].append(f"тональність {obj['key']!r}, потрібно {case['key']!r}")
        return res

    # акорди
    chords_raw = [c for bar in bars for c in bar]
    chords = [parse_chord(c) for c in chords_raw]
    bad = [r for r, p in zip(chords_raw, chords) if p is None]
    if bad:
        res["notes"].append(f"невідомі акорди: {bad}")
        return res

    check = case.get("check", "diatonic")
    if check == "diatonic":
        scale = scale_pcs(model_key)
        out = [r for r, p in zip(chords_raw, chords) if not chord_tones(p) <= scale]
        if out:
            res["notes"].append(f"недіатонічні акорди: {sorted(set(out))}")
            return res
    elif check == "blues":
        tonic = model_key[0]
        roots = {tonic, (tonic + 5) % 12, (tonic + 7) % 12}
        allowed = {QUALITIES[""], QUALITIES["7"], QUALITIES["9"]}
        out = [r for r, p in zip(chords_raw, chords) if p[0] not in roots or p[1] not in allowed]
        if out:
            res["notes"].append(f"поза схемою I–IV–V: {sorted(set(out))}")
            return res
    elif check == "exact":
        expected = [parse_chord(c) for c in case["exact"]]
        if chords != expected:
            res["notes"].append(f"очікувалось {case['exact']}, отримано {chords_raw}")
            return res

    if "last" in case and chords[-1] != parse_chord(case["last"]):
        res["notes"].append(f"останній акорд {chords_raw[-1]}, потрібно {case['last']}")
        return res

    res["chords_ok"] = True
    return res
