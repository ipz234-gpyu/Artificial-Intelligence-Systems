import os
import random
import re
import sys
import time

from openai import APIConnectionError, APIStatusError, APITimeoutError, NotFoundError, OpenAI

from music_check import (SYSTEM_PROMPT, chord_tones, extract_json, normalize_request, parse_chord,
                         parse_key, scale_pcs, schema_ok)

BASE_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("MODEL", "qwen3:4b-instruct-2507-q4_K_M")

MAX_TOKENS = 300            # JSON на 12 тактів ≈ 50 токенів; запас у 6 разів, але без безкінечної генерації
TIMEOUT_S = 120             # одна відповідь ≈ 5–10 с; 120 с — запас на повільний старт моделі
MAX_ATTEMPTS = 3
TEMPERATURE = 0.0           # детермінований результат; так само, як у тесті якості
VARIATION_TEMPERATURE = 0.8 # кнопка «Інший варіант»
RETRY_TEMPERATURE = 0.5     # повторна спроба має відрізнятися від провальної
MAX_PROMPT_CHARS = 500

NUMBER_WORDS = {"одному": 1, "двом": 2, "двох": 2, "трьох": 3, "чотирьох": 4}


class GenerationError(Exception):
    """Помилка, текст якої можна показати користувачеві."""


_client = None


def client():
    global _client
    if _client is None:
        # max_retries=0: помилку хочемо бачити одразу, а не після прихованих повторів бібліотеки
        _client = OpenAI(base_url=BASE_URL, api_key="local", timeout=TIMEOUT_S, max_retries=0)
    return _client


def _call_model(messages, temperature, seed):
    """Один запит до моделі. Повертає (текст, кількість токенів, finish_reason)."""
    try:
        r = client().chat.completions.create(
            model=MODEL, messages=messages, temperature=temperature, seed=seed,
            max_tokens=MAX_TOKENS, extra_body={"reasoning_effort": "none"},
        )
    except APITimeoutError:
        raise GenerationError(f"Модель не відповіла за {TIMEOUT_S} с. Спробуйте простіший запит або перезапустіть Ollama.")
    except APIConnectionError:
        raise GenerationError(f"Не вдалося підключитися до Ollama ({BASE_URL}). "
                              "Запустіть Ollama (команда `ollama serve` або застосунок Ollama) і спробуйте ще раз.")
    except NotFoundError:
        raise GenerationError(f"Модель «{MODEL}» не знайдена в Ollama. Завантажте її: ollama pull {MODEL}")
    except APIStatusError as e:
        raise GenerationError(f"Ollama повернула помилку {e.status_code}. Перегляньте її журнал.")
    choice = r.choices[0]
    return choice.message.content or "", (r.usage.completion_tokens if r.usage else 0), choice.finish_reason


def check_health():
    """Чи доступна Ollama і чи завантажена потрібна модель."""
    try:
        ids = [m.id for m in client().models.list().data]
    except Exception:
        return {"ollama": False, "model_present": False, "model": MODEL}
    return {"ollama": True, "model_present": MODEL in ids, "model": MODEL}


def parse_request(text):
    """Що саме просив користувач (з нормалізованого тексту) — для перевірки відповіді."""
    low = text.lower()
    req = {"bars": None, "per_bar": None, "key": None, "mode_given": False, "key_text": None,
           "free": bool(re.search(r"блюз|джаз", low)), "tonic_end": "тонік" in low,
           "musical": bool(re.search(r"такт|акорд|прогресі|тональн", low))}
    m = re.search(r"(\d+)\s*такт", low)
    if m:
        req["bars"] = int(m.group(1))
    m = re.search(r"по\s+(\d+|одному|двом|двох|трьох|чотирьох)\s+акорд", low)
    if m:
        req["per_bar"] = int(m.group(1)) if m.group(1).isdigit() else NUMBER_WORDS[m.group(1)]
    m = re.search(r"тональності\s+([A-G][#b]?(?:\s+(?:major|minor))?)", text)
    if m:
        req["key_text"] = m.group(1)
        req["key"] = parse_key(m.group(1))
        req["mode_given"] = bool(re.search(r"major|minor", m.group(1)))
    return req


def validate(obj, req):
    """Список проблем відповіді (порожній — все гаразд)."""
    if not schema_ok(obj):
        return ["структура JSON не відповідає формату {key, bars}"]
    issues, bars = [], obj["bars"]
    if req["bars"] and len(bars) != req["bars"]:
        issues.append(f"тактів {len(bars)}, потрібно {req['bars']}")
    if req["per_bar"] and any(len(b) != req["per_bar"] for b in bars):
        issues.append(f"в кожному такті має бути рівно {req['per_bar']} акорд(и)")
    key = parse_key(obj["key"])
    if key is None:
        return issues + [f"тональність {obj['key']!r} не розпізнано"]
    if req["key"] and (key[0] != req["key"][0] or (req["mode_given"] and key[1] != req["key"][1])):
        issues.append(f"тональність має бути {req['key_text']}, а не {obj['key']}")
    names = [c for bar in bars for c in bar]
    chords = [parse_chord(c) for c in names]
    bad = [n for n, c in zip(names, chords) if c is None]
    if bad:
        return issues + [f"невідомі акорди: {bad}"]
    if not req["free"]:
        scale = scale_pcs(key)
        outside = sorted({n for n, c in zip(names, chords) if not chord_tones(c) <= scale})
        if outside:
            issues.append(f"акорди {outside} не належать до тональності {obj['key']}")
    if req["tonic_end"] and chords[-1][0] != key[0]:
        issues.append("остання гармонія має бути тонікою")
    return issues


def midi_notes(chord):
    """Ноти акорда для відтворення в браузері: бас + акорд над ним."""
    root, intervals = chord
    base = 48 + root
    notes, prev = [base - 12], -1
    for i in intervals:
        while i < prev:
            i += 12
        notes.append(base + i)
        prev = i
    return notes


def _bars_payload(obj):
    out = []
    for bar in obj["bars"]:
        row = []
        for name in bar:
            chord = parse_chord(name)
            row.append({"name": name, "notes": midi_notes(chord) if chord else []})
        out.append(row)
    return out


def generate(prompt, variation=False):
    prompt = (prompt or "").strip()
    if not prompt:
        raise GenerationError("Введіть запит, наприклад: «сумна прогресія в ля мінорі, 4 такти».")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise GenerationError(f"Запит задовгий (понад {MAX_PROMPT_CHARS} символів).")

    request_text = normalize_request(prompt)
    req = parse_request(request_text)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": request_text}]
    base_temp = VARIATION_TEMPERATURE if variation else TEMPERATURE
    seed = random.randint(1, 10**6) if variation else 1

    t0, tokens_total, history = time.perf_counter(), 0, []
    obj, issues = None, []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        temp = base_temp if attempt == 1 else max(base_temp, RETRY_TEMPERATURE)
        text, tokens, finish = _call_model(messages, temp, seed + attempt - 1)
        tokens_total += tokens
        obj = extract_json(text)
        if obj is None:
            issues = ["відповідь не є коректним JSON" + (" (обрізана лімітом токенів)" if finish == "length" else "")]
        elif "bars" not in obj and isinstance(obj.get("error"), str):
            if not req["musical"]:
                reason = obj["error"].strip()
                if reason.lower() == "коротка причина":
                    reason = "запит не стосується акордових прогресій"
                return {"ok": True, "refusal": reason, "attempts": attempt, "tokens": tokens_total,
                        "elapsed_s": round(time.perf_counter() - t0, 1), "model": MODEL, "request": request_text}
            issues = ["модель відмовилась, хоча запит стосується прогресії"]
        else:
            issues = validate(obj, req)
        history.append({"attempt": attempt, "issues": issues})
        if not issues:
            break
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": "Відповідь не пройшла перевірку: " + "; ".join(issues)
                      + ". Поверни виправлений JSON за тим самим форматом."}]

    result = {"ok": not issues, "issues": issues, "attempts": len(history), "history": history,
              "tokens": tokens_total, "elapsed_s": round(time.perf_counter() - t0, 1),
              "model": MODEL, "request": request_text, "key": None, "bars": None}
    if obj is not None and schema_ok(obj):
        result["key"], result["bars"] = obj["key"], _bars_payload(obj)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Використання: python generator.py "запит" [--variation]')
    try:
        res = generate(sys.argv[1], variation="--variation" in sys.argv)
    except GenerationError as e:
        sys.exit(f"Помилка: {e}")
    if res.get("refusal"):
        print("Відмова:", res["refusal"])
    elif res["bars"]:
        print(f"{res['key']}  |  " + "  |  ".join(" ".join(c["name"] for c in bar) for bar in res["bars"]))
        print("перевірка:", "пройдена" if res["ok"] else "НЕ пройдена: " + "; ".join(res["issues"]))
    else:
        print("Не вдалося отримати прогресію:", "; ".join(res["issues"]))
    print(f"спроб: {res['attempts']}, токенів: {res['tokens']}, {res['elapsed_s']} с, модель: {res['model']}")
