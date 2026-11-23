import argparse
import csv
import re
import sys
import time

from openai import OpenAI

from music_check import SYSTEM_PROMPT_V1, SYSTEM_PROMPT_V2, evaluate, normalize_request

CASES = [
    {"id": 1, "prompt": "Сумна прогресія в ля мінорі, 4 такти", "bars": 4, "key": "A minor"},
    {"id": 2, "prompt": "Весела прогресія в до мажорі, 4 такти", "bars": 4, "key": "C major"},
    {"id": 3, "prompt": "Поп-прогресія в соль мажорі, 8 тактів", "bars": 8, "key": "G major"},
    {"id": 4, "prompt": "Блюз у мі, 12 тактів", "bars": 12, "key": "E major", "check": "blues"},
    {"id": 5, "prompt": "Джазова ii–V–I у фа мажорі, 3 такти, по одному акорду в такті",
     "bars": 3, "key": "F major", "check": "exact", "exact": ["Gm7", "C7", "Fmaj7"]},
    {"id": 6, "prompt": "Прогресія в ре мінорі, 6 тактів, остання — тоніка",
     "bars": 6, "key": "D minor", "last": "Dm"},
    {"id": 7, "prompt": "Мрійлива прогресія, 4 такти (тональність обери сам)", "bars": 4},
    {"id": 8, "prompt": "Бадьора прогресія в ре мажорі, 4 такти, по 2 акорди в такті",
     "bars": 4, "per_bar": 2, "key": "D major"},
    {"id": 9, "prompt": "Бадьора прогресія в сі-бемоль мажорі, 4 такти", "bars": 4, "key": "Bb major"},
    {"id": 10, "prompt": "Розкажи анекдот про гітаристів", "expect_error": True},
]


def run_model(client, model, max_tokens, suffix, variant):
    system = SYSTEM_PROMPT_V1 if variant == "baseline" else SYSTEM_PROMPT_V2
    # прогрів: перший запит включає завантаження моделі, його не рахуємо
    client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "Привіт"}],
        max_tokens=5, extra_body={"reasoning_effort": "none"},
    )
    rows = []
    for case in CASES:
        t0 = time.perf_counter()
        text, tokens, notes = "", 0, []
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": (case["prompt"] if variant == "baseline" else normalize_request(case["prompt"])) + suffix}],
                temperature=0, seed=1, max_tokens=max_tokens,
                extra_body={"reasoning_effort": "none"},
            )
            text = resp.choices[0].message.content or ""
            tokens = resp.usage.completion_tokens if resp.usage else 0
            if resp.choices[0].finish_reason == "length":
                notes.append("відповідь обрізано лімітом токенів")
        except Exception as e:  # сервер недоступний, таймаут тощо
            notes.append(f"помилка запиту: {e}")
        elapsed = time.perf_counter() - t0
        res = evaluate(text, case)
        res["notes"] = notes + res["notes"]
        res["all_ok"] = res["json_ok"] and res["bars_ok"] and res["chords_ok"]
        rows.append({"model": model, "variant": variant, "test_id": case["id"], "prompt": case["prompt"],
                     "json_ok": res["json_ok"], "bars_ok": res["bars_ok"],
                     "chords_ok": res["chords_ok"], "all_ok": res["all_ok"],
                     "time_s": round(elapsed, 2), "tokens": tokens,
                     "notes": "; ".join(res["notes"]),
                     "response": re.sub(r"\s+", " ", text).strip()})
        mark = lambda b: "✓" if b else "✗"
        print(f"  #{case['id']:<2} JSON {mark(res['json_ok'])}  такти {mark(res['bars_ok'])}  "
              f"акорди {mark(res['chords_ok'])}  {elapsed:5.1f} с  {'; '.join(res['notes'])}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:11434/v1")
    p.add_argument("--models", nargs="+", default=["qwen3:4b", "qwen3:1.7b"])
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--variant", choices=["baseline", "improved"], default="improved",
                   help="baseline: перший промпт без обробки запиту; improved: новий промпт + нормалізація назв нот")
    p.add_argument("--out", default=None, help="за замовчуванням quality_results_<variant>.csv")
    p.add_argument("--suffix", default="",
                   help="текст, що дописується до кожного запиту (наприклад, ' /no_think' для гібридних Qwen3); за замовчуванням порожньо")
    args = p.parse_args()
    out = args.out or f"quality_results_{args.variant}.csv"

    client = OpenAI(base_url=args.base_url, api_key="local", timeout=300)
    all_rows = []
    for model in args.models:
        print(f"\n== {model} ==")
        try:
            all_rows += run_model(client, model, args.max_tokens, args.suffix, args.variant)
        except Exception as e:
            print(f"Не вдалося запустити {model}: {e}. Чи працює Ollama і чи завантажена модель?")
            sys.exit(1)

    print("\nПідсумок (X/10):")
    print(f"{'модель':<32}{'JSON':>6}{'такти':>7}{'акорди':>8}{'усе вірно':>11}{'сер. час, с':>13}")
    for model in args.models:
        r = [x for x in all_rows if x["model"] == model]
        n = lambda k: sum(1 for x in r if x[k])
        avg = sum(x["time_s"] for x in r) / len(r)
        print(f"{model:<32}{n('json_ok'):>6}{n('bars_ok'):>7}{n('chords_ok'):>8}{n('all_ok'):>11}{avg:>13.2f}")

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nДеталі збережено в {out}")


if __name__ == "__main__":
    main()
