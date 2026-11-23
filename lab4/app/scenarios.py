import csv
import json
import sys

import generator
from music_check import evaluate
from quality_test import CASES


def main():
    print(f"Модель: {generator.MODEL}\n")
    rows = []
    for case in CASES:
        try:
            res = generator.generate(case["prompt"])
        except generator.GenerationError as e:
            sys.exit(f"Помилка: {e}")
        if res.get("refusal"):
            text = json.dumps({"error": res["refusal"]}, ensure_ascii=False)
        elif res["bars"]:
            text = json.dumps({"key": res["key"], "bars": [[c["name"] for c in b] for b in res["bars"]]}, ensure_ascii=False)
        else:
            text = ""
        verdict = evaluate(text, case)
        passed = verdict["json_ok"] and verdict["bars_ok"] and verdict["chords_ok"]
        answer = text if text else "—"
        notes = "; ".join(verdict["notes"]) or "; ".join(res.get("issues", []))
        rows.append({"test_id": case["id"], "prompt": case["prompt"], "attempts": res["attempts"],
                     "passed": passed, "time_s": res["elapsed_s"], "notes": notes, "answer": answer})
        print(f"#{case['id']:<2} {'✓' if passed else '✗'}  спроб: {res['attempts']}  {res['elapsed_s']:5.1f} с  "
              f"{case['prompt'][:46]:<46} {notes}")

    total = sum(r["passed"] for r in rows)
    retried = sum(r["attempts"] > 1 for r in rows)
    print(f"\nПройдено: {total}/{len(rows)}; запитів, де знадобилась повторна спроба: {retried}; "
          f"середній час: {sum(r['time_s'] for r in rows) / len(rows):.1f} с")
    with open("scenarios_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("Деталі збережено в scenarios_results.csv")


if __name__ == "__main__":
    main()
