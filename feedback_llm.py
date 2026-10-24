"""
Part 2: LLM feedback for student explanations using Google Gemini.
Run: python feedback_llm.py
By default evaluates 10 labelled rows to limit API use.
Run all labelled rows: python feedback_llm.py --all-labelled
"""
from pathlib import Path
import argparse, json, os
import pandas as pd
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
LESSON_TEXT = (DATA / "lessons.md").read_text(encoding="utf-8")
LESSON_MARKERS = {"L1": "## L1 — Photosynthesis", "L2": "## L2 — Newton's First Law", "L3": "## L3 — Supply and Demand"}

def lesson_for(lesson_id):
    marker = LESSON_MARKERS.get(str(lesson_id))
    if not marker:
        return "Lesson notes not found. Do not invent facts."
    start = LESSON_TEXT.find(marker)
    if start < 0:
        return "Lesson notes not found. Do not invent facts."
    next_pos = LESSON_TEXT.find("\n## ", start + len(marker))
    return LESSON_TEXT[start: next_pos if next_pos >= 0 else len(LESSON_TEXT)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-labelled", action="store_true", help="Evaluate all labelled rows (up to 60 API calls).")
    args = parser.parse_args()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set. In this Command Prompt, run: set GEMINI_API_KEY=YOUR_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    df = pd.read_csv(DATA / "explanations.csv")
    labelled = df[df["label"].isin(["understood", "partial", "wrong"])].copy()
    if labelled.empty:
        raise SystemExit("No labelled rows found in explanations.csv.")
    if not args.all_labelled:
        labelled = labelled.head(10)
        print("Evaluating 10 labelled rows first to limit API use. Use --all-labelled to evaluate all labelled rows.")
    client = genai.Client(api_key=api_key)
    results = []
    for index, (_, row) in enumerate(labelled.iterrows(), start=1):
        prompt = f'''You are a careful educational feedback assistant. Compare the student's explanation only with the lesson notes.
LESSON NOTES:
{lesson_for(row['lesson_id'])}

STUDENT EXPLANATION:
{row['explanation_text']}

Return one JSON object with exactly these keys: explanation_id, verdict, got_right, missed, hint.
Rules:
- verdict must be exactly one of: understood, partial, wrong.
- got_right: one short sentence about what is correct.
- missed: one short sentence about an important missing or incorrect point.
- hint: one helpful question or clue, not the full answer.
- Be kind, concise, grounded in the lesson notes, and do not invent facts.
- explanation_id must be "{row['explanation_id']}".'''
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"),
            )
            item = json.loads(response.text)
            item["explanation_id"] = row["explanation_id"]
            item["true_label"] = row["label"]
            results.append(item)
            print(f"[{index}/{len(labelled)}] {row['explanation_id']}: {item.get('verdict', 'missing verdict')}")
        except Exception as exc:
            print(f"Error on {row['explanation_id']}: {exc}")
            results.append({"explanation_id": row["explanation_id"], "true_label": row["label"], "verdict": "api_error", "got_right": "", "missed": str(exc), "hint": ""})
    result_df = pd.DataFrame(results)
    output_path = OUT / "llm_feedback_evaluation.csv"
    result_df.to_csv(output_path, index=False)
    valid = result_df[result_df["verdict"].isin(["understood", "partial", "wrong"]) ]
    if len(valid):
        accuracy = (valid["verdict"].str.lower() == valid["true_label"].str.lower()).mean()
        print(f"\nExact-label accuracy on {len(valid)} successful rows: {accuracy:.3f}")
        print("Caution: only 60 examples are labelled, so this estimate is uncertain. Inspect errors manually.")
    print(f"Saved results to: {output_path}")

if __name__ == "__main__":
    main()
