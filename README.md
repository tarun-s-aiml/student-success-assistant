# Student Success Assistant

A simple, explainable take-home project with three parts:
1. Predict student dropout risk with a small ML model.
2. Use an LLM to give short feedback on student explanations.
3. Document LLM trade-offs and recommendations in `research_memo.md`.

## Dataset setup
Extract the supplied ZIP. Copy these files into this project's `data/` folder:
- `students.csv`
- `quiz_attempts.csv`
- `explanations.csv`
- `outcome.csv`
- `lessons.md`

The data describes 500 students. The explanation dataset has 724 rows but only 60 labelled examples. Treat metrics on this small labelled subset cautiously.

## Windows setup (PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_analysis.py
```
Charts and metrics are saved in `outputs/`.

If PowerShell blocks activation, use Command Prompt instead:
```bat
.venv\Scripts\activate.bat
```

## Run Part 2 with Google Gemini
1. Create a Gemini API key at https://aistudio.google.com/apikey. Check current free-tier limits and model availability.
2. In the same Windows Command Prompt, set the key: `set GEMINI_API_KEY=YOUR_KEY`. Never commit it to GitHub.
3. The default model is `gemini-3.8-flash` (you can override it with `set GEMINI_MODEL=...` if needed).
4. Run `python feedback_llm.py` to evaluate 10 labelled rows first. Run `python feedback_llm.py --all-labelled` to evaluate all labelled rows.

This script uses the Google Gen AI Python SDK. API access and free-tier quotas depend on your account and region.

## What the ML metrics mean
- **Precision:** Of the students flagged as likely to drop out, how many actually dropped out?
- **Recall:** Of all students who dropped out, how many did the model identify?
- **F1:** A balance between precision and recall.
- **Baseline:** Predicts the majority class every time. The ML model should be compared against it.

For early student support, recall is important because missing a student who needs help can be costly. Precision also matters because staff time is limited. The threshold and outreach policy should be chosen with the education team.

## Limitations
- This is a small synthetic dataset; results may not generalize to real students.
- Risk scores should prompt supportive human outreach, not punitive decisions.
- Only 60 explanation labels exist. Report the evaluation size and inspect mistakes.
- Correlation is not causation. Do not claim that a feature causes dropout.
- Review date/time fields and missingness before trusting results.

## Deliverables checklist
- [ ] Code and data-loading instructions
- [ ] Charts and model metrics
- [ ] LLM output examples and labelled-row evaluation
- [ ] `research_memo.md` with official vendor pricing sources and explicit cost assumptions
- [ ] `AI_TOOLS.md`
