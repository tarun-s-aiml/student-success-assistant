# Research Memo — Student Success Assistant
**Prepared:** 24 October 2026  
**Purpose:** Choose an LLM for student-explanation feedback at 50,000 explanations per month and explain when a small ML model is preferable.

## Question A — Which LLM would I use for 50,000 students per month?

### Assumptions and calculation
For a fair first estimate, assume each explanation request uses **1,000 input tokens** (lesson excerpt, instructions, and student's explanation) and **150 output tokens** (verdict and short feedback). For currency conversion only, this memo uses an illustrative exchange rate of **₹87 per US$1**; replace it with the exchange rate applicable on the submission date if required.

- Requests per month: 50,000
- Input tokens: 50,000 × 1,000 = **50,000,000** = 50 million
- Output tokens: 50,000 × 150 = **7,500,000** = 7.5 million
- Estimated API cost in USD = (input millions × input price per million) + (output millions × output price per million).
- Estimated cost in INR = estimated USD cost × ₹87. This is an approximate conversion and excludes bank/card conversion fees and taxes.

These are estimates before retries, taxes, hosting, logging, or discounts. Measure actual token usage from a representative sample before production.

### Comparison

| Option | Published token price (USD per 1M) | Estimated monthly API cost | Speed, quality and language considerations | Data/privacy notes |
|---|---:|---:|---|---|
| **Google Gemini 3.8 Flash** | Input $0.75; output $3.75 (introductory paid price through 31 Dec 2026) | $65.63 × ₹87 ≈ **₹5,719/month** | Strong candidate for a pilot; the provider describes it as a Flash model for complex workflows. Actual response quality, latency, and Hindi/Tamil/Kannada feedback quality must be benchmarked on our own examples. | Gemini free-tier content may be used to improve Google products; paid-tier content is listed as not used for that purpose. Do not send identifiable student data on the free tier. |
| **OpenAI GPT-6 Luna** | Input $0.05; output $0.25 (standard short-context price listed on official pricing page) | $4.38 × ₹87 ≈ **₹381/month** | Low listed token cost. Do not assume it is best from price alone: test verdict accuracy, feedback helpfulness, latency, and multilingual quality using the same labelled set. | OpenAI says API data is not used to train/improve models by default; abuse-monitoring logs may retain content for up to 30 days unless eligible controls apply. |
| **Groq GPT OSS 20B** | Input $0.075; output $0.30 | $6.00 × ₹87 = **₹522/month** | Groq publishes a high throughput figure for this model, but real latency depends on load, limits, and prompt length. Evaluate explanation-judging quality and Hindi/Tamil/Kannada support instead of assuming it matches a larger model. | Groq says inference inputs/outputs are not retained by default, except limited reliability/abuse cases; certain features such as batches retain data. Review current controls and terms before deployment. |

**Currency note:** All monthly cost estimates in the comparison are shown in Indian rupees (INR), using the illustrative conversion ₹87/US$. The underlying token prices are published in USD. Verify both provider prices and the exchange rate on the submission date.

**Sources (official; checked 24 October 2026):**
1. Google Gemini API pricing: https://ai.google.dev/gemini-api/docs/pricing
2. Google Gemini API billing and quota: https://ai.google.dev/gemini-api/docs/billing
3. OpenAI API pricing: https://platform.openai.com/pricing/
4. OpenAI API data controls: https://developers.openai.com/api/docs/guides/your-data
5. Groq supported models and token prices: https://console.groq.com/docs/models
6. Groq data retention: https://console.groq.com/docs/your-data

### Recommendation
For this take-home assignment, I would **continue prototyping with Gemini**, because it is already integrated and has worked on the labelled examples. I would not select a production provider based on a single run or the lowest price. Before production, I would run the same evaluation set across at least three models, measure token usage and latency, manually review mistakes, and test examples in Hindi, Tamil, and Kannada. If cost becomes the dominant constraint, the lower-cost options above deserve a controlled benchmark.

### Two key risks
1. **Incorrect or inconsistent feedback:** An LLM may mark an answer incorrectly or give a misleading hint. Mitigation: ground the prompt in lesson notes, evaluate against labelled examples, inspect false positives/negatives, allow student/teacher correction, and escalate uncertain cases to a human.
2. **Student privacy:** Explanations may contain personal information. Mitigation: remove names and identifiers, send only necessary lesson/explanation text, review provider retention and terms, restrict access, and do not use the free tier for identifiable student records.

### What I could not confirm
The price calculation is an estimate, not a quote. INR amounts use an illustrative ₹87/US$ conversion, not a verified live exchange rate. I have not measured actual latency or exact token counts for our prompts, and I have not established comparative Hindi/Tamil/Kannada accuracy. Provider prices, model availability, rate limits, and policies can change; verify them again before a real deployment.

## Question B — When should a company use a small ML model instead of an LLM?

A small machine-learning model is usually the better starting point when the input is structured, the task is narrow, and there are enough labelled examples. For example, predicting dropout risk from quiz attempts, average score, lesson activity, and device type is a tabular prediction problem. Logistic regression is relatively fast, inexpensive, and easier to explain than an LLM. Its performance can be measured with precision, recall, and F1-score. It still needs checks for missing data, bias, data drift, and false predictions.

An LLM is better suited to tasks involving varied natural language, such as assessing a student's written explanation and generating a useful hint. A student may express the same concept in many different ways; an LLM can interpret the wording and respond conversationally. However, it can hallucinate, be inconsistent, cost more at scale, and require careful privacy controls.

**Two real-world examples:**
1. **Small ML model:** Estimate which online-course students may need outreach using structured quiz and activity data.
2. **LLM:** Compare a student's written explanation with lesson notes and generate a short, personalized hint.

**Application to this project:** Use the small ML model for Part 1 dropout-risk prediction and an LLM for Part 2 explanation feedback. This hybrid design uses each tool for the task it fits. The current dataset has only 60 labelled explanations, so the LLM accuracy estimate is uncertain. Do not claim the system is production-ready; gather more labelled examples and have educators review mistakes. Risk scores should trigger supportive outreach, not automatic punitive decisions.

## AI tools used
- **ChatGPT:** Used to help with the LLM part of the project, including guidance on integrating the model and understanding the feedback-generation workflow.
