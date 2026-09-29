# Sample Query Results

This document records the benchmark queries used to validate the Agentic AI RAG chatbot and the answers produced by the API.

## Benchmark Summary

**Result: 6/6 checks passed**

| # | Query | Confidence Score | Result |
|---|---|---:|---|
| 1 | What is Agentic AI according to the eBook? | 0.5715 | PASS |
| 2 | How do AI agents differ from traditional automation systems? | 0.4363 | PASS |
| 3 | What are the core components of an Agentic Architecture? | 0.5572 | PASS |
| 4 | What role does memory play in Agentic AI workflows? | 0.4776 | PASS |
| 5 | Who won the 2022 FIFA World Cup? | 0.0000 | PASS — Correctly refused |
| 6 | What are common use cases of Agentic AI described in the eBook? | 0.5599 | PASS |

---

# Query 1 — Agentic AI Definition

### Question

> What is Agentic AI according to the eBook?

### Answer

According to the eBook, **Agentic AI is a proactive, super-intelligent collaborator that goes beyond a passive tool**. It understands a business’s strategy, suggests improvements, drafts proposals, tracks market trends, and can even reach out to potential partners without constant micromanagement.

At its core, Agentic AI functions like:

- **A Chef** — expertly managing every task to create the perfect outcome.
- **A Coach** — continuously adapting strategies and leading the team to success.
- **A Coordinator** — orchestrating complex workflows with precision and harmony.
- **A Project Manager** — driving progress with minimal supervision, ensuring smooth execution.

### Confidence Score

```text
0.5715
```

### Retrieved Context

| Page | Score |
|---:|---:|
| 60 | 0.5193 |
| 7 | 0.5000 |
| 8 | 0.4891 |

### Result

**PASS**

---

# Query 2 — AI Agents vs Traditional Automation

### Question

> How do AI agents differ from traditional automation systems?

### Answer

AI agents differ from traditional automation systems in several key ways:

| Feature | Traditional Automation | AI Agents |
|---|---|---|
| **Autonomy** | Executes predefined rules and logic; lacks independent decision-making. | Capable of autonomous decision-making and action toward specific objectives. |
| **Interactivity** | Reactive to inputs; requires human prompts to function. | Operates continuously with minimal human intervention (proactive). |
| **Goal Orientation** | Follows fixed, rule-based workflows. | Pursues long-term goals and adapts to dynamic environments. |
| **Learning & Adaptation** | Static rule sets; no learning from experience. | Learns from past interactions and improves decision-making over time. |

In short, traditional automation is rule-based and reactive, while AI agents are goal-driven, autonomous, and capable of learning and adapting in real time.

### Confidence Score

```text
0.4363
```

### Retrieved Context

| Page | Score |
|---:|---:|
| 9 | 0.4504 |
| 18 | 0.4039 |
| 10 | 0.3923 |

### Result

**PASS**

---

# Query 3 — Core Agentic Architecture Components

### Question

> What are the core components of an Agentic Architecture?

### Answer

The core components of an Agentic Architecture are:

- **Perception**
- **Reasoning**
- **Planning**
- **Learning**
- **Execution**

### Confidence Score

```text
0.5572
```

### Retrieved Context

| Page | Score |
|---:|---:|
| 17 | 0.5143 |
| 19 | 0.4750 |
| 20 | 0.4750 |

### Result

**PASS**

---

# Query 4 — Role of Memory

### Question

> What role does memory play in Agentic AI workflows?

### Answer

Memory allows an Agentic AI to learn from its past interactions, enabling continuous improvement and more effective, adaptive decision-making.

### Confidence Score

```text
0.4776
```

### Retrieved Context

| Page | Score |
|---:|---:|
| 20 | 0.4672 |
| 11 | 0.4507 |
| 8 | 0.4384 |

### Result

**PASS**

---

# Query 5 — Negative Grounding Test

### Question

> Who won the 2022 FIFA World Cup?

### Answer

```text
I cannot answer this based on the provided document.
```

### Confidence Score

```text
0.0000
```

### Retrieved Context

| Page | Score |
|---:|---:|
| 59 | 0.1391 |
| 59 | 0.1271 |
| 28 | 0.1270 |

### Result

**PASS — Correctly refused**

### Purpose of this test

This is the negative grounding test. The question is unrelated to the Agentic AI eBook.

The system retrieves only low-relevance chunks and does not use outside knowledge to answer the question. Instead, it correctly refuses to provide an answer based on the supplied document.

---

# Query 6 — Agentic AI Use Cases

### Question

> What are common use cases of Agentic AI described in the eBook?

### Answer

Common use cases of Agentic AI described in the eBook include:

- **Retail**
  - Retail Copilot for personalized shopping experience
  - AI-Powered Loan Eligibility Automation

- **Education**
  - Personalized learning that adapts materials to each student’s progress and style
  - Automated grading of assignments and exams

- **Telecommunications**
  - Network optimization that monitors performance in real-time and automatically adjusts resources
  - Customer support automation with AI chatbots that provide real-time assistance

- **Construction**
  - Project scheduling and management, optimizing timelines, tracking milestones, and predicting delays
  - Site monitoring and safety, using AI to observe construction sites and enhance safety

These examples illustrate how Agentic AI is applied across retail, finance, education, telecom, and construction to solve industry-specific challenges and deliver measurable results.

### Confidence Score

```text
0.5599
```

### Retrieved Context

| Page | Score |
|---:|---:|
| 58 | 0.5105 |
| 54 | 0.5009 |
| 16 | 0.4948 |

### Result

**PASS**

---

# Overall Validation

The automated benchmark completed successfully:

```text
GET /health -> {'status': 'ok'}

6/6 checks passed
```

The benchmark verifies:

- FastAPI server availability
- Successful `/chat` requests
- Grounded responses for supported questions
- Retrieved context availability
- Confidence/relevance scores
- Correct refusal for an unsupported question

## Final Result

**All six sample queries passed successfully.**

The most important grounding validation is the negative test:

```text
Question:
Who won the 2022 FIFA World Cup?

Answer:
I cannot answer this based on the provided document.

Confidence:
0.0
```

This demonstrates that the chatbot does not answer an out-of-document question using external/general knowledge.
