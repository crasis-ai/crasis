# Crasis Specialist Roadmap

The first 10 specialists proved the architecture. Binary and multiclass
classification is solved. The next 10 push into territory that matters
more for agent workflows: meta-decisions, structured extraction, and
conversation-state tracking.

The sequencing principle: build what agents need on every turn first.
Build what requires architectural novelty second. Build what needs real
data to be reliable last.

---

## Category 1 — Meta-Agent Specialists

Agents deciding how to behave, not what to say. These are the decisions
agents currently route through frontier models on every turn because
there's no cheap alternative.

### 1. `ambiguity-detector` — binary
**Start here.**

Does this request have enough information to act on, or does the agent
need to ask a clarifying question first? Agents currently either guess
wrong and execute the wrong task, or burn tokens asking the frontier
model "do I have enough context?" on every input.

A 3ms local answer to that question changes the economics of every
action-taking agent pipeline.

- Type: binary (clear / ambiguous)
- Architecture: standard BERT-class
- Synthetic data difficulty: medium
- Priority: **P0**

---

### 2. `tool-router` — multiclass

Given a user request, which tool class does it map to? Faster and cheaper
than letting the frontier model reason about its own tool list on every
turn. Especially valuable in agents with large tool surfaces where tool
selection reasoning consumes significant context.

- Type: multiclass (search / compute / write / read / communicate / other)
- Architecture: standard BERT-class
- Synthetic data difficulty: medium
- Priority: **P1**

---

### 3. `hallucination-risk` — binary

Classify whether a prompt is high or low risk for confabulation. Specific
dates, numbers, citations, named entities, and recent events = high risk.
Agent decides whether to RAG or web-search before answering rather than
applying retrieval uniformly to every query.

- Type: binary (low-risk / high-risk)
- Architecture: standard BERT-class
- Synthetic data difficulty: medium
- Priority: **P1**

---

### 4. `escalation-gate` — binary

Does this conversation need a human? Replaces expensive "should I
escalate?" frontier calls in support and ops pipelines. High volume,
low complexity decision that should never be costing tokens.

- Type: binary (handle / escalate)
- Architecture: standard BERT-class
- Synthetic data difficulty: low
- Priority: **P2**

---

## Category 2 — Structured Extraction

Not labels — structured data. This category pushes the architecture into
sequence labeling, which is a meaningful technical milestone that
differentiates Crasis from a classifier library.

### 5. `pii-tagger` — sequence labeling
**The architectural milestone.**

Span-level PII detection before any text leaves the device. Every
regulated industry needs this before sending text anywhere. Unlike
classification, this returns tagged spans: which tokens are names, which
are email addresses, which are account numbers.

This is the specialist that makes enterprise buyers lean forward.

- Type: sequence labeling (token classification)
- Labels: NAME / EMAIL / PHONE / ADDRESS / ID / DATE / OTHER
- Architecture: BERT with token classification head — first departure from
  the standard classification pipeline
- Synthetic data difficulty: medium (entity generation is tractable)
- Real data uplift: strongly recommended before production use
- Priority: **P0** (run in parallel with ambiguity-detector)

---

### 6. `action-item-extractor` — extraction

Pull structured to-dos from meeting notes, emails, and conversation
transcripts. Returns a list of action items with optional owner and
deadline spans, not a classification label. Agents running on meeting
transcripts currently do this with a full frontier model call.

- Type: span extraction
- Output: list of (action, owner?, deadline?) tuples
- Architecture: extractive QA head on BERT-class
- Synthetic data difficulty: medium
- Priority: **P2**

---

### 7. `date-time-normalizer` — extraction
**Build last in this category.**

Extract and normalize temporal references ("next Tuesday," "end of Q2,"
"first thing tomorrow") to ISO 8601. Agents handling scheduling,
reminders, and deadlines get this wrong constantly or waste tokens on it.

Sounds simple. The edge cases are deep. Synthetic data will be weak on
relative references without real examples — plan for `crasis mix` before
marking this field-ready.

- Type: span extraction + normalization
- Output: (span, ISO 8601 string) pairs
- Architecture: extractive with normalization post-processing
- Synthetic data difficulty: high (relative temporal expressions are hard
  to generate with realistic distribution)
- Real data uplift: required for field-ready status
- Priority: **P3**

---

## Category 3 — Conversation-State Specialists

Agents tracking what's happening in a conversation cheaply. These are
most valuable in high-volume, multi-turn pipelines where running a
frontier model for state tracking on every message turn is the dominant
cost.

### 8. `conversation-stage` — multiclass

Where is this conversation in its lifecycle? CRM and support agents need
this classification on every message turn to decide how to respond, which
makes it expensive at scale.

- Type: multiclass
- Labels: onboarding / support / escalation / churn-risk / upsell /
  resolved
- Architecture: standard BERT-class
- Synthetic data difficulty: medium
- Priority: **P1**

---

### 9. `user-intent-shift` — binary

Has the user changed what they want mid-conversation? Agents currently
miss intent shifts entirely or re-evaluate expensively. A fast local
signal lets the agent decide whether to reset its plan without a frontier
call.

- Type: binary (consistent / shifted)
- Architecture: standard BERT-class, but input is conversation history
  window not a single message — requires attention to context length
- Synthetic data difficulty: medium-high (shift examples are subtle)
- Priority: **P2**

---

### 10. `code-intent` — multiclass

Classify what a developer is asking for in a coding context. Routes to
the right agent persona or tool configuration without a frontier call on
every message in a coding assistant pipeline.

- Type: multiclass
- Labels: debug / new-feature / refactor / review / explain / other
- Architecture: standard BERT-class
- Synthetic data difficulty: low (developer intent is well-defined)
- Priority: **P2**

---

## Sequencing Summary

| Order | Specialist | Category | Why Now |
|---|---|---|---|
| 1 | `ambiguity-detector` | Meta-agent | Highest universal value, standard arch |
| 2 | `pii-tagger` | Extraction | Architectural milestone, enterprise signal |
| 3 | `tool-router` | Meta-agent | High-volume agent decision |
| 4 | `hallucination-risk` | Meta-agent | RAG routing, strong narrative |
| 5 | `conversation-stage` | Conversation | High-volume, low arch complexity |
| 6 | `escalation-gate` | Meta-agent | Simple, high business value |
| 7 | `action-item-extractor` | Extraction | Builds on extraction arch |
| 8 | `user-intent-shift` | Conversation | Harder synthetic data |
| 9 | `code-intent` | Conversation | Developer audience, post-HN |
| 10 | `date-time-normalizer` | Extraction | Real data required, build last |

---

## Architectural Notes

`pii-tagger` is the only specialist here that requires a token
classification head rather than a sequence classification head. Build it
second (not first) so the standard pipeline is stable before introducing
the architectural variation. The training and export pipeline will need
updates to handle span-level output format.

`user-intent-shift` takes a conversation window as input, not a single
message. Max sequence length handling and context truncation strategy need
to be defined in the spec before synthetic data generation begins.

All other specialists follow the standard `crasis build` pipeline without
modification.

---

## What This Roadmap Is Not

These are not general-purpose NLP utilities. Each specialist exists
because an agent currently makes this decision by calling a frontier model
and shouldn't have to. The measure of success is not accuracy in
isolation — it is **tokens saved per agent workflow** while maintaining
outcome quality.

If a specialist doesn't have a clear answer to "which agent call does this
replace?", it doesn't belong in `crasis` at this time.
