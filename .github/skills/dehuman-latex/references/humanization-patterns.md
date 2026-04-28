# Humanization Patterns & Banned Phrases

Reference for rewriting AI-generated academic prose into natural student writing.
Calibrated for Indian university B.Tech/M.Tech CSE reports (KTU / APJ Abdul Kalam
Technological University style).

---

## Part 1 — Banned AI Phrases

Remove or replace any of the following. They are the strongest AI-detection signals.

### Transition / Connective Overuse
| Banned | Replace with |
|--------|-------------|
| Furthermore, | Also, / Beyond this, / On top of that, |
| Moreover, | In addition, / As well, / Alongside this, |
| Additionally, | Also, / On a related note, |
| In conclusion, | To summarise, / Overall, / Taken together, |
| In summary, | In short, / Briefly, / To recap, |
| It is worth noting that | Note that / Notably, / It should be said that |
| It is important to note that | Importantly, / As a key point, |
| It is evident that | Clearly, / As can be seen, |
| As mentioned earlier | As noted above / As discussed / Previously |
| This paper/chapter/section presents | This work describes / The following section covers |
| This paper/chapter/section proposes | The proposed system / Here we describe |
| The remainder of this paper is organised as follows | The rest of this report is structured as follows |

### AI Buzzwords (use sparingly / replace in most contexts)
| Banned/Overused | Natural alternative |
|-----------------|-------------------|
| robust | reliable / effective / consistent |
| seamlessly | smoothly / directly / without friction |
| state-of-the-art | current best / leading / competitive |
| leverage (as a verb) | use / apply / draw on / exploit |
| utilise | use |
| facilitate | help / enable / support / allow |
| demonstrate | show / confirm / reveal |
| significant improvement | clear improvement / measurable gain / noticeable gain |
| novel approach | new approach / different strategy / this method |
| comprehensive | thorough / complete / detailed |
| straightforward | simple / direct / not complicated |
| harnessing | using / applying |
| cutting-edge | recent / modern / advanced |
| paradigm | approach / framework / method |
| inherently | naturally / by design / in essence |
| notably | importantly / in particular |
| pivotal | key / central / important |
| streamline | simplify / reduce overhead in |

### Formulaic Sentence Starters
Never start a paragraph with:
- "In the context of..."
- "It can be observed that..."
- "It should be noted that..."
- "As a result of..."
- "Due to the fact that..." → replace with "Because..."
- "In order to..." → replace with "To..."
- "With the aim of..." → replace with "To..."
- "The purpose of this [section/chapter] is to..."

---

## Part 2 — Humanization Techniques

### Sentence Length Variation
AI text is monotonous in rhythm. Break this by:

**Before (AI-like):**
> The proposed system integrates a knowledge graph with a retrieval-augmented generation framework to improve factual accuracy and reduce hallucinations in question answering tasks.

**After (human-like):**
> The proposed system integrates a knowledge graph with a RAG framework. This combination directly targets two recurring problems: factual inaccuracy and hallucination. Together, they limit the usefulness of LLMs in question answering.

Rule: Every 3–4 sentences, insert one that is ≤12 words.

---

### Active vs. Passive Voice
Passive voice is fine in methodology sections. But alternate:

**Over-passive (AI-like):**
> The model is trained on the MetaQA benchmark. The results are then evaluated using the Hit@1 metric. The performance is compared against three baseline configurations.

**Balanced (human-like):**
> We trained the model on the MetaQA benchmark and measured performance using Hit@1. Three baseline configurations serve as comparison points.

---

### First-Person vs. Third-Person
For a student report, limited first-person ("we", "our") is natural and reduces AI
signals. Use it especially in:
- Methodology descriptions: "We implemented...", "Our system uses..."
- Results discussion: "We observed that...", "The results suggest..."
- Avoid first-person in: Literature review facts, abstract (keep abstract third-person)

---

### Specificity over Generality
AI tends toward vague claims. Add specificity by referencing the actual content:

**Vague (AI-like):**
> The system demonstrates improved performance across all evaluated configurations.

**Specific (human-like):**
> On 2-hop and 3-hop questions, the full KG-RAG system outperforms the KAPING baseline by a clear margin — a gain that disappears for 1-hop questions, where decomposition adds unnecessary overhead.

---

### Hedging and Qualification
Real students hedge. AI is often falsely confident:

**Over-confident (AI-like):**
> This approach eliminates hallucinations and ensures accurate factual retrieval.

**Appropriately hedged (human-like):**
> This approach substantially reduces hallucination in practice, though the degree of improvement depends on the quality of the underlying knowledge graph.

---

### Paragraph Opening Variety
Rotate how paragraphs open. Avoid repeating the same structure:

Good openers:
- Start with a result/finding: "The retrieval hit rate for teaching queries reached 100%..."
- Start with a contrast: "Unlike purely embedding-based systems..."
- Start with a condition: "When the query involves multi-hop reasoning..."
- Start with a short declarative: "The knowledge graph serves two purposes here."
- Start with "This", "These", "Such" referring back to prior paragraph

Avoid:
- Starting 2+ consecutive paragraphs with "The"
- Starting with "In this..."
- Starting with a gerund phrase: "Utilising the..."

---

## Part 3 — Domain-Specific Notes for CSE/NLP Reports

These terms are technical and must NOT be rephrased or replaced:
- RAG, KG-RAG, KGQA, LLM, NLP, IR, CoT, ICL
- Model names: Mistral-7B, KAPING, Keqing, RRA, BART, DeepSeek-R1
- Dataset names: MetaQA, Mintaka, WebQSP, LC-QuAD
- Metrics: Hit@1, MRR, NDCG@5, Precision@k, Recall@k
- Library names: ChromaDB, SentenceTransformers, pdfplumber, BeautifulSoup, LangChain
- Architecture terms: breadth-first search, cosine similarity, dot-product similarity,
  vector embedding, semantic chunking, top-K retrieval

These terms can be placed in different sentence positions but their spelling and form
must not change.

---

## Part 4 — Indian Academic Register

KTU reports have a specific academic register. Calibrate rewrites to match:
- Formal but not pretentious
- Complete sentences (no bullet-point prose)
- Third-person preferred in abstract, literature review, and objective statements
- First-person ("we") acceptable in methodology and results
- British English spelling: "behaviour", "colour", "recognise", "modelling", "analyse"
- Avoid American-isms: use "whilst" occasionally, "amongst", "towards"

---

## Part 5 — Scoring Self-Check

After rewriting a passage, mentally score it on these axes (1=bad, 5=good):

| Axis | Question |
|------|----------|
| Rhythm | Do consecutive sentences vary in length? |
| Voice | Is passive/active balance natural? |
| Openers | Do paragraph openers vary? |
| Hedging | Are claims appropriately qualified? |
| Jargon | Are technical terms used correctly and not over-explained? |
| Banned phrases | Are all items from Part 1 absent? |

Aim for ≥4 on all axes before finalising.
