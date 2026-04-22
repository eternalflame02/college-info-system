---
name: dehuman-latex
description: >
  Use this skill whenever the user wants to reduce plagiarism, AI-detection scores, or
  AI-writing patterns in a LaTeX (.tex) document — particularly academic reports, seminar
  reports, mini-project reports, theses, or any university submission. Triggers include:
  "reduce plagiarism", "remove AI writing", "humanize my report", "make this bypass
  Turnitin/GPTZero/Originality", "rewrite for submission", "lower AI score", "edit my .tex
  file", or any request to make LaTeX content sound more human/natural for academic
  submission. ALWAYS use this skill when a .tex file or LaTeX content is involved and the
  goal is to change how the text reads. The skill covers the full pipeline: parse LaTeX
  safely, rewrite prose, verify LaTeX integrity, and return a clean diff-ready output.
---

# DeHuman-LaTeX Skill

Rewrites the **prose sections** of a LaTeX academic report to eliminate AI-writing
patterns and plagiarism signals — while keeping all LaTeX commands, citations,
math, figures, tables, and technical facts 100% intact.

---

## Core Principle

> **Touch only prose. Never touch structure.**

LaTeX documents have two layers:
1. **Markup layer** — commands, environments, labels, refs, math, citations → **NEVER modify**
2. **Prose layer** — natural language sentences inside paragraphs → **ALWAYS rewrite**

The agent's job is to surgically rewrite layer 2 while leaving layer 1 byte-perfect.

---

## Step 0 — Read the Reference Files First

Before processing any file, read:
- `references/latex-safety-rules.md` — mandatory LaTeX-safe editing constraints
- `references/humanization-patterns.md` — rewriting strategies and banned phrases

---

## Step 1 — Parse and Classify the File

When given a `.tex` file (full file at once), scan it and mentally tag every line:

| Tag | Description | Action |
|-----|-------------|--------|
| `PROSE` | Natural language inside `\begin{document}` paragraphs | Rewrite |
| `SKIP` | Commands, environments, math, citations, labels | Leave untouched |
| `CAPTION` | `\caption{...}` content | Rewrite lightly (1-2 word swaps only) |
| `ABSTRACT` | Inside `abstract` environment | Rewrite (high priority — most scrutinised) |

**SKIP zones (never rewrite):**
- Preamble (everything before `\begin{document}`)
- `\begin{equation}` / `\[...\]` / inline `$...$`
- `\cite{...}`, `\ref{...}`, `\label{...}`, `\url{...}`
- `\begin{table}`, `\begin{figure}`, `\begin{lstlisting}`, `\begin{verbatim}`
- Any command arguments that are not prose: `\section{}`, `\subsection{}`, `\textbf{}` used as labels
- Comment lines starting with `%`
- `\begin{itemize}` / `\begin{enumerate}` bullet text that is a technical term or code

**Note on section titles:** Do NOT rewrite `\section{...}` or `\subsection{...}` titles — these are structural and often referenced by `\ref{}`.

---

## Step 2 — Identify High-Risk Prose

Before rewriting, flag prose that is highest risk for detection:

1. **AI-pattern sentences** — opening with "In this paper/chapter/section", "It is worth noting", "Furthermore", "Moreover", "In conclusion", "This approach leverages", "state-of-the-art", "robust", "seamlessly", "straightforward"
2. **Plagiarism-risk sentences** — sentences that are very close to standard textbook or Wikipedia phrasing for well-known concepts (e.g., definitions of RAG, Transformers, KGs)
3. **Uniform sentence length** — paragraphs where every sentence is 20–30 words long (a hallmark of AI writing)
4. **Passive-heavy paragraphs** — excessive use of "is used", "is performed", "was conducted"

---

## Step 3 — Rewrite Rules (Moderate Mode)

Apply these rules to all `PROSE` tagged sections:

### ✅ DO
- **Vary sentence length**: Mix short punchy sentences (8–12 words) with longer ones (25–35 words). Never let three consecutive sentences be the same approximate length.
- **Use first-person active voice** where natural for a student report: "We propose...", "The system processes...", "Our approach differs from..."
- **Break up compound sentences**: Split long sentences joined by "which", "that", "and" into two separate sentences.
- **Use field-specific jargon naturally**: The documents are CSE/NLP focused — terms like "embedding", "retrieval", "graph traversal", "inference" should appear without being over-explained.
- **Add minor hedging where appropriate**: "to a reasonable degree", "in most cases", "under the tested conditions" — sounds like a real student hedging claims.
- **Vary paragraph openings**: Never start two consecutive paragraphs with the same word/phrase.
- **Use occasional contractions in non-formal prose** only in introductions/conclusions where register is slightly informal.
- **Preserve all technical claims and numbers exactly**: Accuracy metrics, model names, dataset sizes, hop counts — do not change.
- **Reorder minor supporting clauses** within sentences when doing so doesn't change meaning.

### ❌ DO NOT
- Change any technical term, acronym, model name, or dataset name
- Change any number, percentage, metric, or formula reference
- Add new claims or remove existing claims
- Change the logical flow between sentences (only the surface form)
- Introduce new citations or remove existing `\cite{}` tags
- Modify anything inside `\begin{...}` / `\end{...}` environments (only the prose between them)
- Use words from the banned list in `references/humanization-patterns.md`
- Make the text sound informal or conversational (this is an academic report)

---

## Step 4 — Output Format

Return the **complete rewritten `.tex` file** — not a diff, not a partial patch. The user compiles manually in Overleaf, so the output must be a drop-in replacement.

**Output format:**
```
[REWRITE COMPLETE]
— Sections modified: <list>
— Sentences rewritten: ~<count>
— LaTeX commands preserved: <confirm>
— High-risk phrases removed: <list top 5>

<full .tex content below>
```

Then output the complete `.tex` content inside a code block with `latex` syntax highlighting.

---

## Step 5 — Self-Check Before Outputting

Run this checklist mentally before returning the output:

- [ ] Every `\cite{}`, `\ref{}`, `\label{}` is byte-identical to original
- [ ] Every math expression `$...$`, `\[...\]`, `equation` environment is byte-identical
- [ ] No section or subsection title was changed
- [ ] The preamble is byte-identical
- [ ] No new technical claims were introduced
- [ ] No existing technical claims were removed
- [ ] All numbers, metrics, and percentages are unchanged
- [ ] Abstract has been rewritten (highest scrutiny zone)
- [ ] Introduction chapter has been rewritten (second highest scrutiny zone)
- [ ] No three consecutive sentences start with the same word
- [ ] No banned AI phrases remain (check references/humanization-patterns.md)

---

## Handling Large Files

If the `.tex` file is very long (>500 lines of prose), process it in logical chapters:
1. Abstract + Chapter 1 (Introduction)
2. Chapter 2 (Literature Review / Background)
3. Chapter 3+ (Methodology, Results, etc.)
4. Conclusion + References section prose (if any)

But always output the **full file** at the end — never partial.

---

## Special Cases

**Tables with prose cells**: Only rewrite cells that contain complete sentences. Leave column headers, numbers, and code strings alone.

**Itemize/enumerate lists**: Rewrite only if the bullet item is a full prose sentence. Leave short technical labels (e.g., "Training-Free Operation") untouched.

**Footnotes `\footnote{}`**: Treat as prose — rewrite.

**`\textit{}` and `\textbf{}`**: Preserve the command but may rewrite the text inside if it is a prose fragment (not a technical term being emphasised).
