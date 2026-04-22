# LaTeX Safety Rules

Critical constraints for editing .tex files without breaking compilation.

---

## Absolute No-Touch Zones

These patterns must never be modified under any circumstances:

### Preamble
Everything from the first line of the file up to and including `\begin{document}` is
off-limits. This includes `\documentclass`, `\usepackage`, `\newcommand`,
`\renewcommand`, `\input`, `\include`, `\geometry`, `\hypersetup`, and all
custom macro definitions.

### Math Environments
```
$...$              % inline math
$$...$$            % display math (deprecated but common)
\(...\)            % inline math
\[...\]            % display math
\begin{equation}   % numbered equation
\begin{align}      % multi-line align
\begin{gather}     % gather environment
\begin{multline}   % multline
\begin{cases}      % cases
```
Even if a math expression appears to contain words (e.g., `\text{subject}` inside
align), do not touch it.

### Cross-Reference Commands
```
\cite{key}
\citep{key}
\citet{key}
\ref{label}
\eqref{label}
\pageref{label}
\label{identifier}
\hyperref[label]{text}   % only preserve the label, text may be rewritten
\nameref{label}
```
The key/label arguments (inside `{}`) must be preserved exactly — including
capitalisation, underscores, and hyphens.

### Structural Commands
```
\section{...}
\subsection{...}
\subsubsection{...}
\paragraph{...}
\chapter{...}
\part{...}
```
Do not change section titles — they may be referenced elsewhere with `\ref`.

### Figure and Table Environments
```
\begin{figure}...\end{figure}
\begin{table}...\end{table}
\begin{tabular}...\end{tabular}
\begin{tabularx}...\end{tabularx}
\begin{longtable}...\end{longtable}
```
Inside these environments, only rewrite `\caption{...}` text (lightly — 1-2 word
swaps). Do not touch column specs, `\hline`, `\toprule`, `\midrule`,
`&`-separated cell content that is numerical or code-like.

### Code and Verbatim Environments
```
\begin{lstlisting}...\end{lstlisting}
\begin{verbatim}...\end{verbatim}
\begin{minted}...\end{minted}
\verb|...|
\texttt{...}   % treat as code — do not rephrase
```

### Bibliography
Everything inside `\begin{thebibliography}` or any `\bibliography{}` command
is off-limits. If using BibTeX/BibLaTeX, the `.bib` file is not your concern.

---

## Safe-to-Touch Zones

### Paragraph Prose
Any continuous natural-language text between commands, not inside a skip-zone
environment. Example:
```latex
This chapter presents the proposed system architecture. The design
consists of three primary modules, each responsible for a distinct
stage of the processing pipeline.
```
Both sentences above are safe to rewrite.

### Abstract Environment
```latex
\begin{abstract}
  [REWRITE THIS TEXT]
\end{abstract}
```
High priority rewrite zone.

### Acknowledgements
```latex
\chapter*{Acknowledgement}
  [REWRITE THIS TEXT — keep all proper names (people, institutions) exact]
```
Keep all names of people, departments, and institutions exactly as written.

### Footnotes
`\footnote{This is a footnote.}` — the prose inside the braces is rewriteable.

---

## Common LaTeX Pitfalls to Avoid

1. **Unbalanced braces**: Every `{` you open must be closed. If you're rewriting
   inside a `\textbf{...}` or `\emph{...}`, count the braces.

2. **Escaped characters**: LaTeX special characters `% $ & # _ ^ ~ { } \` have
   special meaning. If the original text has `\%` or `\$` in prose, preserve the
   backslash-escape — don't strip it.

3. **Tilde `~`**: In LaTeX, `~` is a non-breaking space (e.g., `Figure~\ref{fig:arch}`).
   Never remove or replace tildes that appear before `\ref`, `\cite`, `\eqref`.

4. **Em-dashes and en-dashes**: LaTeX uses `---` for em-dash and `--` for
   en-dash. Do not replace with Unicode `—` or `–`.

5. **Quotation marks**: LaTeX uses `` ` `` and `'` for quotes: `` ``word'' `` not `"word"`.
   Do not introduce Unicode quotes.

6. **Line breaks in prose**: LaTeX ignores single newlines in prose (they are treated
   as spaces). A blank line = new paragraph. Preserve blank lines between
   paragraphs. Do not introduce `\\` (forced line break) into prose.

7. **Comments**: Lines beginning with `%` are comments. Do not remove or rewrite
   them — they may contain author notes or disabled code.

8. **Hyphenation hints**: If original has `\-` inside a word (discretionary hyphen),
   preserve it.

---

## Quick Validation Checklist

After rewriting, verify:
- [ ] Brace count in rewritten sections is balanced
- [ ] No `~\ref`, `~\cite` tildes were removed
- [ ] No `---`, `--` replaced with Unicode dashes
- [ ] No `` ``...'' `` replaced with `"..."`
- [ ] All `\label{...}` identifiers are identical to original
- [ ] All `\cite{...}` keys are identical to original
- [ ] Preamble is byte-identical
