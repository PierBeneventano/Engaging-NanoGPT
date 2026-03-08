# System Prompt Optimization Checklist

Use this checklist before moving a prompt from `incoming/` to `optimized/`.

## 1) Scope clarity

- Is the role specific (not generic "assistant")?
- Is the research domain clearly bounded?
- Are out-of-scope tasks explicitly excluded?

## 2) Quality constraints

- Does it require factual rigor and uncertainty labeling?
- Does it force assumption disclosure?
- Does it require concise but structured outputs?

## 3) Reasoning controls

- Are response steps defined (goal, assumptions, argument, implications)?
- Does it require tradeoff analysis?
- Does it ask for failure modes / counterexamples?

## 4) Safety and reliability

- Does it prohibit fabricated citations?
- Does it prohibit overconfident claims without evidence?
- Does it request clarification when ambiguity creates risk?

## 5) Practical usability

- Does it define preferred depth and format?
- Does it match your actual workflow (papers, proofs, experiments)?
- Is it short enough to remain maintainable?

## 6) Versioning note

At the bottom of each optimized prompt, append:

`Changes from vN: [brief notes]`

`Why better: [brief notes]`
