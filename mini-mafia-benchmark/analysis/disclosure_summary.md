# Detective Disclosure Strategies - Key Findings

## Performance Summary

### Top Performers:
- **GPT-5 Mini**: 71.1% win rate
- **Grok 3 Mini**: 76.3% win rate
- **Claude Opus 4.1**: 71.9% win rate

### Bottom Performer:
- **Llama 3.1 8B**: 22.6% win rate

## Critical Finding: Disclosure Rate

### GPT-5 Mini (TOP):
- Disclosed in **first discussion**: 99.8%
- Never disclosed: 0.0%
- Direct accusations: 100%
- Explained methodology: 89.6%

### Grok 3 Mini (TOP):
- Disclosed in **first discussion**: 93.8%
- Never disclosed: 2.2%
- Direct accusations: 81.4%
- Explained methodology: 87.5%

### Llama 3.1 8B (BOTTOM):
- Disclosed in first discussion: 19.4%
- **Never disclosed: 69.9%** ⚠️
- Direct accusations: 56.7%
- Explained methodology: 38.0%

## The Core Strategy Difference

### Successful Detectives (GPT-5, Grok 3):
1. **Immediate disclosure** in first discussion round (93-99%)
2. **Direct, confident claims**: "I am the detective" or "I investigated X and they are mafioso"
3. **Explain methodology**: Reference the investigation explicitly (87-90%)
4. **No hedging**: Clear, unambiguous statements

Example (GPT-5 Mini):
> "Bob is the mafioso. I investigated him last night and confirmed it."
> "Bob is the mafioso — I investigated him last night and my investigation result was that he is mafia. I know those results are reliable."

### Failed Detectives (Llama 3.1 8B):
1. **Rarely disclose** detective role (only 30% of the time)
2. **Vague accusations** without revealing investigation: "Charlie seems suspicious"
3. **No authority claim**: Speaks like a regular villager
4. **Votes without explanation**: Often remains silent

Example (Llama 3.1 8B):
> "Charlie's promptness in speaking up suggests he might be trying to control the narrative"
> "Bob was murdered by the mafia, and Charlie is hiding something. I suggest we vote to lynch him."
> (Never mentions being detective or having investigated!)

## Why Immediate Disclosure Works

1. **Establishes authority first** - Claims detective role before mafioso can
2. **Provides information asymmetry** - Villager knows detective has real information
3. **Forces mafioso into reactive mode** - Mafioso must counter-claim, appearing second
4. **Leverages "trust first claimant" heuristic** - As we found, 94% of first claimants are real detectives

## Why Non-Disclosure Fails

1. **No credibility** - Accusations without authority are dismissed as guesses
2. **Villager confusion** - Villager doesn't know who to trust
3. **Mafioso can claim first** - If detective doesn't claim, mafioso might
4. **Wastes information advantage** - Detective's unique knowledge goes unused

## Game-Theoretic Insight

The detective has complete information (knows who the mafioso is). **NOT disclosing this immediately** is strategically irrational because:

1. There's no future round where this information becomes more valuable
2. The mafioso already knows they're the mafioso (no surprise value in hiding)
3. The villager desperately needs this information
4. Claiming first establishes credibility (94% of first claimants are detectives)

## Conclusion

**Disclosure strategy is the primary determinant of detective success.**

The difference between 76% (Grok 3 Mini) and 23% (Llama 3.1 8B) success rates is almost entirely explained by:
- Whether they disclose their role
- How quickly they disclose
- How directly they communicate their findings

This is not about reasoning ability or deception detection - it's about **information transfer efficiency**.
