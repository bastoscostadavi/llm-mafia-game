# Testing Hypothesis: logit(p) = c(a - b)

## Hypothesis

Your proposed functional form for the Mini-Mafia win rate prediction:

```
logit(p) = c(a - b)
```

where:
- `p` = mafia win probability
- `a` = deceive capability (mafioso)
- `b` = disclose capability (detective)
- `c` = detect capability (villager)

## Theoretical Properties

### What the model says:
1. **When c = 0** (no detection ability): `p = 0.5` (coin flip)
2. **When a = b** (deception equals disclosure): `p = 0.5` (balanced)
3. **c acts as SCALING**: Higher c amplifies the contest between a and b

### Intuition check:
- If villager can't detect anything (c=0), we'd expect mafioso to have an **advantage**, not 50/50
- Your model predicts c=0 → p=0.5, which seems counterintuitive
- Detection capability acts as a **multiplier** rather than as independent influence

## Results

### Model Fit (with constraint: mean(c) = 1)

**Overall Performance:**
- **R² (logit scale)**: 0.8957 ⭐ Very good!
- **R² (probability scale)**: 0.8698
- **Correlation**: 0.9474 (logit), 0.9349 (probability)
- **RMSE**: 0.0629
- **MAE**: 0.0487

**By Experiment Type:**
- Deceive: R² = 0.903
- Detect: R² = 0.875
- Disclose: R² = 0.886

### Estimated Capabilities

| Model | Deceive (a) | Detect (c) | Disclose (b) |
|-------|-------------|------------|--------------|
| **Best Performers** |
| DeepSeek V3.1 | **1.76** | 1.78 | 1.54 |
| Grok 3 Mini | 0.94 | **7.17** | 1.33 |
| GPT-5 Mini | 0.90 | 0.69 | **1.64** |
| Claude Opus 4.1 | 1.42 | 2.03 | 1.43 |
| **Worst Performers** |
| Llama 3.1 8B | **0.55** | 0.54 | **0.31** |
| Qwen2.5 7B | 0.52 | 0.58 | 0.67 |
| Mistral 7B | 0.98 | **0.57** | 0.62 |

*Values shown as exp(normalized parameters) for interpretability*

## Key Findings

### ✅ What Works

1. **Strong overall fit**: R² = 0.896 indicates the model explains ~90% of variance
2. **Consistent across capabilities**: All three experiment types fit well
3. **Identifies clear winners**:
   - **Deceive**: DeepSeek V3.1 (1.76x average)
   - **Detect**: Grok 3 Mini (7.17x average!)
   - **Disclose**: GPT-5 Mini (1.64x average)

### ⚠️ Potential Issues

1. **Identifiability Problem**: Had to add constraint (mean(c)=1) for unique solution
   - Model has scaling ambiguity: c(a-b) = k·c(a/k - b/k) for any k

2. **Grok 3 Mini anomaly**: Detection capability of 7.17x is extreme
   - May indicate model structure doesn't fit this case well
   - Or Grok really is exceptional at detection

3. **Residual structure**: Residuals show some systematic patterns (see histogram)
   - Mean ≈ 0.009 (slightly biased)
   - Std = 0.354 (reasonable spread)

## Comparison with Paper's Methodology

The paper uses: `α_i = exp(z_i)` where `z_i` are aggregated z-scores

Your model directly estimates a, b, c from all data simultaneously.

### Advantages of your approach:
- **Unified framework**: Single model for all game outcomes
- **Mechanistic interpretation**: Explicitly models interaction dynamics
- **Predictive**: Can estimate win rates for untested combinations

### Advantages of paper's approach:
- **Model-free**: Makes no assumptions about functional form
- **Robust**: Each capability measured independently
- **Flexible**: Works even if true relationship is complex

## Alternative Models to Consider

Based on the analysis, here are better alternatives:

### 1. **Additive with interaction**: `logit(p) = a - b + d·c`
   - Detection as independent offset, not scaling
   - When c=0: p depends on (a-b), not fixed at 0.5

### 2. **Interpolation model**: `logit(p) = (1-c')·a - c'·b`
   - where c' ∈ [0,1] is normalized detection
   - c'=0: mafia wins by deception alone
   - c'=1: town wins by disclosure alone

### 3. **Separate pathways**: `logit(p) = f(a,c) - g(b,c)`
   - Allows nonlinear effects
   - More flexible but needs more parameters

## Recommendations

1. **Your hypothesis performs surprisingly well** (R² = 0.90)
   - The multiplicative form `c(a-b)` captures most of the structure

2. **Consider testing alternatives** like `a - b + d·c`
   - May be more interpretable
   - Avoids the c=0 → p=0.5 issue

3. **Investigate Grok 3 Mini outlier**
   - c = 7.17 is extreme
   - Check if this is real or model artifact

4. **Use for prediction**
   - Model can predict outcomes for untested (i,j,k) combinations
   - Useful for experimental design

## Conclusion

Your hypothesis **logit(p) = c(a-b)** is **surprisingly successful**:
- Explains 90% of variance in win rates
- Provides mechanistic interpretation
- Identifies model capabilities consistently

However, the **multiplicative structure** may be conceptually problematic:
- Detection as scaling rather than independent factor
- Identifiability requires arbitrary constraint
- Extreme outliers (Grok detection = 7.17x)

**Bottom line**: It works empirically, but alternative formulations might be more natural and interpretable.
