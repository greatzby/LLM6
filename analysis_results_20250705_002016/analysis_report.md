# Transformer Composition Analysis Report
Generated: 2025-07-05 00:20:52

## Executive Summary

This analysis investigates three key questions about compositional learning in Transformers:
1. What enables early-stage compositional success?
2. What causes the forgetting of compositional ability?
3. How does mixed training prevent forgetting?

## 1. Early Stage Success Analysis (5k iterations)

### original
- **S2 Cohesion**: 0.854
- **S2 Bridge Score**: 0.6803026045362155
- **Probe Accuracy**: 100.00%
- **S2 Classification**: 1.0

### 5% mixed
- **S2 Cohesion**: 0.899
- **S2 Bridge Score**: 0.5457628478606542
- **Probe Accuracy**: 100.00%
- **S2 Classification**: 1.0

### 10% mixed
- **S2 Cohesion**: 0.906
- **S2 Bridge Score**: 0.5343873133261998
- **Probe Accuracy**: 77.78%
- **S2 Classification**: 0.5

## 2. Forgetting Mechanism Analysis

### original

**S2 Cohesion Degradation:**
- Iteration 5000: 0.854
- Iteration 25000: 0.897
- Iteration 50000: 0.902
- **Total degradation**: 0.048 (5.7%)

### 5% mixed

**S2 Cohesion Degradation:**
- Iteration 5000: 0.899
- Iteration 25000: 0.923
- Iteration 50000: 0.927
- **Total degradation**: 0.028 (3.1%)

### 10% mixed

**S2 Cohesion Degradation:**
- Iteration 5000: 0.906
- Iteration 25000: 0.900
- Iteration 50000: 0.907
- **Total degradation**: 0.001 (0.1%)

## 3. Effect of Mixed Training

### Final Performance Comparison (50k iterations)

| Model | S2 Cohesion | Bridge Score | Probe Accuracy |
|-------|-------------|--------------|----------------|
| original | 0.902 | 0.35111825267473856 | 1.0 |
| 5% mixed | 0.927 | 0.3920530279477437 | 0.9444444444444444 |
| 10% mixed | 0.907 | 0.42914868195851646 | 0.9444444444444444 |

## 4. Key Findings

### S2 Representation Drift
- **original**: Mean drift: 63.130 (±13.879)
- **5% mixed**: Mean drift: 65.213 (±7.934)
- **10% mixed**: Mean drift: 58.751 (±7.105)

## 5. Conclusions

1. **Early Success**: High S2 cohesion and bridge scores enable composition
2. **Forgetting**: S2 representations drift and lose their bridging role
3. **Mixed Training**: Maintains S2 cohesion and prevents representational collapse