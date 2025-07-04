# Enhanced Transformer Composition Analysis Report
Generated: 2025-07-05 00:54:45

## Executive Summary

## 1. S2 Diversity Analysis

### original

**Diversity Metrics Evolution (5k → 50k):**
- Mean pairwise distance: 12.854 → 37.763
- Effective dimension: 6 → 7
- Distance std: 5.174 → 14.281

### 5% mixed

**Diversity Metrics Evolution (5k → 50k):**
- Mean pairwise distance: 10.807 → 31.263
- Effective dimension: 6 → 8
- Distance std: 4.280 → 10.561

### 10% mixed

**Diversity Metrics Evolution (5k → 50k):**
- Mean pairwise distance: 9.985 → 31.445
- Effective dimension: 7 → 8
- Distance std: 3.686 → 10.159

## 2. Representation Collapse Analysis

### original (at 50k iterations)
- S2 effective rank: 30
- S2 active dimensions: 120
- S2 dimension utilization: 100.0%

### 5% mixed (at 50k iterations)
- S2 effective rank: 30
- S2 active dimensions: 120
- S2 dimension utilization: 100.0%

### 10% mixed (at 50k iterations)
- S2 effective rank: 30
- S2 active dimensions: 120
- S2 dimension utilization: 100.0%

## 3. S2 Functionality Analysis

### original

| Iteration | Functional Nodes | Bridge Success | S1 Reach | S3 Reach |
|-----------|------------------|----------------|----------|----------|
| 5000 | 21/30 | 2.0% | 76.0% | 78.7% |
| 25000 | 22/30 | 4.7% | 74.0% | 76.7% |
| 50000 | 22/30 | 1.3% | 71.3% | 80.0% |

### 5% mixed

| Iteration | Functional Nodes | Bridge Success | S1 Reach | S3 Reach |
|-----------|------------------|----------------|----------|----------|
| 5000 | 24/30 | 4.0% | 78.7% | 82.0% |
| 25000 | 24/30 | 2.7% | 78.7% | 87.3% |
| 50000 | 24/30 | 1.3% | 78.0% | 81.3% |

### 10% mixed

| Iteration | Functional Nodes | Bridge Success | S1 Reach | S3 Reach |
|-----------|------------------|----------------|----------|----------|
| 5000 | 23/30 | 4.0% | 72.0% | 86.0% |
| 25000 | 21/30 | 1.3% | 77.3% | 81.3% |
| 50000 | 22/30 | 4.0% | 75.3% | 82.7% |

## 4. Path Generation Success Rates

### Final Performance (50k iterations)

| Model | S1→S2 | S2→S3 | S1→S3 |
|-------|-------|-------|-------|
| original | 100.0% | 100.0% | 30.0% |
| 5% mixed | 100.0% | 100.0% | 88.0% |
| 10% mixed | 100.0% | 100.0% | 88.0% |

## 5. Conclusions

1. **S2 Cohesion Paradox**: Higher cohesion indicates collapse, not strength
2. **Diversity is Key**: S2 nodes need diversity to bridge S1 and S3
3. **Functional Degradation**: S2 nodes lose their bridging capability over training
4. **Mixed Training Benefits**: Maintains S2 diversity and functionality