# Fix 3: Full Outcome Histograms

## Motivation

The original code reported only one number per test: the count of the specific "weak" outcome. This is insufficient for a research study because:

1. **Non-weak outcomes are invisible**: A test reporting "0 weak" tells you nothing about whether the test even ran meaningfully.
2. **Timing effects are hidden**: Knowing the full distribution reveals whether both threads race (producing a mix of outcomes) or one consistently beats the other.
3. **Spec comparison is harder**: The PTX formal model specifies which outcome tuples are *allowed* and which are *forbidden*. A full histogram lets you check all outcomes against the model.
4. **Debugging**: A trivially broken test (like the original LB kernel) shows 100% `(0,0)` — immediately suspicious.

## Implementation

### `Hist2` Struct

For 2-thread tests (SB, MP, LB) with binary outcomes (r0, r1 ∈ {0,1}):

```cpp
struct Hist2 {
    long long c[2][2];   // c[r0][r1]

    void record(int r0, int r1);       // increments c[r0][r1] with bounds check
    long long total() const;           // sum of all cells
    Hist2& operator+=(const Hist2&);   // accumulate across runs
};
```

### Accumulation Strategy

Each test runs `N_RUNS` times. A `Hist2` is built per run, then added into a `total` histogram. Percentages are computed over `total.total()` = N_RUNS × ITERATIONS observations.

### Print Format

```
Test Name  | Weak(r0,r1): COUNT/TOTAL = RATE%  [mean=X% std=Y%]
  (r0, r1) distribution — TOTAL total obs (N runs):
    (0, 0)* :   COUNT  ( RATE%)      ← * marks the forbidden/weak outcome
    (0, 1)  :   COUNT  ( RATE%)
    (1, 0)  :   COUNT  ( RATE%)
    (1, 1)  :   COUNT  ( RATE%)
```

The `*` marker on a row indicates that cell is the theoretically forbidden (weak) outcome for that test variant.

## Interpretations Found from the Histograms

### SB WEAK (inter-block)
```
(0, 0)*: 826  ( 0.02%)  ← WEAK: both read initial 0
(0, 1) : 4999170 (99.98%)
(1, 0) : 0     ( 0.00%)
(1, 1) : 4     ( 0.00%)
```
The (0,1) dominance and (1,0)=0 asymmetry reveals a strong **execution ordering bias**: P0 (block 0) systematically runs slightly ahead of P1 (block 1), so P1 almost always reads x=1 before P0 reads y. The rare (0,0) events are genuine SB reorderings.

### SB RELAXED GPU (inter-block)
```
(0, 0)*: 0     (  0.00%)
(0, 1) : 4370382 (87.4%)
(1, 0) : 11368  ( 0.2%)
(1, 1) : 618250 (12.4%)
```
No weak behavior, but a very different distribution from WEAK. The presence of `(1,1)` (both see each other's write) shows that GPU-scope relaxed atomics allow both threads to be "live" at the same time in a way that plain stores do not. The atomics' participation in the coherence protocol creates more overlapping execution windows.

### MP RELAXED GPU (inter-block)
```
(0, 0) : 0        ( 0.00%)
(0, 1) : 4        ( 0.00%)
(1, 0)*: 0        ( 0.00%)   ← no MP weakness
(1, 1) : 4999996  (99.999%)
```
Consumer almost always reads after producer completes both writes. The (0,1) entries (flag=0, data=1) indicate a tiny number of timing inversions where data arrives before flag — the reverse of the MP weakness.

### LB RELAXED GPU (inter-block, after fix)
```
(0, 0) : 89908   ( 1.8%)
(0, 1) : 857051  (17.1%)
(1, 0) : 4053041 (81.1%)
(1, 1)*: 0        ( 0.0%)
```
P1 (block 1) tends to lead: P0 sees P1's write of y=1 (`r0=1`) most of the time, but P1 rarely sees P0's write of x=1 (`r1=1`) simultaneously. This confirms no causality cycles ever form.

## For IRIW Tests

IRIW uses two separate `Hist2` structs (one per reader thread):
- `hist_p2`: P2's (ld(x), ld(y)) outcomes
- `hist_p3`: P3's (ld(y), ld(x)) outcomes

Plus a joint counter for the full IRIW weak outcome.

## Files Changed
- `litmus_tests.cu`: Added `Hist2` struct, `compute_stats`, `print_hist2` helper
- `litmus_tests.cu`: All `run_*_test` functions now accumulate and print histograms
