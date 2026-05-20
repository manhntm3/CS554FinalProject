# Fix 1: Load Buffering (LB) Kernel — Correct Pattern

## What Was Wrong

The original LB kernel implemented an incorrect memory pattern and a wrong result counter.

### Wrong pattern (before)
```cuda
// P0 (incorrect):
int v = 0;
v = ptx_ld_weak(addr_x);   // loads from x (not y!)
ptx_st_weak(addr_y, v);    // stores loaded value (always 0) to y

// P1 (incorrect):
int v = 0;
v = ptx_ld_weak(addr_y);   // loads from y (not x!)
ptx_st_weak(addr_x, v);    // stores loaded value (always 0) to x

// Counter (incorrect):
if (h_r0[i] == 0 && h_r1[i] == 0) weak++;  // counted r0=0,r1=0 as "no thin air"
```

Two bugs:
1. **Variables swapped**: P0 should read `y` and write `x`; P1 should read `x` and write `y`.
2. **Stores 0 instead of 1**: The stored value `v` is always 0 (since `x` and `y` start at 0), so the value 1 never appears anywhere. The weak outcome `r0==1 && r1==1` can never trigger.
3. **Wrong counter**: The counter checked `r0==0 && r1==0`, which is always 100% because of bug 2.

The test was vacuously correct — it proved nothing.

### Correct pattern (after)
```cuda
// P0 (correct): load from y, store 1 to x
r0 = ptx_ld_*(addr_y);
ptx_st_*(addr_x, 1);

// P1 (correct): load from x, store 1 to y
r1 = ptx_ld_*(addr_x);
ptx_st_*(addr_y, 1);

// Counter (correct):
if (h_r0[i] == 1 && h_r1[i] == 1) weak++;  // both loaded the other's write
```

## What the Correct LB Test Measures

**Pattern**:
```
Initially: x=0, y=0
P0: r0 = ld(y[i]);   st(x[i], 1);
P1: r1 = ld(x[i]);   st(y[i], 1);
```

**Weak outcome**: `r0==1 && r1==1`

This requires a causality cycle:
- For r0=1: P1's write of y=1 must have happened before P0's load of y.
- Program order in P0: load(y) before store(x,1).
- For r1=1: P0's write of x=1 must have happened before P1's load of x.
- Program order in P1: load(x) before store(y,1).

Chain: `st(y,1) → ld(y) →(po)→ st(x,1) → ld(x) →(po)→ st(y,1)`

This is a cycle. Under sequentially consistent memory, it is impossible. Under relaxed models, it is theoretically allowed. On NVIDIA Turing (sm_75), it is **never observed** across all 5M × 5 = 25M trials — confirming the hardware's strong ordering properties.

## Results After Fix

```
LB Inter-Block WEAK     | Weak(1,1): 0 / 5,000,000 = 0.0000%
LB RELAXED (GPU)        | Weak(1,1): 0 / 5,000,000 = 0.0000%
LB RELAXED (SYS)        | Weak(1,1): 0 / 5,000,000 = 0.0000%
LB ACQ/REL (GPU)        | Weak(1,1): 0 / 5,000,000 = 0.0000%
LB FENCE SC (GPU)       | Weak(1,1): 0 / 5,000,000 = 0.0000%
```

The outcome distribution is now **non-trivial and informative**:
- `(0,0)`: both loads executed before both stores (typical, neither sees the other's write)
- `(1,0)` or `(0,1)`: one thread "won the race" and saw the other's write first
- `(1,1)`: **NEVER** — no causality cycle observed on Turing

## Academic Significance

The PTX specification explicitly forbids out-of-thin-air (OOTA) values. The LB outcome `r0==1 && r1==1` is the canonical example of a value appearing "from thin air" when values loop through a cycle. The Lustig et al. (ASPLOS 2019) formal PTX model includes a "no-thin-air" axiom. These results empirically confirm that the Tesla T4 respects this axiom.

## Files Changed
- `litmus_tests.cu`: `lb_kernel` function (read/write variable swapped, store value fixed from `v` to `1`)
- `litmus_tests.cu`: `run_lb_test` counter changed from `r0==0&&r1==0` to `r0==1&&r1==1`
