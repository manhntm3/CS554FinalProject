/*
 * litmus_tests.cpp  —  Host driver for the Metal litmus suite.
 *
 * Mirrors the host portion of ../litmus_tests.cu:
 *   - Hist2 / Stats helpers
 *   - run_{sb,mp,lb,iriw}_test functions
 *   - main() that prints a CUDA-style banner per test
 *
 * Targets Apple Silicon (M1+) with unified memory; uses
 * MTL::ResourceStorageModeShared for zero-copy host <-> GPU buffers.
 *
 * Build:  see Makefile.  Requires the Metal toolchain (full Xcode or the
 *         standalone Metal Developer Tools — Command Line Tools alone
 *         do not ship `xcrun metal`).
 */

// metal-cpp implementation macros — must be defined in exactly one TU.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>

// ============================================================
// Tunables — match the CUDA suite
// ============================================================
static const int ITERATIONS = 1'000'000;
static const int N_RUNS     = 5;

// ============================================================
// Variant enums — MUST match constants in litmus_tests.metal
// ============================================================
enum SBVariant : int {
    SB_WEAK = 0,
    SB_RELAXED_GPU = 1,
    SB_REL_ACQ_GPU = 2,
    SB_FENCE_SC_GPU = 3,
    SB_ATOMIC_RELAXED = 4,
};
enum MPVariant : int {
    MP_WEAK = 0,
    MP_RELAXED_GPU = 1,
    MP_REL_ACQ_GPU = 2,
    MP_FENCE_SC_GPU = 3,
};
enum LBVariant : int {
    LB_WEAK = 0,
    LB_RELAXED_GPU = 1,
    LB_ACQ_REL_GPU = 2,
    LB_FENCE_SC_GPU = 3,
};
enum IRIWVariant : int {
    IRIW_WEAK = 0,
    IRIW_RELAXED_GPU = 1,
    IRIW_REL_ACQ_GPU = 2,
    IRIW_FENCE_SC_GPU = 3,
};

// ============================================================
// Histogram + stats helpers — identical shape to litmus_tests.cu
// ============================================================
struct Hist2 {
    long long c[2][2];
    Hist2() { c[0][0] = c[0][1] = c[1][0] = c[1][1] = 0; }
    void record(int r0, int r1) {
        if (r0 >= 0 && r0 <= 1 && r1 >= 0 && r1 <= 1) c[r0][r1]++;
    }
    long long total() const { return c[0][0] + c[0][1] + c[1][0] + c[1][1]; }
    Hist2& operator+=(const Hist2& o) {
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++) c[i][j] += o.c[i][j];
        return *this;
    }
};

struct Stats { double mean, stddev, mn, mx; };

static Stats compute_stats(const std::vector<double>& v) {
    double s = 0, s2 = 0, mn = v[0], mx = v[0];
    for (double x : v) { s += x; s2 += x*x; mn = std::min(mn,x); mx = std::max(mx,x); }
    int n = (int)v.size();
    double m = s / n;
    double var = std::max(0.0, s2 / n - m * m);
    return { m, std::sqrt(var), mn, mx };
}

static void print_hist2(const Hist2& h,
                        const char* r0_lbl, const char* r1_lbl,
                        bool w00, bool w10, bool w01, bool w11)
{
    long long tot = h.total();
    if (tot == 0) { printf("    (no observations)\n"); return; }
    printf("  (%s, %s) distribution — %lld total obs (%d runs):\n",
           r0_lbl, r1_lbl, tot, N_RUNS);
    auto row = [&](int r0, int r1, bool is_weak) {
        printf("    (%d, %d)%s : %9lld  (%8.4f%%)\n",
               r0, r1, is_weak ? "*" : " ", h.c[r0][r1],
               100.0 * h.c[r0][r1] / tot);
    };
    row(0, 0, w00);
    row(0, 1, w01);
    row(1, 0, w10);
    row(1, 1, w11);
}

// ============================================================
// MetalContext — owns device, queue, library, and pipeline cache
// ============================================================
class MetalContext {
public:
    MTL::Device*       device  = nullptr;
    MTL::CommandQueue* queue   = nullptr;
    MTL::Library*      library = nullptr;

    MetalContext() {
        device = MTL::CreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal device available.\n");
            std::abort();
        }
        queue = device->newCommandQueue();
        if (!queue) {
            fprintf(stderr, "Failed to create command queue.\n");
            std::abort();
        }
        loadLibrary("litmus_tests.metallib");

        printf("Device: %s\n", device->name()->utf8String());
        printf("Unified memory: %s\n", device->hasUnifiedMemory() ? "yes" : "no");
    }

    ~MetalContext() {
        if (library) library->release();
        if (queue)   queue->release();
        if (device)  device->release();
    }

    MTL::ComputePipelineState* pipeline(const char* kernelName) {
        NS::String* name = NS::String::string(kernelName, NS::UTF8StringEncoding);
        MTL::Function* fn = library->newFunction(name);
        if (!fn) {
            fprintf(stderr, "Kernel '%s' not found in metallib.\n", kernelName);
            std::abort();
        }
        NS::Error* err = nullptr;
        MTL::ComputePipelineState* pso =
            device->newComputePipelineState(fn, &err);
        fn->release();
        if (!pso) {
            fprintf(stderr, "Pipeline creation failed for '%s': %s\n",
                    kernelName,
                    err ? err->localizedDescription()->utf8String() : "(null)");
            std::abort();
        }
        return pso;
    }

private:
    void loadLibrary(const char* path) {
        NS::String* p = NS::String::string(path, NS::UTF8StringEncoding);
        NS::URL* url = NS::URL::fileURLWithPath(p);
        NS::Error* err = nullptr;
        library = device->newLibrary(url, &err);
        if (!library) {
            fprintf(stderr, "Failed to load metallib '%s': %s\n", path,
                    err ? err->localizedDescription()->utf8String() : "(null)");
            fprintf(stderr,
                "Hint: run from the directory containing the .metallib,\n"
                "      or pre-build it via `make`.\n");
            std::abort();
        }
    }
};

// ============================================================
// Buffer helpers
// ============================================================
static MTL::Buffer* makeBuffer(MTL::Device* d, size_t bytes) {
    MTL::Buffer* b = d->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (!b) { fprintf(stderr, "newBuffer(%zu) failed\n", bytes); std::abort(); }
    return b;
}
static inline void zeroBuffer(MTL::Buffer* b) {
    std::memset(b->contents(), 0, b->length());
}

// ============================================================
// Dispatch helper — single-kernel run, no internal sync needed.
// `threadgroups` is the X dimension; threads/group is fixed at 64
// to match the CUDA suite (other lanes early-out in get_roles).
// ============================================================
static void dispatch(MetalContext& ctx,
                     MTL::ComputePipelineState* pso,
                     std::vector<MTL::Buffer*> const& buffers,
                     std::vector<std::pair<const void*, size_t>> const& bytes,
                     int threadgroups)
{
    MTL::CommandBuffer*       cb  = ctx.queue->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
    enc->setComputePipelineState(pso);

    NS::UInteger idx = 0;
    for (MTL::Buffer* b : buffers) {
        enc->setBuffer(b, 0, idx++);
    }
    for (auto& [ptr, sz] : bytes) {
        enc->setBytes(ptr, sz, idx++);
    }

    enc->dispatchThreadgroups(MTL::Size(threadgroups, 1, 1),
                              MTL::Size(64, 1, 1));
    enc->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
}

// ============================================================
// run_sb_test
// ============================================================
void run_sb_test(MetalContext& ctx, MTL::ComputePipelineState* pso,
                 int iterations, bool inter_block, int variant,
                 const char* label)
{
    size_t bytes = iterations * sizeof(int);
    MTL::Buffer* bx  = makeBuffer(ctx.device, bytes);
    MTL::Buffer* by  = makeBuffer(ctx.device, bytes);
    MTL::Buffer* br0 = makeBuffer(ctx.device, bytes);
    MTL::Buffer* br1 = makeBuffer(ctx.device, bytes);

    int inter_block_i = inter_block ? 1 : 0;
    int threadgroups  = inter_block ? 2 : 1;

    Hist2 total;
    std::vector<double> rates;

    for (int run = 0; run < N_RUNS; ++run) {
        zeroBuffer(bx);
        zeroBuffer(by);
        // Result buffers don't need clearing — kernel writes every slot.

        dispatch(ctx, pso,
                 { bx, by, br0, br1 },
                 { { &iterations, sizeof(int) },
                   { &variant, sizeof(int) },
                   { &inter_block_i, sizeof(int) } },
                 threadgroups);

        const int* r0 = static_cast<const int*>(br0->contents());
        const int* r1 = static_cast<const int*>(br1->contents());
        Hist2 h; long long weak = 0;
        for (int i = 0; i < iterations; ++i) {
            h.record(r0[i], r1[i]);
            if (r0[i] == 0 && r1[i] == 0) weak++;
        }
        total += h;
        rates.push_back(100.0 * weak / iterations);
    }

    Stats s = compute_stats(rates);
    long long tot = total.total();
    printf("%-38s | Weak(0,0): %8lld/%lld = %7.4f%%  [mean=%7.4f%% std=%6.4f%%]\n",
           label, total.c[0][0], tot,
           100.0 * total.c[0][0] / tot, s.mean, s.stddev);
    print_hist2(total, "r0", "r1", true, false, false, false);
    printf("\n");

    bx->release(); by->release(); br0->release(); br1->release();
}

// ============================================================
// run_mp_test
// ============================================================
void run_mp_test(MetalContext& ctx, MTL::ComputePipelineState* pso,
                 int iterations, bool inter_block, int variant,
                 const char* label)
{
    size_t bytes = iterations * sizeof(int);
    MTL::Buffer* bdata = makeBuffer(ctx.device, bytes);
    MTL::Buffer* bflag = makeBuffer(ctx.device, bytes);
    MTL::Buffer* brf   = makeBuffer(ctx.device, bytes);
    MTL::Buffer* brd   = makeBuffer(ctx.device, bytes);

    int inter_block_i = inter_block ? 1 : 0;
    int threadgroups  = inter_block ? 2 : 1;

    Hist2 total;
    std::vector<double> rates;

    for (int run = 0; run < N_RUNS; ++run) {
        zeroBuffer(bdata);
        zeroBuffer(bflag);

        dispatch(ctx, pso,
                 { bdata, bflag, brf, brd },
                 { { &iterations, sizeof(int) },
                   { &variant, sizeof(int) },
                   { &inter_block_i, sizeof(int) } },
                 threadgroups);

        const int* rf = static_cast<const int*>(brf->contents());
        const int* rd = static_cast<const int*>(brd->contents());
        Hist2 h; long long weak = 0;
        for (int i = 0; i < iterations; ++i) {
            h.record(rf[i], rd[i]);
            if (rf[i] == 1 && rd[i] == 0) weak++;
        }
        total += h;
        rates.push_back(100.0 * weak / iterations);
    }

    Stats s = compute_stats(rates);
    long long tot = total.total();
    printf("%-38s | Weak(f=1,d=0): %8lld/%lld = %7.4f%%  [mean=%7.4f%% std=%6.4f%%]\n",
           label, total.c[1][0], tot,
           100.0 * total.c[1][0] / tot, s.mean, s.stddev);
    print_hist2(total, "r_flag", "r_data", false, true, false, false);
    printf("\n");

    bdata->release(); bflag->release(); brf->release(); brd->release();
}

// ============================================================
// run_lb_test
// ============================================================
void run_lb_test(MetalContext& ctx, MTL::ComputePipelineState* pso,
                 int iterations, bool inter_block, int variant,
                 const char* label)
{
    size_t bytes = iterations * sizeof(int);
    MTL::Buffer* bx  = makeBuffer(ctx.device, bytes);
    MTL::Buffer* by  = makeBuffer(ctx.device, bytes);
    MTL::Buffer* br0 = makeBuffer(ctx.device, bytes);
    MTL::Buffer* br1 = makeBuffer(ctx.device, bytes);

    int inter_block_i = inter_block ? 1 : 0;
    int threadgroups  = inter_block ? 2 : 1;

    Hist2 total;
    std::vector<double> rates;

    for (int run = 0; run < N_RUNS; ++run) {
        zeroBuffer(bx);
        zeroBuffer(by);

        dispatch(ctx, pso,
                 { bx, by, br0, br1 },
                 { { &iterations, sizeof(int) },
                   { &variant, sizeof(int) },
                   { &inter_block_i, sizeof(int) } },
                 threadgroups);

        const int* r0 = static_cast<const int*>(br0->contents());
        const int* r1 = static_cast<const int*>(br1->contents());
        Hist2 h; long long weak = 0;
        for (int i = 0; i < iterations; ++i) {
            h.record(r0[i], r1[i]);
            if (r0[i] == 1 && r1[i] == 1) weak++;
        }
        total += h;
        rates.push_back(100.0 * weak / iterations);
    }

    Stats s = compute_stats(rates);
    long long tot = total.total();
    printf("%-38s | Weak(1,1): %8lld/%lld = %7.4f%%  [mean=%7.4f%% std=%6.4f%%]\n",
           label, total.c[1][1], tot,
           100.0 * total.c[1][1] / tot, s.mean, s.stddev);
    print_hist2(total, "r0", "r1", false, false, false, true);
    printf("\n");

    bx->release(); by->release(); br0->release(); br1->release();
}

// ============================================================
// run_iriw_test
// ============================================================
void run_iriw_test(MetalContext& ctx, MTL::ComputePipelineState* pso,
                   int iterations, int variant, const char* label)
{
    size_t bytes = iterations * sizeof(int);
    MTL::Buffer* bx  = makeBuffer(ctx.device, bytes);
    MTL::Buffer* by  = makeBuffer(ctx.device, bytes);
    MTL::Buffer* br0 = makeBuffer(ctx.device, bytes);
    MTL::Buffer* br1 = makeBuffer(ctx.device, bytes);
    MTL::Buffer* br2 = makeBuffer(ctx.device, bytes);
    MTL::Buffer* br3 = makeBuffer(ctx.device, bytes);

    Hist2 hist_p2, hist_p3;
    long long total_weak = 0, total_obs = 0;
    std::vector<double> rates;

    for (int run = 0; run < N_RUNS; ++run) {
        zeroBuffer(bx);
        zeroBuffer(by);

        dispatch(ctx, pso,
                 { bx, by, br0, br1, br2, br3 },
                 { { &iterations, sizeof(int) },
                   { &variant, sizeof(int) } },
                 4); // 4 threadgroups for IRIW

        const int* r0 = static_cast<const int*>(br0->contents());
        const int* r1 = static_cast<const int*>(br1->contents());
        const int* r2 = static_cast<const int*>(br2->contents());
        const int* r3 = static_cast<const int*>(br3->contents());

        long long weak = 0, obs = 0;
        for (int i = 0; i < iterations; ++i) {
            int a = r0[i], b = r1[i], c = r2[i], d = r3[i];
            bool valid = (a >= 0 && a <= 1 && b >= 0 && b <= 1 &&
                          c >= 0 && c <= 1 && d >= 0 && d <= 1);
            if (valid) {
                obs++;
                hist_p2.record(a, b);
                hist_p3.record(c, d);
                if (a == 1 && b == 0 && c == 1 && d == 0) weak++;
            }
        }
        total_weak += weak;
        total_obs  += obs;
        rates.push_back(obs > 0 ? 100.0 * weak / obs : 0.0);
    }

    Stats s = compute_stats(rates);
    printf("%-38s | Weak(r0=1,r1=0,r2=1,r3=0): %lld/%lld = %7.4f%%  [mean=%7.4f%% std=%6.4f%%]\n",
           label, total_weak, total_obs,
           total_obs > 0 ? 100.0 * total_weak / total_obs : 0.0,
           s.mean, s.stddev);

    long long p2t = hist_p2.total(), p3t = hist_p3.total();
    if (p2t > 0) {
        printf("  P2 (x→y): (0,0)=%lld(%.1f%%) (1,0)*=%lld(%.1f%%) (0,1)=%lld(%.1f%%) (1,1)=%lld(%.1f%%)\n",
               hist_p2.c[0][0], 100.0 * hist_p2.c[0][0] / p2t,
               hist_p2.c[1][0], 100.0 * hist_p2.c[1][0] / p2t,
               hist_p2.c[0][1], 100.0 * hist_p2.c[0][1] / p2t,
               hist_p2.c[1][1], 100.0 * hist_p2.c[1][1] / p2t);
    }
    if (p3t > 0) {
        printf("  P3 (y→x): (0,0)=%lld(%.1f%%) (1,0)*=%lld(%.1f%%) (0,1)=%lld(%.1f%%) (1,1)=%lld(%.1f%%)\n",
               hist_p3.c[0][0], 100.0 * hist_p3.c[0][0] / p3t,
               hist_p3.c[1][0], 100.0 * hist_p3.c[1][0] / p3t,
               hist_p3.c[0][1], 100.0 * hist_p3.c[0][1] / p3t,
               hist_p3.c[1][1], 100.0 * hist_p3.c[1][1] / p3t);
    }
    printf("\n");

    bx->release(); by->release();
    br0->release(); br1->release(); br2->release(); br3->release();
}

// ============================================================
// main
// ============================================================
int main()
{
    // Autorelease pool so transient NS objects (NS::String, NS::URL, etc.)
    // are cleaned up at scope exit. metal-cpp follows Objective-C semantics.
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    MetalContext ctx;

    printf("Test %d runs x %d iter = %lld total obs/test\n\n",
           N_RUNS, ITERATIONS, (long long)N_RUNS * ITERATIONS);

    MTL::ComputePipelineState* psoSB   = ctx.pipeline("sb_kernel");
    MTL::ComputePipelineState* psoMP   = ctx.pipeline("mp_kernel");
    MTL::ComputePipelineState* psoLB   = ctx.pipeline("lb_kernel");
    MTL::ComputePipelineState* psoIRIW = ctx.pipeline("iriw_kernel");

    // ----- SB -----
    printf("══════════════════════════════════════════════════════════════════════════════\n");
    printf("STORE BUFFERING (SB)   P0: st(x,1); r0=ld(y)  ||  P1: st(y,1); r1=ld(x)\n");
    printf("Weak outcome: r0==0 && r1==0   (* marks weak cell in distribution)\n");
    printf("══════════════════════════════════════════════════════════════════════════════\n\n");
    run_sb_test(ctx, psoSB, ITERATIONS, true,  SB_WEAK,           "SB Inter-Block WEAK");
    run_sb_test(ctx, psoSB, ITERATIONS, true,  SB_RELAXED_GPU,    "SB Inter-Block RELAXED (GPU)");
    run_sb_test(ctx, psoSB, ITERATIONS, true,  SB_REL_ACQ_GPU,    "SB Inter-Block ACQ/REL (GPU)");
    run_sb_test(ctx, psoSB, ITERATIONS, true,  SB_FENCE_SC_GPU,   "SB Inter-Block FENCE SC (GPU)");
    run_sb_test(ctx, psoSB, ITERATIONS, true,  SB_ATOMIC_RELAXED, "SB Inter-Block ATOMIC EXCH (relaxed)");

    // ----- MP -----
    printf("══════════════════════════════════════════════════════════════════════════════\n");
    printf("MESSAGE PASSING (MP)   P0: data=1; flag=1  ||  P1: r_flag=ld(flag); r_data=ld(data)\n");
    printf("Weak outcome: r_flag==1 && r_data==0\n");
    printf("══════════════════════════════════════════════════════════════════════════════\n\n");
    run_mp_test(ctx, psoMP, ITERATIONS, true, MP_WEAK,         "MP Inter-Block WEAK");
    run_mp_test(ctx, psoMP, ITERATIONS, true, MP_RELAXED_GPU,  "MP Inter-Block RELAXED (GPU)");
    run_mp_test(ctx, psoMP, ITERATIONS, true, MP_REL_ACQ_GPU,  "MP Inter-Block ACQ/REL (GPU)");
    run_mp_test(ctx, psoMP, ITERATIONS, true, MP_FENCE_SC_GPU, "MP Inter-Block FENCE SC (GPU)");

    // ----- LB -----
    printf("══════════════════════════════════════════════════════════════════════════════\n");
    printf("LOAD BUFFERING (LB)   P0: r0=ld(y); st(x,1)  ||  P1: r1=ld(x); st(y,1)\n");
    printf("Weak outcome: r0==1 && r1==1   (causality cycle)\n");
    printf("══════════════════════════════════════════════════════════════════════════════\n\n");
    run_lb_test(ctx, psoLB, ITERATIONS, true, LB_WEAK,         "LB Inter-Block WEAK");
    run_lb_test(ctx, psoLB, ITERATIONS, true, LB_RELAXED_GPU,  "LB Inter-Block RELAXED (GPU)");
    run_lb_test(ctx, psoLB, ITERATIONS, true, LB_ACQ_REL_GPU,  "LB Inter-Block ACQ/REL (GPU)");
    run_lb_test(ctx, psoLB, ITERATIONS, true, LB_FENCE_SC_GPU, "LB Inter-Block FENCE SC (GPU)");

    // ----- IRIW -----
    printf("══════════════════════════════════════════════════════════════════════════════\n");
    printf("INDEP. READS OF INDEP. WRITES (IRIW)  — 4-thread / 4-block test\n");
    printf("  P0: x=1  ||  P1: y=1  ||  P2: r0=ld(x); r1=ld(y)  ||  P3: r2=ld(y); r3=ld(x)\n");
    printf("Weak (non-MCA) outcome: r0==1 && r1==0 && r2==1 && r3==0\n");
    printf("══════════════════════════════════════════════════════════════════════════════\n\n");
    run_iriw_test(ctx, psoIRIW, ITERATIONS, IRIW_WEAK,         "IRIW WEAK");
    run_iriw_test(ctx, psoIRIW, ITERATIONS, IRIW_RELAXED_GPU,  "IRIW RELAXED (GPU)");
    run_iriw_test(ctx, psoIRIW, ITERATIONS, IRIW_REL_ACQ_GPU,  "IRIW ACQ/REL (GPU)");
    run_iriw_test(ctx, psoIRIW, ITERATIONS, IRIW_FENCE_SC_GPU, "IRIW FENCE SC (GPU)");

    psoSB->release();
    psoMP->release();
    psoLB->release();
    psoIRIW->release();

    pool->release();
    return 0;
}
