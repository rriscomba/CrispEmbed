// Static-shape micro-benchmark for the four CrispEmbed OCR/layout ops on the
// OpenVINO backend vs the ggml CPU backend.
//
// Why not tests/test-backend-ops perf: that harness builds its graphs with a
// dynamic dimension, and OpenVINO's CPU plugin cannot compile a Convolution
// with dynamic weights ("Doesn't support dynamic weights shape"). Real
// CrispEmbed OCR/layout graphs are statically shaped, so this measures the
// shapes that actually occur instead.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

enum bench_kind { K_CONV, K_CONV_DW, K_POOL, K_UPSCALE };

struct bench_case {
    const char * name;
    bench_kind   kind;
    int64_t      W, H, C, N;   // input
    int64_t      KW, KH, OC;   // kernel (conv)
    int          stride, pad;
};

static ggml_tensor * build(ggml_context * ctx, const bench_case & bc, ggml_tensor ** in, ggml_tensor ** ker) {
    *in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, bc.W, bc.H, bc.C, bc.N);
    ggml_set_name(*in, "in");
    switch (bc.kind) {
        case K_CONV:
            *ker = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, bc.KW, bc.KH, bc.C, bc.OC);
            ggml_set_name(*ker, "ker");
            return ggml_conv_2d_direct(ctx, *ker, *in, bc.stride, bc.stride, bc.pad, bc.pad, 1, 1);
        case K_CONV_DW:
            *ker = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, bc.KW, bc.KH, 1, bc.C);
            ggml_set_name(*ker, "ker");
            return ggml_conv_2d_dw_direct(ctx, *ker, *in, bc.stride, bc.stride, bc.pad, bc.pad, 1, 1);
        case K_POOL:
            *ker = nullptr;
            return ggml_pool_2d(ctx, *in, GGML_OP_POOL_MAX, bc.KW, bc.KH, bc.stride, bc.stride, 0, 0);
        case K_UPSCALE:
            *ker = nullptr;
            return ggml_upscale(ctx, *in, 2, GGML_SCALE_MODE_NEAREST);
    }
    return nullptr;
}

static double run_case(ggml_backend_t backend, const bench_case & bc, int iters, bool & ok) {
    ok = false;
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return 0.0;

    ggml_tensor *in = nullptr, *ker = nullptr;
    ggml_tensor * out = build(ctx, bc, &in, &ker);
    if (!out) { ggml_free(ctx); return 0.0; }
    ggml_set_output(out);

    if (!ggml_backend_supports_op(backend, out)) { ggml_free(ctx); return -1.0; }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); return 0.0; }

    // deterministic-ish fill
    std::vector<float> tmp;
    auto fill = [&](ggml_tensor * t) {
        if (!t) return;
        size_t n = ggml_nelements(t);
        tmp.resize(n);
        for (size_t i = 0; i < n; i++) tmp[i] = (float) ((i * 1103515245u + 12345u) % 1000) / 1000.0f - 0.5f;
        ggml_backend_tensor_set(t, tmp.data(), 0, n * sizeof(float));
    };
    fill(in);
    fill(ker);

    // warmup
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf); ggml_free(ctx); return 0.0;
    }
    ggml_backend_synchronize(backend);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buf); ggml_free(ctx); return 0.0;
        }
    }
    ggml_backend_synchronize(backend);
    auto t1 = std::chrono::high_resolution_clock::now();

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ok = true;
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

int main() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t ov  = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (std::string(ggml_backend_dev_name(d)).rfind("OPENVINO", 0) == 0) { ov = ggml_backend_dev_init(d, nullptr); break; }
    }
    if (!ov) { printf("no OpenVINO device\n"); return 1; }

    // Shapes representative of PP-OCRv6 detector / RT-DETRv2 layout stages.
    std::vector<bench_case> cases = {
        { "conv3x3  128x128x64  ->64",  K_CONV,     128,128, 64, 1, 3,3, 64, 1, 1 },
        { "conv3x3  64x64x128  ->128",  K_CONV,      64, 64,128, 1, 3,3,128, 1, 1 },
        { "conv1x1  128x128x64 ->256",  K_CONV,     128,128, 64, 1, 1,1,256, 1, 0 },
        { "conv3x3/s2 256x256x32->64",  K_CONV,     256,256, 32, 1, 3,3, 64, 2, 1 },
        { "dwconv3x3 128x128x64",       K_CONV_DW,  128,128, 64, 1, 3,3,  0, 1, 1 },
        { "maxpool2x2 256x256x64",      K_POOL,     256,256, 64, 1, 2,2,  0, 2, 0 },
        { "upscale x2 64x64x128",       K_UPSCALE,   64, 64,128, 1, 0,0,  0, 0, 0 },
    };

    printf("%-30s %12s %12s %10s\n", "case", "CPU ms", "OpenVINO ms", "speedup");
    printf("%-30s %12s %12s %10s\n", "------------------------------", "------------", "------------", "----------");
    for (auto & bc : cases) {
        bool ok_c = false, ok_o = false;
        double tc = run_case(cpu, bc, 10, ok_c);
        double to = run_case(ov,  bc, 10, ok_o);
        if (!ok_c || !ok_o) {
            printf("%-30s %12s %12s %10s\n", bc.name,
                   ok_c ? "ok" : (tc < 0 ? "unsup" : "ERR"),
                   ok_o ? "ok" : (to < 0 ? "unsup" : "ERR"), "-");
            continue;
        }
        printf("%-30s %12.3f %12.3f %9.2fx\n", bc.name, tc, to, tc / to);
    }

    ggml_backend_free(ov);
    ggml_backend_free(cpu);
    return 0;
}
