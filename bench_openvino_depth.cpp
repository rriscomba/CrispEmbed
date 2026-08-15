// Is the OpenVINO backend's per-compute cost fixed per GRAPH (amortisable over
// a real network) or per OP (fatal)? Chain N conv3x3 layers in one graph and
// see whether OpenVINO's time stays flat while the CPU's grows linearly.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

static double run(ggml_backend_t b, int depth, int iters, bool & ok) {
    ok = false;
    ggml_init_params ip = { ggml_tensor_overhead()*(size_t)(4*depth+16) + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const int64_t W=64,H=64,C=64;
    ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W,H,C,1);
    std::vector<ggml_tensor*> kers;
    for (int i=0;i<depth;i++) {
        ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3,3,C,C);
        kers.push_back(k);
        x = ggml_conv_2d_direct(ctx, k, x, 1,1,1,1,1,1);
    }
    ggml_set_output(x);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, x);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, b);
    if (!buf) { ggml_free(ctx); return 0.0; }
    std::vector<float> tmp;
    auto fill=[&](ggml_tensor*t){ size_t n=ggml_nelements(t); tmp.assign(n,0.01f); ggml_backend_tensor_set(t,tmp.data(),0,n*sizeof(float)); };
    for (auto k:kers) fill(k);
    if (ggml_backend_graph_compute(b,gf)!=GGML_STATUS_SUCCESS){ ggml_backend_buffer_free(buf); ggml_free(ctx); return 0.0; }
    ggml_backend_synchronize(b);
    auto t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<iters;i++) ggml_backend_graph_compute(b,gf);
    ggml_backend_synchronize(b);
    auto t1=std::chrono::high_resolution_clock::now();
    ggml_backend_buffer_free(buf); ggml_free(ctx); ok=true;
    return std::chrono::duration<double,std::milli>(t1-t0).count()/iters;
}

int main(){
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t ov=nullptr;
    for(size_t i=0;i<ggml_backend_dev_count();i++){ auto d=ggml_backend_dev_get(i);
        if(std::string(ggml_backend_dev_name(d)).rfind("OPENVINO",0)==0){ ov=ggml_backend_dev_init(d,nullptr); break; } }
    if(!ov){printf("no ov\n");return 1;}
    printf("%-8s %12s %12s %10s\n","depth","CPU ms","OpenVINO ms","speedup");
    for (int d : {1,2,4,8,16,17,20,24,32}) {
        bool a=false,b2=false;
        double tc=run(cpu,d,5,a), to=run(ov,d,5,b2);
        printf("%-8d %12.2f %12.2f %9.2fx\n", d, tc, to, tc/to);
    }
    return 0;
}
