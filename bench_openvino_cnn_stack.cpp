// End-to-end check for the >=20-op CNN path: a PP-OCRv6-detector-shaped stack
// (conv/relu/pool/upscale/sigmoid) deep enough to miss is_naive(), run on both
// backends and compared elementwise.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static ggml_tensor * build_stack(ggml_context * ctx, int blocks, std::vector<ggml_tensor*> & ws, int & n_ops) {
    const int64_t W=64,H=64,C=16;
    ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W,H,C,1);
    ws.push_back(x);
    n_ops = 0;
    for (int i=0;i<blocks;i++) {
        ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3,3,C,C); ws.push_back(k);
        x = ggml_conv_2d_direct(ctx, k, x, 1,1,1,1,1,1);        n_ops++;
        x = ggml_relu(ctx, x);                                   n_ops++;
        x = ggml_pool_2d(ctx, x, GGML_OP_POOL_MAX, 2,2,2,2,0,0); n_ops++;
        x = ggml_upscale(ctx, x, 2, GGML_SCALE_MODE_NEAREST);    n_ops++;
    }
    ggml_tensor * kf = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1,1,C,1); ws.push_back(kf);
    x = ggml_conv_2d_direct(ctx, kf, x, 1,1,0,0,1,1);            n_ops++;
    x = ggml_sigmoid(ctx, x);                                    n_ops++;
    return x;
}

static bool run(ggml_backend_t b, int blocks, std::vector<float> & out) {
    ggml_init_params ip = { ggml_tensor_overhead()*512 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    std::vector<ggml_tensor*> ws; int n_ops=0;
    ggml_tensor * y = build_stack(ctx, blocks, ws, n_ops);
    ggml_set_output(y);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, b);
    if(!buf){ ggml_free(ctx); return false; }
    std::vector<float> tmp;
    for (auto t : ws) { size_t n=ggml_nelements(t); tmp.resize(n);
        for(size_t i=0;i<n;i++) tmp[i]=std::sin((float)(i%97)*0.31f)*0.2f;
        ggml_backend_tensor_set(t,tmp.data(),0,n*sizeof(float)); }
    if (ggml_backend_graph_compute(b,gf)!=GGML_STATUS_SUCCESS){ ggml_backend_buffer_free(buf); ggml_free(ctx); return false; }
    out.resize(ggml_nelements(y));
    ggml_backend_tensor_get(y,out.data(),0,out.size()*sizeof(float));
    printf("   (graph ops = %d)\n", n_ops);
    ggml_backend_buffer_free(buf); ggml_free(ctx); return true;
}

int main(){
    ggml_backend_t cpu=ggml_backend_cpu_init(); ggml_backend_t ov=nullptr;
    for(size_t i=0;i<ggml_backend_dev_count();i++){auto d=ggml_backend_dev_get(i);
        if(std::string(ggml_backend_dev_name(d)).rfind("OPENVINO",0)==0){ov=ggml_backend_dev_init(d,nullptr);break;}}
    if(!ov){printf("no ov\n");return 1;}
    for (int blocks : {2, 6, 12}) {
        std::vector<float> a,b2;
        printf("blocks=%-3d ", blocks);
        bool oc=run(cpu,blocks,a), oo=run(ov,blocks,b2);
        if(!oc||!oo||a.size()!=b2.size()){ printf("  -> FAILED TO RUN (cpu=%d ov=%d)\n",oc,oo); continue; }
        double md=0; for(size_t i=0;i<a.size();i++) md=std::max(md,(double)std::fabs(a[i]-b2[i]));
        printf("  -> n=%zu  max|diff| = %.3e  %s\n", a.size(), md, md<1e-4?"MATCH":"MISMATCH");
    }
    return 0;
}
