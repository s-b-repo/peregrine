#include "backend_cuda.h"

#include <cuda_runtime.h>
#include <mma.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

struct ColiCudaTensor {
    void *weights;
    float *scales;
    size_t weight_bytes;
    int fmt, I, O, device;
    int tracked;
};

/* One cached, instantiated CUDA graph of an `expert_group` launch shape. */
#define COLI_CUDA_GRAPH_CACHE 16
struct ColiCudaGraph;
typedef struct {
    uint64_t key;              /* launch shape; 0 = empty slot */
    struct ColiCudaGraph *g;
    unsigned gen;              /* ctx->scratch_gen at capture */
    uint64_t used;             /* LRU stamp */
} GraphSlot;

typedef struct {
    int device;
    int compute_major,compute_minor;
    float *x, *y, *gate, *up;
    size_t x_cap, y_cap, gate_cap, up_cap;
    uint8_t *qx; float *qscale;
    size_t qx_cap, qscale_cap;
    float *host_x,*host_y; size_t host_x_cap,host_y_cap;
    /* Fused reduce (COLI_CUDA_FUSED_REDUCE): CSR metadata + the accumulated
     * [s_n, D] output, device side and pinned staging side. */
    void *red_meta; size_t red_meta_cap;
    void *host_red; size_t host_red_cap;
    float *red_out; size_t red_out_cap;
    float *host_red_out; size_t host_red_out_cap;
    float *aq,*al,*ar,*ac; size_t aq_cap,al_cap,ar_cap,ac_cap;
    float *pipe_buf[24]; size_t pipe_cap[24];   /* scratch persistenti del resident pipeline */
    cudaStream_t stream;
    void *group_desc; size_t group_desc_cap;
    void *host_desc; size_t host_desc_cap;      /* pinned staging: capture cannot copy from pageable */
    size_t tensor_count, tensor_bytes;
    /* Bumped whenever a scratch buffer is actually reallocated. A captured graph
     * holds baked device pointers, so it is only valid at the generation it was
     * captured under — see `reserve`. */
    unsigned scratch_gen;
    GraphSlot graphs[COLI_CUDA_GRAPH_CACHE];
    uint64_t graph_clock;
} DeviceContext;

typedef struct {
    const void *g,*u,*d; const float *gs,*us,*ds;
    int gf,uf,df,rows,offset;
} GroupDesc;

static DeviceContext g_ctx[COLI_CUDA_MAX_DEVICES];
static int g_nctx;
static uint64_t g_group_calls,g_group_experts,g_group_rows;
static double g_group_h2d_ms,g_group_kernel_ms,g_group_d2h_ms;
static std::mutex g_group_stats_mu;

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "[CUDA] %s: %s\n", what, cudaGetErrorString(err));
    return 0;
}

static DeviceContext *find_ctx(int device) {
    for (int i = 0; i < g_nctx; i++) if (g_ctx[i].device == device) return &g_ctx[i];
    return nullptr;
}

/* cudaSetDevice on every call doubles expert-matmul time on 2 GPUs when the
 * serial expert loop alternates devices (measured on RTX 5090 + 4090: 14.3s
 * -> 25.4s per 32 tokens). The current device is per-thread in the CUDA
 * runtime, so a thread-local cache skips the redundant switches. */
static thread_local int g_current_device = -1;

static int select_ctx(DeviceContext *ctx) {
    if (!ctx) return 0;
    if (g_current_device == ctx->device) return 1;
    if (!cuda_ok(cudaSetDevice(ctx->device), "select device")) return 0;
    g_current_device = ctx->device;
    return 1;
}

__host__ __device__ static size_t row_bytes(int fmt, int I) {
    if (fmt == 0) return (size_t)I * sizeof(float);
    if (fmt == 1) return (size_t)I;
    if (fmt == 2) return (size_t)(I + 1) / 2;
    if (fmt == 3) return (size_t)(I + 3) / 4;
    return 0;
}

__device__ static float weight_at(const void *weights, int fmt, size_t row, int i) {
    const uint8_t *base = static_cast<const uint8_t *>(weights) + row;
    if (fmt == 0) return reinterpret_cast<const float *>(base)[i];
    if (fmt == 1) return static_cast<float>(reinterpret_cast<const int8_t *>(base)[i]);
    const uint8_t *q = base;
    if (fmt == 2) {
        uint8_t v = q[i >> 1];
        int n=(i&1)?(v>>4):(v&15); return static_cast<float>(n&8?n-16:n);
    }
    uint8_t v = q[i >> 2];
    return static_cast<float>(((v >> ((i & 3) * 2)) & 3) - 2);
}

__global__ static void offset_to_signed_s4(uint8_t *q,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)q[i]^=0x88;
}

__global__ static void quant_matmul(float *y, const float *x, const void *weights,
                                    const float *scales, int fmt, int S, int I, int O,
                                    size_t rb) {
    int o = blockIdx.x;
    int s = blockIdx.y;
    float sum = 0.0f;
    size_t row = (size_t)o * rb;
    const float *xs = x + (size_t)s * I;
    for (int i = threadIdx.x; i < I; i += blockDim.x)
        sum += xs[i] * weight_at(weights, fmt, row, i);

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int n = blockDim.x >> 1; n; n >>= 1) {
        if (threadIdx.x < n) partial[threadIdx.x] += partial[threadIdx.x + n];
        __syncthreads();
    }
    if (!threadIdx.x)
        y[(size_t)s * O + o] = partial[0] * (fmt ? scales[o] : 1.0f);
}

/* Gate-weighted accumulation of an expert group's rows into their batch rows —
 * the layer-level reduce, on the device.
 *
 * `y` is `[total, D]` in call order; output row `s` sums the contributions
 * `row_idx[row_ptr[s] .. row_ptr[s+1]]`, each scaled by its router weight.
 *
 * **CSR, and no atomics anywhere.** `f32 +=` is not associative, so an atomic
 * scatter would give a different answer per run on identical input — the engine
 * would lose reproducibility to gain a reduce. Here every `(s, d)` is written by
 * exactly one thread which sums its contributions in ascending `row_idx` order,
 * so the result is fixed by the CSR layout the host built and by nothing else. */
__global__ static void grouped_reduce(float *out,const float *y,const int *row_ptr,
                                        const int *row_idx,const float *rw,int D){
    int s=blockIdx.x;
    int lo=row_ptr[s],hi=row_ptr[s+1];
    for(int d=threadIdx.x+blockIdx.y*blockDim.x;d<D;d+=blockDim.x*gridDim.y){
        float acc=0.f;
        for(int j=lo;j<hi;j++){int k=row_idx[j];acc+=rw[k]*y[(size_t)k*D+d];}
        out[(size_t)s*D+d]=acc;
    }
}

__global__ static void silu_mul(float *gate, const float *up, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = gate[i];
        gate[i] = (v / (1.0f + expf(-v))) * up[i];
    }
}

/* Four warps share one A tile and compute TM x (4*TN) outputs.  This matters for
 * prefill: the first prototype reloaded/converted A once per 16 output cols.
 *
 * Templated on the WMMA fragment shape because that shape is the one thing about
 * this kernel worth tuning per (M, K, N), and fp16 WMMA admits exactly three:
 * 16x16x16, 32x8x16 and 8x32x16. `WmmaTuner` measures them and picks; `<16,16,16>`
 * is the historical instantiation and stays the default, so an unmeasured run
 * executes precisely the code it always did. */
template<int TM,int TN,int TK>
__global__ static void w4a16_matmul_t(float *y,const float *x,const uint8_t *w,
                                    const float *scale,int M,int K,int N){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    constexpr int AS=TM*TK, BS=TN*TK, CS=TM*TN;
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int m0=blockIdx.y*TM,n0=blockIdx.x*(TN*4)+warp*TN;
    __shared__ __half ah[AS],bh[4][BS];
    wmma::fragment<wmma::accumulator,TM,TN,TK,float> acc;wmma::fill_fragment(acc,0.f);
    size_t rb=(size_t)(K+1)/2;
    for(int k0=0;k0<K;k0+=TK){
        for(int z=threadIdx.x;z<AS;z+=blockDim.x){
            int m=z/TK,k=z%TK,gm=m0+m,gk=k0+k;
            ah[z]=(gm<M&&gk<K)?__float2half(x[(size_t)gm*K+gk]):__float2half(0.f);
        }
        for(int z=lane;z<BS;z+=32){
            int n=z/TK,gk=k0+(z%TK),gn=n0+n;float v=0.f;
            if(gn<N&&gk<K){uint8_t q=w[(size_t)gn*rb+(gk>>1)];int a=(gk&1)?q>>4:q&15;
                v=(float)(a&8?a-16:a)*scale[gn];}
            bh[warp][z]=__float2half(v);           /* [Ntile,Ktile] == B col-major */
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a,TM,TN,TK,__half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,TM,TN,TK,__half,wmma::col_major> bf;
        wmma::load_matrix_sync(af,ah,TK);wmma::load_matrix_sync(bf,bh[warp],TK);
        wmma::mma_sync(acc,af,bf,acc);__syncthreads();
    }
    __shared__ float out[4][CS];wmma::store_matrix_sync(out[warp],acc,TN,wmma::mem_row_major);__syncwarp();
    for(int z=lane;z<CS;z+=32){int m=z/TN,n=z%TN;
        if(m0+m<M&&n0+n<N)y[(size_t)(m0+m)*N+n0+n]=out[warp][z];}
#endif
}

/* Gate and up use the same input.  Eight warps compute both TM x (4*TN)
 * projections while sharing the FP32->FP16 conversion of A. Templated on the
 * fragment shape for the same reason as `w4a16_matmul_t`. */
template<int TM,int TN,int TK>
__global__ static void w4a16_gate_up_t(float *gate,float *up,const float *x,
        const uint8_t *gw,const uint8_t *uw,const float *gs,const float *us,
        int M,int K,int N){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    constexpr int AS=TM*TK, BS=TN*TK, CS=TM*TN;
    int warp=threadIdx.x>>5,lane=threadIdx.x&31,which=warp&1,tile=warp>>1;
    int m0=blockIdx.y*TM,n0=blockIdx.x*(TN*4)+tile*TN;const uint8_t *w=which?uw:gw;
    const float *scale=which?us:gs;float *y=which?up:gate;size_t rb=(size_t)(K+1)/2;
    __shared__ __half ah[AS],bh[8][BS];
    wmma::fragment<wmma::accumulator,TM,TN,TK,float> acc;wmma::fill_fragment(acc,0.f);
    for(int k0=0;k0<K;k0+=TK){
        for(int z=threadIdx.x;z<AS;z+=blockDim.x){int m=z/TK,k=z%TK,gm=m0+m,gk=k0+k;
            ah[z]=(gm<M&&gk<K)?__float2half(x[(size_t)gm*K+gk]):__float2half(0.f);}
        for(int z=lane;z<BS;z+=32){int n=z/TK,gk=k0+(z%TK),gn=n0+n;float v=0.f;
            if(gn<N&&gk<K){uint8_t q=w[(size_t)gn*rb+(gk>>1)];int a=(gk&1)?q>>4:q&15;
                v=(float)(a&8?a-16:a)*scale[gn];}bh[warp][z]=__float2half(v);}
        __syncthreads();
        wmma::fragment<wmma::matrix_a,TM,TN,TK,__half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,TM,TN,TK,__half,wmma::col_major> bf;
        wmma::load_matrix_sync(af,ah,TK);wmma::load_matrix_sync(bf,bh[warp],TK);
        wmma::mma_sync(acc,af,bf,acc);__syncthreads();
    }
    __shared__ float out[8][CS];wmma::store_matrix_sync(out[warp],acc,TN,wmma::mem_row_major);__syncwarp();
    for(int z=lane;z<CS;z+=32){int m=z/TN,n=z%TN;
        if(m0+m<M&&n0+n<N)y[(size_t)(m0+m)*N+n0+n]=out[warp][z];}
#endif
}

/* The three fp16 WMMA fragment shapes, dispatched at runtime.
 *
 * A `switch` over template instantiations, not a runtime tile: WMMA fragment
 * shapes are compile-time — `wmma::fragment<...,M,N,K,...>` has no runtime form
 * — so "tunable tile size" for this kernel can only mean selecting among
 * instantiations. Anything outside the three legal shapes falls back to
 * 16x16x16, which is what an unmeasured or corrupt `kernel_tuning.json` gets. */
#define COLI_W4A16_TILES(EMIT) \
    EMIT(16,16,16) EMIT(32,8,16) EMIT(8,32,16)

static void w4a16_gate_up_dispatch(dim3 grid,cudaStream_t s,int tm,int tn,int tk,
        float *gate,float *up,const float *x,const uint8_t *gw,const uint8_t *uw,
        const float *gs,const float *us,int M,int K,int N){
#define COLI_EMIT(A,B,C) if(tm==A&&tn==B&&tk==C){ \
        w4a16_gate_up_t<A,B,C><<<grid,256,0,s>>>(gate,up,x,gw,uw,gs,us,M,K,N); return; }
    COLI_W4A16_TILES(COLI_EMIT)
#undef COLI_EMIT
    w4a16_gate_up_t<16,16,16><<<grid,256,0,s>>>(gate,up,x,gw,uw,gs,us,M,K,N);
}

static void w4a16_matmul_dispatch(dim3 grid,cudaStream_t s,int tm,int tn,int tk,
        float *y,const float *x,const uint8_t *w,const float *scale,int M,int K,int N){
#define COLI_EMIT(A,B,C) if(tm==A&&tn==B&&tk==C){ \
        w4a16_matmul_t<A,B,C><<<grid,128,0,s>>>(y,x,w,scale,M,K,N); return; }
    COLI_W4A16_TILES(COLI_EMIT)
#undef COLI_EMIT
    w4a16_matmul_t<16,16,16><<<grid,128,0,s>>>(y,x,w,scale,M,K,N);
}

__global__ static void quantize_s4_rows(uint8_t *q,float *scale,const float *x,int S,int K){
    int s=blockIdx.x; if(s>=S)return; const float *xs=x+(size_t)s*K;
    float v=0; for(int i=threadIdx.x;i<K;i+=blockDim.x)v=fmaxf(v,fabsf(xs[i]));
    __shared__ float m[256]; m[threadIdx.x]=v; __syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)m[threadIdx.x]=fmaxf(m[threadIdx.x],m[threadIdx.x+n]);__syncthreads();}
    float sc=m[0]>0?m[0]/7.f:1.f; if(!threadIdx.x)scale[s]=sc;
    uint8_t *dst=q+(size_t)s*((K+1)/2);
    for(int b=threadIdx.x;b<(K+1)/2;b+=blockDim.x){
        int i=b*2,a=__float2int_rn(xs[i]/sc),c=i+1<K?__float2int_rn(xs[i+1]/sc):0;
        a=max(-8,min(7,a)); c=max(-8,min(7,c)); dst[b]=(uint8_t)((a&15)|((c&15)<<4));
    }
}

__global__ static void grouped_s4_wmma(float *y,const uint8_t *x,const float *xscale,
                                        const GroupDesc *desc,int K,int O,int which){
#if __CUDA_ARCH__ >= 750
    using namespace nvcuda;
    int warp=threadIdx.x/32,lane=threadIdx.x%32,tile=blockIdx.x*8+warp,c=blockIdx.y;
    if(tile*8>=O)return; GroupDesc d=desc[c];
    const void *w=which==0?d.g:(which==1?d.u:d.d);
    const float *ws=which==0?d.gs:(which==1?d.us:d.ds);
    int fmt=which==0?d.gf:(which==1?d.uf:d.df);
    if(fmt!=2)return;
    wmma::fragment<wmma::accumulator,8,8,32,int> acc; wmma::fill_fragment(acc,0);
    const uint8_t *a=x+(size_t)d.offset*((K+1)/2);
    const uint8_t *b=(const uint8_t*)w+(size_t)(tile*8)*((K+1)/2);
    for(int k=0;k<K;k+=32){
        wmma::fragment<wmma::matrix_a,8,8,32,wmma::experimental::precision::s4,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,8,8,32,wmma::experimental::precision::s4,wmma::col_major> bf;
        wmma::load_matrix_sync(af,a+k/2,K);
        wmma::load_matrix_sync(bf,b+k/2,K);
        wmma::mma_sync(acc,af,bf,acc);
    }
    __shared__ int out[8][64]; wmma::store_matrix_sync(out[warp],acc,8,wmma::mem_row_major);
    for(int i=lane;i<64;i+=32){int s=i/8,o=tile*8+i%8;
        if(s<d.rows&&o<O)y[(size_t)(d.offset+s)*O+o]=(float)out[warp][i]*xscale[d.offset+s]*ws[o];}
#endif
}

__global__ static void grouped_hidden(float *y,const float *x,const GroupDesc *desc,
                                      int I,int D,int which){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z; GroupDesc d=desc[c];
    if(s>=d.rows) return;
    const void *w=which?d.u:d.g; const float *sc=which?d.us:d.gs; int fmt=which?d.uf:d.gf;
    size_t rb=row_bytes(fmt,D),row=(size_t)o*rb; const float *xs=x+(size_t)(d.offset+s)*D;
    float sum=0; for(int i=threadIdx.x;i<D;i+=blockDim.x) sum+=xs[i]*weight_at(w,fmt,row,i);
    __shared__ float p[256]; p[threadIdx.x]=sum; __syncthreads();
    for(int n=128;n;n>>=1){ if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n]; __syncthreads(); }
    if(!threadIdx.x) y[(size_t)(d.offset+s)*I+o]=p[0]*(fmt?sc[o]:1.f);
}

__global__ static void grouped_down(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z; GroupDesc d=desc[c];
    if(s>=d.rows) return;
    size_t rb=row_bytes(d.df,I),row=(size_t)o*rb; const float *xs=x+(size_t)(d.offset+s)*I;
    float sum=0; for(int i=threadIdx.x;i<I;i+=blockDim.x) sum+=xs[i]*weight_at(d.d,d.df,row,i);
    __shared__ float p[256]; p[threadIdx.x]=sum; __syncthreads();
    for(int n=128;n;n>>=1){ if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n]; __syncthreads(); }
    if(!threadIdx.x) y[(size_t)(d.offset+s)*D+o]=p[0]*(d.df?d.ds[o]:1.f);
}

__device__ static void unpack_s4(uint8_t v,float *lo,float *hi){
    int a=v&15,b=v>>4; *lo=(float)(a&8?a-16:a); *hi=(float)(b&8?b-16:b);
}

/* Exact low-row W4A32 path. It consumes each packed weight byte once instead
 * of routing both nibbles through weight_at(), preserving FP32 activations. */
__global__ static void grouped_hidden_w4(float *y,const float *x,const GroupDesc *desc,
                                         int I,int D,int which){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *w=(const uint8_t*)(which?d.u:d.g);const float *sc=which?d.us:d.gs;
    const uint8_t *row=w+(size_t)o*((D+1)/2);const float *xs=x+(size_t)(d.offset+s)*D;
    float sum=0;for(int b=threadIdx.x;b<(D+1)/2;b+=blockDim.x){float a,z;unpack_s4(row[b],&a,&z);
        int i=b*2;sum+=xs[i]*a;if(i+1<D)sum+=xs[i+1]*z;}
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*I+o]=p[0]*sc[o];
}

__global__ static void grouped_hidden_w4_dual(float *gate,float *up,const float *x,
                                               const GroupDesc *desc,int I,int D){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*((D+1)/2);
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*((D+1)/2);
    const float *xs=x+(size_t)(d.offset+s)*D;float ga=0,ua=0;
    for(int b=threadIdx.x;b<(D+1)/2;b+=blockDim.x){float g0,g1,u0,u1;unpack_s4(gr[b],&g0,&g1);unpack_s4(ur[b],&u0,&u1);
        int i=b*2;ga+=xs[i]*g0;ua+=xs[i]*u0;if(i+1<D){ga+=xs[i+1]*g1;ua+=xs[i+1]*u1;}}
    __shared__ float gp[256],upv[256];gp[threadIdx.x]=ga;upv[threadIdx.x]=ua;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){gp[threadIdx.x]+=gp[threadIdx.x+n];upv[threadIdx.x]+=upv[threadIdx.x+n];}__syncthreads();}
    if(!threadIdx.x){size_t z=(size_t)(d.offset+s)*I+o;gate[z]=gp[0]*d.gs[o];up[z]=upv[0]*d.us[o];}
}

__global__ static void grouped_down_w4(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *row=(const uint8_t*)d.d+(size_t)o*((I+1)/2);
    const float *xs=x+(size_t)(d.offset+s)*I;float sum=0;
    for(int b=threadIdx.x;b<(I+1)/2;b+=blockDim.x){float a,z;unpack_s4(row[b],&a,&z);
        int i=b*2;sum+=xs[i]*a;if(i+1<I)sum+=xs[i+1]*z;}
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*D+o]=p[0]*d.ds[o];
}

__global__ static void attention_absorb_kernel(float *ctx,const float *q,const float *latent,
                                                const float *rope,const void *weights,const float *wscale,
                                                int fmt,int H,int Q,int R,int V,int K,int T,float scale){
    int h=blockIdx.x,tid=threadIdx.x,rbase=h*(Q+V);extern __shared__ float sm[];
    float *qa=sm,*cl=qa+K,*scores=cl+K;
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=q[(size_t)h*(Q+R)+d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*(fmt?wscale[rbase+d]:1.f);qa[k]=a;}
    __syncthreads();
    for(int t=tid;t<T;t+=blockDim.x){float a=0;const float *lt=latent+(size_t)t*K,*rt=rope+(size_t)t*R;
        for(int k=0;k<K;k++)a+=qa[k]*lt[k];for(int d=0;d<R;d++)a+=q[(size_t)h*(Q+R)+Q+d]*rt[d];scores[t]=a*scale;}
    __syncthreads();
    if(!tid){float mx=scores[0];for(int t=1;t<T;t++)mx=fmaxf(mx,scores[t]);float z=0;
        for(int t=0;t<T;t++){scores[t]=expf(scores[t]-mx);z+=scores[t];}for(int t=0;t<T;t++)scores[t]/=z;}
    __syncthreads();
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int t=0;t<T;t++)a+=scores[t]*latent[(size_t)t*K+k];cl[k]=a;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k);ctx[(size_t)h*V+v]=a*(fmt?wscale[row]:1.f);}
}

__global__ static void attention_absorb_batch_kernel(float *ctx,const float *q,
        const float *latent,const float *rope,const void *weights,const float *wscale,
        int fmt,int S,int H,int Q,int R,int V,int K,int T,float scale){
    int s=blockIdx.y,h=blockIdx.x,tid=threadIdx.x,nt=T-S+s+1,rbase=h*(Q+V);
    if(s>=S||nt<1)return;
    extern __shared__ float sm[];float *qa=sm,*cl=qa+K,*scores=cl+K,*red=scores+T;
    const float *qs=q+((size_t)s*H+h)*(Q+R);
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=qs[d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*
          (fmt?wscale[rbase+d]:1.f);qa[k]=a;}
    __syncthreads();
    for(int t=tid;t<nt;t+=blockDim.x){float a=0;const float *lt=latent+(size_t)t*K;
        const float *rt=rope+(size_t)t*R;for(int k=0;k<K;k++)a+=qa[k]*lt[k];
        for(int d=0;d<R;d++)a+=qs[Q+d]*rt[d];scores[t]=a*scale;}
    __syncthreads();
    float local=-3.402823466e+38F;for(int t=tid;t<nt;t+=blockDim.x)local=fmaxf(local,scores[t]);
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]=fmaxf(red[tid],red[tid+n]);__syncthreads();}
    float mx=red[0];local=0;for(int t=tid;t<nt;t+=blockDim.x){float e=expf(scores[t]-mx);scores[t]=e;local+=e;}
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]+=red[tid+n];__syncthreads();}
    float inv=1.f/red[0];for(int t=tid;t<nt;t+=blockDim.x)scores[t]*=inv;
    __syncthreads();
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int t=0;t<nt;t++)
        a+=scores[t]*latent[(size_t)t*K+k];cl[k]=a;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k);
        ctx[((size_t)s*H+h)*V+v]=a*(fmt?wscale[row]:1.f);}
}

/* ---- scratch reservation, and the generation counter graph capture needs ----
 *
 * Every reserve here is grow-only and FREES BEFORE IT REALLOCATES, so a buffer's
 * device address is stable only until something asks for a larger one. A
 * captured CUDA graph bakes the addresses it was recorded with, so a replay
 * after any growth would read and write freed VRAM — silently, since the
 * allocator will happily hand those pages to something else.
 *
 * `ctx->scratch_gen` is bumped on every ACTUAL reallocation (not on a satisfied
 * request), and a cached graph carries the generation it was captured under.
 * Mismatch means discard, not replay. Threading `ctx` through these three
 * helpers rather than bumping at the call sites is deliberate: `dc->y` is the
 * *same* buffer as `ctx->y`, so an attention call can invalidate an
 * expert-group graph, and a scheme that required each caller to remember would
 * fail exactly at the call site nobody thought about. */
static void note_realloc(DeviceContext *ctx) {
    if (ctx) ctx->scratch_gen++;
}

static int reserve(DeviceContext *ctx, float **ptr, size_t *cap, size_t bytes) {
    if (*cap >= bytes) return 1;
    if (*ptr) cudaFree(*ptr);
    *ptr = nullptr;
    *cap = 0;
    note_realloc(ctx);
    if (!cuda_ok(cudaMalloc(ptr, bytes), "scratch allocation")) return 0;
    *cap = bytes;
    return 1;
}

static int reserve_bytes(DeviceContext *ctx,void **ptr,size_t *cap,size_t bytes){
    if(*cap>=bytes) return 1; if(*ptr) cudaFree(*ptr); *ptr=nullptr; *cap=0;
    note_realloc(ctx);
    if(!cuda_ok(cudaMalloc(ptr,bytes),"descriptor allocation")) return 0; *cap=bytes; return 1;
}

static int reserve_pinned(DeviceContext *ctx,float **ptr,size_t *cap,size_t bytes){
    if(*cap>=bytes)return 1;if(*ptr)cudaFreeHost(*ptr);*ptr=nullptr;*cap=0;
    note_realloc(ctx);
    if(!cuda_ok(cudaMallocHost(ptr,bytes),"pinned staging allocation"))return 0;*cap=bytes;return 1;
}

/* Pinned host staging for the group descriptors.
 *
 * Capture rejects an async copy out of pageable memory, and `expert_group` built
 * its descriptors in a `GroupDesc host[256]` on the stack. Staging them in
 * pinned memory is what makes the H2D copy recordable — and, because a graph
 * replays the copy rather than the values, it is also what lets one captured
 * graph serve a *different set of experts* at the same shape: rewrite the pinned
 * buffer, replay, and the descriptors the kernels read are the new ones. */
static int reserve_pinned_bytes(DeviceContext *ctx,void **ptr,size_t *cap,size_t bytes){
    if(*cap>=bytes)return 1;if(*ptr)cudaFreeHost(*ptr);*ptr=nullptr;*cap=0;
    note_realloc(ctx);
    if(!cuda_ok(cudaMallocHost(ptr,bytes),"pinned descriptor staging"))return 0;*cap=bytes;return 1;
}

extern "C" int coli_cuda_init(const int *devices, int count) {
    int available = 0;
    if (!devices || count < 1 || count > COLI_CUDA_MAX_DEVICES) return 0;
    if (!cuda_ok(cudaGetDeviceCount(&available), "device discovery")) return 0;
    g_nctx = 0;
    for (int i = 0; i < count; i++) {
        int device = devices[i];
        if (device < 0 || device >= available) {
            std::fprintf(stderr, "[CUDA] invalid device %d (available: 0..%d)\n", device, available - 1);
            g_nctx = 0;
            return 0;
        }
        if (find_ctx(device)) {
            std::fprintf(stderr, "[CUDA] duplicate device %d\n", device);
            g_nctx = 0;
            return 0;
        }
        DeviceContext *ctx = &g_ctx[g_nctx];
        *ctx = {};
        ctx->device = device;
        if (!select_ctx(ctx)) { g_nctx = 0; return 0; }
        cudaDeviceProp prop{};
        if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "device properties")) { g_nctx = 0; return 0; }
        ctx->compute_major=prop.major;ctx->compute_minor=prop.minor;
        if(!cuda_ok(cudaStreamCreateWithFlags(&ctx->stream,cudaStreamNonBlocking),"stream creation")){
            g_nctx=0;return 0;
        }
        g_nctx++;
        std::fprintf(stderr, "[CUDA] device %d: %s, %.1f GB VRAM, sm_%d%d\n",
                     device, prop.name, prop.totalGlobalMem / 1e9, prop.major, prop.minor);
    }
    return 1;
}

/* Defined with the rest of the graph cache below; declared here because
 * shutdown must free the instantiated graphs before it frees the stream and the
 * scratch they were captured against. */
static void graph_cache_clear(DeviceContext *ctx);

extern "C" void coli_cuda_shutdown(void) {
    for (int i = 0; i < g_nctx; i++) {
        DeviceContext *ctx = &g_ctx[i];
        if (!select_ctx(ctx)) continue;
        /* Graphs first: they hold device pointers into the buffers freed below
         * and an exec handle bound to the stream destroyed below. */
        graph_cache_clear(ctx);
        if (ctx->x) cudaFree(ctx->x);
        if (ctx->y) cudaFree(ctx->y);
        if (ctx->gate) cudaFree(ctx->gate);
        if (ctx->up) cudaFree(ctx->up);
        if (ctx->qx) cudaFree(ctx->qx);
        if (ctx->qscale) cudaFree(ctx->qscale);
        if(ctx->aq)cudaFree(ctx->aq);if(ctx->al)cudaFree(ctx->al);if(ctx->ar)cudaFree(ctx->ar);if(ctx->ac)cudaFree(ctx->ac);
        for(int b=0;b<24;b++) if(ctx->pipe_buf[b]) cudaFree(ctx->pipe_buf[b]);
        if (ctx->red_meta) cudaFree(ctx->red_meta);
        if (ctx->red_out) cudaFree(ctx->red_out);
        if (ctx->host_x) cudaFreeHost(ctx->host_x);
        if (ctx->host_y) cudaFreeHost(ctx->host_y);
        if (ctx->host_desc) cudaFreeHost(ctx->host_desc);
        if (ctx->host_red) cudaFreeHost(ctx->host_red);
        if (ctx->host_red_out) cudaFreeHost(ctx->host_red_out);
        if (ctx->stream) cudaStreamDestroy(ctx->stream);
        if (ctx->group_desc) cudaFree(ctx->group_desc);
        ctx->x = ctx->y = ctx->gate = ctx->up = nullptr;
        ctx->qx=nullptr; ctx->qscale=nullptr;
        ctx->aq=ctx->al=ctx->ar=ctx->ac=nullptr;
        ctx->host_x=ctx->host_y=nullptr;ctx->stream=nullptr;
        ctx->x_cap = ctx->y_cap = ctx->gate_cap = ctx->up_cap = 0;
        ctx->qx_cap=ctx->qscale_cap=0;
        ctx->aq_cap=ctx->al_cap=ctx->ar_cap=ctx->ac_cap=0;
        ctx->host_x_cap=ctx->host_y_cap=0;
        ctx->host_desc=nullptr; ctx->host_desc_cap=0;
        ctx->red_meta=nullptr; ctx->red_meta_cap=0;
        ctx->red_out=nullptr; ctx->red_out_cap=0;
        ctx->host_red=nullptr; ctx->host_red_cap=0;
        ctx->host_red_out=nullptr; ctx->host_red_out_cap=0;
        ctx->group_desc=nullptr; ctx->group_desc_cap=0;
    }
    g_nctx = 0;
}

extern "C" int coli_cuda_device_count(void) { return g_nctx; }

/* How many CUDA devices the DRIVER reports, independent of whether this process
 * has initialized any.
 *
 * `coli_cuda_device_count` above returns `g_nctx` — the number of contexts
 * *this process built* — which is the right answer for "which devices may I
 * address" and the wrong one for "does this host have a GPU". Anything using
 * the latter to decide whether to call `coli_cuda_init` is circular: the count
 * is 0 until init runs, so the gate never opens. That is exactly what
 * `peregrine_cuda::is_available()` did, and it reported "unavailable" on a
 * working RTX 3060.
 *
 * Creates no context: `cudaGetDeviceCount` only queries the driver, so this is
 * safe to call before init and cheap enough for a startup banner. Returns 0
 * rather than a negative on driver errors (no driver, no permission) because
 * every caller wants "how many can I use", and "none" is the honest answer to
 * that in all of those cases. */
extern "C" int coli_cuda_probe_device_count(void) {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess || n < 0) {
        /* Clear the sticky error so a later real call is not misattributed. */
        cudaGetLastError();
        return 0;
    }
    return n;
}

extern "C" int coli_cuda_device_at(int index) {
    return index >= 0 && index < g_nctx ? g_ctx[index].device : -1;
}

extern "C" int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes) {
    DeviceContext *ctx = find_ctx(device);
    if (!free_bytes || !total_bytes || !select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemGetInfo(free_bytes, total_bytes), "memory info");
}

extern "C" void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes) {
    size_t count = 0, bytes = 0;
    for (int i = 0; i < g_nctx; i++) if (device < 0 || g_ctx[i].device == device) {
        count += g_ctx[i].tensor_count;
        bytes += g_ctx[i].tensor_bytes;
    }
    if (tensor_count) *tensor_count = count;
    if (tensor_bytes) *tensor_bytes = bytes;
}

extern "C" void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                                        double *h2d_ms, double *kernel_ms, double *d2h_ms) {
    if(calls) *calls=g_group_calls; if(experts) *experts=g_group_experts; if(rows) *rows=g_group_rows;
    if(h2d_ms) *h2d_ms=g_group_h2d_ms; if(kernel_ms) *kernel_ms=g_group_kernel_ms;
    if(d2h_ms) *d2h_ms=g_group_d2h_ms;
}

/* ---- CUDA Graphs: capture a device's managed stream once, replay it many times.
 * The steady-state decode step (stable shapes) is captured between begin/end;
 * every replay skips the per-op launch cost. Ops to capture must run on
 * `ctx->stream` (the `pipe_*` primitives) — synchronous copies break capture. */
struct ColiCudaGraph { cudaGraph_t graph; cudaGraphExec_t exec; int device; };

extern "C" int coli_cuda_graph_begin(int device) {
    DeviceContext *ctx = find_ctx(device);
    if (!select_ctx(ctx)) return 0;
    return cuda_ok(cudaStreamBeginCapture(ctx->stream, cudaStreamCaptureModeThreadLocal),
                   "graph begin capture");
}

extern "C" int coli_cuda_graph_end(int device, ColiCudaGraph **out) {
    DeviceContext *ctx = find_ctx(device);
    if (!out || !select_ctx(ctx)) return 0;
    cudaGraph_t graph = NULL;
    if (!cuda_ok(cudaStreamEndCapture(ctx->stream, &graph), "graph end capture")) return 0;
    cudaGraphExec_t exec = NULL;
    if (!cuda_ok(cudaGraphInstantiate(&exec, graph, 0), "graph instantiate")) {
        cudaGraphDestroy(graph);
        return 0;
    }
    ColiCudaGraph *g = (ColiCudaGraph *)std::calloc(1, sizeof(*g));
    if (!g) { cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); return 0; }
    g->graph = graph; g->exec = exec; g->device = device;
    *out = g;
    return 1;
}

extern "C" int coli_cuda_graph_launch(ColiCudaGraph *g) {
    if (!g) return 0;
    DeviceContext *ctx = find_ctx(g->device);
    if (!select_ctx(ctx)) return 0;
    if (!cuda_ok(cudaGraphLaunch(g->exec, ctx->stream), "graph launch")) return 0;
    return cuda_ok(cudaStreamSynchronize(ctx->stream), "graph synchronize");
}

extern "C" void coli_cuda_graph_free(ColiCudaGraph *g) {
    if (!g) return;
    if (g->exec) cudaGraphExecDestroy(g->exec);
    if (g->graph) cudaGraphDestroy(g->graph);
    std::free(g);
}

/* ---- the expert-group graph cache (COLI_CUDA_GRAPH) ----
 *
 * The decode loop calls `expert_group` once per sparse layer per token, and at
 * B=1 every routed expert contributes exactly one row — so the *launch shape*
 * (arm, expert count, D, I, rows) repeats constantly while the *contents*
 * (which experts, what activations) change every call. That is exactly the
 * split CUDA Graphs exist for: the shape becomes an instantiated graph, and the
 * contents ride in through pinned staging buffers the graph copies from.
 *
 * The counters are here because "is it replaying" is not observable otherwise,
 * and a shape key that churns would leave this capturing on every call — slower
 * than not having it, with nothing in the output to say so. */
static uint64_t g_graph_captures, g_graph_replays, g_graph_invalidations, g_graph_misses;
static std::mutex g_graph_stats_mu;

extern "C" void coli_cuda_graph_cache_stats(uint64_t *captures, uint64_t *replays,
                                              uint64_t *invalidations, uint64_t *uncacheable) {
    std::lock_guard<std::mutex> lock(g_graph_stats_mu);
    if (captures) *captures = g_graph_captures;
    if (replays) *replays = g_graph_replays;
    if (invalidations) *invalidations = g_graph_invalidations;
    if (uncacheable) *uncacheable = g_graph_misses;
}

/* FNV-1a over the launch shape. Not a hash of the *inputs*: two calls with the
 * same shape and different experts must collide here, because that is the whole
 * benefit — one graph serving every generation at that shape. */
static uint64_t graph_key(int arm, int count, int D, int I, const int *rows, int extra) {
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&h](uint64_t v) {
        for (int b = 0; b < 8; b++) { h ^= (v >> (b * 8)) & 0xFF; h *= 1099511628211ULL; }
    };
    mix((uint64_t)arm); mix((uint64_t)count); mix((uint64_t)D); mix((uint64_t)I); mix((uint64_t)extra);
    for (int c = 0; c < count; c++) mix((uint64_t)rows[c]);
    /* 0 marks an empty slot, so a shape that hashes to it takes 1 instead —
     * a collision with one other shape at worst, never a lost entry. */
    return h ? h : 1;
}

/* Look up a live graph for `key`, discarding any entry captured under a stale
 * scratch generation. Returns NULL on miss. */
static ColiCudaGraph *graph_lookup(DeviceContext *ctx, uint64_t key) {
    for (int i = 0; i < COLI_CUDA_GRAPH_CACHE; i++) {
        GraphSlot *s = &ctx->graphs[i];
        if (s->key != key || !s->g) continue;
        if (s->gen != ctx->scratch_gen) {
            /* A scratch buffer grew since capture, so this graph's baked device
             * pointers are dangling. Free it rather than replay it. */
            coli_cuda_graph_free(s->g);
            s->g = NULL; s->key = 0;
            std::lock_guard<std::mutex> lock(g_graph_stats_mu);
            g_graph_invalidations++;
            return NULL;
        }
        s->used = ++ctx->graph_clock;
        return s->g;
    }
    return NULL;
}

/* Store `g` under `key`, evicting the least recently used slot when full. */
static void graph_store(DeviceContext *ctx, uint64_t key, ColiCudaGraph *g) {
    int victim = 0;
    for (int i = 0; i < COLI_CUDA_GRAPH_CACHE; i++) {
        GraphSlot *s = &ctx->graphs[i];
        if (!s->g) { victim = i; break; }
        if (s->used < ctx->graphs[victim].used || ctx->graphs[victim].g == NULL) victim = i;
    }
    GraphSlot *s = &ctx->graphs[victim];
    if (s->g) coli_cuda_graph_free(s->g);
    s->key = key; s->g = g; s->gen = ctx->scratch_gen; s->used = ++ctx->graph_clock;
}

/* Drop every cached graph on a device (shutdown, and any wholesale change). */
static void graph_cache_clear(DeviceContext *ctx) {
    for (int i = 0; i < COLI_CUDA_GRAPH_CACHE; i++) {
        if (ctx->graphs[i].g) coli_cuda_graph_free(ctx->graphs[i].g);
        ctx->graphs[i].g = NULL; ctx->graphs[i].key = 0;
    }
}

extern "C" int coli_cuda_tensor_upload(ColiCudaTensor **tensor,
                                        const void *weights, const float *scales,
                                        int fmt, int I, int O, int device) {
    DeviceContext *ctx = find_ctx(device);
    if (!tensor || !weights || I < 1 || O < 1 || !select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    if (!rb || (fmt && !scales)) return 0;
    if (*tensor) {
        ColiCudaTensor *t = *tensor;
        return t->fmt == fmt && t->I == I && t->O == O && t->device == device;
    }
    ColiCudaTensor *t = static_cast<ColiCudaTensor *>(std::calloc(1, sizeof(*t)));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->device = device; t->weight_bytes = rb * (size_t)O;
    if (!cuda_ok(cudaMalloc(&t->weights, t->weight_bytes), "tensor allocation") ||
        !cuda_ok(cudaMemcpy(t->weights, weights, t->weight_bytes, cudaMemcpyHostToDevice), "tensor upload")) {
        coli_cuda_tensor_free(t);
        return 0;
    }
    /* The conversion runs on the default stream, but every consumer of these
     * weights (`expert_group`, `pipe_gemm`) runs on `ctx->stream`, which is
     * NON-BLOCKING and therefore not ordered against it. Without the sync below
     * a kernel could read unconverted offset-encoded nibbles — rare, because
     * there is always host work between an upload and its first use, and
     * invisible when it happens, because the GPU arm is already documented as
     * not token-identical. Uploading is a blocking multi-megabyte H2D copy
     * already, so making it also *finish* before returning costs nothing and
     * gives the function the contract every caller assumes it had. */
    if(fmt==2){offset_to_signed_s4<<<(unsigned)((t->weight_bytes+255)/256),256>>>((uint8_t*)t->weights,t->weight_bytes);
        if(!cuda_ok(cudaGetLastError(),"int4 weight conversion")||
           !cuda_ok(cudaStreamSynchronize(0),"int4 weight conversion sync")){coli_cuda_tensor_free(t);return 0;}}
    if (fmt) {
        if (!cuda_ok(cudaMalloc(&t->scales, (size_t)O * sizeof(float)), "scale allocation") ||
            !cuda_ok(cudaMemcpy(t->scales, scales, (size_t)O * sizeof(float), cudaMemcpyHostToDevice), "scale upload")) {
            coli_cuda_tensor_free(t);
            return 0;
        }
    }
    t->tracked = 1;
    ctx->tensor_count++;
    ctx->tensor_bytes += t->weight_bytes + (fmt ? (size_t)O * sizeof(float) : 0);
    *tensor = t;
    return 1;
}

extern "C" int coli_cuda_tensor_update(ColiCudaTensor *tensor,
                                          const void *weights,
                                          const float *scales) {
    if (!tensor || !weights || (tensor->fmt && !scales)) return 0;
    DeviceContext *ctx=find_ctx(tensor->device);
    if (!select_ctx(ctx)) return 0;
    if (!cuda_ok(cudaMemcpy(tensor->weights,weights,tensor->weight_bytes,
                            cudaMemcpyHostToDevice),"tensor refresh")) return 0;
    if(tensor->fmt==2){
        offset_to_signed_s4<<<(unsigned)((tensor->weight_bytes+255)/256),256>>>(
            (uint8_t*)tensor->weights,tensor->weight_bytes);
        /* Same ordering hazard as tensor_upload, and worse here: a refresh
         * happens during `reheat`, i.e. between decode steps, so the window
         * before the next `expert_group` is far shorter than at startup. */
        if(!cuda_ok(cudaGetLastError(),"int4 weight refresh")||
           !cuda_ok(cudaStreamSynchronize(0),"int4 weight refresh sync")) return 0;
    }
    return !tensor->fmt || cuda_ok(cudaMemcpy(tensor->scales,scales,
        (size_t)tensor->O*sizeof(float),cudaMemcpyHostToDevice),"scale refresh");
}

extern "C" int coli_cuda_matmul(ColiCudaTensor **tensor,
                                 float *y, const float *x,
                                 const void *weights, const float *scales,
                                 int fmt, int S, int I, int O, int device) {
    if (S < 1 || !coli_cuda_tensor_upload(tensor, weights, scales, fmt, I, O, device)) return 0;
    ColiCudaTensor *t = *tensor;
    DeviceContext *ctx = find_ctx(t->device);
    if (!select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);
    if (!reserve(ctx, &ctx->x, &ctx->x_cap, xb) || !reserve(ctx, &ctx->y, &ctx->y_cap, yb)) return 0;
    if (!cuda_ok(cudaMemcpy(ctx->x, x, xb, cudaMemcpyHostToDevice), "input upload")) return 0;
    dim3 grid((unsigned)O, (unsigned)S);
    quant_matmul<<<grid, 256>>>(ctx->y, ctx->x, t->weights, t->scales, fmt, S, I, O, rb);
    if (!cuda_ok(cudaGetLastError(), "matmul launch") ||
        !cuda_ok(cudaMemcpy(y, ctx->y, yb, cudaMemcpyDeviceToHost), "output download")) return 0;
    return 1;
}

extern "C" int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                                      ColiCudaTensor *down, float *y,
                                      const float *x, int S) {
    if (!gate || !up || !down || !x || !y || S < 1 ||
        gate->device != up->device || gate->device != down->device ||
        gate->I != up->I || gate->O != up->O ||
        down->I != gate->O || down->O != gate->I) return 0;
    DeviceContext *ctx = find_ctx(gate->device);
    if (!select_ctx(ctx)) return 0;
    int D = gate->I, I = gate->O;
    size_t xb=(size_t)S*D*sizeof(float), ib=(size_t)S*I*sizeof(float);
    size_t yb=(size_t)S*D*sizeof(float);
    if (!reserve(ctx, &ctx->x,&ctx->x_cap,xb) || !reserve(ctx, &ctx->y,&ctx->y_cap,yb) ||
        !reserve(ctx, &ctx->gate,&ctx->gate_cap,ib) || !reserve(ctx, &ctx->up,&ctx->up_cap,ib)) return 0;
    if (!cuda_ok(cudaMemcpy(ctx->x,x,xb,cudaMemcpyHostToDevice),"expert input upload")) return 0;
    dim3 hidden_grid((unsigned)I,(unsigned)S), output_grid((unsigned)D,(unsigned)S);
    quant_matmul<<<hidden_grid,256>>>(ctx->gate,ctx->x,gate->weights,gate->scales,
        gate->fmt,S,D,I,row_bytes(gate->fmt,D));
    quant_matmul<<<hidden_grid,256>>>(ctx->up,ctx->x,up->weights,up->scales,
        up->fmt,S,D,I,row_bytes(up->fmt,D));
    size_t n=(size_t)S*I;
    silu_mul<<<(unsigned)((n+255)/256),256>>>(ctx->gate,ctx->up,n);
    quant_matmul<<<output_grid,256>>>(ctx->y,ctx->gate,down->weights,down->scales,
        down->fmt,S,I,D,row_bytes(down->fmt,I));
    if (!cuda_ok(cudaGetLastError(),"expert MLP launch") ||
        !cuda_ok(cudaMemcpy(y,ctx->y,yb,cudaMemcpyDeviceToHost),"expert output download")) return 0;
    return 1;
}

extern "C" int coli_cuda_shared_mlp_w4a16(ColiCudaTensor *gate,ColiCudaTensor *up,
        ColiCudaTensor *down,float *y,const float *x,int S){
    if(!gate||!up||!down||!x||!y||S<1||gate->fmt!=2||up->fmt!=2||down->fmt!=2||
       gate->device!=up->device||gate->device!=down->device||gate->I!=up->I||
       gate->O!=up->O||down->I!=gate->O||down->O!=gate->I)return 0;
    DeviceContext *ctx=find_ctx(gate->device);if(!select_ctx(ctx)||ctx->compute_major<7)return 0;
    int D=gate->I,I=gate->O;size_t xb=(size_t)S*D*sizeof(float),ib=(size_t)S*I*sizeof(float);
    if(!reserve(ctx, &ctx->x,&ctx->x_cap,xb)||!reserve(ctx, &ctx->gate,&ctx->gate_cap,ib)||
       !reserve(ctx, &ctx->up,&ctx->up_cap,ib)||!reserve(ctx, &ctx->y,&ctx->y_cap,xb)||
       !reserve_pinned(ctx, &ctx->host_x,&ctx->host_x_cap,xb)||
       !reserve_pinned(ctx, &ctx->host_y,&ctx->host_y_cap,xb))return 0;
    std::memcpy(ctx->host_x,x,xb);
    if(!cuda_ok(cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream),
                               "shared w4a16 input upload"))return 0;
    dim3 hidden((unsigned)((I+63)/64),(unsigned)((S+15)/16));
    dim3 output((unsigned)((D+63)/64),(unsigned)((S+15)/16));
    /* The shared expert is not part of the routed group and is not tuned: it
     * runs at the default fragment shape, which is what this call has always
     * used. Kept explicit rather than defaulted so a future tuner has to decide
     * to include it rather than inherit it by accident. */
    w4a16_gate_up_t<16,16,16><<<hidden,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,
        (const uint8_t*)gate->weights,(const uint8_t*)up->weights,gate->scales,up->scales,S,D,I);
    silu_mul<<<(unsigned)(((size_t)S*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)S*I);
    w4a16_matmul_t<16,16,16><<<output,128,0,ctx->stream>>>(ctx->y,ctx->gate,(const uint8_t*)down->weights,down->scales,S,I,D);
    if(!cuda_ok(cudaGetLastError(),"shared w4a16 launch")||
       !cuda_ok(cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream),
                               "shared w4a16 output download")||
       !cuda_ok(cudaStreamSynchronize(ctx->stream),"shared w4a16 synchronize"))return 0;
    std::memcpy(y,ctx->host_y,xb);
    return 1;
}

/* Which kernel arm a call takes. Also the first component of the graph key: two
 * calls of the same shape on different arms are different launch sequences. */
enum { ARM_TC_INT4 = 0, ARM_W4A16 = 1, ARM_W4_PACKED = 2, ARM_GENERIC = 3 };

static int select_arm(DeviceContext *ctx, int all_s4, int D, int I, const int *rows, int count) {
    int tc = getenv("COLI_CUDA_TC_INT4") && atoi(getenv("COLI_CUDA_TC_INT4"));
    tc = tc && all_s4 && D % 32 == 0 && I % 32 == 0 && D % 8 == 0 && I % 8 == 0;
    int tc_min = getenv("COLI_CUDA_TC_MIN_ROWS") ? atoi(getenv("COLI_CUDA_TC_MIN_ROWS")) : 8;
    for (int c = 0; c < count && tc; c++) tc = rows[c] >= tc_min;
    if (tc) return ARM_TC_INT4;
    if (all_s4 && ctx->compute_major >= 7 && getenv("COLI_CUDA_TC_W4A16") && atoi(getenv("COLI_CUDA_TC_W4A16")))
        return ARM_W4A16;
    if (all_s4 && (!getenv("COLI_CUDA_W4_PACKED") || atoi(getenv("COLI_CUDA_W4_PACKED")))) return ARM_W4_PACKED;
    return ARM_GENERIC;
}

/* Scratch an arm needs beyond the common buffers.
 *
 * Split out of the dispatch because `cudaMalloc` is illegal during stream
 * capture: every allocation an arm can make has to happen before `graph_begin`,
 * or the capture aborts on the one call whose activation quantization buffer
 * happened to need growing. */
static int prepare_arm(DeviceContext *ctx, int arm, int total, int D, int I) {
    if (arm != ARM_TC_INT4) return 1;
    size_t qb = (size_t)(total + 7) * (size_t)(D > I ? D : I) / 2;
    return reserve_bytes(ctx, (void **)&ctx->qx, &ctx->qx_cap, qb) &&
           reserve(ctx, &ctx->qscale, &ctx->qscale_cap, (size_t)(total + 7) * sizeof(float));
}

/* Issue one arm's kernels on `ctx->stream`. Launches only — no allocation, no
 * synchronization, no host memory access — so the identical call is valid
 * whether the stream is capturing or executing. That equivalence is what makes
 * "the graph does what eager mode does" true by construction rather than by a
 * second implementation that has to be kept in step. */
/* The WMMA fragment shape the W4A16 arm should use, `{0,0,0}` meaning "the
 * default". Travels as one struct because a partially-overridden tile is not a
 * tile — WMMA shapes are legal only as complete triples. */
typedef struct { int m, n, k; } WmmaTile;

static void dispatch_arm(DeviceContext *ctx, int arm, const GroupDesc *host, GroupDesc *dev,
                         const int *rows, int count, int D, int I, int total, int max_rows,
                         WmmaTile tile) {
    if (arm == ARM_TC_INT4) {
        size_t qb = (size_t)(total + 7) * (size_t)(D > I ? D : I) / 2;
        cudaMemsetAsync(ctx->qx, 0, qb, ctx->stream);
        quantize_s4_rows<<<total,256,0,ctx->stream>>>(ctx->qx,ctx->qscale,ctx->x,total,D);
        grouped_s4_wmma<<<dim3((unsigned)((I+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->gate,ctx->qx,ctx->qscale,dev,D,I,0);
        grouped_s4_wmma<<<dim3((unsigned)((I+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->up,ctx->qx,ctx->qscale,dev,D,I,1);
        silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        quantize_s4_rows<<<total,256,0,ctx->stream>>>(ctx->qx,ctx->qscale,ctx->gate,total,I);
        grouped_s4_wmma<<<dim3((unsigned)((D+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->y,ctx->qx,ctx->qscale,dev,I,D,2);
    } else if (arm == ARM_W4A16) {
        /* W4A16 Tensor Core per gruppo: attivazioni fp16 per tile (lossless al
         * contrario del path W4A4), un lancio per expert dentro lo stream —
         * l'overhead di lancio e' trascurabile rispetto ai GEMM.
         *
         * NOTE: this arm passes `host[c].g/u/d` — DEVICE WEIGHT POINTERS — as
         * kernel arguments, so a captured graph would be bound to the expert set
         * it was recorded with. That is why it is excluded from the graph cache;
         * see `graph_cacheable_arm`. */
        int tc16_min=getenv("COLI_CUDA_TC_W4A16_MIN")?atoi(getenv("COLI_CUDA_TC_W4A16_MIN")):16;
        int off16=0;
        for(int c=0;c<count;c++){
            int r=rows[c];
            float *g16=ctx->gate+(size_t)off16*I,*u16=ctx->up+(size_t)off16*I;
            float *x16=ctx->x+(size_t)off16*D,*y16=ctx->y+(size_t)off16*D;
            if(r>=tc16_min){
                /* Grid follows the tile: each block covers TM rows and 4*TN
                 * columns, so a hardcoded (63/64, 15/16) would under-cover
                 * every shape but 16x16x16 and silently leave columns unwritten. */
                int tm=tile.m?tile.m:16,tn=tile.n?tile.n:16,tk=tile.k?tile.k:16;
                dim3 hg16((unsigned)((I+tn*4-1)/(tn*4)),(unsigned)((r+tm-1)/tm));
                dim3 og16((unsigned)((D+tn*4-1)/(tn*4)),(unsigned)((r+tm-1)/tm));
                w4a16_gate_up_dispatch(hg16,ctx->stream,tm,tn,tk,g16,u16,x16,
                    (const uint8_t*)host[c].g,(const uint8_t*)host[c].u,host[c].gs,host[c].us,r,D,I);
                silu_mul<<<(unsigned)(((size_t)r*I+255)/256),256,0,ctx->stream>>>(g16,u16,(size_t)r*I);
                w4a16_matmul_dispatch(og16,ctx->stream,tm,tn,tk,y16,g16,
                    (const uint8_t*)host[c].d,host[c].ds,r,I,D);
            }else{
                /* piccoli batch: tile TC quasi vuoti + overhead di lancio — il
                 * kernel naive per-elemento resta piu' veloce (misurato in decode) */
                quant_matmul<<<dim3((unsigned)I,(unsigned)r),256,0,ctx->stream>>>(g16,x16,
                    host[c].g,host[c].gs,host[c].gf,r,D,I,row_bytes(host[c].gf,D));
                quant_matmul<<<dim3((unsigned)I,(unsigned)r),256,0,ctx->stream>>>(u16,x16,
                    host[c].u,host[c].us,host[c].uf,r,D,I,row_bytes(host[c].uf,D));
                silu_mul<<<(unsigned)(((size_t)r*I+255)/256),256,0,ctx->stream>>>(g16,u16,(size_t)r*I);
                quant_matmul<<<dim3((unsigned)D,(unsigned)r),256,0,ctx->stream>>>(y16,g16,
                    host[c].d,host[c].ds,host[c].df,r,I,D,row_bytes(host[c].df,I));
            }
            off16+=r;
        }
    } else if (arm == ARM_W4_PACKED) {
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        int dual=!getenv("COLI_CUDA_DUAL_PROJ")||atoi(getenv("COLI_CUDA_DUAL_PROJ"));
        if(dual)grouped_hidden_w4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
        else{
            grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D,0);
            grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->up,ctx->x,dev,I,D,1);
        }
        silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        grouped_down_w4<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    } else {
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        grouped_hidden<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D,0);
        grouped_hidden<<<hg,256,0,ctx->stream>>>(ctx->up,ctx->x,dev,I,D,1);
        silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        grouped_down<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }
}

/* An arm is cacheable when every kernel argument it passes is either a stable
 * scratch pointer or a shape — i.e. when the only thing that varies between
 * calls of one shape rides in through the descriptor buffer the graph copies. */
static int graph_cacheable_arm(int arm) { return arm != ARM_W4A16; }

static int graph_enabled(void) {
    const char *v = getenv("COLI_CUDA_GRAPH");
    return v && atoi(v);
}

/* Launch a cached graph and wait for it, so the pinned output buffer is settled
 * before the caller copies out of it. */
static int graph_run(DeviceContext *ctx, ColiCudaGraph *g) {
    return cuda_ok(cudaGraphLaunch(g->exec, ctx->stream), "expert group graph launch") &&
           cuda_ok(cudaStreamSynchronize(ctx->stream), "expert group graph synchronize");
}

/* The optional layer-level reduce fused onto the end of a group dispatch.
 *
 * Present ⇒ the kernels write `ctx->y` as usual and one `grouped_reduce` folds
 * it into `[s_n, D]` on the device, so the D2H carries `s_n` rows instead of
 * `total`. At batch saturation those differ by the batch's expert-per-row
 * factor — ~5× at B=16 on the measured GLM-5.2 unions — which is the whole
 * point, and also why a B=1 measurement of this cannot tell you anything. */
typedef struct {
    const int *row_ptr;   /* [s_n + 1], ascending, row_ptr[s_n] == total */
    const int *row_idx;   /* [total], each entry an index into the y rows */
    const float *rw;      /* [total], the router weight of each y row */
    int s_n;
    float *out;           /* host destination, [s_n, D] */
} ReduceSpec;

static int expert_group_impl(ColiCudaTensor *const *gates,
                             ColiCudaTensor *const *ups,
                             ColiCudaTensor *const *downs,
                             const int *rows, int count,
                             float *y, const float *x,
                             const ReduceSpec *red, WmmaTile tile, int *arm_out) {
    if (!gates || !ups || !downs || !rows || !x || count < 1) return 0;
    if (!red && !y) return 0;
    if (red && (!red->row_ptr || !red->row_idx || !red->rw || !red->out || red->s_n < 1)) return 0;
    ColiCudaTensor *first=gates[0];
    if (!first) return 0;
    int device=first->device,D=first->I,I=first->O,total=0,max_rows=0;
    // At batch saturation a layer's routed union can reach all 256 experts, so a
    // saturated union dispatches in one launch instead of chunking into ≤64 groups.
    GroupDesc host[256]; if(count>256) return 0;
    int all_s4=1;
    for(int c=0;c<count;c++){
        ColiCudaTensor *g=gates[c],*u=ups[c],*d=downs[c];
        if(!g||!u||!d||rows[c]<1||g->device!=device||u->device!=device||d->device!=device||
           g->I!=D||u->I!=D||g->O!=I||u->O!=I||d->I!=I||d->O!=D) return 0;
        host[c]={g->weights,u->weights,d->weights,g->scales,u->scales,d->scales,
                 g->fmt,u->fmt,d->fmt,rows[c],total};
        all_s4&=g->fmt==2&&u->fmt==2&&d->fmt==2;
        total+=rows[c]; if(rows[c]>max_rows) max_rows=rows[c];
    }
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    size_t xb=(size_t)total*D*sizeof(float), ib=(size_t)total*I*sizeof(float);
    if(!reserve(ctx, &ctx->x,&ctx->x_cap,xb)||!reserve(ctx, &ctx->y,&ctx->y_cap,xb)||
       !reserve(ctx, &ctx->gate,&ctx->gate_cap,ib)||!reserve(ctx, &ctx->up,&ctx->up_cap,ib)||
       !reserve_bytes(ctx, &ctx->group_desc,&ctx->group_desc_cap,(size_t)count*sizeof(GroupDesc))) return 0;
    int async=!getenv("COLI_CUDA_ASYNC")||atoi(getenv("COLI_CUDA_ASYNC"));
    if(async&&(!reserve_pinned(ctx, &ctx->host_x,&ctx->host_x_cap,xb)||
               !reserve_pinned(ctx, &ctx->host_y,&ctx->host_y_cap,xb)))return 0;
    int profile=getenv("COLI_CUDA_PROFILE")&&atoi(getenv("COLI_CUDA_PROFILE"));
    GroupDesc *dev=(GroupDesc*)ctx->group_desc;
    int arm=select_arm(ctx,all_s4,D,I,rows,count);
    /* Report which arm actually ran. The tile only reaches ARM_W4A16, so a
     * caller timing tiles has to know whether its tile was even consulted —
     * mirroring the arm selection on the host would be a second copy of this
     * decision, and the two drifting is how a tuner starts recording noise. */
    if(arm_out)*arm_out=arm;
    if(!prepare_arm(ctx,arm,total,D,I))return 0;

    /* Fused-reduce scratch: one device buffer holding row_ptr | row_idx | rw,
     * plus the [s_n, D] output. Reserved here, before any capture, because
     * `cudaMalloc` is illegal on a capturing stream. The pinned mirror is
     * staged unconditionally on this path (not just under the graph knob) so
     * the H2D is async either way — the metadata is small and the copy is on
     * the critical path of every layer. */
    size_t rmb=0,rob=0; int *red_ptr_dev=NULL,*red_idx_dev=NULL; float *red_rw_dev=NULL;
    if(red){
        rmb=(size_t)(red->s_n+1+total)*sizeof(int)+(size_t)total*sizeof(float);
        rob=(size_t)red->s_n*D*sizeof(float);
        if(!reserve_bytes(ctx,&ctx->red_meta,&ctx->red_meta_cap,rmb)||
           !reserve(ctx,&ctx->red_out,&ctx->red_out_cap,rob)||
           !reserve_pinned_bytes(ctx,&ctx->host_red,&ctx->host_red_cap,rmb)||
           !reserve_pinned(ctx,&ctx->host_red_out,&ctx->host_red_out_cap,rob))return 0;
        /* Lay the three arrays out contiguously and identically on both sides,
         * so one copy moves all of them and the device offsets are arithmetic
         * rather than three separate allocations to keep in step. */
        int *hp=(int*)ctx->host_red;
        std::memcpy(hp,red->row_ptr,(size_t)(red->s_n+1)*sizeof(int));
        std::memcpy(hp+red->s_n+1,red->row_idx,(size_t)total*sizeof(int));
        std::memcpy((float*)(hp+red->s_n+1+total),red->rw,(size_t)total*sizeof(float));
        red_ptr_dev=(int*)ctx->red_meta;
        red_idx_dev=red_ptr_dev+red->s_n+1;
        red_rw_dev=(float*)(red_idx_dev+total);
    }

    /* ---- graph-cached path (COLI_CUDA_GRAPH) ----
     *
     * Requires async staging (capture cannot record a copy from pageable
     * memory), a cacheable arm, and no profiling (the event records that
     * measure the phases are not part of the work being replayed). Anything
     * else falls through to the eager path below, unchanged. */
    if(graph_enabled()&&async&&graph_cacheable_arm(arm)&&!profile){
        size_t db=(size_t)count*sizeof(GroupDesc);
        if(!reserve_pinned_bytes(ctx,&ctx->host_desc,&ctx->host_desc_cap,db))return 0;
        /* Staged BEFORE the lookup: `reserve_pinned_bytes` can bump the scratch
         * generation, and a graph looked up first would then be validated
         * against the generation it is about to be invalidated by. */
        std::memcpy(ctx->host_desc,host,db);
        std::memcpy(ctx->host_x,x,xb);
        /* `s_n` joins the key: the same expert shape reducing into a different
         * number of batch rows is a different `grouped_reduce` grid, and a
         * plain dispatch is a different sequence again (hence the -1). */
        /* The tile changes which kernel instantiation was recorded, so it has to
         * separate cache entries or a tuner switching tiles would keep replaying
         * the tile it started with. */
        uint64_t key=graph_key(arm,count,D,I,rows,
                               (red?red->s_n:-1)*1000003+tile.m*10007+tile.n*101+tile.k);
        ColiCudaGraph *g=graph_lookup(ctx,key);
        if(!g){
            if(!coli_cuda_graph_begin(device))return 0;
            /* The recorded sequence is exactly the eager one: descriptors in,
             * activations in, the arm's kernels, results out. The two copies
             * are inside the graph on purpose — that is what lets one graph
             * serve every later call at this shape with different experts and
             * different activations. */
            cudaMemcpyAsync(ctx->group_desc,ctx->host_desc,db,cudaMemcpyHostToDevice,ctx->stream);
            cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream);
            if(red)cudaMemcpyAsync(ctx->red_meta,ctx->host_red,rmb,cudaMemcpyHostToDevice,ctx->stream);
            dispatch_arm(ctx,arm,host,dev,rows,count,D,I,total,max_rows,tile);
            if(red){
                grouped_reduce<<<dim3((unsigned)red->s_n,(unsigned)((D+255)/256)),256,0,ctx->stream>>>(
                    ctx->red_out,ctx->y,red_ptr_dev,red_idx_dev,red_rw_dev,D);
                cudaMemcpyAsync(ctx->host_red_out,ctx->red_out,rob,cudaMemcpyDeviceToHost,ctx->stream);
            }else{
                cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream);
            }
            if(!coli_cuda_graph_end(device,&g)){
                /* Capture failed and the stream is no longer capturing. Drop
                 * every cached graph: a failed capture can leave sibling
                 * entries recorded against a stream state we can no longer
                 * reason about, and replaying one of those is the silent
                 * failure this whole mechanism is built to avoid. */
                graph_cache_clear(ctx);
                return 0;
            }
            graph_store(ctx,key,g);
            { std::lock_guard<std::mutex> lock(g_graph_stats_mu); g_graph_captures++; }
        }else{
            { std::lock_guard<std::mutex> lock(g_graph_stats_mu); g_graph_replays++; }
        }
        if(!graph_run(ctx,g))return 0;
        if(red)std::memcpy(red->out,ctx->host_red_out,rob);
        else std::memcpy(y,ctx->host_y,xb);
        { std::lock_guard<std::mutex> lock(g_group_stats_mu);
          g_group_calls++; g_group_experts+=(uint64_t)count; g_group_rows+=(uint64_t)total; }
        return 1;
    }
    if(graph_enabled()){ std::lock_guard<std::mutex> lock(g_graph_stats_mu); g_graph_misses++; }

    cudaError_t copy_desc=async?cudaMemcpyAsync(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),
                                                cudaMemcpyHostToDevice,ctx->stream)
                               :cudaMemcpy(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),cudaMemcpyHostToDevice);
    if(!cuda_ok(copy_desc,"expert group descriptors"))return 0;
    if(red&&!cuda_ok(cudaMemcpyAsync(ctx->red_meta,ctx->host_red,rmb,cudaMemcpyHostToDevice,ctx->stream),
                     "expert group reduce metadata"))return 0;
    cudaEvent_t ev[4]={};
    if(profile) for(int i=0;i<4;i++) if(!cuda_ok(cudaEventCreate(&ev[i]),"profile event")) profile=0;
    if(profile) cudaEventRecord(ev[0],ctx->stream);
    if(async)std::memcpy(ctx->host_x,x,xb);
    cudaError_t copy_x=async?cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream)
                            :cudaMemcpy(ctx->x,x,xb,cudaMemcpyHostToDevice);
    if(!cuda_ok(copy_x,"expert group input upload")) return 0;
    if(profile) cudaEventRecord(ev[1],ctx->stream);
    dispatch_arm(ctx,arm,host,dev,rows,count,D,I,total,max_rows,tile);
    if(red)grouped_reduce<<<dim3((unsigned)red->s_n,(unsigned)((D+255)/256)),256,0,ctx->stream>>>(
        ctx->red_out,ctx->y,red_ptr_dev,red_idx_dev,red_rw_dev,D);
    if(profile) cudaEventRecord(ev[2],ctx->stream);
    if(!async&&!cuda_ok(cudaStreamSynchronize(ctx->stream),"expert group synchronize"))return 0;
    cudaError_t copy_y;
    if(red){
        /* The whole point: `s_n` rows leave the device instead of `total`. */
        copy_y=async?cudaMemcpyAsync(ctx->host_red_out,ctx->red_out,rob,cudaMemcpyDeviceToHost,ctx->stream)
                    :cudaMemcpy(red->out,ctx->red_out,rob,cudaMemcpyDeviceToHost);
    }else{
        copy_y=async?cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream)
                    :cudaMemcpy(y,ctx->y,xb,cudaMemcpyDeviceToHost);
    }
    if(!cuda_ok(cudaGetLastError(),"expert group launch")||!cuda_ok(copy_y,"expert group output download"))return 0;
    if(async){if(!cuda_ok(cudaStreamSynchronize(ctx->stream),"expert group synchronize"))return 0;
        if(red)std::memcpy(red->out,ctx->host_red_out,rob);
        else std::memcpy(y,ctx->host_y,xb);}
    if(profile){
        cudaEventRecord(ev[3],ctx->stream); cudaEventSynchronize(ev[3]); float a=0,b=0,c=0;
        cudaEventElapsedTime(&a,ev[0],ev[1]); cudaEventElapsedTime(&b,ev[1],ev[2]);
        cudaEventElapsedTime(&c,ev[2],ev[3]);
        { std::lock_guard<std::mutex> lock(g_group_stats_mu);
          g_group_h2d_ms+=a; g_group_kernel_ms+=b; g_group_d2h_ms+=c; }
        for(int i=0;i<4;i++) cudaEventDestroy(ev[i]);
    }
    { std::lock_guard<std::mutex> lock(g_group_stats_mu);
      g_group_calls++; g_group_experts+=(uint64_t)count; g_group_rows+=(uint64_t)total; }
    return 1;
}

extern "C" int coli_cuda_expert_group(ColiCudaTensor *const *gates,
                                        ColiCudaTensor *const *ups,
                                        ColiCudaTensor *const *downs,
                                        const int *rows, int count,
                                        float *y, const float *x) {
    WmmaTile none = {0,0,0};
    return expert_group_impl(gates,ups,downs,rows,count,y,x,NULL,none,NULL);
}

extern "C" int coli_cuda_expert_group_tiled(ColiCudaTensor *const *gates,
                                              ColiCudaTensor *const *ups,
                                              ColiCudaTensor *const *downs,
                                              const int *rows, int count,
                                              float *y, const float *x,
                                              int tile_m, int tile_n, int tile_k,
                                              int *arm_out) {
    WmmaTile tile = {tile_m,tile_n,tile_k};
    return expert_group_impl(gates,ups,downs,rows,count,y,x,NULL,tile,arm_out);
}

extern "C" int coli_cuda_expert_group_reduce(ColiCudaTensor *const *gates,
                                               ColiCudaTensor *const *ups,
                                               ColiCudaTensor *const *downs,
                                               const int *rows, int count,
                                               const int *row_ptr, const int *row_idx,
                                               const float *rw, int s_n,
                                               float *out, const float *x) {
    ReduceSpec red = { row_ptr, row_idx, rw, s_n, out };
    WmmaTile none = {0,0,0};
    return expert_group_impl(gates,ups,downs,rows,count,NULL,x,&red,none,NULL);
}

extern "C" int coli_cuda_attention_absorb(ColiCudaTensor *w,float *ctx,const float *q,
                                            const float *latent,const float *rope,int H,int Q,
                                            int R,int V,int K,int T,float scale){
    if(!w||!ctx||!q||!latent||!rope||H<1||Q<1||R<1||V<1||K<1||K>512||T<1||T>4096||
       w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)H*(Q+R)*sizeof(float),lb=(size_t)T*K*sizeof(float);
    size_t rb=(size_t)T*R*sizeof(float),cb=(size_t)H*V*sizeof(float);
    if(!reserve(dc, &dc->aq,&dc->aq_cap,qb)||!reserve(dc, &dc->al,&dc->al_cap,lb)||
       !reserve(dc, &dc->ar,&dc->ar_cap,rb)||!reserve(dc, &dc->ac,&dc->ac_cap,cb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"attention q upload")||
       !cuda_ok(cudaMemcpyAsync(dc->al,latent,lb,cudaMemcpyHostToDevice,dc->stream),"attention latent upload")||
       !cuda_ok(cudaMemcpyAsync(dc->ar,rope,rb,cudaMemcpyHostToDevice,dc->stream),"attention rope upload"))return 0;
    size_t shared=(size_t)(2*K+T)*sizeof(float);
    attention_absorb_kernel<<<H,256,shared,dc->stream>>>(dc->ac,dc->aq,dc->al,dc->ar,w->weights,w->scales,
        w->fmt,H,Q,R,V,K,T,scale);
    if(!cuda_ok(cudaGetLastError(),"attention absorb launch")||
       !cuda_ok(cudaMemcpyAsync(ctx,dc->ac,cb,cudaMemcpyDeviceToHost,dc->stream),"attention context download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"attention synchronize"))return 0;
    return 1;
}

static int attention_absorb_batch_run(ColiCudaTensor *w,ColiCudaTensor *proj,float *out,
        const float *q,const float *latent,const float *rope,int S,int H,int Q,int R,int V,
        int K,int T,float scale){
    if(!w||!out||!q||!latent||!rope||S<1||H<1||Q<1||R<1||V<1||K<1||K>512||
       T<S||T>8192||w->I!=K||w->O!=H*(Q+V))return 0;
    if(proj&&(proj->device!=w->device||proj->I!=H*V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)S*H*(Q+R)*sizeof(float),lb=(size_t)T*K*sizeof(float);
    size_t rb=(size_t)T*R*sizeof(float),cb=(size_t)S*H*V*sizeof(float);
    if(!reserve(dc, &dc->aq,&dc->aq_cap,qb)||!reserve(dc, &dc->al,&dc->al_cap,lb)||
       !reserve(dc, &dc->ar,&dc->ar_cap,rb)||!reserve(dc, &dc->ac,&dc->ac_cap,cb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"attention batch q upload")||
       !cuda_ok(cudaMemcpyAsync(dc->al,latent,lb,cudaMemcpyHostToDevice,dc->stream),"attention batch latent upload")||
       !cuda_ok(cudaMemcpyAsync(dc->ar,rope,rb,cudaMemcpyHostToDevice,dc->stream),"attention batch rope upload"))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,S),256,shared,dc->stream>>>(dc->ac,dc->aq,dc->al,
        dc->ar,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale);
    if(!cuda_ok(cudaGetLastError(),"attention batch launch"))return 0;
    const float *src=dc->ac;size_t ob=cb;
    if(proj){
        ob=(size_t)S*proj->O*sizeof(float);if(!reserve(dc, &dc->y,&dc->y_cap,ob))return 0;
        quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(dc->y,dc->ac,proj->weights,
            proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I));
        if(!cuda_ok(cudaGetLastError(),"attention o_proj launch"))return 0;src=dc->y;
    }
    if(!cuda_ok(cudaMemcpyAsync(out,src,ob,cudaMemcpyDeviceToHost,dc->stream),
                               proj?"attention projected output download":"attention batch context download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"attention batch synchronize"))return 0;
    return 1;
}

extern "C" int coli_cuda_attention_absorb_batch(ColiCudaTensor *w,float *ctx,const float *q,
        const float *latent,const float *rope,int S,int H,int Q,int R,int V,int K,int T,
        float scale){
    return attention_absorb_batch_run(w,nullptr,ctx,q,latent,rope,S,H,Q,R,V,K,T,scale);
}

extern "C" int coli_cuda_attention_project_batch(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out,const float *q,const float *latent,const float *rope,int S,int H,int Q,
        int R,int V,int K,int T,float scale){
    return attention_absorb_batch_run(w,proj,out,q,latent,rope,S,H,Q,R,V,K,T,scale);
}

extern "C" void coli_cuda_tensor_free(ColiCudaTensor *tensor) {
    if (!tensor) return;
    DeviceContext *ctx = find_ctx(tensor->device);
    if (ctx) select_ctx(ctx);
    if (tensor->tracked && ctx) {
        size_t bytes = tensor->weight_bytes + (tensor->fmt ? (size_t)tensor->O * sizeof(float) : 0);
        if (ctx->tensor_count) ctx->tensor_count--;
        if (ctx->tensor_bytes >= bytes) ctx->tensor_bytes -= bytes;
    }
    if (tensor->weights) cudaFree(tensor->weights);
    if (tensor->scales) cudaFree(tensor->scales);
    std::free(tensor);
}

extern "C" size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor) {
    return tensor ? tensor->weight_bytes + (tensor->fmt ? (size_t)tensor->O * sizeof(float) : 0) : 0;
}

extern "C" int coli_cuda_tensor_device(const ColiCudaTensor *tensor) {
    return tensor ? tensor->device : -1;
}

/* ==== resident-pipeline primitives (Inc.0, 2026-07-13) ====
 * Device-side building blocks so the residual stream can stay on the layer's
 * home device across a whole layer. Control flow stays on CPU; only the data
 * plane lives here. All entry points take DEVICE pointers (no transfers) —
 * the caller owns staging via the pipe buffer API below. */

__global__ static void pipe_rmsnorm_rows(float *y,const float *x,const float *w,
                                         int D,float eps,int xstride,int ystride){
    const float *xr=x+(size_t)blockIdx.x*xstride; float *yr=y+(size_t)blockIdx.x*ystride;
    __shared__ double sh[256];
    double a=0; for(int i=threadIdx.x;i<D;i+=blockDim.x){ double v=xr[i]; a+=v*v; }
    sh[threadIdx.x]=a; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
    float r=rsqrtf((float)(sh[0]/D)+eps);
    for(int i=threadIdx.x;i<D;i+=blockDim.x) yr[i]=xr[i]*r*w[i];
}

/* RoPE interleaved, identical math to glm.c rope_interleave. One block per row;
 * row layout: v + row*stride + offset holds R floats. pos index = row/heads
 * (heads=1 for k_rot rows, heads=H for [S,H,qh] query rows). */
__global__ static void pipe_rope_rows(float *v,const int *pos,int pos_base,int stride,
                                      int offset,int R,int heads,float theta){
    float *p=v+(size_t)blockIdx.x*stride+offset;
    int half=R/2, ps=pos?pos[blockIdx.x/heads]:pos_base+(int)(blockIdx.x/heads);
    __shared__ float in[256];
    for(int j=threadIdx.x;j<R;j+=blockDim.x) in[j]=p[j];
    __syncthreads();
    for(int j=threadIdx.x;j<half;j+=blockDim.x){
        float inv=__powf(theta,-2.0f*j/R);
        float ang=ps*inv, cs=__cosf(ang), sn=__sinf(ang);
        float a=in[2*j], b=in[2*j+1];
        p[j]=a*cs-b*sn; p[half+j]=b*cs+a*sn;
    }
}

__global__ static void pipe_add_n(float *x,const float *t,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) x[i]+=t[i];
}

/* Fixed-order partial merge: block b adds partial row b into x row rows[b].
 * Target rows are unique by construction (CPU pre-sums per token), so no
 * atomics — the 9.20.7 lesson. */
__global__ static void pipe_rows_add(float *x,const float *partial,const int *rows,
                                     int D){
    float *xr=x+(size_t)rows[blockIdx.x]*D;
    const float *pr=partial+(size_t)blockIdx.x*D;
    for(int i=threadIdx.x;i<D;i+=blockDim.x) xr[i]+=pr[i];
}

/* scratch persistente per (device,slot): cresce e resta — niente cudaMalloc/Free
 * per layer (78 x ~10 alloc/richiesta erano puro churn). */
extern "C" float *coli_cuda_pipe_scratch(int device,int slot,size_t bytes){
    DeviceContext *ctx=find_ctx(device);
    if(slot<0||slot>=24||!select_ctx(ctx)) return NULL;
    if(!reserve(ctx, &ctx->pipe_buf[slot],&ctx->pipe_cap[slot],bytes)) return NULL;
    return ctx->pipe_buf[slot];
}
extern "C" void *coli_cuda_pipe_alloc(int device,size_t bytes){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return NULL;
    void *p=NULL;
    if(!cuda_ok(cudaMalloc(&p,bytes),"pipe alloc")) return NULL;
    return p;
}
extern "C" void coli_cuda_pipe_free(int device,void *p){
    DeviceContext *ctx=find_ctx(device); if(!p||!select_ctx(ctx)) return;
    cudaFree(p);
}
extern "C" int coli_cuda_pipe_upload(int device,void *dst,const void *src,size_t bytes){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemcpy(dst,src,bytes,cudaMemcpyHostToDevice),"pipe upload");
}
extern "C" int coli_cuda_pipe_download(int device,const void *src,void *dst,size_t bytes){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemcpy(dst,src,bytes,cudaMemcpyDeviceToHost),"pipe download");
}
/* Every `pipe_*` op below runs on `ctx->stream`. That is not a style choice.
 *
 * `ctx->stream` is created with `cudaStreamNonBlocking` (see coli_cuda_init),
 * which means it does **not** implicitly synchronize with the legacy default
 * stream. Until 2026-08-07 `pipe_rmsnorm`, `pipe_rmsnorm_s`, `pipe_rope`,
 * `pipe_rope_base` and `pipe_rows_add` launched with no stream argument — i.e.
 * on the default stream — while `pipe_silu_mul` and `pipe_add` used
 * `ctx->stream`. A chain mixing them therefore had **no ordering guarantee
 * whatsoever**: a `pipe_silu_mul` could read a buffer a `pipe_rmsnorm` had not
 * finished writing, nondeterministically.
 *
 * Nothing was wrong in practice only because no live path builds such a chain —
 * the `pipe_*` set is exercised solely by the graph-capture tests, which use
 * silu_mul/add. The bug would have surfaced the moment the device-resident
 * forward wired one, as intermittently wrong logits with no failing test.
 *
 * The same change is what makes capture possible at all: `cudaStreamBeginCapture`
 * records `ctx->stream`, so an op on any other stream is silently not in the
 * graph. */
extern "C" int coli_cuda_pipe_rmsnorm(int device,float *y_dev,const float *x_dev,
                                      const float *w_dev,int S,int D,float eps){
    DeviceContext *ctx=find_ctx(device);
    if(S<1||D<1||!select_ctx(ctx)) return 0;
    pipe_rmsnorm_rows<<<S,256,0,ctx->stream>>>(y_dev,x_dev,w_dev,D,eps,D,D);
    return cuda_ok(cudaGetLastError(),"pipe rmsnorm");
}
extern "C" int coli_cuda_pipe_rmsnorm_s(int device,float *y_dev,const float *x_dev,
                                        const float *w_dev,int S,int D,float eps,
                                        int xstride,int ystride){
    DeviceContext *ctx=find_ctx(device);
    if(S<1||D<1||xstride<D||ystride<D||!select_ctx(ctx)) return 0;
    pipe_rmsnorm_rows<<<S,256,0,ctx->stream>>>(y_dev,x_dev,w_dev,D,eps,xstride,ystride);
    return cuda_ok(cudaGetLastError(),"pipe rmsnorm strided");
}
extern "C" int coli_cuda_pipe_rope(int device,float *v_dev,const int *pos_dev,
                                   int rows,int stride,int offset,int R,int heads,
                                   float theta){
    DeviceContext *ctx=find_ctx(device);
    if(rows<1||R<2||R>256||heads<1||!select_ctx(ctx)) return 0;
    pipe_rope_rows<<<rows,128,0,ctx->stream>>>(v_dev,pos_dev,0,stride,offset,R,heads,theta);
    return cuda_ok(cudaGetLastError(),"pipe rope");
}
extern "C" int coli_cuda_pipe_rope_base(int device,float *v_dev,int pos_base,int rows,
                                        int stride,int offset,int R,int heads,float theta){
    DeviceContext *ctx=find_ctx(device);
    if(rows<1||R<2||R>256||heads<1||!select_ctx(ctx)) return 0;
    pipe_rope_rows<<<rows,128,0,ctx->stream>>>(v_dev,NULL,pos_base,stride,offset,R,heads,theta);
    return cuda_ok(cudaGetLastError(),"pipe rope base");
}
/* Device-to-device, so async-on-stream is both capturable and correctly ordered
 * against the ops around it. Host visibility, if a caller ever needs it, is
 * `coli_cuda_pipe_sync` — the same contract the rest of the pipe_* set has. */
extern "C" int coli_cuda_pipe_copy2d(int device,float *dst,int dpitch,const float *src,
                                     int spitch,int width,int height){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemcpy2DAsync(dst,(size_t)dpitch*4,src,(size_t)spitch*4,
        (size_t)width*4,height,cudaMemcpyDeviceToDevice,ctx->stream),"pipe copy2d");
}
/* attention batch + fused o_proj with DEVICE-resident q/latent/rope: the whole
 * upstream projection chain stayed on this device, so nothing is uploaded here.
 * Only the final [S,O] projection is downloaded to host. */
extern "C" int coli_cuda_attention_project_batch_dev(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out,const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!w||!proj||!out||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||T>8192||w->I!=K||w->O!=H*(Q+V)||
       proj->device!=w->device||proj->I!=H*V)return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t cb=(size_t)S*H*V*sizeof(float);
    if(!reserve(dc, &dc->ac,&dc->ac_cap,cb))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,S),256,shared,dc->stream>>>(dc->ac,q_dev,latent_dev,
        rope_dev,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale);
    if(!cuda_ok(cudaGetLastError(),"pipe attention launch"))return 0;
    size_t ob=(size_t)S*proj->O*sizeof(float);
    if(!reserve(dc, &dc->y,&dc->y_cap,ob))return 0;
    quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(dc->y,dc->ac,proj->weights,
        proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I));
    if(!cuda_ok(cudaGetLastError(),"pipe o_proj launch"))return 0;
    if(!cuda_ok(cudaMemcpyAsync(out,dc->y,ob,cudaMemcpyDeviceToHost,dc->stream),"pipe attention download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"pipe attention sync"))return 0;
    return 1;
}
extern "C" int coli_cuda_pipe_silu_mul(int device,float *gate_dev,const float *up_dev,
                                       size_t n){
    DeviceContext *ctx=find_ctx(device); if(!n||!select_ctx(ctx)) return 0;
    // on ctx->stream so it composes into an overlapped / graph-captured chain
    silu_mul<<<(unsigned)((n+255)/256),256,0,ctx->stream>>>(gate_dev,up_dev,n);
    return cuda_ok(cudaGetLastError(),"pipe silu mul");
}
extern "C" int coli_cuda_pipe_add(int device,float *x_dev,const float *t_dev,size_t n){
    DeviceContext *ctx=find_ctx(device); if(!n||!select_ctx(ctx)) return 0;
    pipe_add_n<<<(unsigned)((n+255)/256),256,0,ctx->stream>>>(x_dev,t_dev,n);
    return cuda_ok(cudaGetLastError(),"pipe add");
}
extern "C" int coli_cuda_pipe_rows_add(int device,float *x_dev,const float *partial_dev,
                                       const int *rows_dev,int nrows,int D){
    DeviceContext *ctx=find_ctx(device); if(nrows<1||D<1||!select_ctx(ctx)) return 0;
    pipe_rows_add<<<nrows,256,0,ctx->stream>>>(x_dev,partial_dev,rows_dev,D);
    return cuda_ok(cudaGetLastError(),"pipe rows add");
}
/* GEMM with device-resident activations: same quant_matmul kernel as
 * coli_cuda_matmul, zero host transfers. */
extern "C" int coli_cuda_pipe_gemm(ColiCudaTensor *t,float *y_dev,const float *x_dev,
                                   int S){
    if(!t||S<1) return 0;
    DeviceContext *ctx=find_ctx(t->device); if(!select_ctx(ctx)) return 0;
    dim3 grid((unsigned)t->O,(unsigned)S);
    /* on ctx->stream with the rest of the pipe_* set — see the ordering note
     * above coli_cuda_pipe_rmsnorm. This one was the only capturable op left on
     * the default stream. */
    quant_matmul<<<grid,256,0,ctx->stream>>>(y_dev,x_dev,t->weights,t->scales,t->fmt,S,t->I,t->O,
        row_bytes(t->fmt,t->I));
    return cuda_ok(cudaGetLastError(),"pipe gemm");
}
/* copia diretta scheda->scheda (P2P se disponibile, altrimenti staging driver) */
extern "C" int coli_cuda_pipe_peer_copy(int dst_dev,float *dst,int src_dev,
                                        const float *src,size_t bytes){
    if(!dst||!src) return 0;
    if(dst_dev==src_dev){ DeviceContext *c=find_ctx(dst_dev); if(!select_ctx(c)) return 0;
        return cuda_ok(cudaMemcpy(dst,src,bytes,cudaMemcpyDeviceToDevice),"pipe intra copy"); }
    return cuda_ok(cudaMemcpyPeer(dst,dst_dev,src,src_dev,bytes),"pipe peer copy");
}
/* come attention_project_batch_dev ma l'uscita di o_proj RESTA sul device (out_dev). */
extern "C" int coli_cuda_attention_project_batch_dev_out(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!w||!proj||!out_dev||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||T>8192||w->I!=K||w->O!=H*(Q+V)||
       proj->device!=w->device||proj->I!=H*V)return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t cb=(size_t)S*H*V*sizeof(float);
    if(!reserve(dc, &dc->ac,&dc->ac_cap,cb))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,S),256,shared,dc->stream>>>(dc->ac,q_dev,latent_dev,
        rope_dev,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale);
    if(!cuda_ok(cudaGetLastError(),"pipe attention launch (dev out)"))return 0;
    quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(out_dev,dc->ac,proj->weights,
        proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I));
    if(!cuda_ok(cudaGetLastError(),"pipe o_proj launch (dev out)"))return 0;
    return cuda_ok(cudaStreamSynchronize(dc->stream),"pipe attention sync (dev out)");
}
/* absorb batch con TUTTO su device (q/latent/rope gia' residenti sulla scheda
 * dello shard, ctx resta sul device): il cuore della attention head-shardata
 * dentro il pipeline. Nessun trasferimento host. */
extern "C" int coli_cuda_attention_absorb_batch_dev(ColiCudaTensor *w,float *ctx_dev,
        const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!w||!ctx_dev||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||T>8192||w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,S),256,shared,dc->stream>>>(ctx_dev,q_dev,latent_dev,
        rope_dev,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale);
    if(!cuda_ok(cudaGetLastError(),"pipe shard attention launch"))return 0;
    return cuda_ok(cudaStreamSynchronize(dc->stream),"pipe shard attention sync");
}
/* absorb per il DECODE con KV gia' residente: carica solo q (poche KB),
 * latent/rope arrivano dall'ombra device. ctx torna a host (S piccolo). */
extern "C" int coli_cuda_attention_absorb_kvdev(ColiCudaTensor *w,float *ctx,const float *q,
        const float *latent_dev,const float *rope_dev,int H,int Q,int R,int V,int K,int T,
        float scale){
    if(!w||!ctx||!q||!latent_dev||!rope_dev||H<1||Q<1||R<1||V<1||K<1||K>512||T<1||T>8192||
       w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)H*(Q+R)*sizeof(float),cb=(size_t)H*V*sizeof(float);
    if(!reserve(dc, &dc->aq,&dc->aq_cap,qb)||!reserve(dc, &dc->ac,&dc->ac_cap,cb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"kvdev q upload"))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,1),256,shared,dc->stream>>>(dc->ac,dc->aq,latent_dev,
        rope_dev,w->weights,w->scales,w->fmt,1,H,Q,R,V,K,T,scale);
    if(!cuda_ok(cudaGetLastError(),"kvdev absorb launch")||
       !cuda_ok(cudaMemcpyAsync(ctx,dc->ac,cb,cudaMemcpyDeviceToHost,dc->stream),"kvdev ctx download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"kvdev absorb sync"))return 0;
    return 1;
}
extern "C" int coli_cuda_pipe_sync(int device){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaDeviceSynchronize(),"pipe sync");
}
