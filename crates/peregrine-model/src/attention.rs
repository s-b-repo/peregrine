//! MLA (Multi-head Latent Attention) — `attention_rows` (`c/glm.c:2329-2638`).
//!
//! Two cores share one projection + KV-append front end:
//!   - [`mla_attention`] — dense reconstruction: rebuild `[k_nope|v]` for every
//!     cached position via `kv_b`, then standard scored attention.
//!   - [`mla_attention_absorb`] — the decode optimization: absorb `kv_b`'s
//!     `k_nope` rows into the query (`qabs`) and score against the latent `Lc`
//!     directly, and project the latent context average through `kv_b`'s value
//!     rows once — no per-token reconstruction. Algebraically identical to dense.
//!
//! DSA sparse selection is deferred (M5 remainder); attending over all cached
//! keys is exactly the "DSA selects everything" case the C engine validates as
//! reproducing dense attention.

use crate::math::{rmsnorm_inplace, rope_interleave, softmax};
use crate::weight::QtWeight;
use peregrine_core::{Cfg, Error};
use std::sync::Arc;

/// An immutable, refcounted run of cached positions, shared by every sequence
/// seeded from the same prompt prefix.
///
/// Sound for the same reason the prefix cache above it is: each position
/// attends only its causal prefix, so two prompts agreeing on their first `n`
/// tokens produce bit-identical KV for those positions. Once built it is never
/// written again — a sequence's own new positions go in its private tail — so
/// sharing needs nothing beyond the refcount.
struct KvPrefix {
    lc: Vec<f32>,
    rc: Vec<f32>,
}

/// Per-layer KV cache holding the compressed latents and roped keys. Positions
/// are appended in order (`pos == len`), matching sequential prefill/decode.
///
/// Storage is a shared immutable prefix plus this sequence's private tail. The
/// prefix is what an admission from the cross-request prefix cache used to
/// **deep-copy** — ~350 MB per request for a 2 k-token shared system prompt at
/// GLM-5.2's 175.5 KiB/token — and is now a refcount bump. Readers never see
/// the seam: they go through [`KvSpan`], and
/// `kv_span_split_attention_is_bit_identical_to_contiguous` pins that a split
/// cache produces the same bits as a contiguous one.
#[derive(Clone)]
pub struct LayerKv {
    /// Shared immutable prefix, if this cache was seeded from one.
    shared: Option<Arc<KvPrefix>>,
    /// How many of `shared`'s rows this sequence uses. Always `<=` the rows the
    /// prefix holds, and may be *fewer*: that is what lets one frozen prefix
    /// serve requests matching it to different depths without a copy each.
    shared_rows: usize,
    /// This sequence's own positions, appended after the shared prefix.
    lc: Vec<f32>, // [len - shared_rows, kv_lora]
    rc: Vec<f32>, // [len - shared_rows, qk_rope]
    len: usize,
    kv_lora: usize,
    qk_rope: usize,
}

impl LayerKv {
    pub fn new(kv_lora: usize, qk_rope: usize) -> LayerKv {
        LayerKv { shared: None, shared_rows: 0, lc: Vec::new(), rc: Vec::new(), len: 0, kv_lora, qk_rope }
    }

    /// Positions cached so far — the shared prefix this sequence uses plus its
    /// own tail.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether no positions are cached yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Append one position's latents. Positions must arrive in order: an
    /// out-of-order append would silently corrupt attention for the rest of the
    /// sequence, so the invariant is checked on every call. The failure is
    /// reported rather than asserted — a scheduler bug that mismatches a
    /// sequence with its cache must fail that one request, not abort the
    /// process (the release profile sets `panic = "abort"`, so a panic here
    /// would take every concurrent sequence down with it).
    fn append(&mut self, pos: usize, lc_row: &[f32], rc_row: &[f32]) -> Result<(), Error> {
        if pos != self.len {
            return Err(Error::Format(format!(
                "KV cache append out of order: position {pos} into a cache of length {}",
                self.len
            )));
        }
        if lc_row.len() != self.kv_lora || rc_row.len() != self.qk_rope {
            return Err(Error::Format(format!(
                "KV cache append shape mismatch: got ({}, {}), expected ({}, {})",
                lc_row.len(),
                rc_row.len(),
                self.kv_lora,
                self.qk_rope
            )));
        }
        self.lc.extend_from_slice(lc_row);
        self.rc.extend_from_slice(rc_row);
        self.len += 1;
        Ok(())
    }

    /// Drop cached positions back to `new_len` — the speculative-decode rewind
    /// after rejected draft tokens. A no-op when `new_len >= len`.
    pub fn truncate(&mut self, new_len: usize) {
        if new_len >= self.len {
            return;
        }
        if new_len >= self.shared_rows {
            let own = new_len - self.shared_rows;
            self.lc.truncate(own * self.kv_lora);
            self.rc.truncate(own * self.qk_rope);
        } else {
            // Rewinding into the shared prefix. It is immutable and other
            // sequences may still be reading it, so narrow this cache's *view*
            // instead of the buffer. The dropped rows stay allocated until the
            // last holder goes — exactly the lifetime they had before sharing.
            self.shared_rows = new_len;
            self.lc.clear();
            self.rc.clear();
        }
        self.len = new_len;
    }

    /// Logical bytes of the cached latents: what this sequence would cost if it
    /// held every position privately. Unchanged by sharing, so a budget built on
    /// it stays correct (conservatively) however prefixes are shared — see
    /// [`Self::owned_bytes`] and [`Self::shared_bytes`] for the exact split.
    pub fn bytes(&self) -> usize {
        (self.shared_rows * (self.kv_lora + self.qk_rope) + self.lc.len() + self.rc.len()) * std::mem::size_of::<f32>()
    }

    /// Bytes this cache holds *by itself*, excluding any shared prefix.
    pub fn owned_bytes(&self) -> usize {
        (self.lc.len() + self.rc.len()) * std::mem::size_of::<f32>()
    }

    /// Bytes of the whole shared prefix allocation — not just this cache's view
    /// of it, because the allocation is what actually occupies RAM while any
    /// sequence holds it.
    pub fn shared_bytes(&self) -> usize {
        self.shared.as_ref().map_or(0, |p| (p.lc.len() + p.rc.len()) * std::mem::size_of::<f32>())
    }

    /// Identity of the shared prefix allocation, if any. A budget summing many
    /// sequences uses it to count one shared prefix once rather than once per
    /// viewer. Pointer-derived and never dereferenced; two caches seeded from
    /// the same entry compare equal.
    pub fn shared_id(&self) -> Option<usize> {
        self.shared.as_ref().map(|p| Arc::as_ptr(p) as usize)
    }

    /// A cache holding only the first `n` positions, sharing storage with this
    /// one wherever it can.
    ///
    /// **This is the admission path's hot copy.** Every request matching a
    /// cached prompt prefix deep-copied the whole thing. When `n` falls inside
    /// an already-frozen prefix — the prefix cache's steady state, since its
    /// entries are themselves produced by this function — the result is a
    /// refcount bump and nothing is copied at all.
    ///
    /// Bit-identical by construction: the child sees the same bytes in the same
    /// order and never writes them.
    pub fn clone_prefix(&self, n: usize) -> LayerKv {
        let n = n.min(self.len);
        if n <= self.shared_rows {
            if let Some(pre) = &self.shared {
                return LayerKv {
                    shared: Some(Arc::clone(pre)),
                    shared_rows: n,
                    lc: Vec::new(),
                    rc: Vec::new(),
                    len: n,
                    kv_lora: self.kv_lora,
                    qk_rope: self.qk_rope,
                };
            }
        }
        // Freeze the first `n` positions into a new shared prefix. This copies
        // once, so that every later `clone_prefix` of the result takes the
        // refcount path above.
        let mut lc = Vec::with_capacity(n * self.kv_lora);
        let mut rc = Vec::with_capacity(n * self.qk_rope);
        for run in self.lc_span(n).runs() {
            lc.extend_from_slice(run);
        }
        for run in self.rc_span(n).runs() {
            rc.extend_from_slice(run);
        }
        LayerKv {
            shared: Some(Arc::new(KvPrefix { lc, rc })),
            shared_rows: n,
            lc: Vec::new(),
            rc: Vec::new(),
            len: n,
            kv_lora: self.kv_lora,
            qk_rope: self.qk_rope,
        }
    }

    /// The compressed latents of the first `n` positions, in causal order.
    pub fn lc_span(&self, n: usize) -> KvSpan<'_> {
        let shared = self.shared.as_ref().map_or(&[][..], |p| &p.lc);
        span_of(shared, &self.lc, self.shared_rows, n, self.kv_lora)
    }

    /// The roped keys of the first `n` positions, in causal order.
    pub fn rc_span(&self, n: usize) -> KvSpan<'_> {
        let shared = self.shared.as_ref().map_or(&[][..], |p| &p.rc);
        span_of(shared, &self.rc, self.shared_rows, n, self.qk_rope)
    }

    /// The whole cache as an attention view.
    pub fn view(&self) -> RowAttn<'_> {
        self.view_prefix(self.len)
    }

    /// The first `n` positions as an attention view — one row's causal prefix.
    pub fn view_prefix(&self, n: usize) -> RowAttn<'_> {
        let n = n.min(self.len);
        RowAttn { len: n, lc: self.lc_span(n), rc: self.rc_span(n) }
    }
}

/// The first `n` rows of `shared ++ own` as a span, where `shared_rows` of the
/// shared buffer belong to this cache.
///
/// Total by construction — every bound is clamped to what the buffers actually
/// hold. A slicing panic here would abort the process under the release
/// profile's `panic = "abort"`, taking every concurrent sequence with it, which
/// is the same reason [`LayerKv::append`] reports its errors instead of
/// asserting.
fn span_of<'a>(shared: &'a [f32], own: &'a [f32], shared_rows: usize, n: usize, width: usize) -> KvSpan<'a> {
    let head_rows = shared_rows.min(n).min(shared.len().checked_div(width).unwrap_or(0));
    let tail_rows = (n - head_rows).min(own.len().checked_div(width).unwrap_or(0));
    let head = &shared[..head_rows * width];
    if tail_rows == 0 {
        KvSpan::contiguous(head)
    } else {
        KvSpan::split(head, &own[..tail_rows * width])
    }
}

/// An ordered view of cached KV that may be split across **two** contiguous
/// runs: a shared, immutable prefix and this sequence's own tail.
///
/// **Why the readers stopped taking `&[f32]`.** Every KV improvement on the
/// roadmap — sharing a prefix by refcount instead of copying it, block-pooling
/// the allocation, narrowing the element type — makes the cache discontiguous
/// or changes its type, and every reader here demanded one contiguous slice.
/// That single requirement is what coupled four separate items into one
/// refactor; this is the seam that decouples them.
///
/// Two runs rather than an arbitrary block list, deliberately: it is
/// `Copy`, allocation-free, and index arithmetic stays two branches — enough
/// for prefix+tail, which is the case that exists today. A block table
/// generalises `row` and `runs` without touching any caller.
///
/// **Rows never straddle the boundary.** `head` always holds a whole number of
/// rows, so `row` is a bounds check and an offset, never a stitch across two
/// allocations. Callers construct spans from row-aligned prefixes only.
#[derive(Clone, Copy)]
pub struct KvSpan<'a> {
    head: &'a [f32],
    tail: &'a [f32],
}

impl<'a> KvSpan<'a> {
    /// The whole cache in one run — what every caller had before.
    pub fn contiguous(all: &'a [f32]) -> KvSpan<'a> {
        KvSpan { head: all, tail: &[] }
    }

    /// A shared prefix followed by an owned tail.
    pub fn split(head: &'a [f32], tail: &'a [f32]) -> KvSpan<'a> {
        KvSpan { head, tail }
    }

    /// Row `t` of `width` floats, in causal order across both runs.
    #[inline]
    pub fn row(&self, t: usize, width: usize) -> &'a [f32] {
        let hrows = self.head.len().checked_div(width).unwrap_or(0);
        if t < hrows {
            &self.head[t * width..t * width + width]
        } else {
            let k = t - hrows;
            &self.tail[k * width..k * width + width]
        }
    }

    /// Whole rows visible through this span.
    pub fn rows(&self, width: usize) -> usize {
        (self.head.len() + self.tail.len()).checked_div(width).unwrap_or(0)
    }

    /// The runs in causal order, skipping empties — for bulk operations that
    /// work per row and can be applied run-by-run, then concatenated.
    pub fn runs(&self) -> impl Iterator<Item = &'a [f32]> {
        [self.head, self.tail].into_iter().filter(|r| !r.is_empty())
    }
}

/// A view of one row's KV history for batched MLA decode: that row's own cached
/// latents (`lc[len, kv_lora]`, `rc[len, qk_rope]`) and causal length `len`.
/// Rows come from independent sequences, so each carries its own view.
pub struct RowAttn<'a> {
    pub len: usize,
    pub lc: KvSpan<'a>,
    pub rc: KvSpan<'a>,
}

/// The five projection weights of one attention block.
pub struct AttnWeights<'a> {
    pub q_a: &'a QtWeight,
    pub q_a_ln: &'a [f32],
    pub q_b: &'a QtWeight,
    pub kv_a: &'a QtWeight,
    pub kv_a_ln: &'a [f32],
    pub kv_b: &'a QtWeight,
    pub o: &'a QtWeight,
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Batched q/kv projection front end for `s_n` independent rows (one new token
/// per sequence): the q_a/q_b/kv_a matmuls stay batched across all rows, while
/// per-head query RoPE and the compressed KV (`Lc` normalized latent, `Rc` roped
/// key) use each row's own position `pos_of[s]`. Returns the roped queries
/// `Q[s_n, H*qk_head]` plus the per-row latents to append to each row's own
/// cache (`lc_rows[s_n, kv_lora]`, `rc_rows[s_n, qk_rope]`). Performs no cache
/// mutation, so it is storage-agnostic (single cache or per-sequence caches).
fn project_batched(w: &AttnWeights, x: &[f32], s_n: usize, pos_of: &[usize], c: &Cfg) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let h_n = c.n_heads as usize;
    let qk_nope = c.qk_nope as usize;
    let qk_rope = c.qk_rope as usize;
    let qh = qk_nope + qk_rope;
    let kvl = c.kv_lora as usize;
    let cw = kvl + qk_rope;
    let q_lora = c.q_lora as usize;

    let mut qr = w.q_a.apply_vec(x, s_n);
    for s in 0..s_n {
        let row = &mut qr[s * q_lora..s * q_lora + q_lora];
        // In-place: `rmsnorm` reads the whole row before writing any element, so
        // the scratch copy this used to make (one allocation per row per token
        // per layer) bought nothing.
        rmsnorm_inplace(row, w.q_a_ln, c.eps);
    }
    let mut q = w.q_b.apply_vec(&qr, s_n);
    let comp = w.kv_a.apply_vec(x, s_n);

    let mut lc_rows = vec![0f32; s_n * kvl];
    let mut rc_rows = vec![0f32; s_n * qk_rope];
    for s in 0..s_n {
        let pos = pos_of[s];
        for h in 0..h_n {
            let off = s * h_n * qh + h * qh + qk_nope;
            rope_interleave(&mut q[off..off + qk_rope], pos, c);
        }
        let cs = &comp[s * cw..s * cw + cw];
        let dst_lc = &mut lc_rows[s * kvl..s * kvl + kvl];
        dst_lc.copy_from_slice(&cs[..kvl]);
        rmsnorm_inplace(dst_lc, w.kv_a_ln, c.eps);
        let dst_rc = &mut rc_rows[s * qk_rope..s * qk_rope + qk_rope];
        dst_rc.copy_from_slice(&cs[kvl..cw]);
        rope_interleave(dst_rc, pos, c);
    }
    (q, lc_rows, rc_rows)
}

/// Shared single-sequence front end: [`project_batched`] over the consecutive
/// positions `pos_base..pos_base+s_n`, appending each new latent to the one
/// `cache` in order. Bit-identical to the pre-batching projection (the appends
/// are independent of the query/latent arithmetic, so hoisting them is exact).
fn project(w: &AttnWeights, x: &[f32], s_n: usize, pos_base: usize, cache: &mut LayerKv, c: &Cfg) -> Result<Vec<f32>, Error> {
    let kvl = c.kv_lora as usize;
    let qk_rope = c.qk_rope as usize;
    let pos_of: Vec<usize> = (0..s_n).map(|s| pos_base + s).collect();
    let (q, lc_rows, rc_rows) = project_batched(w, x, s_n, &pos_of, c);
    for s in 0..s_n {
        cache.append(pos_base + s, &lc_rows[s * kvl..s * kvl + kvl], &rc_rows[s * qk_rope..s * qk_rope + qk_rope])?;
    }
    Ok(q)
}

/// Dense core: reconstruct `[k_nope|v]` for all cached positions via `kv_b`,
/// then causal scored attention. Returns `ctx[s_n, H*v_head]`.
///
/// `sel`, when `Some`, restricts each query `s` to attend only the cached key
/// indices in `sel[s]` (the DSA lightning-indexer selection); `None` attends all
/// causal keys (dense). Selecting every causal key is identical to dense.
fn attend_dense(w: &AttnWeights, q: &[f32], s_n: usize, pos_base: usize, kv: &RowAttn, c: &Cfg, sel: Option<&[Vec<usize>]>) -> Vec<f32> {
    let h_n = c.n_heads as usize;
    let qk_nope = c.qk_nope as usize;
    let qk_rope = c.qk_rope as usize;
    let qh = qk_nope + qk_rope;
    let vh = c.v_head as usize;
    let kvl = c.kv_lora as usize;
    let kvb_head = qk_nope + vh;

    let tk = kv.len;
    // Reconstruct `[k_nope|v]` for every cached position. When the cache is
    // split across a shared prefix and an owned tail, the matmul runs once per
    // run and the *outputs* concatenate — bit-identically, because `apply_vec`
    // is row-independent: it already splits rows into chunks across the pool,
    // and `apply_vec_parallel_matches_serial` pins that as equal to one whole
    // serial call. Splitting the input on a row boundary is the same split.
    let mut kvb_all: Vec<f32> = Vec::new();
    for run in kv.lc.runs() {
        let rows = run.len().checked_div(kvl).unwrap_or(0);
        let part = w.kv_b.apply_vec(run, rows);
        if kvb_all.is_empty() {
            kvb_all = part; // the contiguous case: no extra copy
        } else {
            kvb_all.extend_from_slice(&part);
        }
    }

    let mut ctx = vec![0f32; s_n * h_n * vh];
    for s in 0..s_n {
        let pos = pos_base + s;
        // keys this query attends: the DSA selection (clamped causal), or all 0..=pos
        let mut keys: Vec<usize> = match sel.and_then(|sets| sets.get(s)) {
            // DSA selection for this row, clamped to its causal prefix.
            Some(set) => set.iter().copied().filter(|&t| t <= pos).collect(),
            // Dense (`sel` is None) — or a selection that does not cover this
            // row, which the public entry point rejects before we get here.
            None => match sel {
                Some(_) => Vec::new(),
                None => (0..=pos).collect(),
            },
        };
        // A query must always attend at least its own position. An empty
        // selection (indexer configured with no keys, or every selected key
        // beyond `pos`) would otherwise softmax an empty score row and leave the
        // context vector identically zero — silently blanking attention.
        if keys.is_empty() {
            keys.push(pos.min(tk.saturating_sub(1)));
        }
        for h in 0..h_n {
            let qp = &q[s * h_n * qh + h * qh..s * h_n * qh + h * qh + qh];
            let (q_nope, q_rope) = qp.split_at(qk_nope);
            let mut sc = vec![0f32; keys.len()];
            for (i, &t) in keys.iter().enumerate() {
                let base = t * h_n * kvb_head + h * kvb_head;
                let kn = &kvb_all[base..base + qk_nope];
                let kr = kv.rc.row(t, qk_rope);
                sc[i] = (dot(q_nope, kn) + dot(q_rope, kr)) * c.attn_scale;
            }
            softmax(&mut sc);
            let cx = &mut ctx[(s * h_n + h) * vh..(s * h_n + h) * vh + vh];
            for (i, &t) in keys.iter().enumerate() {
                let base = t * h_n * kvb_head + h * kvb_head + qk_nope;
                let vv = &kvb_all[base..base + vh];
                let a = sc[i];
                for d in 0..vh {
                    cx[d] += a * vv[d];
                }
            }
        }
    }
    ctx
}

/// Absorb core over `s_n` independent rows: row `s` folds `kv_b`'s k_nope rows
/// into its query, scores against its own `rows[s].lc` (0..len), and projects the
/// latent average through `kv_b`'s value rows. Row `s` depends only on `q[s]` and
/// its own KV view, so a batched decode step is arithmetically identical to
/// running each sequence's decode alone (the `batched_matches_sequential` guard).
fn attend_absorb_batched(w: &AttnWeights, q: &[f32], rows: &[RowAttn], c: &Cfg, par_min: usize) -> Vec<f32> {
    let s_n = rows.len();
    let h_n = c.n_heads as usize;
    let qk_nope = c.qk_nope as usize;
    let qk_rope = c.qk_rope as usize;
    let qh = qk_nope + qk_rope;
    let vh = c.v_head as usize;
    let kvl = c.kv_lora as usize;

    // Row `s` reads only its own KV view + `q[s]` and writes only its own head-block
    // of `ctx` (stride `h_n*vh`), so rows run on the pool above `par_min` with a
    // result bit-identical to the serial loop. `par_min` is a parameter only so the
    // `attend_absorb_parallel_matches_serial` test can force each branch.
    let mut ctx = vec![0f32; s_n * h_n * vh];
    peregrine_par::par_rows_mut(&mut ctx, h_n * vh, s_n, par_min, |s, ctx_row| {
        let cache_lc = rows[s].lc;
        let cache_rc = rows[s].rc;
        let nt = rows[s].len;
        // Scratch reused across every head and every kv_b row of this token.
        // These dequantizations re-derive an immutable weight, so allocating per
        // call cost ~h_n*(qk_nope+v_head) heap allocations per token per layer.
        let mut kvb_row = vec![0f32; kvl];
        let mut qabs = vec![0f32; kvl];
        let mut clat = vec![0f32; kvl];
        let mut sc = vec![0f32; nt];
        for h in 0..h_n {
            let qp = &q[s * h_n * qh + h * qh..s * h_n * qh + h * qh + qh];
            let (q_nope, q_rope) = qp.split_at(qk_nope);
            let rbase = h * (qk_nope + vh);

            // qabs = Σ_d q_nope[d] · kv_b_row(rbase+d)   (absorb k_nope into q)
            qabs.fill(0.0);
            for (d, &qd) in q_nope.iter().enumerate().take(qk_nope) {
                w.kv_b.dequant_row_into(rbase + d, &mut kvb_row);
                for (acc, &kv) in qabs.iter_mut().zip(kvb_row.iter()).take(kvl) {
                    *acc += qd * kv;
                }
            }

            sc.clear();
            sc.resize(nt, 0.0);
            for (t, sct) in sc.iter_mut().enumerate().take(nt) {
                let lt = cache_lc.row(t, kvl);
                let kr = cache_rc.row(t, qk_rope);
                *sct = (dot(&qabs, lt) + dot(q_rope, kr)) * c.attn_scale;
            }
            softmax(&mut sc);

            // clat = Σ_t sc[t] · Lc[t]; ctx = kv_b value rows · clat
            clat.fill(0.0);
            for (t, &a) in sc.iter().enumerate().take(nt) {
                let lt = cache_lc.row(t, kvl);
                for i in 0..kvl {
                    clat[i] += a * lt[i];
                }
            }
            let cx = &mut ctx_row[h * vh..h * vh + vh];
            for (j, out) in cx.iter_mut().enumerate() {
                w.kv_b.dequant_row_into(rbase + qk_nope + j, &mut kvb_row);
                *out = dot(&kvb_row, &clat);
            }
        }
    });
    ctx
}

/// Absorb core (single sequence): the `s_n` new tokens each attend their own
/// causal prefix of the shared `cache`. A thin wrapper over
/// [`attend_absorb_batched`] with per-row prefix views — bit-identical to the
/// pre-batching core (same `nt`, same slices, same arithmetic).
fn attend_absorb(w: &AttnWeights, q: &[f32], s_n: usize, pos_base: usize, cache: &LayerKv, c: &Cfg) -> Vec<f32> {
    let rows: Vec<RowAttn> = (0..s_n).map(|s| cache.view_prefix(pos_base + s + 1)).collect();
    attend_absorb_batched(w, q, &rows, c, peregrine_par::PAR_ATTN_MIN)
}

/// MLA attention (dense reconstruction). `s_n` new tokens from `pos_base`,
/// appended to `cache`; returns `out[s_n, hidden]`.
pub fn mla_attention(w: &AttnWeights, x: &[f32], s_n: usize, pos_base: usize, cache: &mut LayerKv, c: &Cfg) -> Result<Vec<f32>, Error> {
    let q = project(w, x, s_n, pos_base, cache, c)?;
    let ctx = attend_dense(w, &q, s_n, pos_base, &cache.view(), c, None);
    Ok(w.o.apply_vec(&ctx, s_n))
}

/// MLA attention with a DSA lightning-indexer selection: each new query attends
/// only the `sel[s]` cached keys (top-`index_topk`) instead of all — the sparse
/// path for long context. Selecting every causal key reproduces [`mla_attention`].
pub fn mla_attention_dsa(
    w: &AttnWeights,
    x: &[f32],
    s_n: usize,
    pos_base: usize,
    cache: &mut LayerKv,
    c: &Cfg,
    sel: &[Vec<usize>],
) -> Result<Vec<f32>, Error> {
    if sel.len() != s_n {
        return Err(Error::Format(format!(
            "DSA selection covers {} rows but {s_n} queries were projected",
            sel.len()
        )));
    }
    let q = project(w, x, s_n, pos_base, cache, c)?;
    let ctx = attend_dense(w, &q, s_n, pos_base, &cache.view(), c, Some(sel));
    Ok(w.o.apply_vec(&ctx, s_n))
}

/// MLA attention (weight absorption). Same result as [`mla_attention`], via the
/// decode-optimized latent-space core.
pub fn mla_attention_absorb(
    w: &AttnWeights,
    x: &[f32],
    s_n: usize,
    pos_base: usize,
    cache: &mut LayerKv,
    c: &Cfg,
) -> Result<Vec<f32>, Error> {
    let q = project(w, x, s_n, pos_base, cache, c)?;
    let ctx = attend_absorb(w, &q, s_n, pos_base, cache, c);
    Ok(w.o.apply_vec(&ctx, s_n))
}

/// Batched MLA decode attention (absorb core) over `s_n` **independent
/// sequences**: `x[s_n, hidden]` are one new token each, `pos_of[s]` is that
/// token's absolute position in its sequence, and `caches[s]` is that sequence's
/// per-layer KV (the new latent is appended in place). Returns `out[s_n, hidden]`.
///
/// Precondition: exactly one new token per sequence (pure decode), so after the
/// append each row attends its sequence's whole causal cache. Output row `s` is
/// identical to running [`mla_attention_absorb`] on sequence `s` alone — the
/// property the `batched_matches_sequential` test guards. The MoE lane needs no
/// change: it is already row-batch-union'd over `s_n` rows regardless of which
/// sequence each row belongs to, so B sequences share one set of expert reads.
pub fn mla_attention_batched(
    w: &AttnWeights,
    x: &[f32],
    pos_of: &[usize],
    caches: &mut [&mut LayerKv],
    c: &Cfg,
) -> Result<Vec<f32>, Error> {
    let s_n = pos_of.len();
    if caches.len() != s_n {
        return Err(Error::Format(format!(
            "batched attention: {} caches for {s_n} sequences",
            caches.len()
        )));
    }
    let kvl = c.kv_lora as usize;
    let qk_rope = c.qk_rope as usize;
    let (q, lc_rows, rc_rows) = project_batched(w, x, s_n, pos_of, c);
    for s in 0..s_n {
        caches[s].append(pos_of[s], &lc_rows[s * kvl..s * kvl + kvl], &rc_rows[s * qk_rope..s * qk_rope + qk_rope])?;
    }
    let rows: Vec<RowAttn> = caches.iter().map(|k| k.view()).collect();
    let ctx = attend_absorb_batched(w, &q, &rows, c, peregrine_par::PAR_ATTN_MIN);
    Ok(w.o.apply_vec(&ctx, s_n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight::test_support::quant_i4;

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
    }

    fn cfg() -> Result<Cfg, peregrine_core::Error> {
        let j = serde_json::json!({
            "hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2,
            "n_routed_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 8,
            "intermediate_size": 8, "first_k_dense_replace": 1, "q_lora_rank": 12,
            "kv_lora_rank": 8, "qk_nope_head_dim": 4, "qk_rope_head_dim": 4,
            "v_head_dim": 6, "n_shared_experts": 1, "vocab_size": 32, "n_group": 1,
            "topk_group": 1, "rope_parameters": {"rope_theta": 10000.0}, "rms_norm_eps": 1e-5
        });
        Cfg::from_json(&j)
    }

    struct Weights {
        q_a: QtWeight,
        q_a_ln: Vec<f32>,
        q_b: QtWeight,
        kv_a: QtWeight,
        kv_a_ln: Vec<f32>,
        kv_b: QtWeight,
        o: QtWeight,
    }
    impl Weights {
        fn view(&self) -> AttnWeights<'_> {
            AttnWeights {
                q_a: &self.q_a,
                q_a_ln: &self.q_a_ln,
                q_b: &self.q_b,
                kv_a: &self.kv_a,
                kv_a_ln: &self.kv_a_ln,
                kv_b: &self.kv_b,
                o: &self.o,
            }
        }
    }

    fn make_weights(c: &Cfg, seed: u64) -> Weights {
        let mut r = Lcg(seed);
        let (hidden, h, qh, vh) = (c.hidden as usize, c.n_heads as usize, c.qk_head as usize, c.v_head as usize);
        let (ql, kvl, qkr, qkn) = (c.q_lora as usize, c.kv_lora as usize, c.qk_rope as usize, c.qk_nope as usize);
        let w = |r: &mut Lcg, n: usize| (0..n).map(|_| r.f()).collect::<Vec<f32>>();
        Weights {
            q_a: quant_i4(&w(&mut r, ql * hidden), ql, hidden),
            q_a_ln: (0..ql).map(|_| 0.5 + r.f() * 0.1).collect(),
            q_b: quant_i4(&w(&mut r, h * qh * ql), h * qh, ql),
            kv_a: quant_i4(&w(&mut r, (kvl + qkr) * hidden), kvl + qkr, hidden),
            kv_a_ln: (0..kvl).map(|_| 0.5 + r.f() * 0.1).collect(),
            kv_b: quant_i4(&w(&mut r, h * (qkn + vh) * kvl), h * (qkn + vh), kvl),
            o: quant_i4(&w(&mut r, hidden * h * vh), hidden, h * vh),
        }
    }

    fn new_cache(c: &Cfg) -> LayerKv {
        LayerKv::new(c.kv_lora as usize, c.qk_rope as usize)
    }

    /// Both cores behind one signature, so a property can be asserted for each.
    fn attend_either(
        dense: bool,
        w: &AttnWeights,
        x: &[f32],
        s_n: usize,
        pos: usize,
        cache: &mut LayerKv,
        c: &Cfg,
    ) -> Result<Vec<f32>, Error> {
        if dense {
            mla_attention(w, x, s_n, pos, cache, c)
        } else {
            mla_attention_absorb(w, x, s_n, pos, cache, c)
        }
    }

    #[test]
    fn single_token_is_value_projection() -> Result<(), peregrine_core::Error> {
        let c = cfg()?;
        let w = make_weights(&c, 1);
        let (h, qkn, vh, kvl) = (c.n_heads as usize, c.qk_nope as usize, c.v_head as usize, c.kv_lora as usize);
        let mut r = Lcg(555);
        let x: Vec<f32> = (0..c.hidden as usize).map(|_| r.f()).collect();
        let mut cache = new_cache(&c);
        let out = mla_attention(&w.view(), &x, 1, 0, &mut cache, &c)?;
        let kvb = w.kv_b.apply_vec(cache.lc_span(1).row(0, kvl), 1);
        let kvb_head = qkn + vh;
        let mut v = vec![0f32; h * vh];
        for hh in 0..h {
            let base = hh * kvb_head + qkn;
            v[hh * vh..hh * vh + vh].copy_from_slice(&kvb[base..base + vh]);
        }
        let expect = w.o.apply_vec(&v, 1);
        for d in 0..c.hidden as usize {
            assert!((out[d] - expect[d]).abs() < 1e-4);
        }
        Ok(())
    }

    #[test]
    fn attention_is_causal() -> Result<(), peregrine_core::Error> {
        let c = cfg()?;
        let w = make_weights(&c, 2);
        let hidden = c.hidden as usize;
        let s_n = 4;
        let mut r = Lcg(999);
        let mut x: Vec<f32> = (0..s_n * hidden).map(|_| r.f()).collect();
        let mut ca = new_cache(&c);
        let out_a = mla_attention(&w.view(), &x, s_n, 0, &mut ca, &c)?;
        for d in 0..hidden {
            x[(s_n - 1) * hidden + d] += 1.0;
        }
        let mut cb = new_cache(&c);
        let out_b = mla_attention(&w.view(), &x, s_n, 0, &mut cb, &c)?;
        for p in 0..s_n - 1 {
            for d in 0..hidden {
                assert!((out_a[p * hidden + d] - out_b[p * hidden + d]).abs() < 1e-6);
            }
        }
        Ok(())
    }

    #[test]
    fn decode_step_matches_prefill() -> Result<(), peregrine_core::Error> {
        let c = cfg()?;
        let w = make_weights(&c, 3);
        let hidden = c.hidden as usize;
        let mut r = Lcg(24680);
        let x: Vec<f32> = (0..4 * hidden).map(|_| r.f()).collect();
        let mut full = new_cache(&c);
        let out_full = mla_attention(&w.view(), &x, 4, 0, &mut full, &c)?;
        let mut inc = new_cache(&c);
        mla_attention(&w.view(), &x[..3 * hidden], 3, 0, &mut inc, &c)?;
        let out_dec = mla_attention(&w.view(), &x[3 * hidden..4 * hidden], 1, 3, &mut inc, &c)?;
        for d in 0..hidden {
            assert!((out_full[3 * hidden + d] - out_dec[d]).abs() < 1e-4);
        }
        Ok(())
    }

    #[test]
    fn absorb_approximates_dense() -> Result<(), peregrine_core::Error> {
        // absorb is an algebraic rearrangement of dense; they differ only by
        // kv_b activation quantization in the dense reconstruction.
        let c = cfg()?;
        let w = make_weights(&c, 7);
        let hidden = c.hidden as usize;
        let s_n = 5;
        let mut r = Lcg(31415);
        let x: Vec<f32> = (0..s_n * hidden).map(|_| r.f()).collect();
        let mut cd = new_cache(&c);
        let dense = mla_attention(&w.view(), &x, s_n, 0, &mut cd, &c)?;
        let mut cabs = new_cache(&c);
        let absorb = mla_attention_absorb(&w.view(), &x, s_n, 0, &mut cabs, &c)?;
        for z in 0..s_n * hidden {
            let scale = dense[z].abs().max(1.0);
            assert!((dense[z] - absorb[z]).abs() < 0.1 * scale, "z={z} dense={} absorb={}", dense[z], absorb[z]);
        }
        Ok(())
    }

    #[test]
    fn out_of_order_append_is_an_error_not_an_abort() -> Result<(), peregrine_core::Error> {
        // A scheduler bug that pairs a sequence with the wrong cache must fail
        // that request, not take the process down (release builds abort on panic,
        // so a panic here would kill every concurrent sequence).
        let c = cfg()?;
        let w = make_weights(&c, 13);
        let hidden = c.hidden as usize;
        let mut r = Lcg(0xDEAD);
        let x: Vec<f32> = (0..hidden).map(|_| r.f()).collect();
        let mut cache = new_cache(&c);
        // position 3 into an empty cache — out of order
        let err = mla_attention(&w.view(), &x, 1, 3, &mut cache, &c);
        assert!(err.is_err(), "out-of-order append must be reported");
        // the engine stays usable: the correct position still works
        let ok = mla_attention(&w.view(), &x, 1, 0, &mut cache, &c)?;
        assert_eq!(ok.len(), hidden);
        Ok(())
    }

    #[test]
    fn empty_dsa_selection_attends_self_not_nothing() -> Result<(), peregrine_core::Error> {
        // An empty selection used to softmax an empty score row, leaving the
        // context vector identically zero — attention silently blanked with no
        // error. Every query must still attend at least its own position.
        let c = cfg()?;
        let w = make_weights(&c, 17);
        let hidden = c.hidden as usize;
        let s_n = 3;
        let mut r = Lcg(0xBEEF);
        let x: Vec<f32> = (0..s_n * hidden).map(|_| r.f()).collect();
        let mut cs = new_cache(&c);
        let empty: Vec<Vec<usize>> = (0..s_n).map(|_| Vec::new()).collect();
        let sparse = mla_attention_dsa(&w.view(), &x, s_n, 0, &mut cs, &c, &empty)?;
        assert!(sparse.iter().any(|v| v.abs() > 1e-9), "attention output must not be all zeros");
        // self-only selection produces the same thing (that is what it falls back to)
        let mut cs2 = new_cache(&c);
        let self_only: Vec<Vec<usize>> = (0..s_n).map(|s| vec![s]).collect();
        let expect = mla_attention_dsa(&w.view(), &x, s_n, 0, &mut cs2, &c, &self_only)?;
        for z in 0..s_n * hidden {
            assert!((sparse[z] - expect[z]).abs() < 1e-6, "z={z}");
        }
        // a selection with the wrong row count is a reported error
        let mut cs3 = new_cache(&c);
        assert!(mla_attention_dsa(&w.view(), &x, s_n, 0, &mut cs3, &c, &empty[..1]).is_err());
        Ok(())
    }

    #[test]
    fn dsa_selecting_all_keys_equals_dense() -> Result<(), peregrine_core::Error> {
        // The DSA sparse path must reproduce dense attention when the selection
        // covers every causal key (the indexer's "select everything" case).
        let c = cfg()?;
        let w = make_weights(&c, 11);
        let hidden = c.hidden as usize;
        let s_n = 4;
        let mut r = Lcg(0x5A5A);
        let x: Vec<f32> = (0..s_n * hidden).map(|_| r.f()).collect();
        let mut cd = new_cache(&c);
        let dense = mla_attention(&w.view(), &x, s_n, 0, &mut cd, &c)?;
        let sel: Vec<Vec<usize>> = (0..s_n).map(|s| (0..=s).collect()).collect();
        let mut cs = new_cache(&c);
        let sparse = mla_attention_dsa(&w.view(), &x, s_n, 0, &mut cs, &c, &sel)?;
        for z in 0..s_n * hidden {
            assert!((dense[z] - sparse[z]).abs() < 1e-6, "z={z} dense={} sparse={}", dense[z], sparse[z]);
        }
        Ok(())
    }

    #[test]
    fn absorb_is_causal() -> Result<(), peregrine_core::Error> {
        // the absorb core must be causal too (exact, no tolerance)
        let c = cfg()?;
        let w = make_weights(&c, 8);
        let hidden = c.hidden as usize;
        let s_n = 4;
        let mut r = Lcg(271);
        let mut x: Vec<f32> = (0..s_n * hidden).map(|_| r.f()).collect();
        let mut ca = new_cache(&c);
        let a = mla_attention_absorb(&w.view(), &x, s_n, 0, &mut ca, &c)?;
        for d in 0..hidden {
            x[(s_n - 1) * hidden + d] += 1.0;
        }
        let mut cb = new_cache(&c);
        let b = mla_attention_absorb(&w.view(), &x, s_n, 0, &mut cb, &c)?;
        for p in 0..s_n - 1 {
            for d in 0..hidden {
                assert!((a[p * hidden + d] - b[p * hidden + d]).abs() < 1e-6);
            }
        }
        Ok(())
    }

    #[test]
    fn batched_matches_sequential() -> Result<(), peregrine_core::Error> {
        // B independent sequences, each prefilled to a different length, then one
        // decode step. The batched absorb output for row s must equal running the
        // single-sequence absorb decode on sequence s alone — the amortization the
        // whole batching design rides on is only valid if this holds.
        let c = cfg()?;
        let w = make_weights(&c, 5);
        let hidden = c.hidden as usize;
        let b = 3usize;
        let lens = [2usize, 1, 3]; // prior context length per sequence
        let mut r = Lcg(13579);

        // per-sequence caches, prefilled with `lens[s]` tokens via the absorb path
        let mut seq_caches: Vec<LayerKv> = (0..b).map(|_| new_cache(&c)).collect();
        for (s, cache) in seq_caches.iter_mut().enumerate() {
            let pre: Vec<f32> = (0..lens[s] * hidden).map(|_| r.f()).collect();
            mla_attention_absorb(&w.view(), &pre, lens[s], 0, cache, &c)?;
        }

        // one new decode token per sequence
        let newtok: Vec<Vec<f32>> = (0..b).map(|_| (0..hidden).map(|_| r.f()).collect()).collect();

        // sequential reference: decode each sequence on a clone of its cache
        let mut seq_out = vec![0f32; b * hidden];
        for s in 0..b {
            let mut refc = seq_caches[s].clone();
            let o = mla_attention_absorb(&w.view(), &newtok[s], 1, lens[s], &mut refc, &c)?;
            seq_out[s * hidden..s * hidden + hidden].copy_from_slice(&o);
        }

        // batched: all b tokens in one call, each with its own pos + cache
        let x: Vec<f32> = (0..b).flat_map(|s| newtok[s].clone()).collect();
        let pos_of: Vec<usize> = lens.iter().take(b).copied().collect();
        let mut refs: Vec<&mut LayerKv> = seq_caches.iter_mut().collect();
        let bat_out = mla_attention_batched(&w.view(), &x, &pos_of, &mut refs, &c)?;

        for z in 0..b * hidden {
            assert!((seq_out[z] - bat_out[z]).abs() < 1e-6, "z={z} seq={} bat={}", seq_out[z], bat_out[z]);
        }
        Ok(())
    }

    #[test]
    fn kv_span_row_lookup_crosses_the_boundary_in_causal_order() {
        // The span's one non-obvious job: rows before the split come from the
        // head, rows after from the tail, and the numbering is continuous. An
        // off-by-one here would silently attend the wrong position.
        let head: Vec<f32> = (0..6).map(|k| k as f32).collect(); // 3 rows of 2
        let tail: Vec<f32> = (6..12).map(|k| k as f32).collect(); // 3 more
        let split = KvSpan::split(&head, &tail);
        let all: Vec<f32> = (0..12).map(|k| k as f32).collect();
        let contig = KvSpan::contiguous(&all);
        assert_eq!(split.rows(2), 6);
        assert_eq!(contig.rows(2), 6);
        for t in 0..6 {
            assert_eq!(split.row(t, 2), contig.row(t, 2), "row {t} must be the same either way");
        }
        // Degenerate shapes return empty rather than dividing by zero.
        assert_eq!(KvSpan::contiguous(&[]).rows(2), 0);
        assert_eq!(contig.rows(0), 0);
        // `runs` skips empties, so a contiguous span is one run, not two.
        assert_eq!(contig.runs().count(), 1);
        assert_eq!(split.runs().count(), 2);
        assert_eq!(KvSpan::split(&head, &[]).runs().count(), 1);
    }

    #[test]
    fn kv_span_split_attention_is_bit_identical_to_contiguous() -> Result<(), peregrine_core::Error> {
        // **The property the whole refactor rests on.** Sharing a prefix by
        // refcount, block-pooling the cache, or narrowing its element type all
        // make the KV discontiguous; none of them may move a single output bit.
        //
        // Achievable here — unlike vLLM's paged attention, which breaks batch
        // invariance by reducing over cached and current K/V separately —
        // because splitting changes only *where a row is read from*, never the
        // order rows are accumulated in. This test is what keeps that true.
        let c = cfg()?;
        let (kvl, qkr) = (c.kv_lora as usize, c.qk_rope as usize);
        let mut rng = Lcg(0xB0A75);
        let w = make_weights(&c, 11);
        let nt = 7usize;

        let lc: Vec<f32> = (0..nt * kvl).map(|_| rng.f()).collect();
        let rc: Vec<f32> = (0..nt * qkr).map(|_| rng.f()).collect();
        let (h_n, qh) = (c.n_heads as usize, c.qk_head as usize);
        let q: Vec<f32> = (0..h_n * qh).map(|_| rng.f()).collect();

        let whole = attend_absorb_batched(
            &w.view(),
            &q,
            &[RowAttn { len: nt, lc: KvSpan::contiguous(&lc), rc: KvSpan::contiguous(&rc) }],
            &c,
            usize::MAX,
        );
        // Every split point, including the degenerate ends, must agree.
        for cut in 0..=nt {
            let (lh, lt) = lc.split_at(cut * kvl);
            let (rh, rt) = rc.split_at(cut * qkr);
            let got = attend_absorb_batched(
                &w.view(),
                &q,
                &[RowAttn { len: nt, lc: KvSpan::split(lh, lt), rc: KvSpan::split(rh, rt) }],
                &c,
                usize::MAX,
            );
            for (k, (a, b)) in whole.iter().zip(&got).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "split at row {cut}, output {k}: not bit-identical");
            }
        }
        Ok(())
    }

    #[test]
    fn a_shared_prefix_produces_the_same_bits_as_a_private_copy() -> Result<(), peregrine_core::Error> {
        // Prefix sharing's whole claim: `clone_prefix` hands out a refcounted
        // view instead of the deep copy every admission used to pay (~350 MB for
        // a 2 k-token system prompt at GLM-5.2's 175.5 KiB/token), and nothing
        // downstream can tell. Both cores are checked — the dense one because
        // its `kv_b` reconstruction now runs once per run and concatenates, the
        // absorb one because it reads every row through the span.
        let c = cfg()?;
        let w = make_weights(&c, 23);
        let hidden = c.hidden as usize;
        let (n, m) = (5usize, 3usize); // shared prefix, then this sequence's own tail
        let mut r = Lcg(0xC0FFEE);
        let x: Vec<f32> = (0..(n + m) * hidden).map(|_| r.f()).collect();

        for dense in [true, false] {
            // Private: one cache owning every position, as before this change.
            let mut owned = new_cache(&c);
            attend_either(dense, &w.view(), &x[..n * hidden], n, 0, &mut owned, &c)?;
            let want = attend_either(dense, &w.view(), &x[n * hidden..], m, n, &mut owned, &c)?;

            // Shared: freeze the prefix, then continue into a private tail.
            let mut src = new_cache(&c);
            attend_either(dense, &w.view(), &x[..n * hidden], n, 0, &mut src, &c)?;
            let mut shared = src.clone_prefix(n);
            assert!(shared.shared_id().is_some(), "clone_prefix must freeze into a shared prefix");
            assert_eq!(shared.owned_bytes(), 0, "a freshly seeded cache owns no rows of its own");
            assert_eq!(shared.len(), n);
            let got = attend_either(dense, &w.view(), &x[n * hidden..], m, n, &mut shared, &c)?;

            assert_eq!(shared.len(), n + m);
            assert!(shared.owned_bytes() > 0, "the tail must be private, not written into the prefix");
            for (k, (a, b)) in want.iter().zip(&got).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "dense={dense}, output {k}: sharing moved a bit");
            }
        }
        Ok(())
    }

    #[test]
    fn cloning_an_already_frozen_prefix_is_a_refcount_bump() -> Result<(), peregrine_core::Error> {
        // The steady state that makes 3a worth anything: the prefix cache's
        // entries are themselves `clone_prefix` results, so every admission
        // matching one takes the shared path. Identical allocation identity is
        // the direct evidence that nothing was copied.
        let c = cfg()?;
        let w = make_weights(&c, 29);
        let hidden = c.hidden as usize;
        let n = 6usize;
        let mut r = Lcg(0x51A7ED);
        let x: Vec<f32> = (0..n * hidden).map(|_| r.f()).collect();
        let mut src = new_cache(&c);
        mla_attention_absorb(&w.view(), &x, n, 0, &mut src, &c)?;

        let entry = src.clone_prefix(n); // the one copy: freezes the prefix
        assert!(src.shared_id().is_none(), "the source owned its rows privately");
        let full = entry.clone_prefix(n);
        let partial = entry.clone_prefix(n - 2); // a shallower match reuses the same buffer
        assert_eq!(entry.shared_id(), full.shared_id(), "a full re-clone must not copy");
        assert_eq!(entry.shared_id(), partial.shared_id(), "a shallower match must not copy either");
        assert_eq!(partial.len(), n - 2);
        assert_eq!(partial.owned_bytes(), 0);
        // The shallower view charges its own logical size but shares the whole
        // allocation — the distinction `resident_kv` budgets on.
        assert!(partial.bytes() < entry.bytes(), "a shallower view is logically smaller");
        assert_eq!(partial.shared_bytes(), entry.shared_bytes(), "…but the allocation is the same one");

        // Rows read through the shallower view are the same rows.
        let kvl = c.kv_lora as usize;
        for t in 0..n - 2 {
            assert_eq!(partial.lc_span(n - 2).row(t, kvl), entry.lc_span(n).row(t, kvl), "row {t}");
        }
        Ok(())
    }

    #[test]
    fn rewinding_into_a_shared_prefix_leaves_other_holders_intact() -> Result<(), peregrine_core::Error> {
        // Speculative decode rewinds the cache on a rejected draft, and the
        // rewind can land *inside* a prefix other sequences are still reading.
        // It must narrow this cache's view, never the shared buffer — truncating
        // the buffer would silently shorten every concurrent sequence seeded
        // from it, which is corruption no test downstream would catch.
        let c = cfg()?;
        let w = make_weights(&c, 31);
        let (hidden, kvl) = (c.hidden as usize, c.kv_lora as usize);
        let n = 6usize;
        let mut r = Lcg(0x2E17);
        let x: Vec<f32> = (0..n * hidden).map(|_| r.f()).collect();
        let mut src = new_cache(&c);
        mla_attention_absorb(&w.view(), &x, n, 0, &mut src, &c)?;

        let entry = src.clone_prefix(n);
        let mut a = entry.clone_prefix(n);
        let b = entry.clone_prefix(n);
        a.truncate(n - 3);
        assert_eq!(a.len(), n - 3);
        assert_eq!(b.len(), n, "the other holder must be untouched by a's rewind");
        for t in 0..n {
            assert_eq!(b.lc_span(n).row(t, kvl), entry.lc_span(n).row(t, kvl), "row {t} of the untouched holder");
        }
        for t in 0..n - 3 {
            assert_eq!(a.lc_span(n - 3).row(t, kvl), entry.lc_span(n).row(t, kvl), "surviving row {t}");
        }
        // The rewound cache still accepts the replacement tokens, in order.
        mla_attention_absorb(&w.view(), &x[..2 * hidden], 2, n - 3, &mut a, &c)?;
        assert_eq!(a.len(), n - 1);
        assert_eq!(b.len(), n, "…and appending to a still does not touch b");
        Ok(())
    }

    #[test]
    fn attend_absorb_parallel_matches_serial() -> Result<(), peregrine_core::Error> {
        // attend_absorb_batched runs rows on the pool above its gate; the parallel
        // and forced-serial branches must be bit-identical (rows are independent).
        // Synthetic q + per-row KV caches suffice — the arithmetic is the same.
        let c = cfg()?;
        let w = make_weights(&c, 9);
        let (h_n, qh, kvl, qkr) = (c.n_heads as usize, c.qk_head as usize, c.kv_lora as usize, c.qk_rope as usize);
        let s_n = 8usize;
        let lens = [1usize, 2, 3, 1, 4, 2, 5, 3]; // varied causal lengths per row
        let mut r = Lcg(0x5eed);
        let q: Vec<f32> = (0..s_n * h_n * qh).map(|_| r.f()).collect();
        let lcs: Vec<Vec<f32>> = (0..s_n).map(|s| (0..lens[s] * kvl).map(|_| r.f()).collect()).collect();
        let rcs: Vec<Vec<f32>> = (0..s_n).map(|s| (0..lens[s] * qkr).map(|_| r.f()).collect()).collect();
        let rows: Vec<RowAttn> = (0..s_n)
            .map(|s| RowAttn { len: lens[s], lc: KvSpan::contiguous(&lcs[s]), rc: KvSpan::contiguous(&rcs[s]) })
            .collect();

        let par = attend_absorb_batched(&w.view(), &q, &rows, &c, 2); // parallel (s_n >= 2)
        let ser = attend_absorb_batched(&w.view(), &q, &rows, &c, usize::MAX); // forced serial
        assert!(par.iter().zip(&ser).all(|(a, b)| a.to_bits() == b.to_bits()), "attend_absorb must be bit-identical parallel vs serial");
        Ok(())
    }
}
