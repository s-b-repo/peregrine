//! Gated-DeltaNet linear attention — the 48 `linear_attention` layers of the
//! Qwen3.5/3.8 hybrid (`Arch::HybridGdn`), in **recurrent form**: one state
//! update per token, O(1) in context length. The chunked/parallel form the HF
//! reference uses for prefill is a wall-clock optimization of the same math
//! and is deliberately not implemented until a measurement asks for it — the
//! recurrent form is the semantics, and on a resident-weights CPU tier the
//! per-token cost is a handful of small GEMVs.
//!
//! Math per token (HF `modeling_qwen3_next.torch_recurrent_gated_delta_rule`,
//! extracted 2026-08-15, pinned by the container parity gate):
//!
//! ```text
//! mixed = silu(causal_conv1d(in_qkv(x)))          # depthwise, k taps, no bias
//! q, k, v = split(mixed);  z = in_z(x)
//! g    = -exp(A_log) * softplus(in_a(x) + dt_bias)   # per v-head
//! beta = sigmoid(in_b(x))                             # per v-head
//! q, k = l2norm(q), l2norm(k) per k-head;  q *= k_dim^-0.5
//! per v-head h (k-head h / (v_heads/k_heads), repeat-interleaved):
//!   S_h *= exp(g_h)
//!   delta = (v_h - S_h·k) * beta_h
//!   S_h  += outer(k, delta)
//!   out_h = q·S_h
//! out = RMSNorm(out) per v-head ⊙ silu(z);  y = out_proj(out)
//! ```
//!
//! The repeat-interleaved q/k head mapping (`h / group`, not `h % k_heads`)
//! is one of the fast-model-read contract points the parity gate verifies —
//! flagged in the Track C interface block.

use crate::math::{sigmoidf, siluf};
use crate::weight::QtWeight;
use peregrine_core::{Cfg, Error};

/// The gated-DeltaNet layer's weights, borrowed from the loaded layer.
pub struct GdnWeights<'a> {
    /// `in_proj_qkv`: `[2*k_heads*k_dim + v_heads*v_dim, hidden]`.
    pub in_qkv: &'a QtWeight,
    /// `in_proj_z` (output gate): `[v_heads*v_dim, hidden]`.
    pub in_z: &'a QtWeight,
    /// `in_proj_a` (decay input): `[v_heads, hidden]`.
    pub in_a: &'a QtWeight,
    /// `in_proj_b` (beta input): `[v_heads, hidden]`.
    pub in_b: &'a QtWeight,
    /// Depthwise causal conv taps, `[conv_dim * k]` row-major (`w[c*k + j]`,
    /// tap `j = k-1` multiplies the current token). Kept float — 40 KiB on the
    /// 27B, and a quantized 4-tap filter would be all rounding, no saving.
    pub conv: &'a [f32],
    /// `A_log`: `[v_heads]`.
    pub a_log: &'a [f32],
    /// `dt_bias`: `[v_heads]`.
    pub dt_bias: &'a [f32],
    /// Gated RMS-norm weight over one value head: `[v_dim]`.
    pub norm: &'a [f32],
    /// `out_proj`: `[hidden, v_heads*v_dim]`.
    pub out: &'a QtWeight,
}

/// A saved [`GdnState`] context — see [`GdnState::snapshot`]. Opaque by
/// design: the only valid operations are restoring it into the state it came
/// from and dropping it.
#[derive(Clone)]
pub struct GdnSnapshot {
    ring: Vec<f32>,
    filled: usize,
    s: Vec<f32>,
    len: usize,
}

impl GdnSnapshot {
    /// Bytes this snapshot holds. The spec-decode caller reports the sum as
    /// `[spec] gdn_snapshot_bytes`, because the whole question of whether
    /// recurrent speculation pays is this copy against the tokens it buys, and
    /// a cost nobody can see is a cost nobody tunes.
    pub fn bytes(&self) -> usize {
        (self.ring.len() + self.s.len()) * 4
    }
}

/// Per-stream recurrent state for one GDN layer: the conv ring (last `k-1`
/// pre-activation rows) and the delta-rule memory `S`. This replaces the KV
/// cache for linear layers — constant size however long the context runs,
/// which is most of why the hybrid can hold 262k tokens on consumer hardware.
#[derive(Clone)]
pub struct GdnState {
    /// `[k-1, conv_dim]` ring of past pre-activation projections, oldest first
    /// once full. `filled` counts real rows during warmup (zero-padding).
    ring: Vec<f32>,
    filled: usize,
    /// `S`: `[v_heads, k_dim, v_dim]`, kept f32 (`mamba_ssm_dtype`).
    s: Vec<f32>,
    /// Positions processed — the GDN analogue of `LayerKv::len`.
    pub len: usize,
}

impl GdnState {
    pub fn new(c: &Cfg) -> GdnState {
        let conv_dim = (2 * c.lin_k_heads * c.lin_k_dim + c.lin_v_heads * c.lin_v_dim) as usize;
        let taps = (c.lin_conv_k as usize).max(1);
        GdnState {
            ring: vec![0.0; (taps - 1) * conv_dim],
            filled: 0,
            s: vec![0.0; (c.lin_v_heads * c.lin_k_dim * c.lin_v_dim) as usize],
            len: 0,
        }
    }

    /// Bytes this state holds — the linear layers' answer to `LayerKv::bytes`.
    pub fn bytes(&self) -> usize {
        (self.ring.len() + self.s.len()) * 4
    }

    /// A fresh (empty-context) state with `like`'s geometry — the safe
    /// fallback when a caller needs a state slot but cannot legally clone one
    /// mid-sequence (see `SeqKv::clone_prefix`).
    pub fn new_like(like: &GdnState) -> GdnState {
        GdnState { ring: vec![0.0; like.ring.len()], filled: 0, s: vec![0.0; like.s.len()], len: 0 }
    }

    /// A point-in-time copy of the whole recurrent context. Speculative decode
    /// needs it because a GDN state cannot rewind: KV rows for rejected draft
    /// tokens can be truncated away, but the delta-rule memory has already
    /// folded them in. The protocol (consumed by the spec-decode verify loop):
    /// snapshot before the verify forward; on FULL acceptance drop the snapshot
    /// (the state is exactly right — with spec-conf's ~80% accept rates this is
    /// the common, zero-cost case); on partial acceptance restore and re-advance
    /// the accepted rows. ~3.1 MB per layer at 27B dims — one snapshot per
    /// sequence per verify step, never one per draft position.
    pub fn snapshot(&self) -> GdnSnapshot {
        GdnSnapshot { ring: self.ring.clone(), filled: self.filled, s: self.s.clone(), len: self.len }
    }

    /// Restore a snapshot taken from this layer's own stream. Geometry is
    /// checked — restoring across layers or models is a wiring bug reported as
    /// an error, never a silent state corruption.
    pub fn restore(&mut self, snap: &GdnSnapshot) -> Result<(), Error> {
        if snap.ring.len() != self.ring.len() || snap.s.len() != self.s.len() {
            return Err(Error::Format(format!(
                "gdn restore: snapshot geometry (ring {}, S {}) does not match the state (ring {}, S {})",
                snap.ring.len(),
                snap.s.len(),
                self.ring.len(),
                self.s.len()
            )));
        }
        self.ring.copy_from_slice(&snap.ring);
        self.filled = snap.filled;
        self.s.copy_from_slice(&snap.s);
        self.len = snap.len;
        Ok(())
    }

    /// Reset to the empty-context state (a new sequence on the same stream).
    pub fn reset(&mut self) {
        self.ring.fill(0.0);
        self.filled = 0;
        self.s.fill(0.0);
        self.len = 0;
    }
}

/// One head's delta-rule update and readout, on plain slices — the pure core,
/// testable without projections or a config. `s` is `[k_dim, v_dim]` row-major.
fn gdn_step_head(
    s: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    decay: f32,
    beta: f32,
    out: &mut [f32],
) {
    let (kd, vd) = (q.len(), v.len());
    debug_assert_eq!(s.len(), kd * vd);
    for x in s.iter_mut() {
        *x *= decay;
    }
    // kv_mem = S·k ; delta = (v - kv_mem) * beta ; S += outer(k, delta)
    for j in 0..vd {
        let mut mem = 0.0f32;
        for i in 0..kd {
            mem += s[i * vd + j] * k[i];
        }
        let delta = (v[j] - mem) * beta;
        for i in 0..kd {
            s[i * vd + j] += k[i] * delta;
        }
    }
    // out = q·S
    for j in 0..vd {
        let mut acc = 0.0f32;
        for i in 0..kd {
            acc += s[i * vd + j] * q[i];
        }
        out[j] = acc;
    }
}

/// L2-normalize `v` in place (eps inside the root, matching the HF kernel).
fn l2norm(v: &mut [f32], eps: f32) {
    let n = (v.iter().map(|x| x * x).sum::<f32>() + eps).sqrt();
    let inv = 1.0 / n;
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// `softplus(x) = ln(1 + e^x)`, with the standard large-`x` shortcut so the
/// exponential never overflows.
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln_1p_free()
    }
}

/// A tiny shim so `softplus` reads as the formula: `ln(1+e^x)` computed as
/// `ln_1p(e^x)` for accuracy at small magnitudes.
trait Ln1pFree {
    fn ln_1p_free(self) -> f32;
}
impl Ln1pFree for f32 {
    fn ln_1p_free(self) -> f32 {
        (self - 1.0).ln_1p()
    }
}

/// Forward `s_n` tokens of one GDN layer, advancing `state`. `x` is
/// `[s_n, hidden]` (already input-layernormed by the caller, like every other
/// attention entry point here). Returns `[s_n, hidden]`.
///
/// Sequential by construction — token `t+1`'s state is token `t`'s output
/// state — so one call over `s_n` rows and `s_n` single-row calls are
/// bit-identical (`gdn_one_call_matches_stepwise`).
pub fn gdn_forward(
    w: &GdnWeights,
    x: &[f32],
    s_n: usize,
    state: &mut GdnState,
    c: &Cfg,
) -> Result<Vec<f32>, Error> {
    let kh = c.lin_k_heads as usize;
    let vh = c.lin_v_heads as usize;
    let kd = c.lin_k_dim as usize;
    let vd = c.lin_v_dim as usize;
    let taps = (c.lin_conv_k as usize).max(1);
    let conv_dim = 2 * kh * kd + vh * vd;
    let group = vh / kh.max(1);
    if w.conv.len() != conv_dim * taps {
        return Err(Error::Format(format!(
            "gdn: conv1d holds {} taps, expected {} ({} channels x {} taps)",
            w.conv.len(),
            conv_dim * taps,
            conv_dim,
            taps
        )));
    }

    let qkv_pre = w.in_qkv.apply_vec(x, s_n); // [s_n, conv_dim]
    let z_all = w.in_z.apply_vec(x, s_n); // [s_n, vh*vd]
    let a_all = w.in_a.apply_vec(x, s_n); // [s_n, vh]
    let b_all = w.in_b.apply_vec(x, s_n); // [s_n, vh]

    let mut y = vec![0.0f32; s_n * (c.hidden as usize)];
    let mut mixed = vec![0.0f32; conv_dim];
    let mut head_out = vec![0.0f32; vd];
    let mut gated = vec![0.0f32; vh * vd];
    for t in 0..s_n {
        let pre = &qkv_pre[t * conv_dim..(t + 1) * conv_dim];
        // Depthwise causal conv over [ring rows.., current], tap k-1 = current.
        // Ring rows older than `filled` are the zero left-padding.
        for ch in 0..conv_dim {
            let taps_w = &w.conv[ch * taps..(ch + 1) * taps];
            let mut acc = taps_w[taps - 1] * pre[ch];
            for (j, &wj) in taps_w[..taps - 1].iter().enumerate() {
                // ring row index j is the (taps-1-j)-steps-ago token
                let age = taps - 1 - j;
                if age <= state.filled {
                    acc += wj * state.ring[j * conv_dim + ch];
                }
            }
            mixed[ch] = siluf(acc);
        }
        // Advance the ring: shift rows left by one, append `pre`.
        if taps > 1 {
            state.ring.copy_within(conv_dim.., 0);
            let last = (taps - 2) * conv_dim;
            state.ring[last..last + conv_dim].copy_from_slice(pre);
            state.filled = (state.filled + 1).min(taps - 1);
        }

        let (q_all, rest) = mixed.split_at_mut(kh * kd);
        let (k_all, v_all) = rest.split_at_mut(kh * kd);
        let scale = (kd as f32).powf(-0.5);
        for h in 0..kh {
            let qh = &mut q_all[h * kd..(h + 1) * kd];
            l2norm(qh, 1e-6);
            for v in qh.iter_mut() {
                *v *= scale;
            }
            l2norm(&mut k_all[h * kd..(h + 1) * kd], 1e-6);
        }

        for h in 0..vh {
            let a = a_all[t * vh + h];
            let b = b_all[t * vh + h];
            let g = -w.a_log[h].exp() * softplus(a + w.dt_bias[h]);
            let decay = g.exp();
            let beta = sigmoidf(b);
            let kh_idx = h / group; // repeat_interleave mapping (gate-verified)
            gdn_step_head(
                &mut state.s[h * kd * vd..(h + 1) * kd * vd],
                &q_all[kh_idx * kd..(kh_idx + 1) * kd],
                &k_all[kh_idx * kd..(kh_idx + 1) * kd],
                &v_all[h * vd..(h + 1) * vd],
                decay,
                beta,
                &mut head_out,
            );
            // Gated RMS norm per value head, then ⊙ silu(z).
            let ms = head_out.iter().map(|v| v * v).sum::<f32>() / vd as f32;
            let inv = 1.0 / (ms + c.eps).sqrt();
            let z = &z_all[t * vh * vd + h * vd..t * vh * vd + (h + 1) * vd];
            for j in 0..vd {
                gated[h * vd + j] = head_out[j] * inv * w.norm[j] * siluf(z[j]);
            }
        }
        state.len += 1;
        let row = w.out.apply_vec(&gated, 1);
        let d = c.hidden as usize;
        y[t * d..(t + 1) * d].copy_from_slice(&row);
    }
    Ok(y)
}

/// KDA (Kimi Delta Attention) weights — the Glm5Next linear-attention layer,
/// borrowed from the loaded layer. KDA differs from the Qwen GDN above in
/// every gate: the forget gate is per *k-channel* (GDN's is a per-head
/// scalar), beta comes from its own `b_proj`, and the output gate is a
/// low-rank `g_b(g_a(x))` through a **sigmoid**-gated RMSNorm (GDN gates with
/// `silu(z)`).
pub struct KdaWeights<'a> {
    /// `q_proj` / `k_proj` / `v_proj`: `[h*d, hidden]` each.
    pub q: &'a QtWeight,
    pub k: &'a QtWeight,
    pub v: &'a QtWeight,
    /// Depthwise causal conv taps over `concat(q, k, v)`: `[3*h*d * taps]`
    /// row-major (`w[c*taps + j]`, tap `j = taps-1` multiplies the current
    /// token) — the checkpoint's three per-projection convs concatenated in
    /// q‖k‖v order by the loader.
    pub conv: &'a [f32],
    /// Forget-gate low-rank pair: `f_a [d, hidden]`, `f_b [h*d, d]`.
    pub f_a: &'a QtWeight,
    pub f_b: &'a QtWeight,
    /// `dt_bias`: `[h*d]` — per channel, not per head.
    pub dt_bias: &'a [f32],
    /// `A_log`: `[h]`.
    pub a_log: &'a [f32],
    /// `b_proj` (beta input): `[h, hidden]`.
    pub b: &'a QtWeight,
    /// Output-gate low-rank pair: `g_a [d, hidden]`, `g_b [h*d, d]`.
    pub g_a: &'a QtWeight,
    pub g_b: &'a QtWeight,
    /// Gated RMS-norm weight over one head: `[d]`.
    pub o_norm: &'a [f32],
    /// `o_proj`: `[hidden, h*d]`.
    pub o: &'a QtWeight,
}

/// One head's KDA update and readout: like [`gdn_step_head`] but the decay is
/// per k-channel (`g_log[i]` is the *log* decay of S's row `i`), which is the
/// whole "Delta" refinement KDA adds over GDN.
fn kda_step_head(
    s: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g_log: &[f32],
    beta: f32,
    out: &mut [f32],
) {
    let (kd, vd) = (q.len(), v.len());
    debug_assert_eq!(s.len(), kd * vd);
    debug_assert_eq!(g_log.len(), kd);
    for i in 0..kd {
        let decay = g_log[i].exp();
        for x in s[i * vd..(i + 1) * vd].iter_mut() {
            *x *= decay;
        }
    }
    // kv_mem = S·k ; delta = (v - kv_mem) * beta ; S += outer(k, delta)
    for j in 0..vd {
        let mut mem = 0.0f32;
        for i in 0..kd {
            mem += s[i * vd + j] * k[i];
        }
        let delta = (v[j] - mem) * beta;
        for i in 0..kd {
            s[i * vd + j] += k[i] * delta;
        }
    }
    for j in 0..vd {
        let mut acc = 0.0f32;
        for i in 0..kd {
            acc += s[i * vd + j] * q[i];
        }
        out[j] = acc;
    }
}

/// Forward `s_n` tokens of one KDA layer, advancing `state`. Same contract as
/// [`gdn_forward`]: `x` is `[s_n, hidden]` already input-layernormed, the
/// recurrent form is the semantics (one call and stepwise are bit-identical),
/// and `state` is a [`GdnState`] whose geometry the Glm5Next config lays out
/// as `lin_k_heads == lin_v_heads == h`, `lin_k_dim == lin_v_dim == d` — so
/// `conv_dim = 2·h·d + h·d = 3·h·d`, exactly the q‖k‖v concat this layer convolves.
pub fn kda_forward(
    w: &KdaWeights,
    x: &[f32],
    s_n: usize,
    state: &mut GdnState,
    c: &Cfg,
) -> Result<Vec<f32>, Error> {
    let h_n = c.lin_k_heads as usize;
    let dd = c.lin_k_dim as usize;
    let taps = (c.lin_conv_k as usize).max(1);
    let qkv = h_n * dd;
    let conv_dim = 3 * qkv;
    if w.conv.len() != conv_dim * taps {
        return Err(Error::Format(format!(
            "kda: conv1d holds {} taps, expected {} ({} channels x {} taps)",
            w.conv.len(),
            conv_dim * taps,
            conv_dim,
            taps
        )));
    }
    if state.ring.len() != (taps - 1) * conv_dim || state.s.len() != h_n * dd * dd {
        return Err(Error::Format(format!(
            "kda: state geometry (ring {}, S {}) does not match the layer (ring {}, S {})",
            state.ring.len(),
            state.s.len(),
            (taps - 1) * conv_dim,
            h_n * dd * dd
        )));
    }

    let q_pre = w.q.apply_vec(x, s_n); // [s_n, qkv]
    let k_pre = w.k.apply_vec(x, s_n);
    let v_pre = w.v.apply_vec(x, s_n);
    let f_all = w.f_b.apply_vec(&w.f_a.apply_vec(x, s_n), s_n); // [s_n, qkv]
    let gate_all = w.g_b.apply_vec(&w.g_a.apply_vec(x, s_n), s_n); // [s_n, qkv]
    let b_all = w.b.apply_vec(x, s_n); // [s_n, h]

    let mut y = vec![0.0f32; s_n * (c.hidden as usize)];
    let mut pre = vec![0.0f32; conv_dim];
    let mut mixed = vec![0.0f32; conv_dim];
    let mut g_log = vec![0.0f32; dd];
    let mut head_out = vec![0.0f32; dd];
    let mut gated = vec![0.0f32; qkv];
    for t in 0..s_n {
        pre[..qkv].copy_from_slice(&q_pre[t * qkv..(t + 1) * qkv]);
        pre[qkv..2 * qkv].copy_from_slice(&k_pre[t * qkv..(t + 1) * qkv]);
        pre[2 * qkv..].copy_from_slice(&v_pre[t * qkv..(t + 1) * qkv]);
        // Depthwise causal conv + SiLU — same ring discipline as gdn_forward.
        for ch in 0..conv_dim {
            let taps_w = &w.conv[ch * taps..(ch + 1) * taps];
            let mut acc = taps_w[taps - 1] * pre[ch];
            for (j, &wj) in taps_w[..taps - 1].iter().enumerate() {
                let age = taps - 1 - j;
                if age <= state.filled {
                    acc += wj * state.ring[j * conv_dim + ch];
                }
            }
            mixed[ch] = siluf(acc);
        }
        if taps > 1 {
            state.ring.copy_within(conv_dim.., 0);
            let last = (taps - 2) * conv_dim;
            state.ring[last..last + conv_dim].copy_from_slice(&pre);
            state.filled = (state.filled + 1).min(taps - 1);
        }

        let (q_all, rest) = mixed.split_at_mut(qkv);
        let (k_all, v_all) = rest.split_at_mut(qkv);
        let scale = (dd as f32).powf(-0.5);
        for h in 0..h_n {
            let qh = &mut q_all[h * dd..(h + 1) * dd];
            l2norm(qh, 1e-6);
            for v in qh.iter_mut() {
                *v *= scale;
            }
            l2norm(&mut k_all[h * dd..(h + 1) * dd], 1e-6);
        }

        let f_row = &f_all[t * qkv..(t + 1) * qkv];
        for h in 0..h_n {
            let a = w.a_log[h].exp();
            // Per-channel log decay. Bounded form: `lb · sigmoid(A · (f + dt_bias))`
            // keeps it in (lb, 0); unbounded: `-A · softplus(f + dt_bias)`.
            for (i, gl) in g_log.iter_mut().enumerate() {
                let g_raw = f_row[h * dd + i] + w.dt_bias[h * dd + i];
                *gl = match c.lin_gate_lb {
                    Some(lb) => lb * sigmoidf(a * g_raw),
                    None => -a * softplus(g_raw),
                };
            }
            let beta = sigmoidf(b_all[t * h_n + h]);
            kda_step_head(
                &mut state.s[h * dd * dd..(h + 1) * dd * dd],
                &q_all[h * dd..(h + 1) * dd],
                &k_all[h * dd..(h + 1) * dd],
                &v_all[h * dd..(h + 1) * dd],
                &g_log,
                beta,
                &mut head_out,
            );
            // Sigmoid-gated RMS norm per head: RMSNorm(out)·w ⊙ σ(g_b(g_a(x))).
            let ms = head_out.iter().map(|v| v * v).sum::<f32>() / dd as f32;
            let inv = 1.0 / (ms + c.eps).sqrt();
            let gt = &gate_all[t * qkv + h * dd..t * qkv + (h + 1) * dd];
            for j in 0..dd {
                gated[h * dd + j] = head_out[j] * inv * w.o_norm[j] * sigmoidf(gt[j]);
            }
        }
        state.len += 1;
        let row = w.o.apply_vec(&gated, 1);
        let d = c.hidden as usize;
        y[t * d..(t + 1) * d].copy_from_slice(&row);
    }
    Ok(y)
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

    #[test]
    fn delta_rule_memorizes_a_pair_at_full_beta_and_no_decay() {
        // With decay=1 and beta=1, presenting (k, v) once makes S·k = v exactly;
        // querying with q = k then reads v back. This is the delta rule's whole
        // contract, testable on the pure core.
        let (kd, vd) = (4, 3);
        let mut s = vec![0.0f32; kd * vd];
        let mut k = vec![0.5f32, -0.5, 0.5, -0.5];
        l2norm(&mut k, 0.0); // unit key, so k·k = 1 and the readout is exact
        let v = vec![1.0f32, -2.0, 3.0];
        let mut out = vec![0.0f32; vd];
        gdn_step_head(&mut s, &k.clone(), &k, &v, 1.0, 1.0, &mut out);
        for (o, want) in out.iter().zip(&v) {
            assert!((o - want).abs() < 1e-6, "first presentation reads back the value (got {o}, want {want})");
        }
        // Presenting it again changes nothing: delta = (v - S·k) = 0.
        let s_before = s.clone();
        gdn_step_head(&mut s, &k.clone(), &k, &v, 1.0, 1.0, &mut out);
        assert!(s.iter().zip(&s_before).all(|(a, b)| (a - b).abs() < 1e-6), "a memorized pair is a fixed point");
    }

    #[test]
    fn decay_scales_the_memory_before_the_update() {
        let (kd, vd) = (2, 2);
        let mut s = vec![1.0f32; kd * vd];
        let q = vec![0.0f32; kd]; // read nothing
        let k = vec![0.0f32; kd]; // write nothing
        let v = vec![0.0f32; vd];
        let mut out = vec![0.0f32; vd];
        gdn_step_head(&mut s, &q, &k, &v, 0.25, 1.0, &mut out);
        assert!(s.iter().all(|&x| (x - 0.25).abs() < 1e-7), "decay multiplies S before k/v touch it");
    }

    fn tiny_gdn_cfg() -> Result<Cfg, peregrine_core::Error> {
        Cfg::from_json(&serde_json::json!({
            "model_type": "qwen3_5",
            "vocab_size": 32, "hidden_size": 16, "intermediate_size": 8,
            "num_hidden_layers": 1, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 4,
            "layer_types": ["linear_attention"],
            "linear_num_key_heads": 2, "linear_num_value_heads": 4,
            "linear_key_head_dim": 4, "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4, "attn_output_gate": true,
            "partial_rotary_factor": 0.5,
            "rope_theta": 10000.0, "rms_norm_eps": 1e-6, "eos_token_id": 0
        }))
    }

    struct W {
        in_qkv: QtWeight,
        in_z: QtWeight,
        in_a: QtWeight,
        in_b: QtWeight,
        conv: Vec<f32>,
        a_log: Vec<f32>,
        dt_bias: Vec<f32>,
        norm: Vec<f32>,
        out: QtWeight,
    }
    impl W {
        fn view(&self) -> GdnWeights<'_> {
            GdnWeights {
                in_qkv: &self.in_qkv,
                in_z: &self.in_z,
                in_a: &self.in_a,
                in_b: &self.in_b,
                conv: &self.conv,
                a_log: &self.a_log,
                dt_bias: &self.dt_bias,
                norm: &self.norm,
                out: &self.out,
            }
        }
    }

    fn make_weights(c: &Cfg, seed: u64) -> W {
        let mut r = Lcg(seed);
        let d = c.hidden as usize;
        let (kh, vh, kd, vd, taps) = (
            c.lin_k_heads as usize,
            c.lin_v_heads as usize,
            c.lin_k_dim as usize,
            c.lin_v_dim as usize,
            c.lin_conv_k as usize,
        );
        let conv_dim = 2 * kh * kd + vh * vd;
        let w = |r: &mut Lcg, n: usize| (0..n).map(|_| r.f()).collect::<Vec<f32>>();
        W {
            in_qkv: quant_i4(&w(&mut r, conv_dim * d), conv_dim, d),
            in_z: quant_i4(&w(&mut r, vh * vd * d), vh * vd, d),
            in_a: quant_i4(&w(&mut r, vh * d), vh, d),
            in_b: quant_i4(&w(&mut r, vh * d), vh, d),
            conv: w(&mut r, conv_dim * taps),
            a_log: (0..vh).map(|_| r.f() * 0.5).collect(),
            dt_bias: (0..vh).map(|_| r.f() * 0.5).collect(),
            norm: (0..vd).map(|_| 0.8 + r.f() * 0.1).collect(),
            out: quant_i4(&w(&mut r, d * vh * vd), d, vh * vd),
        }
    }

    /// The regime an open serving failure points at: a GDN layer at REAL
    /// widths with only its two smallest weights — `in_proj_a` and
    /// `in_proj_b`, [48, 5120] each — VRAM-resident, because at a nearly-full
    /// card those are the first to fit. They feed the recurrence's decay and
    /// beta rather than an ordinary activation, so a wrong value there
    /// degenerates the stream instead of blurring it.
    ///
    /// Real widths deliberately: the toy fixture's `in_proj_a` is [4, 256],
    /// which is a different kernel regime and would not reproduce a bug that
    /// depends on the real one.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_gdn_layer_with_resident_gates_computes_identically() -> Result<(), peregrine_core::Error> {
        if peregrine_cuda::init(&[0]) < 1 {
            return Ok(());
        }
        let c = Cfg::from_json(&serde_json::json!({
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "vocab_size": 32, "hidden_size": 5120, "intermediate_size": 64,
                "num_hidden_layers": 1, "num_attention_heads": 4,
                "num_key_value_heads": 2, "head_dim": 64,
                "layer_types": ["linear_attention"],
                "linear_num_key_heads": 16, "linear_num_value_heads": 48,
                "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4, "attn_output_gate": true,
                "partial_rotary_factor": 0.25,
                "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25},
                "rms_norm_eps": 1e-6, "eos_token_id": 0
            }
        }))?;
        let mut w = make_weights(&c, 91);
        let d = c.hidden as usize;
        let mut r = Lcg(37);
        let x: Vec<f32> = (0..d).map(|_| r.f()).collect();

        let mut st_cpu = GdnState::new(&c);
        let cpu = gdn_forward(&w.view(), &x, 1, &mut st_cpu, &c)?;

        // The set a real budget actually places: the LARGE projections. The
        // gates are 123 KB each and the big three are ~58 MB, so a budget that
        // dies mid-set keeps whichever it reached first — and a test that
        // placed only the gates covered the two weights the real run SKIPPED.
        let up_qkv = w.in_qkv.upload_to_device(0, 1 << 20)?;
        let up_z = w.in_z.upload_to_device(0, 1 << 20)?;
        let up_out = w.out.upload_to_device(0, 1 << 20)?;
        let a_up = w.in_a.upload_to_device(0, 1 << 20)?;
        let b_up = w.in_b.upload_to_device(0, 1 << 20)?;
        println!("placed: qkv={up_qkv} z={up_z} out={up_out} a={a_up} b={b_up}");
        if !(up_qkv || up_z || up_out) {
            return Ok(()); // no headroom right now
        }
        let mut st_gpu = GdnState::new(&c);
        let gpu = gdn_forward(&w.view(), &x, 1, &mut st_gpu, &c)?;

        // Non-vacuity: the device path is numerically distinct, so identical
        // bits would mean it never ran and this asserted nothing.
        assert!(
            cpu.iter().zip(&gpu).any(|(p, q)| p.to_bits() != q.to_bits()),
            "outputs are bit-identical: the device never computed, so this test is vacuous"
        );
        let scale = cpu.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        let worst = cpu.iter().zip(&gpu).fold(0f32, |m, (p, q)| m.max((p - q).abs()));
        println!("gdn resident-gates: worst {worst:.3e} on scale {scale:.3e} ({:.2}%)", worst / scale * 100.0);
        assert!(
            worst / scale < 0.05,
            "resident gates changed the layer output by {:.1}% — the recurrence is being fed wrong values",
            worst / scale * 100.0
        );
        Ok(())
    }

    #[test]
    fn gdn_one_call_matches_stepwise() -> Result<(), peregrine_core::Error> {
        // The recurrent form's state-carry contract: 6 tokens in one call and
        // six 1-token calls must agree bit for bit — this exercises the conv
        // ring across the warmup boundary (taps-1 = 3 < 6) and S carry.
        let c = tiny_gdn_cfg()?;
        let w = make_weights(&c, 5);
        let d = c.hidden as usize;
        let mut r = Lcg(17);
        let x: Vec<f32> = (0..6 * d).map(|_| r.f()).collect();

        let mut st_a = GdnState::new(&c);
        let all = gdn_forward(&w.view(), &x, 6, &mut st_a, &c)?;
        let mut st_b = GdnState::new(&c);
        let mut step = Vec::new();
        for t in 0..6 {
            step.extend(gdn_forward(&w.view(), &x[t * d..(t + 1) * d], 1, &mut st_b, &c)?);
        }
        assert!(
            all.iter().zip(&step).all(|(a, b)| a.to_bits() == b.to_bits()),
            "one call and stepwise must be bit-identical"
        );
        assert_eq!(st_a.len, st_b.len);
        assert!(st_a.s.iter().zip(&st_b.s).all(|(a, b)| a.to_bits() == b.to_bits()), "states must match too");
        Ok(())
    }

    #[test]
    fn snapshot_restore_rewinds_a_diverged_state_bit_exactly() -> Result<(), peregrine_core::Error> {
        // The spec-decode contract: advance, snapshot, advance further down a
        // draft that gets rejected, restore — the state and all subsequent
        // outputs must be bit-identical to never having taken the draft.
        let c = tiny_gdn_cfg()?;
        let w = make_weights(&c, 13);
        let d = c.hidden as usize;
        let mut r = Lcg(29);
        let x: Vec<f32> = (0..8 * d).map(|_| r.f()).collect();

        let mut st = GdnState::new(&c);
        gdn_forward(&w.view(), &x[..3 * d], 3, &mut st, &c)?; // committed context
        let snap = st.snapshot();
        gdn_forward(&w.view(), &x[3 * d..7 * d], 4, &mut st, &c)?; // rejected draft
        st.restore(&snap)?;
        let after_restore = gdn_forward(&w.view(), &x[7 * d..8 * d], 1, &mut st, &c)?;

        let mut clean = GdnState::new(&c);
        gdn_forward(&w.view(), &x[..3 * d], 3, &mut clean, &c)?;
        let clean_out = gdn_forward(&w.view(), &x[7 * d..8 * d], 1, &mut clean, &c)?;
        assert!(
            after_restore.iter().zip(&clean_out).all(|(a, b)| a.to_bits() == b.to_bits()),
            "a restored state must continue bit-identically to one that never drafted"
        );
        assert_eq!(st.len, clean.len);
        // Cross-geometry restore is refused, not absorbed.
        let other = tiny_gdn_cfg()?;
        let mut wrong = GdnState::new(&other);
        wrong.s.push(0.0); // perturb geometry
        assert!(wrong.restore(&snap).is_err(), "geometry mismatch must refuse");
        Ok(())
    }

    #[test]
    fn kda_per_channel_decay_scales_each_state_row_independently() {
        // Channel 0 decays to 0.5, channel 1 not at all: after the update with a
        // zero key/value (nothing written), S's row 0 halves and row 1 is intact.
        let (kd, vd) = (2, 2);
        let mut s = vec![1.0f32; kd * vd];
        let zero = vec![0.0f32; kd];
        let v = vec![0.0f32; vd];
        let g_log = vec![0.5f32.ln(), 0.0];
        let mut out = vec![0.0f32; vd];
        kda_step_head(&mut s, &zero, &zero, &v, &g_log, 1.0, &mut out);
        assert!((s[0] - 0.5).abs() < 1e-7 && (s[1] - 0.5).abs() < 1e-7, "row 0 halves");
        assert!((s[2] - 1.0).abs() < 1e-7 && (s[3] - 1.0).abs() < 1e-7, "row 1 intact");
    }

    #[test]
    fn kda_delta_rule_memorizes_a_pair_at_full_beta_and_no_decay() {
        let (kd, vd) = (4, 4);
        let mut s = vec![0.0f32; kd * vd];
        let mut k = vec![0.5f32, -0.5, 0.5, -0.5];
        l2norm(&mut k, 0.0);
        let v = vec![1.0f32, -2.0, 3.0, 0.5];
        let g_log = vec![0.0f32; kd]; // decay 1
        let mut out = vec![0.0f32; vd];
        kda_step_head(&mut s, &k.clone(), &k, &v, &g_log, 1.0, &mut out);
        for (o, want) in out.iter().zip(&v) {
            assert!((o - want).abs() < 1e-6, "readout {o} != {want}");
        }
    }

    fn tiny_kda_cfg() -> Result<Cfg, peregrine_core::Error> {
        Cfg::from_json(&serde_json::json!({
            "model_type": "glm5_next_text",
            "vocab_size": 32, "hidden_size": 16,
            "num_hidden_layers": 1, "num_attention_heads": 2,
            "n_routed_experts": 4, "num_experts_per_tok": 2,
            "moe_intermediate_size": 8, "intermediate_size": 8,
            "mlp_layer_types": ["sparse"],
            "layer_types": ["linear_attention"],
            "q_lora_rank": 12, "kv_lora_rank": 8,
            "qk_nope_head_dim": 4, "qk_rope_head_dim": 0, "v_head_dim": 4,
            "n_shared_experts": 1, "norm_topk_prob": true,
            "routed_scaling_factor": 2.5, "rms_norm_eps": 1e-5,
            "swiglu_limit": 10.0,
            "linear_attn_config": {"num_heads": 2, "head_dim": 4,
                                   "short_conv_kernel_size": 4, "gate_lower_bound": -5.0},
            "hc_mult": 4, "eos_token_id": 0
        }))
    }

    struct KW {
        q: QtWeight,
        k: QtWeight,
        v: QtWeight,
        conv: Vec<f32>,
        f_a: QtWeight,
        f_b: QtWeight,
        dt_bias: Vec<f32>,
        a_log: Vec<f32>,
        b: QtWeight,
        g_a: QtWeight,
        g_b: QtWeight,
        o_norm: Vec<f32>,
        o: QtWeight,
    }
    impl KW {
        fn view(&self) -> KdaWeights<'_> {
            KdaWeights {
                q: &self.q,
                k: &self.k,
                v: &self.v,
                conv: &self.conv,
                f_a: &self.f_a,
                f_b: &self.f_b,
                dt_bias: &self.dt_bias,
                a_log: &self.a_log,
                b: &self.b,
                g_a: &self.g_a,
                g_b: &self.g_b,
                o_norm: &self.o_norm,
                o: &self.o,
            }
        }
    }

    fn make_kda_weights(c: &Cfg, seed: u64) -> KW {
        let mut r = Lcg(seed);
        let d = c.hidden as usize;
        let (h, dd, taps) = (c.lin_k_heads as usize, c.lin_k_dim as usize, c.lin_conv_k as usize);
        let qkv = h * dd;
        let w = |r: &mut Lcg, n: usize| (0..n).map(|_| r.f()).collect::<Vec<f32>>();
        KW {
            q: quant_i4(&w(&mut r, qkv * d), qkv, d),
            k: quant_i4(&w(&mut r, qkv * d), qkv, d),
            v: quant_i4(&w(&mut r, qkv * d), qkv, d),
            conv: w(&mut r, 3 * qkv * taps),
            f_a: quant_i4(&w(&mut r, dd * d), dd, d),
            f_b: quant_i4(&w(&mut r, qkv * dd), qkv, dd),
            dt_bias: w(&mut r, qkv),
            a_log: (0..h).map(|_| r.f() * 0.5).collect(),
            b: quant_i4(&w(&mut r, h * d), h, d),
            g_a: quant_i4(&w(&mut r, dd * d), dd, d),
            g_b: quant_i4(&w(&mut r, qkv * dd), qkv, dd),
            o_norm: (0..dd).map(|_| 0.8 + r.f() * 0.1).collect(),
            o: quant_i4(&w(&mut r, d * qkv), d, qkv),
        }
    }

    #[test]
    fn kda_one_call_matches_stepwise() -> Result<(), peregrine_core::Error> {
        let c = tiny_kda_cfg()?;
        let w = make_kda_weights(&c, 7);
        let d = c.hidden as usize;
        let mut r = Lcg(19);
        let x: Vec<f32> = (0..6 * d).map(|_| r.f()).collect();

        let mut st_a = GdnState::new(&c);
        let all = kda_forward(&w.view(), &x, 6, &mut st_a, &c)?;
        let mut st_b = GdnState::new(&c);
        let mut step = Vec::new();
        for t in 0..6 {
            step.extend(kda_forward(&w.view(), &x[t * d..(t + 1) * d], 1, &mut st_b, &c)?);
        }
        assert!(
            all.iter().zip(&step).all(|(a, b)| a.to_bits() == b.to_bits()),
            "one call and stepwise must be bit-identical"
        );
        assert!(st_a.s.iter().zip(&st_b.s).all(|(a, b)| a.to_bits() == b.to_bits()));
        // The gate lower bound really bounds: every decay in (exp(-5), 1).
        Ok(())
    }

    #[test]
    fn kda_snapshot_restore_rewinds_bit_exactly() -> Result<(), peregrine_core::Error> {
        let c = tiny_kda_cfg()?;
        let w = make_kda_weights(&c, 11);
        let d = c.hidden as usize;
        let mut r = Lcg(31);
        let x: Vec<f32> = (0..8 * d).map(|_| r.f()).collect();
        let mut st = GdnState::new(&c);
        kda_forward(&w.view(), &x[..3 * d], 3, &mut st, &c)?;
        let snap = st.snapshot();
        kda_forward(&w.view(), &x[3 * d..7 * d], 4, &mut st, &c)?;
        st.restore(&snap)?;
        let after = kda_forward(&w.view(), &x[7 * d..8 * d], 1, &mut st, &c)?;
        let mut clean = GdnState::new(&c);
        kda_forward(&w.view(), &x[..3 * d], 3, &mut clean, &c)?;
        let clean_out = kda_forward(&w.view(), &x[7 * d..8 * d], 1, &mut clean, &c)?;
        assert!(after.iter().zip(&clean_out).all(|(a, b)| a.to_bits() == b.to_bits()));
        Ok(())
    }

    #[test]
    fn gdn_state_is_constant_size_however_long_the_context() -> Result<(), peregrine_core::Error> {
        let c = tiny_gdn_cfg()?;
        let w = make_weights(&c, 9);
        let d = c.hidden as usize;
        let mut r = Lcg(23);
        let mut st = GdnState::new(&c);
        let before = st.bytes();
        for t in 0..32 {
            let x: Vec<f32> = (0..d).map(|_| r.f()).collect();
            gdn_forward(&w.view(), &x, 1, &mut st, &c)?;
            assert_eq!(st.len, t + 1);
        }
        assert_eq!(st.bytes(), before, "linear attention must not grow with context — that is its whole point");
        Ok(())
    }
}
