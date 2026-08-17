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

        // Exactly the two gates, exactly as a nearly-full card would place them.
        let a_up = w.in_a.upload_to_device(0, 1 << 20)?;
        let b_up = w.in_b.upload_to_device(0, 1 << 20)?;
        if !(a_up && b_up) {
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
