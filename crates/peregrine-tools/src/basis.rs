//! Cross-expert factorization (`W_e = B + Δ_e`), measured as **rate–distortion
//! on activations** rather than on weight reconstruction.
//!
//! The idea: experts in one layer are trained alike, so a shared basis `B` may
//! carry most of what they have in common. Put `B` resident and stream only the
//! per-expert residual `Δ_e`. Unlike quantization — which makes *one* expert
//! smaller — this makes a *group* of experts smaller *together*, which is the
//! only lever left once 600 routed experts × ~18.9 MB = 11.3 GB/token is the
//! binding constraint.
//!
//! # Why this module measures what it measures
//!
//! The obvious experiment is to fit `B`, look at `‖W_e − (B + Δ_e)‖_F`, and
//! declare victory when it is small. **That experiment cannot fail**, and its
//! success means nothing:
//!
//! - Frobenius error is minimized *by construction* when `B` is the group mean
//!   and `Δ_e` is stored exactly. The number it reports is about the fit, not
//!   about the container.
//! - A basis can lower reconstruction error while making `Δ_e` **high-entropy
//!   and hostile to the int4/block quantization the residual then has to
//!   survive**. The fit improves and the bytes do not move. Weight-space error
//!   is structurally blind to that, because it never quantizes anything.
//!
//! So the quantity here is the **distortion the engine would actually see, at
//! the bytes it would actually move**:
//!
//! ```text
//!   rate       = residual bytes streamed per routed expert
//!                (+ the basis, charged once, because it is resident)
//!   distortion = Σ_rj ( c_j · (Δ_e − dequant(quant(Δ_e)))[r,j] )²
//! ```
//!
//! with `c_j` the calibrated per-channel activation magnitude from
//! [`crate::requant::CalibWeights`]. Weighting the error by `c` is what makes
//! this *activation*-space rather than weight-space: an error in a channel the
//! model never excites is free, and one in a hot channel is not. It is a
//! **diagonal** approximation of `E_x‖(W − Ŵ)x‖²` — it treats channels as
//! independent, which they are not — and that limitation is reported rather
//! than hidden ([`Report::caveats`]).
//!
//! Note what the distortion expression does *not* contain: the basis. `B` is
//! resident and stored exactly, so `W_e − (B + Δ̂_e) = Δ_e − Δ̂_e`. **All of the
//! error is the residual's quantization error**, which is precisely why the
//! comparison below is the whole experiment.
//!
//! # The comparison that decides it
//!
//! Against every basis arm the tool runs a **baseline**: quantize `W_e`
//! directly, at the same target precision, with no basis at all. Then:
//!
//! - basis wins if `Δ_e` quantizes *better* than `W_e` does — i.e. subtracting
//!   `B` narrowed the dynamic range the quantizer has to cover.
//! - basis loses if the two are comparable, because it has spent resident
//!   capacity on `B` and bought nothing. **This is the reported outcome, not a
//!   failed run.**
//!
//! # The control
//!
//! A basis fit to a *learned* grouping is compared against one fit to a
//! **shuffled** grouping of the same expert set, at equal rank and equal
//! residual precision ([`Grouping::Shuffled`]). If the learned grouping does
//! not beat random, the basis is capturing **layer-wide structure** — something
//! every expert in the layer shares, obtainable with one per-layer mean and no
//! grouping search — rather than cross-expert redundancy. The saving would be
//! real and the *explanation* would be wrong, which is the same error as
//! reporting a warm-cache hit rate as a routing statistic.
//!
//! Prior art in this repo sets the bar for how that gets reported:
//! `peregrine-skipbound` prints the gate-only baseline beside its own number,
//! because a raw fraction alone reads as success. Both arms print here too.

use std::collections::BTreeMap;

use peregrine_core::config::Cfg;
use peregrine_core::pack::QtView;
use peregrine_core::qt::QtInfo;
use peregrine_core::safetensors::SafeTensors;
use peregrine_core::Error;

use crate::requant::{CalibWeights, Target};

/// The three routed-expert projections, and the input width each is indexed by.
///
/// `down_proj` takes the *intermediate* width, not `hidden`, so the calibration
/// sidecar — which is per hidden channel at the layer's MoE input — has nothing
/// to say about its columns. That is a real gap and it is reported, not papered
/// over with a flat weighting that would look like a measurement.
pub const PROJECTIONS: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];

/// Which experts share a basis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grouping {
    /// Cluster by weighted cosine similarity — experts that are already alike
    /// share a basis. Deterministic (farthest-first seeding, no RNG).
    Learned,
    /// Random partition into the same number of same-sized groups. **The
    /// control.** If this matches [`Grouping::Learned`], the grouping search is
    /// not what produced the win.
    Shuffled(u64),
}

impl Grouping {
    pub fn label(&self) -> String {
        match self {
            Grouping::Learned => "learned".into(),
            Grouping::Shuffled(seed) => format!("shuffled/{seed:x}"),
        }
    }
}

/// What to fit and at what precision.
#[derive(Debug, Clone)]
pub struct FitConfig {
    /// Basis vectors *beyond* the group mean. `0` is the plain `W_e = B + Δ_e`
    /// form; higher ranks are the `W_e = Σ_k a_ek B_k` form.
    pub rank: usize,
    /// How many groups to split each layer's experts into. `1` is one basis per
    /// layer. More groups mean a tighter fit and more resident bytes — the
    /// trade this tool exists to price.
    pub groups: usize,
    /// Precision the **residual** is stored at.
    pub residual: Target,
    /// Precision the **baseline** stores the whole weight at, with no basis.
    /// Defaults to mirroring [`FitConfig::residual`] so both arms stream the
    /// same bytes and the only variable is whether the basis helped.
    pub baseline: Target,
    pub grouping: Grouping,
}

impl Default for FitConfig {
    fn default() -> FitConfig {
        // Both arms default to int2-g64, one rung *below* the int4 these
        // containers ship at, and to the SAME precision as each other. That is
        // deliberate on both counts. Defaulting to the container's own
        // precision would make the baseline lossless by construction and every
        // comparison degenerate; defaulting the two arms to different
        // precisions would confound "does the basis help" with "is int2 worse
        // than int4", which is already a closed question here.
        FitConfig {
            rank: 0,
            groups: 1,
            residual: Target::Int2G64,
            baseline: Target::Int2G64,
            grouping: Grouping::Learned,
        }
    }
}

/// One (layer, projection) measured under one [`FitConfig`].
#[derive(Debug, Clone, PartialEq)]
pub struct Arm {
    pub layer: usize,
    pub projection: String,
    /// Σ of weighted squared error over the group's experts, residual path.
    pub distortion: f64,
    /// The same quantity with no basis: `W_e` quantized directly at
    /// [`FitConfig::baseline`]. The number the basis has to beat.
    pub distortion_baseline: f64,
    /// Σ of weighted squared *signal* — the denominator that turns the two
    /// above into relative errors comparable across layers.
    pub signal: f64,
    /// Bytes streamed per routed expert on the residual path.
    pub residual_bytes: usize,
    /// Bytes streamed per routed expert on the baseline path.
    pub baseline_bytes: usize,
    /// Resident bytes for every basis of this (layer, projection), charged once.
    pub basis_bytes: usize,
    /// Bytes one expert of this projection occupies in the container **as
    /// shipped**. Read amplification is measured against this, because the
    /// question is "fewer bytes than what I run today" — the baseline arm is a
    /// second experiment, not the status quo.
    pub container_bytes: usize,
    /// Experts that contributed.
    pub experts: usize,
    /// Whether the activation weighting was available for this projection.
    /// `false` for `down_proj`, whose input width the sidecar does not cover.
    pub activation_weighted: bool,
}

impl Arm {
    /// Relative activation-space distortion of the residual path.
    pub fn relative(&self) -> f64 {
        if self.signal > 0.0 {
            self.distortion / self.signal
        } else {
            0.0
        }
    }

    /// The same, for the no-basis baseline.
    pub fn relative_baseline(&self) -> f64 {
        if self.signal > 0.0 {
            self.distortion_baseline / self.signal
        } else {
            0.0
        }
    }
}

/// Everything one run measured.
#[derive(Debug, Clone, Default)]
pub struct Report {
    pub arms: Vec<Arm>,
    /// Routed experts per token, from `config.json` — the multiplier that turns
    /// per-expert bytes into bytes/token.
    pub topk: usize,
    pub sparse_layers: usize,
    /// Projections that had no calibration vector, so their error is unweighted.
    pub unweighted: Vec<String>,
}

/// Summed measurement over every arm in a report.
#[derive(Debug, Default, Clone, Copy)]
struct Totals {
    distortion: f64,
    distortion_baseline: f64,
    signal: f64,
    residual_bytes: usize,
    baseline_bytes: usize,
    basis_bytes: usize,
    container_bytes: usize,
}

/// Per-token traffic on each path, in bytes.
#[derive(Debug, Default, Clone, Copy)]
pub struct TokenBytes {
    /// Streamed per token with the basis in place.
    pub residual: f64,
    /// Streamed per token by the no-basis arm at the same precision.
    pub baseline: f64,
    /// Streamed per token by the container as it ships today.
    pub container: f64,
    /// One-off resident cost of every basis measured — the charge that applies
    /// only while the basis actually stays in memory.
    pub resident_basis: f64,
    /// The same basis charged **per token** instead: what it costs if it is
    /// evicted and re-read every forward. Each layer's basis is read once per
    /// token, not once per routed expert, so this is far smaller than the
    /// residual traffic — but it is not zero, and at a 0.6 % warm-cache hit
    /// rate against a ~180 GB working set it is the likelier of the two
    /// charges. Reported alongside rather than instead: which one applies is a
    /// property of the deployment, not of the fit.
    pub basis_per_token: f64,
}

impl Report {
    fn totals(&self) -> Totals {
        let mut t = Totals::default();
        for a in &self.arms {
            t.distortion += a.distortion;
            t.distortion_baseline += a.distortion_baseline;
            t.signal += a.signal;
            t.residual_bytes += a.residual_bytes;
            t.baseline_bytes += a.baseline_bytes;
            t.basis_bytes += a.basis_bytes;
            t.container_bytes += a.container_bytes;
        }
        t
    }

    /// Bytes moved per token on the residual path, on the baseline path, and
    /// as the container ships today — plus the basis's resident cost.
    ///
    /// The per-arm byte figures are per *expert*; one measured arm is one
    /// (layer, projection), so the mean over arms is a per-expert-per-projection
    /// cost and a token pays `topk` experts × 3 projections × sparse layers.
    /// Extrapolating measured layers to the whole model is deliberate and is
    /// stated in the caveats — a partial sweep must not silently report a
    /// partial token.
    pub fn bytes_per_token(&self) -> TokenBytes {
        let t = self.totals();
        let n = self.arms.len().max(1) as f64;
        let per_expert = |total: usize| {
            total as f64 / n * PROJECTIONS.len() as f64 * (self.topk * self.sparse_layers) as f64
        };
        // The basis is per (layer, projection), not per expert: one read per
        // layer per token serves every expert routed in that layer. So it
        // scales with `sparse_layers`, where the residual scales with
        // `topk * sparse_layers`.
        let basis_per_arm = t.basis_bytes as f64 / n;
        TokenBytes {
            residual: per_expert(t.residual_bytes),
            baseline: per_expert(t.baseline_bytes),
            container: per_expert(t.container_bytes),
            resident_basis: t.basis_bytes as f64,
            basis_per_token: basis_per_arm * PROJECTIONS.len() as f64 * self.sparse_layers as f64,
        }
    }

    /// Read amplification against **the container as shipped**: `< 1` means
    /// fewer bytes move per token than today, which is the only direction that
    /// changes anything for an operator.
    ///
    /// Assumes the basis stays resident. [`Self::read_amplification_evicted`]
    /// is the same figure when it does not.
    pub fn read_amplification(&self) -> f64 {
        let b = self.bytes_per_token();
        if b.container > 0.0 {
            b.residual / b.container
        } else {
            0.0
        }
    }

    /// Read amplification when the basis is **evicted between tokens** and has
    /// to be re-read, so its bytes join the streamed traffic instead of sitting
    /// in the resident column.
    ///
    /// This is not a pessimistic variant to be quoted only when convenient: the
    /// warm cache hits 0.6 % on sustained decode because a 10 GB cache cannot
    /// hold a ~180 GB working set, and a basis competing for that capacity is
    /// exactly what gets evicted. Reporting only the resident charge would be
    /// assuming the favourable half of the one question this tool cannot
    /// answer offline.
    pub fn read_amplification_evicted(&self) -> f64 {
        let b = self.bytes_per_token();
        if b.container > 0.0 {
            (b.residual + b.basis_per_token) / b.container
        } else {
            0.0
        }
    }

    /// Limitations that travel with every number above.
    pub fn caveats(&self) -> Vec<String> {
        let mut v = vec![
            "Distortion is a DIAGONAL approximation of E_x||(W-W')x||^2: channels are weighted \
             by calibrated mean|x| and treated as independent. Cross-channel structure is \
             invisible to it."
                .into(),
            "Distortion is not flip rate. A relative error that looks small can still move \
             top-1 predictions; gate the winning arm with `peregrine flip-rate` before believing it."
                .into(),
        ];
        if !self.unweighted.is_empty() {
            v.push(format!(
                "{} projection(s) had no calibration vector and are scored UNWEIGHTED \
                 (plain Frobenius): {}. The sidecar covers hidden channels at the MoE input, \
                 and down_proj is indexed by the intermediate width.",
                self.unweighted.len(),
                self.unweighted.join(", ")
            ));
        }
        v
    }

    /// The verdict, in the blunt form the negative case needs.
    pub fn verdict(&self) -> String {
        let t = self.totals();
        let b = self.bytes_per_token();
        let rel = |x: f64| if t.signal > 0.0 { x / t.signal } else { 0.0 };
        let (r, rb) = (rel(t.distortion), rel(t.distortion_baseline));
        let mut out = format!(
            "{} arm(s) over {} sparse layer(s), topk={}\n\
             {:>22} {:>16} {:>16}\n\
             {:>22} {:>15.4e} {:>15.4e}\n\
             {:>22} {:>15.4} {:>15.4}\n",
            self.arms.len(),
            self.sparse_layers,
            self.topk,
            "",
            "basis + residual",
            "no basis",
            "rel. distortion",
            r,
            rb,
            "MB/token",
            b.residual / 1e6,
            b.baseline / 1e6,
        );
        out.push_str(&format!(
            "{:>22} {:>15.4}\n{:>22} {:>15.4}\n{:>22} {:>15.4}\n",
            "container MB/token",
            b.container / 1e6,
            "resident basis (MB)",
            b.resident_basis / 1e6,
            "basis MB/token if evicted",
            b.basis_per_token / 1e6,
        ));
        // Both charges, always, and never one without the other. Which applies
        // is a property of the deployment — whether the basis survives in a
        // cache that hits 0.6 % on sustained decode — and quoting only the
        // resident figure would be assuming the favourable answer to the one
        // question this offline tool cannot settle.
        out.push_str(&format!(
            "{:>22} {:>15.3} {:>15.3}\n{:>22} {:>15} {:>15}\n",
            "read amplification",
            self.read_amplification(),
            self.read_amplification_evicted(),
            "",
            "basis resident",
            "basis evicted",
        ));

        if t.signal <= 0.0 {
            out.push_str(
                "\nVERDICT: nothing measured. Every expert had zero weighted energy, which means \n\
                 the calibration vector is all zeros or the container is empty — not that the \n\
                 basis is perfect.",
            );
            return out;
        }
        // The setup error that would otherwise produce a confident wrong answer.
        // Requantizing to the precision the container already uses is lossless
        // by construction, so the baseline column is a structural zero rather
        // than a measurement, and every ratio against it is infinite. This has
        // to be refused, not rendered: a verdict computed from it would read as
        // "the basis is infinitely worse" when the truth is "you compared the
        // container against itself".
        if t.distortion_baseline <= 0.0 && t.distortion > 0.0 {
            out.push_str(
                "\nVERDICT: DEGENERATE BASELINE — no comparison was made. The baseline precision \n\
                 reproduces the container exactly (you asked to requantize int4 to int4, which is \n\
                 a no-op), so its distortion is a structural zero and every ratio against it is \n\
                 meaningless. Set --residual and --baseline BELOW the container's precision, so \n\
                 both arms actually lose something and the question becomes which loses less.",
            );
            return out;
        }

        // Before any arm-vs-arm comparison: does either arm beat the container
        // it would replace? The two arms can be identical in bytes and the
        // better one can still stream MORE than what ships today — which
        // happens whenever the chosen format's group size exceeds the row
        // width, since the per-group scales then cost more than the payload
        // saves. Reporting "the basis wins" there would be true and useless.
        if b.container > 0.0 && b.residual < b.container && b.residual + b.basis_per_token >= b.container {
            out.push_str(&format!(
                "\nVERDICT: beats the container ONLY while the basis stays resident ({:.3}x resident, \n\
                 {:.3}x evicted). On this engine that is the unfavourable case: the warm cache hits \n\
                 0.6 % on sustained decode because a 10 GB cache cannot hold a ~180 GB working set, \n\
                 so a basis competing for that capacity is what gets evicted. Shrink the basis \n\
                 (fewer --groups, lower --rank) until it wins on the evicted figure too.",
                self.read_amplification(),
                self.read_amplification_evicted()
            ));
            return out;
        }
        if b.container > 0.0 && b.residual >= b.container {
            out.push_str(&format!(
                "\nVERDICT: streams MORE than the container ships ({:.3}x). Whatever the two arms \n\
                 do relative to each other, this precision is a loss against doing nothing. The \n\
                 usual cause is a grouped format whose group is wider than the row: the per-group \n\
                 scales then cost more than the narrower payload saves. Pick a format whose group \n\
                 divides the input width, or a lower --residual.",
                self.read_amplification()
            ));
            return out;
        }

        // Three regimes, and only the third is the rate-distortion trade.
        //
        // Equal streamed bytes is the *default* and the most informative: both
        // arms move the same bytes, so the basis is judged purely on whether
        // the residual quantizes better than the weight, against a resident
        // cost that is stated rather than folded into the comparison.
        let saving = 1.0 - b.residual / b.container.max(1.0);
        if b.residual > b.baseline {
            out.push_str(
                "\nVERDICT: no rate saving. The residual costs MORE bytes per token than the \n\
                 no-basis arm, so the basis is pure overhead whatever its fit looks like. Lower \n\
                 --residual before reading the distortion column.",
            );
            return out;
        }
        let ratio = if rb > 0.0 { r / rb } else { f64::INFINITY };
        if (b.residual - b.baseline).abs() <= f64::EPSILON * b.baseline.max(1.0) {
            // Equal bytes: the whole question is the distortion ratio.
            out.push_str(&format!(
                "\nEqual streamed bytes on both arms ({:.2} MB/token, {:.1}% of the container). \n\
                 The basis is therefore judged only on whether the residual quantizes better than \n\
                 the weight, at a resident cost of {:.1} MB.\n",
                b.residual / 1e6,
                (1.0 - saving) * 100.0,
                b.resident_basis / 1e6,
            ));
            if ratio < 0.75 {
                out.push_str(&format!(
                    "\nVERDICT: worth pursuing. The residual carries {:.0}% of the baseline's \n\
                     distortion at identical streamed bytes — subtracting the basis genuinely \n\
                     narrowed the range the quantizer has to cover. Next step is NOT the loader: \n\
                     run --control at this rank, then gate on flip rate.",
                    ratio * 100.0
                ));
            } else if ratio <= 1.0 {
                out.push_str(&format!(
                    "\nVERDICT: marginal. The residual is only {:.0}% of the baseline's distortion \n\
                     at equal bytes, and that margin has to pay for {:.1} MB of resident capacity \n\
                     the cache would otherwise hold experts in. Price it against the warm cache \n\
                     before pursuing.",
                    ratio * 100.0,
                    b.resident_basis / 1e6
                ));
            } else {
                out.push_str(&format!(
                    "\nVERDICT: the basis moved the entropy around. At identical streamed bytes \n\
                     the residual is {:.2}x the baseline's distortion — it is HARDER to quantize \n\
                     than the weight it replaced, which is exactly the failure mode weight-space \n\
                     reconstruction error cannot see. The resident basis buys nothing here.",
                    ratio
                ));
            }
            return out;
        }
        // Fewer bytes than the baseline arm: a genuine rate-distortion trade.
        let rate_gain = 1.0 - b.residual / b.baseline.max(1.0);
        let distortion_cost = ratio - 1.0;
        if distortion_cost <= 0.0 {
            out.push_str(&format!(
                "\nVERDICT: worth pursuing. {:.1}% fewer bytes than the no-basis arm AND no \n\
                 distortion penalty. Run --control at this rank before believing the explanation, \n\
                 then gate on flip rate.",
                rate_gain * 100.0
            ));
        } else if distortion_cost < rate_gain {
            out.push_str(&format!(
                "\nVERDICT: a real trade rather than a win. Bytes fall {:.1}% while relative \n\
                 distortion rises {:.1}%. Price it against flip rate, not against itself.",
                rate_gain * 100.0,
                distortion_cost * 100.0
            ));
        } else {
            out.push_str(&format!(
                "\nVERDICT: the basis moved the entropy around. Bytes fall {:.1}% but relative \n\
                 distortion rises {:.1}% — the residual is harder to quantize than the weight it \n\
                 replaced, which is the failure mode weight-space reconstruction error cannot see.",
                rate_gain * 100.0,
                distortion_cost * 100.0
            ));
        }
        out
    }
}

/// Compare a learned grouping against shuffled controls at equal rank.
///
/// **Several** shuffled draws, not one. The learned grouping is fit and scored
/// on the same experts, so it carries a selection advantage that exists even
/// when there is no cross-expert structure at all: clustering by similarity
/// optimizes exactly the quantity the distortion column then measures. A single
/// random draw is too weak a null to see through that — it can land anywhere in
/// the spread — so the learned arm has to beat the **best** shuffled draw, and
/// the spread itself is reported so the margin can be read against it.
#[derive(Debug, Clone)]
pub struct Control {
    pub learned: Report,
    pub shuffled: Vec<Report>,
}

impl Control {
    fn rel(r: &Report) -> f64 {
        let t = r.totals();
        if t.signal > 0.0 {
            t.distortion / t.signal
        } else {
            0.0
        }
    }

    /// `(learned, best shuffled, worst shuffled)` relative distortion — lower
    /// is better, so the learned grouping earns its search only by coming in
    /// below **best**.
    pub fn relatives(&self) -> (f64, f64, f64) {
        let mut best = f64::INFINITY;
        let mut worst: f64 = 0.0;
        for r in &self.shuffled {
            let v = Self::rel(r);
            best = best.min(v);
            worst = worst.max(v);
        }
        if !best.is_finite() {
            best = 0.0;
        }
        (Self::rel(&self.learned), best, worst)
    }

    /// The control's verdict. Deliberately harsh: a learned grouping that only
    /// matches random is a *negative* result about the explanation, even when
    /// the byte saving it reports is real.
    pub fn verdict(&self) -> String {
        let (l, best, worst) = self.relatives();
        let mut out = format!(
            "grouping control ({} shuffled draw(s), equal rank, equal residual precision)\n\
             {:>22} {:>15.4e}\n{:>22} {:>15.4e}\n{:>22} {:>15.4e}\n",
            self.shuffled.len(),
            "learned",
            l,
            "best shuffled",
            best,
            "worst shuffled",
            worst
        );
        let margin = if best > 0.0 { (best - l) / best } else { 0.0 };
        let spread = if best > 0.0 { (worst - best) / best } else { 0.0 };
        out.push_str(&format!(
            "{:>22} {:>14.2}%\n{:>22} {:>14.2}%\n",
            "learned beats best by", margin * 100.0, "shuffled spread", spread * 100.0
        ));

        // The learned arm is fit and scored on the same experts, so a positive
        // margin is the null expectation, not evidence. The floor has to clear
        // both that bias and the spread of the shuffled draws themselves —
        // otherwise "learned wins" is indistinguishable from "this draw was
        // unlucky". `structureless_experts_still_favour_the_learned_grouping`
        // pins that the bias is real and non-zero on random weights.
        let floor = 0.15f64.max(spread);
        if margin < floor {
            out.push_str(&format!(
                "\nVERDICT: the grouping search is not what produced this. Learned beats the best \n\
                 random partition by {:.1}%, under the {:.1}% floor this comparison needs — the \n\
                 learned arm is fit and scored on the same experts, so a positive margin is the \n\
                 NULL expectation rather than evidence, and the shuffled draws themselves spread \n\
                 {:.1}%. Read this as: the basis is capturing LAYER-WIDE structure that one \n\
                 per-layer mean would also capture, not cross-expert redundancy. Any byte saving \n\
                 is real; the explanation is wrong and the cheap implementation is the right one.",
                margin * 100.0,
                floor * 100.0,
                spread * 100.0
            ));
        } else {
            out.push_str(&format!(
                "\nVERDICT: the grouping is load-bearing — learned beats the BEST of {} random \n\
                 partitions by {:.1}%, clear of both the fit-and-score selection bias and the \n\
                 {:.1}% spread of the draws. Confirm on a second layer range before trusting it; \n\
                 this margin is a property of the checkpoint, not of the method.",
                self.shuffled.len(),
                margin * 100.0,
                spread * 100.0
            ));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Linear algebra. Small and self-contained: the matrices that get factorized
// here are `E x E` over one group's experts (tens, not thousands), so a Jacobi
// sweep is both simpler and more obviously correct than pulling in a solver.
// ---------------------------------------------------------------------------

/// Symmetric eigendecomposition by cyclic Jacobi rotation.
///
/// Returns `(eigenvalues, eigenvectors)` with eigenvectors stored column-wise
/// in an `n x n` row-major buffer, sorted by descending eigenvalue. Operates on
/// the Gram matrix of a group's experts, so `n` is the group size.
fn jacobi_eigh(a: &mut [f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0f64; n * n];
    for k in 0..n {
        if let Some(s) = v.get_mut(k * n + k) {
            *s = 1.0;
        }
    }
    let at = |a: &[f64], r: usize, c: usize| a.get(r * n + c).copied().unwrap_or(0.0);
    // 64 sweeps is far past convergence for the sizes here; the off-diagonal
    // test exits earlier in practice.
    for _ in 0..64 {
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += at(a, p, q) * at(a, p, q);
            }
        }
        if off <= 1e-24 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = at(a, p, q);
                if apq.abs() < 1e-18 {
                    continue;
                }
                let (app, aqq) = (at(a, p, p), at(a, q, q));
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (at(a, k, p), at(a, k, q));
                    if let Some(x) = a.get_mut(k * n + p) {
                        *x = c * akp - s * akq;
                    }
                    if let Some(x) = a.get_mut(k * n + q) {
                        *x = s * akp + c * akq;
                    }
                }
                for k in 0..n {
                    let (apk, aqk) = (at(a, p, k), at(a, q, k));
                    if let Some(x) = a.get_mut(p * n + k) {
                        *x = c * apk - s * aqk;
                    }
                    if let Some(x) = a.get_mut(q * n + k) {
                        *x = s * apk + c * aqk;
                    }
                }
                for k in 0..n {
                    let (vkp, vkq) = (at(&v, k, p), at(&v, k, q));
                    if let Some(x) = v.get_mut(k * n + p) {
                        *x = c * vkp - s * vkq;
                    }
                    if let Some(x) = v.get_mut(k * n + q) {
                        *x = s * vkp + c * vkq;
                    }
                }
            }
        }
    }
    let mut idx: Vec<usize> = (0..n).collect();
    let eig: Vec<f64> = (0..n).map(|k| at(a, k, k)).collect();
    idx.sort_by(|&x, &y| {
        eig.get(y)
            .unwrap_or(&0.0)
            .partial_cmp(eig.get(x).unwrap_or(&0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let vals: Vec<f64> = idx.iter().map(|&k| eig.get(k).copied().unwrap_or(0.0)).collect();
    let mut vecs = vec![0f64; n * n];
    for (newc, &oldc) in idx.iter().enumerate() {
        for r in 0..n {
            if let Some(x) = vecs.get_mut(r * n + newc) {
                *x = at(&v, r, oldc);
            }
        }
    }
    (vals, vecs)
}

/// Deterministic xorshift64*, for the shuffled control only.
///
/// A control that varied run to run would make "learned did not beat random"
/// unfalsifiable — you could always reroll. The seed is part of the arm label.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

// ---------------------------------------------------------------------------
// The fit
// ---------------------------------------------------------------------------

/// A layer's experts for one projection, dequantized and activation-weighted.
struct Stack {
    /// `experts[e]` is a dense `[o, i]` weight in **weight space**.
    experts: Vec<Vec<f32>>,
    /// Bytes one expert of this projection actually occupies in the container
    /// as shipped. Read amplification is measured against **this**, not against
    /// the baseline arm: the question an operator has is "fewer bytes than what
    /// I run today", and the baseline arm is a second experiment, not the
    /// status quo.
    container_bytes: usize,
    /// Per-column activation weight, or all-ones when the sidecar does not
    /// cover this projection's input width.
    cw: Vec<f32>,
    weighted: bool,
    o: usize,
    i: usize,
}

impl Stack {
    /// Weighted inner product of two experts, both mean-centred by `mean`.
    fn dot_centred(&self, a: &[f32], b: &[f32], mean: &[f32]) -> f64 {
        let mut acc = 0.0f64;
        for r in 0..self.o {
            for j in 0..self.i {
                let k = r * self.i + j;
                let (Some(&av), Some(&bv), Some(&mv), Some(&c)) =
                    (a.get(k), b.get(k), mean.get(k), self.cw.get(j))
                else {
                    continue;
                };
                let c = f64::from(c);
                acc += c * c * f64::from(av - mv) * f64::from(bv - mv);
            }
        }
        acc
    }

    /// Weighted squared norm of `x` — the denominator relative errors use.
    fn energy(&self, x: &[f32]) -> f64 {
        let mut acc = 0.0f64;
        for r in 0..self.o {
            for j in 0..self.i {
                let (Some(&v), Some(&c)) = (x.get(r * self.i + j), self.cw.get(j)) else {
                    continue;
                };
                let c = f64::from(c);
                acc += c * c * f64::from(v) * f64::from(v);
            }
        }
        acc
    }

    /// Weighted squared error between `x` and `y`.
    fn sq_err(&self, x: &[f32], y: &[f32]) -> f64 {
        let mut acc = 0.0f64;
        for r in 0..self.o {
            for j in 0..self.i {
                let k = r * self.i + j;
                let (Some(&xv), Some(&yv), Some(&c)) = (x.get(k), y.get(k), self.cw.get(j)) else {
                    continue;
                };
                let c = f64::from(c);
                let d = f64::from(xv - yv);
                acc += c * c * d * d;
            }
        }
        acc
    }
}

/// Read every expert of one (layer, projection) into a dense stack.
///
/// Memory is `n_experts × o × i × 4` bytes and this is an offline batch tool,
/// so the caller controls the cost with `--layers` and `--experts`. There is no
/// streaming variant because the Gram matrix needs every pair, and a two-pass
/// version would re-dequantize the whole layer per pair.
fn load_stack(
    st: &SafeTensors,
    cfg: &Cfg,
    calib: Option<&CalibWeights>,
    layer: usize,
    proj: &str,
    max_experts: usize,
) -> Result<Option<Stack>, Error> {
    let (hidden, inter) = (cfg.hidden as usize, cfg.moe_inter as usize);
    let (o, i) = if proj == "down_proj" { (hidden, inter) } else { (inter, hidden) };
    let n = (cfg.n_experts as usize).min(max_experts);
    let mut experts = Vec::with_capacity(n);
    let mut container_bytes = 0usize;
    for e in 0..n {
        let name = format!("model.layers.{layer}.mlp.experts.{e}.{proj}.weight");
        if !st.tensors().iter().any(|t| t.name == name) {
            return Ok(None);
        }
        if e == 0 {
            let info = QtInfo::detect(st, &name, o as i64, i as i64);
            let payload = match QtView::row_bytes(info.fmt, i) {
                Some(rb) => o * rb,
                None => o * i * std::mem::size_of::<f32>(),
            };
            container_bytes = payload
                + info.scale_count.max(0) as usize * std::mem::size_of::<f32>();
        }
        experts.push(dequant_dense(st, &name, o, i)?);
    }
    if experts.len() < 2 {
        // One expert cannot have anything in common with anything.
        return Ok(None);
    }
    // `down_proj` is indexed by the intermediate width; the sidecar is per
    // hidden channel at the MoE input. Falling back to ones is *unweighted*,
    // and the caller records that rather than presenting it as calibrated.
    let cw_layer = calib.and_then(|c| c.layers.get(layer)).filter(|v| v.len() == i);
    let (cw, weighted) = match cw_layer {
        Some(v) => (v.clone(), true),
        None => (vec![1.0f32; i], false),
    };
    Ok(Some(Stack { experts, container_bytes, cw, weighted, o, i }))
}

/// Dequantize a container tensor into a dense `[o, i]` buffer.
fn dequant_dense(st: &SafeTensors, name: &str, o: usize, i: usize) -> Result<Vec<f32>, Error> {
    let info = QtInfo::detect(st, name, o as i64, i as i64);
    let Some(row_bytes) = QtView::row_bytes(info.fmt, i) else {
        let mut dense = vec![0f32; o * i];
        st.read_f32(name, &mut dense)?;
        return Ok(dense);
    };
    let mut q = vec![0u8; o * row_bytes];
    st.read_raw(name, &mut q)?;
    let mut scale = vec![0f32; info.scale_count.max(0) as usize];
    st.read_f32(&format!("{name}.qs"), &mut scale)?;
    let view = QtView { fmt: info.fmt, o, i, gs: info.gs as usize, q: &q, scale: &scale };
    let mut dense = vec![0f32; o * i];
    let mut row = vec![0f32; i];
    for r in 0..o {
        view.dequant_row_into(r, &mut row);
        if let Some(dst) = dense.get_mut(r * i..(r + 1) * i) {
            dst.copy_from_slice(&row);
        }
    }
    Ok(dense)
}

/// Partition expert indices into `groups` groups.
///
/// [`Grouping::Learned`] seeds farthest-first on weighted cosine distance and
/// assigns each expert to its nearest seed, then balances so no group is empty
/// — an empty group would silently reduce the effective group count and make
/// the control compare different things.
fn partition(stack: &Stack, mean_all: &[f32], groups: usize, how: Grouping) -> Vec<Vec<usize>> {
    let n = stack.experts.len();
    let g = groups.clamp(1, n);
    if g == 1 {
        return vec![(0..n).collect()];
    }
    let mut out: Vec<Vec<usize>> = vec![Vec::new(); g];
    match how {
        Grouping::Shuffled(seed) => {
            let mut rng = Rng(seed | 1);
            let mut order: Vec<usize> = (0..n).collect();
            for k in (1..n).rev() {
                let j = (rng.next() % (k as u64 + 1)) as usize;
                order.swap(k, j);
            }
            for (slot, &e) in order.iter().enumerate() {
                if let Some(bucket) = out.get_mut(slot % g) {
                    bucket.push(e);
                }
            }
        }
        Grouping::Learned => {
            let norm: Vec<f64> = stack
                .experts
                .iter()
                .map(|w| stack.dot_centred(w, w, mean_all).max(0.0).sqrt())
                .collect();
            let cos = |a: usize, b: usize| -> f64 {
                let (Some(wa), Some(wb)) = (stack.experts.get(a), stack.experts.get(b)) else {
                    return 0.0;
                };
                let d = stack.dot_centred(wa, wb, mean_all);
                let (na, nb) = (norm.get(a).copied().unwrap_or(0.0), norm.get(b).copied().unwrap_or(0.0));
                if na > 0.0 && nb > 0.0 {
                    d / (na * nb)
                } else {
                    0.0
                }
            };
            // Farthest-first seeding: start at 0, then repeatedly take the
            // expert least similar to every seed so far.
            let mut seeds = vec![0usize];
            while seeds.len() < g {
                let mut best = (f64::INFINITY, 0usize);
                for e in 0..n {
                    if seeds.contains(&e) {
                        continue;
                    }
                    let worst = seeds.iter().map(|&s| cos(e, s)).fold(f64::NEG_INFINITY, f64::max);
                    if worst < best.0 {
                        best = (worst, e);
                    }
                }
                seeds.push(best.1);
            }
            for e in 0..n {
                let mut best = (f64::NEG_INFINITY, 0usize);
                for (gi, &s) in seeds.iter().enumerate() {
                    let c = cos(e, s);
                    if c > best.0 {
                        best = (c, gi);
                    }
                }
                if let Some(bucket) = out.get_mut(best.1) {
                    bucket.push(e);
                }
            }
        }
    }
    // An empty group would change the effective rank budget between arms, so
    // the control would no longer be comparing equal ranks.
    let mut spare: Vec<usize> = Vec::new();
    for bucket in out.iter_mut() {
        while bucket.len() > 1 && spare.len() < g {
            if let Some(x) = bucket.pop() {
                spare.push(x);
            } else {
                break;
            }
        }
    }
    for bucket in out.iter_mut() {
        if bucket.is_empty() {
            if let Some(x) = spare.pop() {
                bucket.push(x);
            }
        }
    }
    for x in spare {
        if let Some(b) = out.iter_mut().min_by_key(|b| b.len()) {
            b.push(x);
        }
    }
    out.retain(|b| !b.is_empty());
    out
}

/// Fit one group: mean plus `rank` principal directions, in activation-weighted
/// space, via the Gram trick (the group is tens of experts; the weight is
/// millions of elements, so never form the covariance).
fn fit_group(stack: &Stack, members: &[usize], rank: usize) -> (Vec<f32>, Vec<Vec<f32>>) {
    let (o, i) = (stack.o, stack.i);
    let mut mean = vec![0f32; o * i];
    for &e in members {
        let Some(w) = stack.experts.get(e) else { continue };
        for (m, &v) in mean.iter_mut().zip(w.iter()) {
            *m += v;
        }
    }
    let inv = 1.0 / members.len().max(1) as f32;
    for m in mean.iter_mut() {
        *m *= inv;
    }
    let r = rank.min(members.len().saturating_sub(1));
    if r == 0 {
        return (mean, Vec::new());
    }
    let n = members.len();
    let mut gram = vec![0f64; n * n];
    for (a, &ea) in members.iter().enumerate() {
        for (b, &eb) in members.iter().enumerate().skip(a) {
            let (Some(wa), Some(wb)) = (stack.experts.get(ea), stack.experts.get(eb)) else {
                continue;
            };
            let d = stack.dot_centred(wa, wb, &mean);
            if let Some(x) = gram.get_mut(a * n + b) {
                *x = d;
            }
            if let Some(x) = gram.get_mut(b * n + a) {
                *x = d;
            }
        }
    }
    let (vals, vecs) = jacobi_eigh(&mut gram, n);
    let mut basis: Vec<Vec<f32>> = Vec::with_capacity(r);
    for k in 0..r {
        let lambda = vals.get(k).copied().unwrap_or(0.0);
        if lambda <= 1e-18 {
            break;
        }
        // The eigenvector of the Gram matrix gives the principal direction as a
        // combination of the centred experts; normalize so the direction is
        // unit-norm under the same weighted inner product.
        let mut dir = vec![0f32; o * i];
        for (a, &ea) in members.iter().enumerate() {
            let coef = vecs.get(a * n + k).copied().unwrap_or(0.0) as f32;
            let Some(w) = stack.experts.get(ea) else { continue };
            for (d, (&wv, &mv)) in dir.iter_mut().zip(w.iter().zip(mean.iter())) {
                *d += coef * (wv - mv);
            }
        }
        let scale = 1.0 / lambda.sqrt() as f32;
        for d in dir.iter_mut() {
            *d *= scale;
        }
        basis.push(dir);
    }
    (mean, basis)
}

/// Project `w` onto `mean + span(basis)` under the weighted inner product.
fn reconstruct(stack: &Stack, w: &[f32], mean: &[f32], basis: &[Vec<f32>]) -> Vec<f32> {
    let mut out = mean.to_vec();
    for dir in basis {
        let coef = stack.dot_centred(w, dir, mean) / stack.dot_centred(dir, dir, mean).max(1e-30);
        let coef = coef as f32;
        for (o, &d) in out.iter_mut().zip(dir.iter()) {
            *o += coef * d;
        }
    }
    out
}

/// Measure one (layer, projection) under one config.
fn measure_arm(stack: &Stack, layer: usize, proj: &str, cfg: &FitConfig) -> Arm {
    let (o, i) = (stack.o, stack.i);
    let mut mean_all = vec![0f32; o * i];
    for w in &stack.experts {
        for (m, &v) in mean_all.iter_mut().zip(w.iter()) {
            *m += v;
        }
    }
    let inv = 1.0 / stack.experts.len().max(1) as f32;
    for m in mean_all.iter_mut() {
        *m *= inv;
    }

    let parts = partition(stack, &mean_all, cfg.groups, cfg.grouping);
    let mut arm = Arm {
        layer,
        projection: proj.to_string(),
        distortion: 0.0,
        distortion_baseline: 0.0,
        signal: 0.0,
        residual_bytes: 0,
        baseline_bytes: 0,
        basis_bytes: 0,
        container_bytes: stack.container_bytes,
        experts: stack.experts.len(),
        activation_weighted: stack.weighted,
    };
    // Residual and baseline byte costs are per expert and identical across
    // experts, so they are computed once from the shapes.
    arm.residual_bytes =
        cfg.residual.payload_bytes(o, i) + cfg.residual.scale_count(o, i) * std::mem::size_of::<f32>();
    arm.baseline_bytes =
        cfg.baseline.payload_bytes(o, i) + cfg.baseline.scale_count(o, i) * std::mem::size_of::<f32>();

    for members in &parts {
        let (mean, basis) = fit_group(stack, members, cfg.rank);
        // The basis is resident and stored exactly: f32, charged once. Storing
        // it quantized would be a second experiment, and would put quantization
        // error into the term this measurement deliberately isolates.
        arm.basis_bytes += (1 + basis.len()) * o * i * std::mem::size_of::<f32>();
        for &e in members {
            let Some(w) = stack.experts.get(e) else { continue };
            let recon = reconstruct(stack, w, &mean, &basis);
            let delta: Vec<f32> = w.iter().zip(recon.iter()).map(|(&a, &b)| a - b).collect();
            let delta_hat = cfg.residual.roundtrip(&delta, o, i);
            // W - (recon + delta_hat) == delta - delta_hat: the basis is exact,
            // so all of the error is the residual's quantization error.
            arm.distortion += stack.sq_err(&delta, &delta_hat);
            let w_hat = cfg.baseline.roundtrip(w, o, i);
            arm.distortion_baseline += stack.sq_err(w, &w_hat);
            arm.signal += stack.energy(w);
        }
    }
    arm
}

/// How much of the container to walk.
#[derive(Debug, Clone)]
pub struct Scope {
    /// Sparse layers to measure, from the first sparse layer. `0` = all.
    pub layers: usize,
    /// Experts per layer to include. `0` = all.
    pub experts: usize,
}

impl Default for Scope {
    fn default() -> Scope {
        Scope { layers: 2, experts: 0 }
    }
}

/// Run the sweep over a container.
pub fn measure(
    indir: &std::path::Path,
    calib: Option<&CalibWeights>,
    fit: &FitConfig,
    scope: &Scope,
) -> Result<Report, Error> {
    let cfg = Cfg::load(indir)?;
    let st = SafeTensors::open(indir)?;
    let first = cfg.first_dense as usize;
    let last = cfg.n_layers as usize;
    let want = if scope.layers == 0 { last - first } else { scope.layers.min(last - first) };
    let max_e = if scope.experts == 0 { usize::MAX } else { scope.experts };
    let mut rep = Report {
        topk: cfg.topk.max(0) as usize,
        sparse_layers: want,
        ..Report::default()
    };
    let mut seen_unweighted: BTreeMap<String, ()> = BTreeMap::new();
    for layer in first..(first + want) {
        for proj in PROJECTIONS {
            let Some(stack) = load_stack(&st, &cfg, calib, layer, proj, max_e)? else {
                continue;
            };
            if !stack.weighted {
                seen_unweighted.insert(proj.to_string(), ());
            }
            rep.arms.push(measure_arm(&stack, layer, proj, fit));
        }
    }
    if rep.arms.is_empty() {
        return Err(Error::Format(format!(
            "{}: no routed experts found to factorize",
            indir.display()
        )));
    }
    rep.unweighted = seen_unweighted.into_keys().collect();
    Ok(rep)
}

/// Number of shuffled draws the control runs. One draw cannot separate "the
/// learned grouping found something" from "that draw was unlucky", and the
/// learned arm's fit-and-score bias means the comparison needs a spread to be
/// read against.
pub const CONTROL_DRAWS: usize = 4;

/// Run the learned arm and its shuffled controls at equal rank and precision.
///
/// Every arm is produced by the same code path with only [`Grouping`]
/// differing, which is the point: a control implemented separately from the
/// thing it controls would be measuring two implementations, not two groupings.
pub fn measure_control(
    indir: &std::path::Path,
    calib: Option<&CalibWeights>,
    fit: &FitConfig,
    scope: &Scope,
    seed: u64,
) -> Result<Control, Error> {
    let mut learned = fit.clone();
    learned.grouping = Grouping::Learned;
    let learned = measure(indir, calib, &learned, scope)?;
    let mut draws = Vec::with_capacity(CONTROL_DRAWS);
    let mut rng = Rng(seed | 1);
    for _ in 0..CONTROL_DRAWS {
        let mut arm = fit.clone();
        arm.grouping = Grouping::Shuffled(rng.next());
        draws.push(measure(indir, calib, &arm, scope)?);
    }
    Ok(Control { learned, shuffled: draws })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Experts drawn from `clusters` distinct shared directions plus noise —
    /// genuine cross-expert redundancy, and genuinely *grouped*, which is the
    /// only case a grouping search should be able to exploit.
    fn clustered(n: usize, clusters: usize, o: usize, i: usize, shared: f32, noise: f32) -> Stack {
        let mut rng = Rng(0x5EED_C105);
        let centres: Vec<Vec<f32>> = (0..clusters.max(1))
            .map(|_| (0..o * i).map(|_| shared * ((rng.next() % 2000) as f32 / 1000.0 - 1.0)).collect())
            .collect();
        let experts = (0..n)
            .map(|e| {
                let c = centres[e % clusters.max(1)].clone();
                c.iter().map(|&v| v + noise * ((rng.next() % 2000) as f32 / 1000.0 - 1.0)).collect()
            })
            .collect();
        Stack { experts, container_bytes: o * i / 2, cw: vec![1.0; i], weighted: false, o, i }
    }

    /// A stack whose experts genuinely share a direction, plus per-expert noise.
    fn synthetic(n: usize, o: usize, i: usize, shared: f32, noise: f32) -> Stack {
        let mut rng = Rng(0xBA515F17);
        let mut common = vec![0f32; o * i];
        for c in common.iter_mut() {
            *c = shared * ((rng.next() % 2000) as f32 / 1000.0 - 1.0);
        }
        let experts = (0..n)
            .map(|_| {
                common
                    .iter()
                    .map(|&c| c + noise * ((rng.next() % 2000) as f32 / 1000.0 - 1.0))
                    .collect()
            })
            .collect();
        Stack { experts, container_bytes: o * i / 2, cw: vec![1.0; i], weighted: false, o, i }
    }

    #[test]
    fn the_basis_is_exact_so_all_reported_error_is_the_residuals() {
        // The identity the whole measurement rests on: W - (recon + dhat)
        // equals delta - dhat. If the basis were stored lossily this would not
        // hold and the distortion column would silently include basis error.
        let stack = synthetic(6, 4, 8, 1.0, 0.25);
        let mean_all = vec![0f32; stack.o * stack.i];
        let parts = partition(&stack, &mean_all, 1, Grouping::Learned);
        assert_eq!(parts.len(), 1, "groups=1 must produce exactly one group");
        let members = &parts[0];
        let (mean, basis) = fit_group(&stack, members, 2);
        assert!(!stack.experts.is_empty(), "the fixture must have experts");
        let w = &stack.experts[0];
        let recon = reconstruct(&stack, w, &mean, &basis);
        let delta: Vec<f32> = w.iter().zip(recon.iter()).map(|(&a, &b)| a - b).collect();
        let dhat = Target::Int4.roundtrip(&delta, stack.o, stack.i);
        let direct = stack.sq_err(&delta, &dhat);
        let full: Vec<f32> = recon.iter().zip(dhat.iter()).map(|(&a, &b)| a + b).collect();
        let composed = stack.sq_err(w, &full);
        assert!(
            (direct - composed).abs() <= 1e-6 * direct.max(1e-12),
            "basis must be exact: residual-only error {direct} vs end-to-end {composed}"
        );
    }

    #[test]
    fn a_higher_rank_never_increases_the_residual_energy() {
        // Projection onto a larger subspace cannot leave more residual. This is
        // the property that makes rank a meaningful knob rather than a search.
        let stack = synthetic(8, 4, 16, 1.0, 0.5);
        let mean_all = vec![0f32; stack.o * stack.i];
        let parts = partition(&stack, &mean_all, 1, Grouping::Learned);
        assert_eq!(parts.len(), 1, "groups=1 must produce exactly one group");
        let members = &parts[0];
        let mut last = f64::INFINITY;
        for rank in 0..4 {
            let (mean, basis) = fit_group(&stack, members, rank);
            let mut energy = 0.0;
            for &e in members {
                let Some(w) = stack.experts.get(e) else { continue };
                let recon = reconstruct(&stack, w, &mean, &basis);
                energy += stack.sq_err(w, &recon);
            }
            assert!(
                energy <= last * (1.0 + 1e-9) + 1e-12,
                "rank {rank} left more residual ({energy}) than rank {} ({last})",
                rank.saturating_sub(1)
            );
            last = energy;
        }
    }

    #[test]
    fn the_shuffled_control_is_deterministic_and_differs_from_learned() {
        // A control that varied run to run would be unfalsifiable — you could
        // always reroll until the learned arm won.
        let stack = synthetic(8, 4, 8, 1.0, 0.4);
        let mean_all = vec![0f32; stack.o * stack.i];
        let a = partition(&stack, &mean_all, 3, Grouping::Shuffled(7));
        let b = partition(&stack, &mean_all, 3, Grouping::Shuffled(7));
        assert_eq!(a, b, "the same seed must give the same partition");
        let c = partition(&stack, &mean_all, 3, Grouping::Shuffled(9));
        assert_ne!(a, c, "different seeds must give different partitions");
        for p in [&a, &b, &c] {
            assert!(p.iter().all(|g| !g.is_empty()), "an empty group changes the effective rank budget");
            let mut all: Vec<usize> = p.iter().flatten().copied().collect();
            all.sort_unstable();
            assert_eq!(all, (0..stack.experts.len()).collect::<Vec<_>>(), "every expert exactly once");
        }
    }

    #[test]
    fn activation_weighting_is_what_makes_this_not_frobenius() {
        // The load-bearing claim of the module: error in a channel the model
        // never excites is free. Two stacks differing only in `cw` must score
        // the same weight differently, or the "on activations" framing is a
        // label rather than a measurement.
        let base = synthetic(4, 2, 4, 1.0, 0.3);
        assert!(!base.experts.is_empty(), "the fixture must have experts");
        let w: Vec<f32> = base.experts[0].clone();
        let mut hat = w.clone();
        if let Some(v) = hat.get_mut(0) {
            *v += 1.0; // perturb column 0 only
        }
        let flat = Stack { cw: vec![1.0; base.i], ..synthetic(4, 2, 4, 1.0, 0.3) };
        let mut cold = vec![1.0f32; base.i];
        if let Some(c) = cold.get_mut(0) {
            *c = 0.0; // column 0 is never excited
        }
        let weighted = Stack { cw: cold, weighted: true, ..synthetic(4, 2, 4, 1.0, 0.3) };
        assert!(flat.sq_err(&w, &hat) > 0.0, "a flat weighting must see the perturbation");
        assert_eq!(
            weighted.sq_err(&w, &hat),
            0.0,
            "an error in a dead channel must be free, or this is Frobenius with extra steps"
        );
    }

    /// A one-arm report with the byte and distortion figures set directly, so
    /// the verdict arithmetic is pinned independently of any fit.
    fn report(d: f64, db: f64, res: usize, base: usize, container: usize, basis: usize) -> Report {
        Report {
            arms: vec![Arm {
                layer: 0,
                projection: "gate_proj".into(),
                distortion: d,
                distortion_baseline: db,
                signal: 100.0,
                residual_bytes: res,
                baseline_bytes: base,
                basis_bytes: basis,
                container_bytes: container,
                experts: 4,
                activation_weighted: true,
            }],
            topk: 2,
            sparse_layers: 1,
            unweighted: Vec::new(),
        }
    }

    #[test]
    fn a_hostile_residual_is_reported_as_a_loss_not_a_win() {
        // The failure mode the module exists to catch: at identical streamed
        // bytes the residual quantizes WORSE than the weight, so the basis has
        // spent resident capacity to make the output worse. Weight-space
        // reconstruction error would have called this a success.
        // Basis kept small enough that it still beats the container once
        // evicted — otherwise the eviction guard fires first, which is correct
        // behaviour but a different assertion.
        let rep = report(4.0, 1.0, 50, 50, 100, 50);
        let v = rep.verdict();
        assert!(v.contains("moved the entropy around"), "expected the entropy verdict, got:\n{v}");
        assert!(rep.read_amplification() < 1.0, "bytes did fall vs the container — the verdict is not about rate");
    }

    #[test]
    fn a_residual_that_quantizes_better_is_reported_as_a_win() {
        let rep = report(0.4, 1.0, 50, 50, 100, 10);
        assert!(rep.verdict().contains("worth pursuing"), "{}", rep.verdict());
    }

    #[test]
    fn a_lossless_baseline_is_refused_rather_than_scored() {
        // Requantizing int4 to int4 is a no-op, so the baseline column is a
        // structural zero. Every ratio against it is infinite, and rendering a
        // verdict from that would confidently report "infinitely worse" when
        // the truth is "you compared the container against itself". This is the
        // exact class of error the domain gate and the skipbound bound both
        // were, so it gets its own branch rather than a footnote.
        let rep = report(1.0, 0.0, 50, 50, 100, 10);
        let v = rep.verdict();
        assert!(v.contains("DEGENERATE BASELINE"), "a lossless baseline must be refused: {v}");
        assert!(!v.contains("worth pursuing"), "it must not also render a verdict: {v}");
    }

    #[test]
    fn a_residual_costing_more_than_the_baseline_short_circuits() {
        // No rate saving means distortion is irrelevant — reading it would
        // invite a conclusion about fit that the bytes have already refuted.
        // Both arms stay under the container so the container guard, which is
        // checked first and deliberately outranks this one, does not fire.
        let rep = report(0.1, 1.0, 90, 50, 100, 1);
        assert!(rep.verdict().contains("no rate saving"), "{}", rep.verdict());
    }

    #[test]
    fn a_zero_signal_is_not_reported_as_a_perfect_basis() {
        // An all-zero calibration vector makes every weighted error zero. That
        // is a broken sidecar, not a flawless fit, and the two must not print
        // the same way.
        let mut rep = report(0.0, 0.0, 50, 50, 100, 10);
        if let Some(a) = rep.arms.first_mut() {
            a.signal = 0.0;
        }
        assert!(rep.verdict().contains("nothing measured"), "{}", rep.verdict());
    }

    #[test]
    fn an_arm_that_streams_more_than_the_container_cannot_win() {
        // Both arms equal in bytes, the basis arm strictly better in
        // distortion — and both streaming 2x what ships. "The basis wins" is
        // true and useless here, so the container comparison has to come first.
        let rep = report(0.1, 1.0, 200, 200, 100, 10);
        let v = rep.verdict();
        assert!(v.contains("streams MORE than the container"), "{v}");
        assert!(!v.contains("worth pursuing"), "a loss against doing nothing is not a win: {v}");
    }

    #[test]
    fn a_basis_that_only_wins_while_resident_is_not_reported_as_a_win() {
        // The charging assumption `miacollective` flagged. The residual alone
        // beats the container; add the basis re-read and it does not. On an
        // engine whose warm cache hits 0.6 % that is the likely case, so
        // reporting only the resident charge would be assuming the favourable
        // answer to a question this offline tool cannot settle.
        //
        // sparse_layers=1, topk=2, 1 arm: residual scales by topk*layers*3 and
        // the basis by layers*3, so a basis of 200 lands 600 B/token against
        // residual 50 -> 300 and container 100 -> 600.
        let rep = report(0.1, 1.0, 50, 50, 100, 200);
        let b = rep.bytes_per_token();
        assert!(b.residual < b.container, "the residual alone must beat the container here");
        assert!(
            b.residual + b.basis_per_token >= b.container,
            "and the basis re-read must push it back over"
        );
        let v = rep.verdict();
        assert!(v.contains("ONLY while the basis stays resident"), "{v}");
        assert!(rep.read_amplification_evicted() > rep.read_amplification(), "eviction must cost bytes");
    }

    #[test]
    fn the_basis_is_charged_per_layer_not_per_expert() {
        // A basis serves every expert routed in its layer, so it scales with
        // sparse_layers where the residual scales with topk * sparse_layers.
        // Charging it per expert would overstate the evicted case by `topk`
        // and make every basis look unaffordable.
        let rep = report(0.1, 1.0, 50, 50, 100, 10);
        let b = rep.bytes_per_token();
        let ratio = b.residual / b.basis_per_token;
        assert!(
            (ratio - (rep.topk as f64) * (50.0 / 10.0)).abs() < 1e-9,
            "residual/basis should be topk x their byte ratio, got {ratio}"
        );
    }

    #[test]
    fn read_amplification_is_against_the_container_not_the_baseline_arm() {
        // The operator question is "fewer bytes than what I run today". A
        // baseline-relative figure would read as 1.0 in the equal-bytes mode
        // and hide the entire saving.
        let rep = report(1.0, 1.0, 50, 50, 100, 0);
        assert!(
            (rep.read_amplification() - 0.5).abs() < 1e-9,
            "expected 0.5 against the container, got {}",
            rep.read_amplification()
        );
    }

    #[test]
    fn unweighted_projections_are_named_in_the_caveats() {
        // down_proj cannot be activation-weighted from this sidecar, and a
        // report that quietly scored it as if it could would be the exact
        // "weight-space wearing an activations label" failure.
        let rep = Report {
            arms: Vec::new(),
            topk: 8,
            sparse_layers: 1,
            unweighted: vec!["down_proj".into()],
        };
        let c = rep.caveats().join("\n");
        assert!(c.contains("down_proj"), "the unweighted projection must be named: {c}");
        assert!(c.contains("DIAGONAL"), "the diagonal approximation must always be stated: {c}");
        assert!(c.contains("not flip rate"), "distortion must never be read as a quality gate: {c}");
    }

    /// Relative distortion of one arm over a stack, at fixed rank and groups.
    fn arm_rel(stack: &Stack, g: Grouping) -> f64 {
        let cfg = FitConfig { rank: 1, groups: 2, grouping: g, ..FitConfig::default() };
        measure_arm(stack, 0, "gate_proj", &cfg).relative()
    }

    /// Best (lowest) relative distortion over the control's shuffled draws.
    fn best_shuffled(stack: &Stack) -> f64 {
        (0..CONTROL_DRAWS)
            .map(|k| arm_rel(stack, Grouping::Shuffled(0x9E37_79B9 ^ k as u64)))
            .fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn the_control_does_not_fire_on_experts_with_no_structure() {
        // The control's own control. These experts are iid noise with NO shared
        // direction, so a grouping search has nothing to find and must not
        // appear to find something.
        //
        // This is where an assumption of mine was wrong and the test corrected
        // it: I expected the learned arm to win anyway, on the theory that
        // clustering optimizes the same quantity the distortion column scores,
        // so fit and evaluation are the same sample. That bias is real, but
        // taking the BEST of several shuffled draws is itself a selection over
        // the null, and empirically the two cancel — the best draw comes in at
        // or below the learned arm on structureless data. That cancellation is
        // the reason `Control` compares against the best draw rather than one
        // draw or the mean, and it is worth pinning: if it ever stops holding,
        // every positive verdict this tool has ever emitted needs re-reading.
        let stack = synthetic(8, 4, 16, 0.0, 1.0);
        let learned = arm_rel(&stack, Grouping::Learned);
        let best = best_shuffled(&stack);
        assert!(learned > 0.0 && best.is_finite(), "both arms must measure something");
        let margin = (best - learned) / best;
        assert!(
            margin < 0.15,
            "structureless experts must not clear the control's floor: learned {learned:e} vs \
             best shuffled {best:e} is a {:.1}% margin",
            margin * 100.0
        );
    }

    #[test]
    fn the_control_does_fire_when_the_grouping_is_real() {
        // The other direction, without which the test above only proves the
        // control is inert. Two genuine clusters, two groups: a grouping search
        // has something to find, and finding it has to clear the same floor
        // that structureless data does not.
        let stack = clustered(8, 2, 4, 16, 4.0, 0.15);
        let learned = arm_rel(&stack, Grouping::Learned);
        let best = best_shuffled(&stack);
        let margin = (best - learned) / best;
        assert!(
            margin >= 0.15,
            "real cluster structure must clear the floor: learned {learned:e} vs best shuffled \
             {best:e} is only a {:.1}% margin",
            margin * 100.0
        );
    }

    #[test]
    fn jacobi_recovers_a_known_spectrum() {
        // The eigensolver is the one piece here with a checkable ground truth.
        let mut a = vec![2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0];
        let (vals, vecs) = jacobi_eigh(&mut a, 3);
        let want = [2.0 + 2f64.sqrt(), 2.0, 2.0 - 2f64.sqrt()];
        for (got, exp) in vals.iter().zip(want.iter()) {
            assert!((got - exp).abs() < 1e-9, "eigenvalue {got} != {exp}");
        }
        for k in 0..3 {
            let n: f64 = (0..3).map(|r| vecs[r * 3 + k] * vecs[r * 3 + k]).sum();
            assert!((n - 1.0).abs() < 1e-9, "eigenvector {k} is not unit-norm");
        }
    }

    #[test]
    fn the_sweep_runs_on_a_real_container() -> Result<(), Error> {
        let dir = std::env::temp_dir().join(format!("peregrine_basis_{}", std::process::id()));
        if let Err(e) = std::fs::remove_dir_all(&dir) {
            assert_eq!(e.kind(), std::io::ErrorKind::NotFound, "stale fixture: {e}");
        }
        peregrine_model::testkit::build_tiny_model(&dir)?;
        let fit = FitConfig { rank: 1, groups: 1, ..FitConfig::default() };
        let rep = measure(&dir, None, &fit, &Scope { layers: 1, experts: 0 })?;
        assert!(!rep.arms.is_empty(), "the fixture has routed experts to factorize");
        assert!(rep.arms.iter().all(|a| a.signal > 0.0), "a zero signal makes every relative error 0");
        assert!(
            rep.arms.iter().all(|a| !a.activation_weighted),
            "no sidecar was passed, so nothing may claim to be activation-weighted"
        );
        // Without a calibration sidecar every projection is unweighted, and the
        // caveat list must say so rather than letting the run read as calibrated.
        assert_eq!(rep.unweighted.len(), PROJECTIONS.len(), "all three projections are unweighted here");
        assert!(!rep.verdict().is_empty());
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn the_control_runs_both_arms_through_one_code_path() -> Result<(), Error> {
        let dir = std::env::temp_dir().join(format!("peregrine_basisctl_{}", std::process::id()));
        if let Err(e) = std::fs::remove_dir_all(&dir) {
            assert_eq!(e.kind(), std::io::ErrorKind::NotFound, "stale fixture: {e}");
        }
        peregrine_model::testkit::build_tiny_model(&dir)?;
        let fit = FitConfig { rank: 1, groups: 2, ..FitConfig::default() };
        let ctl = measure_control(&dir, None, &fit, &Scope { layers: 1, experts: 0 }, 0xC0FFEE)?;
        let (l, best, worst) = ctl.relatives();
        assert!(
            l.is_finite() && best.is_finite() && worst.is_finite(),
            "every arm must produce a number: {l} {best} {worst}"
        );
        assert_eq!(ctl.shuffled.len(), CONTROL_DRAWS, "one draw cannot separate signal from luck");
        // Equal rank and equal precision on every side is what makes the
        // comparison a control rather than several unrelated runs.
        for d in &ctl.shuffled {
            assert_eq!(
                ctl.learned.arms.len(),
                d.arms.len(),
                "every arm must cover the same (layer, projection) set"
            );
        }
        assert!(!ctl.verdict().is_empty());
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }
}
