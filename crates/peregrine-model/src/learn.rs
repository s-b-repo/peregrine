//! Online learned schedulers — the pragmatic forms of the roadmap's
//! "learning-based scheduler" and "reinforcement-learning scheduler".
//!
//! Both learn *online* from observed decode latency (no offline training
//! pipeline) and both act on the same knob set the hand-tuned governors use, so
//! a bad policy can never leave the safe envelope:
//!
//! - [`BanditScheduler`] — ε-greedy multi-armed bandit over a small set of knob
//!   configurations (prefetch distance × worker count). Reward = EWMA of
//!   1 / decode-µs. ε decays as arms accumulate pulls.
//! - [`QScheduler`] — tabular Q-learning over (lane bias × phase-stability
//!   bucket) states and knob-delta actions, reward = latency improvement.
//!
//! Determinism for tests: exploration uses a seeded LCG, never wall-clock or
//! `rand`. Both policies serialize to JSON and persist inside
//! `route_stats.json`, so a session starts from the previous session's policy
//! ("cross-session memory" of the living-engine capstone).
//!
//! Correctness-neutral: knobs only shape performance.

use crate::lane::Bias;

/// One knob configuration a learner can choose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KnobArm {
    /// Prefetch distance (warm-tier breadth) to apply.
    pub prefetch_distance: usize,
    /// CPU-lane worker count to apply.
    pub workers: usize,
}

/// Deterministic LCG (same constants as the corpus synthesizer) — exploration
/// randomness without process-global state.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

/// ε-greedy bandit over `arms`. Call [`Self::choose`] before a forward, apply
/// the returned arm, then [`Self::reward`] with the observed decode latency.
pub struct BanditScheduler {
    arms: Vec<KnobArm>,
    /// per-arm EWMA of reward (1e6 / decode_us → "forwards per second"-ish).
    value: Vec<f32>,
    pulls: Vec<u32>,
    current: usize,
    rng: Lcg,
}

impl BanditScheduler {
    /// A bandit over the default arm grid around `base_workers`: three prefetch
    /// distances × two worker counts.
    pub fn new(base_workers: usize, seed: u64) -> BanditScheduler {
        let w_hi = base_workers.max(2);
        let w_lo = (base_workers / 2).max(2);
        let mut arms = Vec::new();
        for &d in &[2usize, 4, 8] {
            for &w in &[w_lo, w_hi] {
                let arm = KnobArm { prefetch_distance: d, workers: w };
                if !arms.contains(&arm) {
                    arms.push(arm);
                }
            }
        }
        let n = arms.len();
        BanditScheduler { arms, value: vec![0.0; n], pulls: vec![0; n], current: 0, rng: Lcg(seed) }
    }

    /// Pick the next arm: explore with probability `ε = 1/(1+total_pulls/16)`
    /// (decays as evidence accumulates), else exploit the best EWMA.
    pub fn choose(&mut self) -> KnobArm {
        let total: u32 = self.pulls.iter().sum();
        let eps_denom = 1 + (total / 16);
        let explore = self.rng.next().is_multiple_of(eps_denom as u64 + 1);
        self.current = if explore || total == 0 {
            (self.rng.next() as usize) % self.arms.len()
        } else {
            self.value
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal).then(b.0.cmp(&a.0)))
                .map(|(i, _)| i)
                .unwrap_or(0)
        };
        self.arms[self.current]
    }

    /// Report the decode latency observed under the last-chosen arm.
    pub fn reward(&mut self, decode_us: u64) {
        let r = 1e6 / decode_us.max(1) as f32;
        let i = self.current;
        self.pulls[i] = self.pulls[i].saturating_add(1);
        self.value[i] = if self.pulls[i] == 1 { r } else { 0.8 * self.value[i] + 0.2 * r };
    }

    /// The best-known arm (exploit-only view, for introspection).
    pub fn best(&self) -> KnobArm {
        let i = self
            .value
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal).then(b.0.cmp(&a.0)))
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.arms[i]
    }

    pub fn to_json(&self) -> serde_json::Value {
        let arms: Vec<serde_json::Value> = self
            .arms
            .iter()
            .zip(self.value.iter().zip(&self.pulls))
            .map(|(a, (v, p))| serde_json::json!([a.prefetch_distance, a.workers, v, p]))
            .collect();
        serde_json::json!({ "kind": "bandit", "arms": arms })
    }

    /// Restore value/pull statistics for matching arms (mismatched arms keep
    /// their fresh zeros — a changed grid degrades gracefully).
    pub fn restore(&mut self, v: &serde_json::Value) {
        let Some(arms) = v.get("arms").and_then(|a| a.as_array()) else { return };
        for row in arms {
            let Some(a) = row.as_array() else { continue };
            if a.len() != 4 {
                continue;
            }
            let (Some(d), Some(w), Some(val), Some(p)) =
                (a[0].as_u64(), a[1].as_u64(), a[2].as_f64(), a[3].as_u64())
            else {
                continue;
            };
            if let Some(i) = self
                .arms
                .iter()
                .position(|arm| arm.prefetch_distance == d as usize && arm.workers == w as usize)
            {
                self.value[i] = val as f32;
                self.pulls[i] = p as u32;
            }
        }
    }
}

/// Q-learning state: the lane bias × a phase-stability bucket.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct QState {
    pub bias: u8,      // 0 balanced, 1 io, 2 cpu, 3 gpu
    pub stability: u8, // 0 stable (low entropy) / 1 shifting (high entropy)
}

impl QState {
    pub fn from(bias: Bias, entropy: f32) -> QState {
        let b = match bias {
            Bias::Balanced => 0,
            Bias::TowardIo => 1,
            Bias::TowardCpu => 2,
            Bias::TowardGpu => 3,
        };
        QState { bias: b, stability: u8::from(entropy > 0.7) }
    }
}

/// Actions: adjust one knob one step (or hold).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QAction {
    Hold,
    PrefetchUp,
    PrefetchDown,
    WorkersUp,
    WorkersDown,
}

pub const Q_ACTIONS: [QAction; 5] =
    [QAction::Hold, QAction::PrefetchUp, QAction::PrefetchDown, QAction::WorkersUp, QAction::WorkersDown];

/// Tabular Q-learning over 8 states × 5 actions. Reward = latency improvement
/// vs the previous forward (positive = faster). α = 0.3, γ = 0.5, ε decays with
/// visits.
pub struct QScheduler {
    /// `q[state_index][action_index]`
    q: Vec<[f32; 5]>,
    visits: Vec<u32>,
    last: Option<(usize, usize)>, // (state, action) awaiting reward
    prev_latency_us: Option<u64>,
    rng: Lcg,
}

impl QScheduler {
    pub fn new(seed: u64) -> QScheduler {
        QScheduler { q: vec![[0.0; 5]; 8], visits: vec![0; 8], last: None, prev_latency_us: None, rng: Lcg(seed) }
    }

    fn state_index(s: QState) -> usize {
        ((s.bias as usize) & 3) * 2 + (s.stability as usize)
    }

    /// Choose an action for `state` (ε-greedy, ε = 1/(1+visits/8)).
    pub fn choose(&mut self, state: QState) -> QAction {
        let si = Self::state_index(state);
        self.visits[si] = self.visits[si].saturating_add(1);
        let eps_denom = 1 + (self.visits[si] / 8);
        let explore = self.rng.next().is_multiple_of(eps_denom as u64 + 1);
        let ai = if explore {
            (self.rng.next() as usize) % 5
        } else {
            let row = &self.q[si];
            (0..5)
                .max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap_or(std::cmp::Ordering::Equal).then(b.cmp(&a)))
                .unwrap_or(0)
        };
        self.last = Some((si, ai));
        Q_ACTIONS[ai]
    }

    /// Report the decode latency observed after the last action; updates Q with
    /// reward = (prev_latency − latency) / prev_latency (bounded ±1).
    pub fn reward(&mut self, latency_us: u64, next_state: QState) {
        let Some((si, ai)) = self.last.take() else {
            self.prev_latency_us = Some(latency_us);
            return;
        };
        let r = match self.prev_latency_us {
            Some(prev) if prev > 0 => {
                ((prev as f32 - latency_us as f32) / prev as f32).clamp(-1.0, 1.0)
            }
            _ => 0.0,
        };
        self.prev_latency_us = Some(latency_us);
        let ni = Self::state_index(next_state);
        let max_next = self.q[ni].iter().fold(f32::MIN, |m, &v| m.max(v));
        let max_next = if max_next == f32::MIN { 0.0 } else { max_next };
        let cur = self.q[si][ai];
        self.q[si][ai] = cur + 0.3 * (r + 0.5 * max_next - cur);
    }

    /// The greedy action for `state` (exploit-only view, for tests).
    pub fn greedy(&self, state: QState) -> QAction {
        let row = &self.q[Self::state_index(state)];
        let ai = (0..5)
            .max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap_or(std::cmp::Ordering::Equal).then(b.cmp(&a)))
            .unwrap_or(0);
        Q_ACTIONS[ai]
    }

    pub fn to_json(&self) -> serde_json::Value {
        let rows: Vec<serde_json::Value> =
            self.q.iter().map(|row| serde_json::json!(row.to_vec())).collect();
        serde_json::json!({ "kind": "q", "q": rows, "visits": self.visits })
    }

    pub fn restore(&mut self, v: &serde_json::Value) {
        if let Some(rows) = v.get("q").and_then(|r| r.as_array()) {
            for (i, row) in rows.iter().take(self.q.len()).enumerate() {
                if let Some(vals) = row.as_array() {
                    for (j, x) in vals.iter().take(5).enumerate() {
                        if let Some(f) = x.as_f64() {
                            self.q[i][j] = f as f32;
                        }
                    }
                }
            }
        }
        if let Some(vs) = v.get("visits").and_then(|r| r.as_array()) {
            for (i, x) in vs.iter().take(self.visits.len()).enumerate() {
                if let Some(n) = x.as_u64() {
                    self.visits[i] = n as u32;
                }
            }
        }
    }
}

/// Which learned scheduler is enabled: bandit (`COLI_LEARN_SCHED=1`) wins over
/// Q-learning (`COLI_RL_SCHED=1`); neither → `None`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LearnMode {
    Bandit,
    Q,
}

pub fn learn_mode() -> Option<LearnMode> {
    if matches!(std::env::var("COLI_LEARN_SCHED").as_deref(), Ok("1") | Ok("true")) {
        return Some(LearnMode::Bandit);
    }
    if matches!(std::env::var("COLI_RL_SCHED").as_deref(), Ok("1") | Ok("true")) {
        return Some(LearnMode::Q);
    }
    None
}

/// The active learned scheduler, unified so the model holds one field.
pub enum Learner {
    Bandit(BanditScheduler),
    Q(QScheduler),
}

impl Learner {
    /// Build from the env mode (`None` when learning is off).
    pub fn from_env(base_workers: usize) -> Option<Learner> {
        match learn_mode()? {
            LearnMode::Bandit => Some(Learner::Bandit(BanditScheduler::new(base_workers, 0x5EED))),
            LearnMode::Q => Some(Learner::Q(QScheduler::new(0x5EED))),
        }
    }

    /// Choose the next knob configuration given the current state signals.
    /// The Q learner emits deltas; they are resolved against `cur` here so both
    /// learners return an absolute [`KnobArm`].
    pub fn choose(&mut self, bias: Bias, entropy: f32, cur: KnobArm, max_workers: usize) -> KnobArm {
        match self {
            Learner::Bandit(b) => b.choose(),
            Learner::Q(q) => {
                let a = q.choose(QState::from(bias, entropy));
                let mut next = cur;
                match a {
                    QAction::Hold => {}
                    QAction::PrefetchUp => next.prefetch_distance = (cur.prefetch_distance + 1).min(16),
                    QAction::PrefetchDown => next.prefetch_distance = cur.prefetch_distance.saturating_sub(1).max(1),
                    QAction::WorkersUp => next.workers = (cur.workers + 1).min(max_workers),
                    QAction::WorkersDown => next.workers = cur.workers.saturating_sub(1).max(2),
                }
                next
            }
        }
    }

    /// Report the observed decode latency for the last choice.
    pub fn reward(&mut self, latency_us: u64, bias: Bias, entropy: f32) {
        match self {
            Learner::Bandit(b) => b.reward(latency_us),
            Learner::Q(q) => q.reward(latency_us, QState::from(bias, entropy)),
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Learner::Bandit(b) => b.to_json(),
            Learner::Q(q) => q.to_json(),
        }
    }

    /// Restore a persisted policy when its kind matches the active mode.
    pub fn restore(&mut self, v: &serde_json::Value) {
        let kind = v.get("kind").and_then(|k| k.as_str());
        match (self, kind) {
            (Learner::Bandit(b), Some("bandit")) => b.restore(v),
            (Learner::Q(q), Some("q")) => q.restore(v),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bandit_converges_to_the_faster_arm() {
        let mut b = BanditScheduler::new(8, 42);
        // Synthetic world: arms with prefetch_distance 8 are 4× faster.
        for _ in 0..400 {
            let arm = b.choose();
            let latency = if arm.prefetch_distance == 8 { 1_000 } else { 4_000 };
            b.reward(latency);
        }
        assert_eq!(b.best().prefetch_distance, 8, "bandit must find the fast arm");
    }

    #[test]
    fn bandit_round_trips_policy() {
        let mut b = BanditScheduler::new(8, 42);
        for _ in 0..100 {
            let arm = b.choose();
            b.reward(if arm.prefetch_distance == 8 { 1_000 } else { 4_000 });
        }
        let j = b.to_json();
        let mut b2 = BanditScheduler::new(8, 7);
        b2.restore(&j);
        assert_eq!(b2.best(), b.best(), "restored policy exploits the same arm");
    }

    #[test]
    fn q_learns_the_rewarded_action() {
        let mut q = QScheduler::new(11);
        let s = QState { bias: 2, stability: 0 };
        // Synthetic world: WorkersUp halves latency, everything else doubles it.
        for _ in 0..600 {
            let a = q.choose(s);
            let latency: u64 = match a {
                QAction::WorkersUp => 2_000,
                _ => 8_000,
            };
            q.reward(latency, s);
        }
        assert_eq!(q.greedy(s), QAction::WorkersUp, "Q must converge on the rewarded action");
    }

    #[test]
    fn q_round_trips() {
        let mut q = QScheduler::new(11);
        let s = QState { bias: 1, stability: 1 };
        for _ in 0..100 {
            q.choose(s);
            q.reward(3_000, s);
        }
        let j = q.to_json();
        let mut q2 = QScheduler::new(5);
        q2.restore(&j);
        assert_eq!(q2.greedy(s), q.greedy(s));
    }

    #[test]
    fn deterministic_given_seed() {
        let run = || {
            let mut b = BanditScheduler::new(4, 99);
            (0..50).map(|_| b.choose().prefetch_distance).collect::<Vec<_>>()
        };
        assert_eq!(run(), run(), "seeded LCG → reproducible choices");
    }
}
