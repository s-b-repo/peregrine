//! Unified prediction & scheduling — combined residency scheduler,
//! cancellable speculative reads, learned tiered placement.


/// Combined residency scheduler: jointly scores memory prediction, KV-state
/// prediction, and (for MoE models) expert prediction instead of treating
/// them as separate uncoordinated systems.
#[derive(Debug, Clone, Default)]
pub struct CombinedResidencyScheduler {
    pub memory_weight: f32,
    pub kv_weight: f32,
    pub expert_weight: f32,
}

impl CombinedResidencyScheduler {
    pub fn new() -> Self {
        Self {
            memory_weight: 1.0,
            kv_weight: 1.0,
            expert_weight: 1.0,
        }
    }

    pub fn score(&self, memory: f32, kv: f32, expert: f32) -> f32 {
        self.memory_weight * memory + self.kv_weight * kv + self.expert_weight * expert
    }

    pub fn combine(&self, memory: f32, kv: f32, expert: f32) -> f32 {
        self.score(memory, kv, expert)
    }

    pub fn joint_score(&self, memory: f32, kv: f32, expert: f32) -> f32 {
        self.score(memory, kv, expert)
    }
}

pub type ResidencyScheduler = CombinedResidencyScheduler;
pub type UnifiedScheduler = CombinedResidencyScheduler;

/// Cancellable speculative reads: deprioritise or cancel in-flight prefetches
/// when the conversation shifts direction.
#[derive(Debug, Clone, Default)]
pub struct CancellableSpeculativeReads {
    pub cancelled: Vec<String>,
}

impl CancellableSpeculativeReads {
    pub fn cancel(&mut self, id: &str) {
        if !self.cancelled.iter().any(|x| x == id) {
            self.cancelled.push(id.to_string());
        }
    }
    pub fn is_cancelled(&self, id: &str) -> bool {
        self.cancelled.iter().any(|x| x == id)
    }
    pub fn cancel_all(&mut self, ids: &[&str]) {
        for id in ids {
            self.cancel(id);
        }
    }
}

pub type CancellablePrefetch = CancellableSpeculativeReads;

/// Learned tiered placement: a lightweight cost model deciding promotion/
/// demotion across GPU/RAM/SSD based on predicted reuse and reload cost.
#[derive(Debug, Clone)]
pub struct LearnedTieredPlacement {
    pub reuse_weight: f32,
    pub reload_cost_weight: f32,
    pub bias: f32,
}

impl Default for LearnedTieredPlacement {
    fn default() -> Self {
        Self {
            reuse_weight: 1.0,
            reload_cost_weight: -0.5,
            bias: 0.0,
        }
    }
}

impl LearnedTieredPlacement {
    /// Predicted cost/benefit score; higher means promote to a hotter tier.
    pub fn score(&self, predicted_reuse: f32, reload_cost: f32) -> f32 {
        self.bias + self.reuse_weight * predicted_reuse + self.reload_cost_weight * reload_cost
    }

    pub fn promote(&self, predicted_reuse: f32, reload_cost: f32, threshold: f32) -> bool {
        self.score(predicted_reuse, reload_cost) >= threshold
    }

    pub fn should_promote(&self, predicted_reuse: f32, reload_cost: f32, threshold: f32) -> bool {
        self.promote(predicted_reuse, reload_cost, threshold)
    }

    pub fn demote(&self, predicted_reuse: f32, reload_cost: f32, threshold: f32) -> bool {
        !self.promote(predicted_reuse, reload_cost, threshold)
    }
}

pub type LearnedCostModel = LearnedTieredPlacement;
pub type TieredPlacement = LearnedTieredPlacement;

/// Expert prediction placeholder for MoE models.
#[derive(Debug, Clone, Default)]
pub struct ExpertPrediction {
    pub layer: usize,
    pub expert: usize,
    pub probability: f32,
}

/// KV-state prediction placeholder.
#[derive(Debug, Clone, Default)]
pub struct KvStatePrediction {
    pub session: String,
    pub probability: f32,
}

/// Memory prediction placeholder.
#[derive(Debug, Clone, Default)]
pub struct MemoryPrediction {
    pub node: String,
    pub probability: f32,
}

// Keep HashMap import used.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combined_score() {
        let s = CombinedResidencyScheduler::new();
        let v = s.score(1.0, 2.0, 3.0);
        assert!(v > 0.0);
    }

    #[test]
    fn cancel_prefetch() {
        let mut c = CancellableSpeculativeReads::default();
        c.cancel("x");
        assert!(c.is_cancelled("x"));
        assert!(!c.is_cancelled("y"));
    }

    #[test]
    fn learned_placement_promote() {
        let m = LearnedTieredPlacement::default();
        assert!(m.promote(10.0, 1.0, 0.0));
        assert!(!m.promote(0.0, 10.0, 100.0));
    }
}
