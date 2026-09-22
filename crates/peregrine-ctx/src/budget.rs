//! The attention budget: context assembly as constrained optimisation.
//!
//! ```text
//! maximize   U(S)
//! over       S ⊆ memory objects
//! subject to Σ cost(x) ≤ B_attention
//! ```
//!
//! Because the same node can be admitted at several ladder levels, this is a
//! *multiple-choice* knapsack: pick at most one (node, level) option per node.
//! Solved exactly by DP over quantised token costs — candidate counts here are
//! in the hundreds, so exactness is affordable and removes a whole class of
//! "why did it drop that" debugging.
//!
//! Demotion beats eviction: dropping a node to a cheaper level keeps some
//! utility, evicting keeps none. That falls out of the formulation for free — a
//! cheaper option with non-zero value dominates the skip option — which is why
//! this is an optimiser and not a heuristic sort.

use std::collections::HashMap;

use crate::ladder::Level;
use crate::nodes::NodeIdx;

/// Cost quantisation is chosen so the DP stays this wide regardless of budget:
/// small budgets get exact token resolution, large ones a few tokens of slack.
/// Costs always round *up*, so the solution can never exceed `B_attention`.
pub const MAX_DP_BUCKETS: usize = 400;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Choice {
    pub level: Level,
    pub cost: usize,
    pub utility: f32,
}

impl Choice {
    pub fn new(level: Level, cost: usize, utility: f32) -> Self {
        Self {
            level,
            cost,
            utility,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Candidate {
    pub node: NodeIdx,
    pub options: Vec<Choice>,
    /// Pinned candidates must be admitted at some level (their cheapest, if the
    /// budget is tight). Used for goals, constraints and `explain()` paths the
    /// query explicitly demands.
    pub pinned: bool,
}

impl Candidate {
    pub fn new(node: NodeIdx, options: Vec<Choice>, pinned: bool) -> Self {
        Self {
            node,
            options,
            pinned,
        }
    }

    /// The lowest-cost option, or `None` for an optionless candidate. Total by
    /// construction — the panic the old `expect` documented is now a type the
    /// caller must handle, so a candidate with no options can never reach the
    /// reserved-cost accumulation.
    pub fn cheapest(&self) -> Option<Choice> {
        self.options
            .iter()
            .min_by(|a, b| {
                a.cost
                    .cmp(&b.cost)
                    .then(a.level.order().cmp(&b.level.order()))
            })
            .copied()
    }
}

#[derive(Debug, Default, Clone)]
pub struct Allocation {
    pub chosen: HashMap<NodeIdx, Choice>,
    pub tokens: usize,  // audit-allow: LM token count, not a credential
    pub utility: f32,
    pub budget: usize,
    pub dropped: Vec<NodeIdx>,
    /// `(node, wanted_level, admitted_level)` — the demotions the budget forced.
    pub demoted: Vec<(NodeIdx, Level, Level)>,
    /// True when even the pinned set did not fit; the caller must shrink the
    /// plan rather than silently blow the budget.
    pub overflow: bool,
}

fn ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// Exact multiple-choice knapsack over quantised costs.
pub fn solve(
    candidates: &[Candidate],
    budget: usize,
    quantum: Option<usize>,
    preferred: &HashMap<NodeIdx, Level>,
) -> Allocation {
    let mut alloc = Allocation {
        budget,
        ..Default::default()
    };
    if budget == 0 || candidates.is_empty() {
        alloc.dropped = candidates.iter().map(|c| c.node).collect();
        alloc.overflow = candidates.iter().any(|c| c.pinned);
        return alloc;
    }

    // Pinned candidates get their cheapest option reserved off the top so a
    // flood of cheap high-utility trivia can never squeeze out a constraint.
    let mut forced: HashMap<NodeIdx, Choice> = HashMap::new();
    let mut reserved = 0usize;
    for candidate in candidates.iter().filter(|c| c.pinned) {
        let Some(option) = candidate.cheapest() else {
            continue;
        };
        reserved += option.cost;
        forced.insert(candidate.node, option);
    }

    if reserved > budget {
        // Not even the mandatory set fits: admit in utility order until full.
        let mut ordered: Vec<(NodeIdx, Choice)> = forced.into_iter().collect();
        ordered.sort_by(|a, b| b.1.utility.total_cmp(&a.1.utility).then(a.0.cmp(&b.0)));
        for (node, option) in ordered {
            if alloc.tokens + option.cost <= budget {
                alloc.tokens += option.cost;
                alloc.utility += option.utility;
                alloc.chosen.insert(node, option);
            }
        }
        alloc.overflow = true;
        alloc.dropped = candidates
            .iter()
            .map(|c| c.node)
            .filter(|n| !alloc.chosen.contains_key(n))
            .collect();
        return alloc;
    }

    let free_budget = budget - reserved;
    let quantum = quantum.unwrap_or_else(|| ceil_div(free_budget, MAX_DP_BUCKETS).max(1));
    let buckets = (free_budget / quantum).max(1);

    // For a pinned node the DP optimises the *upgrade* from its reserved
    // cheapest option; for everyone else it optimises admission from nothing.
    struct Item {
        node: NodeIdx,
        choices: Vec<(usize, f32, Option<Choice>)>,
    }
    let mut items: Vec<Item> = Vec::new();
    for candidate in candidates {
        if candidate.options.is_empty() {
            continue;
        }
        let base = forced.get(&candidate.node).copied();
        let mut choices: Vec<(usize, f32, Option<Choice>)> = Vec::new();
        match base {
            None => {
                choices.push((0, 0.0, None)); // skip
                for option in &candidate.options {
                    choices.push((
                        ceil_div(option.cost, quantum),
                        option.utility,
                        Some(*option),
                    ));
                }
            }
            Some(base) => {
                choices.push((0, 0.0, Some(base))); // keep the reserved cheapest
                for option in &candidate.options {
                    if option.cost <= base.cost {
                        continue;
                    }
                    choices.push((
                        ceil_div(option.cost - base.cost, quantum),
                        option.utility - base.utility,
                        Some(*option),
                    ));
                }
            }
        }
        items.push(Item {
            node: candidate.node,
            choices,
        });
    }

    let neg = f32::NEG_INFINITY;
    let mut dp = vec![0.0f32; buckets + 1];
    let mut back: Vec<Vec<i32>> = Vec::with_capacity(items.len());
    for item in &items {
        let mut next = vec![neg; buckets + 1];
        let mut picks = vec![-1i32; buckets + 1];
        for capacity in 0..=buckets {
            let mut best_value = neg;
            let mut best_choice = -1i32;
            for (ci, (cost_b, value, _)) in item.choices.iter().enumerate() {
                if *cost_b > capacity {
                    continue;
                }
                let prev = dp[capacity - cost_b];
                if prev == neg {
                    continue;
                }
                let total = prev + value;
                if total > best_value {
                    best_value = total;
                    best_choice = ci as i32;
                }
            }
            next[capacity] = best_value;
            picks[capacity] = best_choice;
        }
        dp = next;
        back.push(picks);
    }

    let mut capacity = (0..=buckets)
        .max_by(|a, b| dp[*a].total_cmp(&dp[*b]).then(b.cmp(a)))
        .unwrap_or(0);
    let mut selections: HashMap<NodeIdx, Option<Choice>> = HashMap::new();
    for i in (0..items.len()).rev() {
        let pick = back[i][capacity];
        if pick < 0 {
            selections.insert(items[i].node, forced.get(&items[i].node).copied());
            continue;
        }
        let (cost_b, _value, option) = items[i].choices[pick as usize];
        selections.insert(items[i].node, option);
        capacity -= cost_b;
    }

    for candidate in candidates {
        let option = selections
            .get(&candidate.node)
            .copied()
            .flatten()
            .or_else(|| forced.get(&candidate.node).copied());
        match option {
            None => alloc.dropped.push(candidate.node),
            Some(option) => {
                alloc.tokens += option.cost;
                alloc.utility += option.utility;
                alloc.chosen.insert(candidate.node, option);
                if let Some(&wanted) = preferred.get(&candidate.node) {
                    if wanted != option.level {
                        alloc.demoted.push((candidate.node, wanted, option.level));
                    }
                }
            }
        }
    }
    alloc.demoted.sort_by_key(|(n, _, _)| *n);
    alloc.dropped.sort();
    alloc
}
