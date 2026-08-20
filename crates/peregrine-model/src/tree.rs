//! Token trees: several candidate continuations verified in one forward.
//!
//! A speculative *chain* proposes one continuation and keeps its longest
//! correct prefix. A *tree* proposes several and keeps the longest correct
//! path through any of them, which is what makes an extra verify row buy more
//! than one extra token of expected progress. It is the shared substrate under
//! "tree speculative decoding", "token-tree execution", "blockwise parallel
//! decoding" and the tree half of EAGLE-2 — those differ in how the tree is
//! *built*, not in how it is run.
//!
//! # The layout, and why `LayerKv` did not have to change
//!
//! The obvious way to hold a tree is a cache indexed by tree position, with
//! siblings sharing a slot. That would mean rewriting `LayerKv`, whose
//! `append` deliberately refuses any position that is not its current length —
//! the invariant that turns a scheduler bug into a failed request instead of a
//! sequence silently attending someone else's history.
//!
//! It is not necessary. Lay the candidates out in **DFS order as ordinary
//! consecutive slots** and the cache never sees a tree at all; what makes them
//! a tree is two things the row layout already carries:
//!
//! - [`RowLayout::sel`] gives each row its ancestors, so siblings cannot see
//!   each other — neither is in the other's key set.
//! - [`RowLayout::rope_pos`] gives each row its *tree depth*, so two siblings
//!   are rotated as alternatives at one position rather than as a sequence.
//!
//! A rejected branch then disappears through the same `truncate` a rejected
//! chain does, and the prefix-sharing, the DSA key stream and the recurrent
//! rewind all keep working unchanged.
//!
//! ```text
//!   tokens   A   B   C   D        slot   0   1   2   3
//!   parent   -   0   0   2        depth  0   1   1   2
//!
//!   sel[0] = prefix + {0}                  A          depth 0
//!   sel[1] = prefix + {0, 1}              / \
//!   sel[2] = prefix + {0, 2}             B   C        depth 1
//!   sel[3] = prefix + {0, 2, 3}               \
//!                                              D      depth 2
//! ```
//!
//! B and C are alternatives at depth 1. Slot 1 sits before slot 2 in the cache
//! and B is nonetheless invisible to C, because `sel[2]` does not contain 1.
//!
//! # What this does *not* work on
//!
//! Recurrent (GDN) layers. A linear-attention layer keeps one delta-rule state
//! per sequence and advances it row by row, so two sibling rows would chain
//! into each other's state rather than branch from a shared one — and giving
//! each branch its own state costs a full [`crate::gdn::GdnState`] copy per
//! branch per layer (~3.1 MB × 48 at 27B dims). Trees are therefore an
//! MLA-track mechanism here; the hybrid speculates in chains.

use peregrine_core::Error;

/// A tree of candidate tokens for one sequence, in DFS order.
///
/// Node 0 is the **root**: the already-committed token being fed this tick, not
/// a draft. Its logits are what decide which of its children is accepted, so a
/// tree always has at least one node and a one-node tree is an unspeculated
/// decode step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateTree {
    tokens: Vec<i32>,
    parent: Vec<usize>,
}

impl CandidateTree {
    /// Build and validate. `parent[i] < i` for every non-root node — DFS order
    /// is not a convention here but the thing that makes a tree expressible as
    /// an append-only cache, so it is checked rather than assumed.
    pub fn new(tokens: Vec<i32>, parent: Vec<usize>) -> Result<CandidateTree, Error> {
        if tokens.is_empty() {
            return Err(Error::Format("candidate tree: needs at least the root token".into()));
        }
        if parent.len() != tokens.len() {
            return Err(Error::Format(format!(
                "candidate tree: {} tokens but {} parents",
                tokens.len(),
                parent.len()
            )));
        }
        for (i, &p) in parent.iter().enumerate().skip(1) {
            if p >= i {
                return Err(Error::Format(format!(
                    "candidate tree: node {i}'s parent is {p} — parents must precede their children so the \
                     tree can be laid out as ascending cache slots"
                )));
            }
        }
        Ok(CandidateTree { tokens, parent })
    }

    /// The degenerate tree: a root plus a single branch. This is exactly what
    /// `COLI_DRAFT`'s chain is, which is why every tree rule below has to
    /// reduce to the chain rule on it.
    pub fn chain(root: i32, drafts: &[i32]) -> CandidateTree {
        let mut tokens = Vec::with_capacity(drafts.len() + 1);
        tokens.push(root);
        tokens.extend_from_slice(drafts);
        let parent = (0..tokens.len()).map(|i| i.saturating_sub(1)).collect();
        CandidateTree { tokens, parent }
    }

    pub fn tokens(&self) -> &[i32] {
        &self.tokens
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        false // `new` rejects an empty tree; the root always exists
    }

    /// Rows this tree contributes to a forward — its nodes, root included.
    pub fn rows(&self) -> usize {
        self.tokens.len()
    }

    /// Depth of node `i`, root = 0. This is the node's **logical position**
    /// offset, which is what RoPE must see.
    pub fn depth_of(&self, i: usize) -> usize {
        let mut d = 0;
        let mut cur = i;
        while cur > 0 {
            cur = self.parent[cur];
            d += 1;
        }
        d
    }

    /// Node indices from the root down to `i` inclusive, ascending — which is
    /// both the attention key order and, offset by the block's base slot, the
    /// row's `sel` entry.
    pub fn path_to(&self, i: usize) -> Vec<usize> {
        let mut up = vec![i];
        let mut cur = i;
        while cur > 0 {
            cur = self.parent[cur];
            up.push(cur);
        }
        up.reverse();
        up
    }

    /// Children of `i`, in DFS order.
    pub fn children(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        (i + 1..self.tokens.len()).filter(move |&c| self.parent[c] == i)
    }

    /// Per-row RoPE positions for a block based at cache slot `base`.
    pub fn rope_positions(&self, base: usize) -> Vec<usize> {
        (0..self.tokens.len()).map(|i| base + self.depth_of(i)).collect()
    }

    /// Per-row allowed key sets for a block based at cache slot `base`, given
    /// that everything before `base` is committed context every row may see.
    pub fn key_sets(&self, base: usize) -> Vec<Vec<usize>> {
        (0..self.tokens.len())
            .map(|i| {
                let path = self.path_to(i);
                let mut keys: Vec<usize> = (0..base).collect();
                keys.extend(path.into_iter().map(|n| base + n));
                keys
            })
            .collect()
    }
}

/// The two extra vectors a tree forward needs, bundled because they are always
/// the same length as each other and as the row list, and a mismatch between
/// them would surface only as a wrong answer.
///
/// Build both from one [`CandidateTree`] and one base slot:
/// [`CandidateTree::rope_positions`] and [`CandidateTree::key_sets`].
#[derive(Clone, Copy, Debug)]
pub struct TreeRows<'a> {
    /// Each row's **tree depth**, offset by the block's base slot. Siblings
    /// share a value; that is what makes them alternatives rather than a
    /// sequence.
    pub rope_pos: &'a [usize],
    /// Each row's ancestors as absolute cache indices, plus the committed
    /// prefix every row may see.
    ///
    /// `Option` per row because this vector has to cover **every** row of the
    /// forward, including rows belonging to sequences that are not speculating
    /// on a tree at all: `None` is "dense", and is what keeps a mixed batch on
    /// the attention cores' untouched loops. [`CandidateTree::key_sets`]
    /// produces the tree's own rows; the caller places them at the right
    /// offsets and leaves the rest `None`.
    pub sel: &'a [Option<Vec<usize>>],
}

/// The tree analogue of [`crate::accept_run`], and it must reduce to it.
///
/// `rows` is `[tree.len(), vocab]` — one logits row per node, in DFS order.
/// Walk from the root: at each node take the child whose token equals that
/// node's own argmax, stop when no child matches. Returns the accepted node
/// path (**excluding** the root, which was already committed) and the token to
/// emit next, which is the argmax of the last node reached.
///
/// This is the same greedy-identity rule a chain uses, so a tree changes only
/// *how many* tokens a forward commits, never *which*: every accepted node
/// equals what one-token-at-a-time decoding would have produced at that
/// position. `accept_tree_reduces_to_accept_run_on_a_chain` is the assertion.
pub fn accept_tree(rows: &[f32], vocab: usize, tree: &CandidateTree) -> (Vec<usize>, i32) {
    let row = |i: usize| -> Option<&[f32]> { rows.get(i * vocab..(i + 1) * vocab) };
    let mut path = Vec::new();
    let mut cur = 0usize;
    loop {
        // A short `rows` means the forward returned less than it was asked for.
        // Committing nothing is the safe reading; the caller falls back to an
        // unspeculated step rather than emitting a token nothing verified.
        let Some(r) = row(cur) else { return (path, 0) };
        let a = crate::sample::argmax(r) as i32;
        match tree.children(cur).find(|&c| tree.tokens()[c] == a) {
            Some(c) => {
                path.push(c);
                cur = c;
            }
            None => return (path, a),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tree drawn in the module docs.
    fn abcd() -> Result<CandidateTree, Error> {
        CandidateTree::new(vec![10, 20, 30, 40], vec![0, 0, 0, 2])
    }

    #[test]
    fn parents_must_precede_their_children() {
        // The DFS ordering is load-bearing: it is what lets a tree be laid out
        // as ascending cache slots, so a violation is an error and not a
        // reordering.
        assert!(CandidateTree::new(vec![1, 2], vec![0, 1]).is_err(), "self-parent");
        assert!(CandidateTree::new(vec![1, 2, 3], vec![0, 2, 0]).is_err(), "forward reference");
        assert!(CandidateTree::new(vec![], vec![]).is_err(), "no root");
        assert!(CandidateTree::new(vec![1, 2], vec![0]).is_err(), "length mismatch");
    }

    #[test]
    fn depths_and_paths_follow_the_tree_not_the_slots() -> Result<(), Error> {
        let t = abcd()?;
        assert_eq!((0..4).map(|i| t.depth_of(i)).collect::<Vec<_>>(), vec![0, 1, 1, 2]);
        // Slot 1 and slot 2 are siblings — same depth, different slots.
        assert_eq!(t.path_to(1), vec![0, 1]);
        assert_eq!(t.path_to(2), vec![0, 2]);
        assert_eq!(t.path_to(3), vec![0, 2, 3]);
        assert_eq!(t.children(0).collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(t.children(2).collect::<Vec<_>>(), vec![3]);
        assert_eq!(t.children(1).count(), 0);
        Ok(())
    }

    #[test]
    fn a_sibling_is_absent_from_the_other_siblings_key_set() -> Result<(), Error> {
        // The whole correctness argument for reusing an append-only cache.
        let t = abcd()?;
        let sel = t.key_sets(5); // five committed positions before the block
        assert_eq!(sel[0], vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(sel[1], vec![0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(sel[2], vec![0, 1, 2, 3, 4, 5, 7], "C must not see B at slot 6");
        assert_eq!(sel[3], vec![0, 1, 2, 3, 4, 5, 7, 8], "D sees its own path only");
        for (i, keys) in sel.iter().enumerate() {
            assert!(keys.windows(2).all(|w| w[0] < w[1]), "row {i}: key sets must be ascending");
            assert!(keys.iter().all(|&k| k <= 5 + i), "row {i}: no key may sit past the row's own slot");
        }
        assert_eq!(t.rope_positions(5), vec![5, 6, 6, 7], "siblings share a logical position");
        Ok(())
    }

    #[test]
    fn a_chain_is_the_degenerate_tree() {
        let c = CandidateTree::chain(7, &[8, 9]);
        assert_eq!(c.tokens(), &[7, 8, 9]);
        assert_eq!((0..3).map(|i| c.depth_of(i)).collect::<Vec<_>>(), vec![0, 1, 2]);
        // Its key sets are exactly the causal prefix — no masking at all.
        assert_eq!(c.key_sets(2), vec![vec![0, 1, 2], vec![0, 1, 2, 3], vec![0, 1, 2, 3, 4]]);
        assert_eq!(c.rope_positions(2), vec![2, 3, 4]);
    }

    /// `[n, vocab]` logits whose row `i` peaks at `want[i]`.
    fn rows_peaking_at(want: &[i32], vocab: usize) -> Vec<f32> {
        assert!(want.iter().all(|&w| (w as usize) < vocab), "test data: a peak outside the vocab writes into the next row");
        let mut r = vec![0.0f32; want.len() * vocab];
        for (i, &w) in want.iter().enumerate() {
            r[i * vocab + w as usize] = 1.0;
        }
        r
    }

    #[test]
    fn accept_tree_reduces_to_accept_run_on_a_chain() {
        // The contract that makes trees safe to introduce: on the shape the
        // engine already runs, the new rule and the shipped one must agree —
        // including on where they stop and what they emit next.
        let vocab = 8;
        for drafts in [vec![], vec![3i32], vec![3i32, 4], vec![3i32, 4, 5]] {
            for peaks in [vec![3i32, 4, 5, 6], vec![9i32 % 8, 4, 5, 6], vec![3i32, 7, 5, 6]] {
                let n = drafts.len() + 1;
                let rows = rows_peaking_at(&peaks[..n], vocab);
                let (k, next) = crate::accept_run(&rows, vocab, &drafts);
                let tree = CandidateTree::chain(1, &drafts);
                let (path, tnext) = accept_tree(&rows, vocab, &tree);
                assert_eq!(path.len(), k, "chain {drafts:?} peaks {:?}: accepted count", &peaks[..n]);
                assert_eq!(tnext, next, "chain {drafts:?} peaks {:?}: next token", &peaks[..n]);
                // And the accepted nodes are the accepted drafts, in order.
                let toks: Vec<i32> = path.iter().map(|&i| tree.tokens()[i]).collect();
                assert_eq!(toks, drafts[..k].to_vec());
            }
        }
    }

    #[test]
    fn the_walk_takes_whichever_branch_the_model_argmaxes() -> Result<(), Error> {
        let vocab = 64;
        let t = abcd()?; // root 10; children 20, 30; 30's child 40
        // Root peaks at 30 → take C (slot 2); C peaks at 40 → take D (slot 3);
        // D peaks at 55 → stop and emit 55.
        let rows = rows_peaking_at(&[30, 0, 40, 55], vocab);
        assert_eq!(accept_tree(&rows, vocab, &t), (vec![2, 3], 55));
        // Root peaks at 20 → take B (slot 1), which has no children → emit B's
        // own argmax. The *other* branch being longer must not matter.
        let rows = rows_peaking_at(&[20, 59, 40, 55], vocab);
        assert_eq!(accept_tree(&rows, vocab, &t), (vec![1], 59));
        // Root peaks at something no child offers → nothing is accepted.
        let rows = rows_peaking_at(&[63, 59, 40, 55], vocab);
        assert_eq!(accept_tree(&rows, vocab, &t), (vec![], 63));
        Ok(())
    }

    #[test]
    fn a_short_rows_buffer_commits_nothing() -> Result<(), Error> {
        // A forward that returned less than it was asked for must not be read
        // as an acceptance; the caller retries unspeculated.
        let t = abcd()?;
        let (path, _) = accept_tree(&[], 8, &t);
        assert!(path.is_empty());
        Ok(())
    }
}
