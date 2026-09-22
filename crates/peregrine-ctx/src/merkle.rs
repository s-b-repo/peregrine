//! Merkle tree over the container's objects.
//!
//! Signing one enormous context file proves only that the file as a whole is
//! intact. A Merkle root proves the same thing *and* lets any single object be
//! verified against it with `log n` hashes, without reading the rest of the
//! store. That is what makes the scrubber cheap: it can check one object and
//! repair its path rather than re-signing everything.
//!
//! Two rules the construction depends on:
//!
//! * **Leaves are ordered by object id**, not by insertion, so the root is a
//!   function of the set rather than of the write order.
//! * **Leaves and interior nodes are domain-separated** (`tagged("merkle:leaf")`
//!   vs `tagged("merkle:node")`). Without that, an attacker who controls object
//!   bytes could submit a crafted "object" whose digest is really an interior
//!   node and forge an inclusion proof for material that was never stored.
//!
//! An odd node at any level is **promoted**, not duplicated. Duplicating the
//! last leaf is the classic CVE-2012-2459 shape: two different leaf sets end up
//! with the same root.

use crate::hash::{Digest, tagged};

pub const LEAF_TAG: &str = "dcr:merkle:leaf";
pub const NODE_TAG: &str = "dcr:merkle:node";

/// Which side of the pair the sibling sits on, when replaying a proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

/// One step of an inclusion proof: a sibling digest and the side it is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Step {
    pub sibling: Digest,
    pub side: Side,
}

/// Hash of a leaf's content, under the leaf domain tag.
pub fn leaf_digest(object_id: &str, content: &Digest) -> Digest {
    tagged(LEAF_TAG, &[object_id.as_bytes(), content.as_bytes()])
}

fn node_digest(left: &Digest, right: &Digest) -> Digest {
    tagged(NODE_TAG, &[left.as_bytes(), right.as_bytes()])
}

/// The root of an empty tree — a distinct, well-defined value rather than a
/// zero digest, so "no objects" cannot be confused with "digest not computed".
pub fn empty_root() -> Digest {
    tagged("dcr:merkle:empty", &[])
}

/// A Merkle tree built over `(object_id, content_hash)` pairs.
#[derive(Debug, Clone, Default)]
pub struct MerkleTree {
    /// Leaves, sorted by object id. Kept so proofs can be produced later.
    leaves: Vec<(String, Digest)>,
    /// `levels[0]` is the leaf digests; the last level is the single root.
    levels: Vec<Vec<Digest>>,
}

impl MerkleTree {
    /// Build from any iterator of `(object_id, content_hash)`. Duplicate ids
    /// are collapsed to the last one seen — the store never holds two objects
    /// under one id, and silently keeping both would make the root ambiguous.
    pub fn build(entries: impl IntoIterator<Item = (String, Digest)>) -> MerkleTree {
        let mut leaves: Vec<(String, Digest)> = entries.into_iter().collect();
        leaves.sort_by(|a, b| a.0.cmp(&b.0));
        leaves.dedup_by(|a, b| a.0 == b.0);

        let mut levels: Vec<Vec<Digest>> = Vec::new();
        let base: Vec<Digest> = leaves
            .iter()
            .map(|(id, content)| leaf_digest(id, content))
            .collect();
        levels.push(base);

        while levels
            .last()
            .map(|level| level.len() > 1)
            .unwrap_or_default()
        {
            let current = levels.last().cloned().unwrap_or_default();
            let mut next = Vec::with_capacity(current.len().div_ceil(2));
            let mut i = 0;
            while i + 1 < current.len() {
                next.push(node_digest(&current[i], &current[i + 1]));
                i += 2;
            }
            if i < current.len() {
                // Odd one out: promote it unchanged. Do not duplicate it.
                next.push(current[i]);
            }
            levels.push(next);
        }

        MerkleTree { leaves, levels }
    }

    pub fn root(&self) -> Digest {
        match self.levels.last().and_then(|level| level.first()) {
            Some(root) => *root,
            None => empty_root(),
        }
    }

    pub fn len(&self) -> usize {
        self.leaves.len()
    }

    pub fn is_empty(&self) -> bool {
        self.leaves.is_empty()
    }

    pub fn contains(&self, object_id: &str) -> bool {
        self.position(object_id).is_some()
    }

    fn position(&self, object_id: &str) -> Option<usize> {
        self.leaves
            .binary_search_by(|(id, _)| id.as_str().cmp(object_id))
            .ok()
    }

    /// Inclusion proof for one object: the siblings needed to rebuild the root.
    pub fn proof(&self, object_id: &str) -> Option<Vec<Step>> {
        let mut index = self.position(object_id)?;
        let mut steps = Vec::new();
        for level in &self.levels {
            if level.len() <= 1 {
                break;
            }
            // A promoted odd node has no sibling at this level: it rises
            // unchanged, so the proof records nothing and the index halves.
            if index == level.len() - 1 && level.len() % 2 == 1 {
                index /= 2;
                continue;
            }
            let (sibling, side) = if index % 2 == 0 {
                (level[index + 1], Side::Right)
            } else {
                (level[index - 1], Side::Left)
            };
            steps.push(Step { sibling, side });
            index /= 2;
        }
        Some(steps)
    }
}

/// Replay a proof: does `(object_id, content)` sit under `root`?
///
/// This is the verifier a repaired replica must pass before it is trusted —
/// it needs the object and the proof, never the rest of the store.
pub fn verify_proof(object_id: &str, content: &Digest, steps: &[Step], root: &Digest) -> bool {
    let mut current = leaf_digest(object_id, content);
    for step in steps {
        current = match step.side {
            Side::Right => node_digest(&current, &step.sibling),
            Side::Left => node_digest(&step.sibling, &current),
        };
    }
    current == *root
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::sha256;

    fn entries(n: usize) -> Vec<(String, Digest)> {
        (0..n)
            .map(|i| (format!("obj_{i:03}"), sha256(format!("content {i}").as_bytes())))
            .collect()
    }

    #[test]
    fn every_leaf_proves_against_the_root() {
        // Odd and even counts, and the single-leaf edge case.
        for n in [1usize, 2, 3, 4, 5, 7, 8, 9, 16, 17, 33] {
            let items = entries(n);
            let tree = MerkleTree::build(items.clone());
            let root = tree.root();
            for (id, content) in &items {
                let proof = tree.proof(id).unwrap_or_else(|| panic!("no proof for {id}"));
                assert!(
                    verify_proof(id, content, &proof, &root),
                    "n={n} leaf {id} failed to verify"
                );
            }
        }
    }

    #[test]
    fn a_changed_object_fails_its_proof() {
        let items = entries(8);
        let tree = MerkleTree::build(items.clone());
        let tampered = sha256(b"content 3 but modified");
        let proof = tree.proof("obj_003").unwrap_or_default();
        assert!(!verify_proof("obj_003", &tampered, &proof, &tree.root()));
    }

    #[test]
    fn root_is_order_independent() {
        let items = entries(6);
        let mut reversed = items.clone();
        reversed.reverse();
        assert_eq!(
            MerkleTree::build(items).root(),
            MerkleTree::build(reversed).root()
        );
    }

    #[test]
    fn root_changes_when_the_set_changes() {
        let base = MerkleTree::build(entries(6)).root();
        assert_ne!(base, MerkleTree::build(entries(7)).root());

        let mut edited = entries(6);
        edited[2].1 = sha256(b"different");
        assert_ne!(base, MerkleTree::build(edited).root());
    }

    /// CVE-2012-2459: duplicating the last leaf would make these two sets
    /// share a root. Promotion keeps them distinct.
    #[test]
    fn duplicating_the_last_leaf_does_not_collide() {
        let three = entries(3);
        let mut four = three.clone();
        four.push(three[2].clone());
        four[3].0 = "obj_003".to_string();
        assert_ne!(
            MerkleTree::build(three).root(),
            MerkleTree::build(four).root()
        );
    }

    #[test]
    fn empty_tree_has_a_defined_root() {
        let tree = MerkleTree::build(Vec::new());
        assert!(tree.is_empty());
        assert_eq!(tree.root(), empty_root());
        assert_ne!(tree.root(), Digest::default());
        assert_eq!(tree.proof("nothing"), None);
    }

    #[test]
    fn an_interior_digest_cannot_pose_as_a_leaf() {
        let items = entries(4);
        let tree = MerkleTree::build(items.clone());
        // The digest of an interior node, offered as if it were object content.
        let interior = node_digest(
            &leaf_digest(&items[0].0, &items[0].1),
            &leaf_digest(&items[1].0, &items[1].1),
        );
        let proof = tree.proof(&items[0].0).unwrap_or_default();
        assert!(!verify_proof(&items[0].0, &interior, &proof, &tree.root()));
    }
}
