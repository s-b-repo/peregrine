//! The context-plane group: the primitives content addressing rests on.
//!
//! SHA-256 and the Merkle tree are the one place in this workspace where a
//! silent error becomes *permanent*: a digest that is subtly wrong makes every
//! content-addressed object addressable only by the same wrong digest, and a
//! Merkle root that ignores the set order makes an inclusion proof attest to a
//! container nobody can audit. Nothing below is an inference path, which is why
//! it is certified rather than merely tested: the failure mode is a trust
//! decision made on bad arithmetic.

use peregrine_ctx::hash::{sha256, tagged};
use peregrine_ctx::merkle::{empty_root, leaf_digest, verify_proof, MerkleTree, Side};

use crate::{cert, exact, rng, with_cases, Certificate};

pub fn certificates() -> Vec<Certificate> {
    vec![
        sha256_published_vectors(),
        tagged_is_domain_separated(),
        tagged_is_length_prefixed(),
        merkle_root_depends_on_the_set_not_the_order(),
        merkle_proofs_verify_and_forgeries_fail(),
        merkle_empty_root_is_distinct(),
        leaf_digest_binds_the_id(),
    ]
}

/// FIPS 180-4 / NIST CAVP vectors. These check the implementation against
/// published output rather than against itself, which is the only kind of check
/// a hash can really have: a self-consistent hash that is not SHA-256 passes
/// every round-trip test in the tree and verifies nothing at all.
fn sha256_published_vectors() -> Certificate {
    // (input, published digest)
    let vectors: [(&[u8], &str); 4] = [
        (b"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        (b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
        (
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
        ),
        (
            b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
            "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1",
        ),
    ];
    let mut bad = 0.0f64;
    for (input, expected) in vectors {
        if sha256(input).to_hex() != expected {
            bad += 1.0;
        }
    }
    // The million-'a' vector is the padding-boundary case, and it is the one
    // that catches a length field that only works below 56 bytes.
    let long = vec![b'a'; 1_000_000];
    if sha256(&long).to_hex() != "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0" {
        bad += 1.0;
    }
    with_cases(
        cert(
            "ctx",
            "sha256_matches_published_vectors",
            "SHA-256 reproduces the FIPS 180-4 vectors for the empty string, \"abc\", the 448-bit and 896-bit cases, and the one-million-'a' case — including the padding boundary above 55 bytes.",
            bad,
            0.0,
        ),
        vectors.len() + 1,
    )
}

/// Domain separation is why a digest computed for one purpose cannot be replayed
/// as a digest for another. Without it, a content hash and a leaf hash over the
/// same bytes would be interchangeable, and a container could present a content
/// digest where a leaf digest was expected.
fn tagged_is_domain_separated() -> Certificate {
    let mut r = rng();
    let mut bad = 0.0f64;
    let mut cases = 0usize;
    for _ in 0..16 {
        let bytes: Vec<u8> = (0..64).map(|_| (r.unit() * 255.0) as u8).collect();
        let bare = sha256(&bytes);
        let with_tag = tagged("dcr:leaf", &[bytes.as_slice()]);
        let other_tag = tagged("dcr:node", &[bytes.as_slice()]);
        if with_tag == bare || with_tag == other_tag || other_tag == bare {
            bad += 1.0;
        }
        cases += 1;
    }
    with_cases(
        cert(
            "ctx",
            "tagged_hashing_is_domain_separated",
            "A tagged digest differs from the bare digest of the same bytes and from a digest under a different tag: a digest cannot be replayed from one structural role into another.",
            bad,
            0.0,
        ),
        cases,
    )
}

/// Length prefixing is what makes a multi-part digest unambiguous. Without it,
/// the parts `["ab", "c"]` and `["a", "bc"]` hash identically — and since ids are
/// the first part of every leaf digest, two distinct objects could share a leaf.
fn tagged_is_length_prefixed() -> Certificate {
    let a: &[u8] = b"ab";
    let b: &[u8] = b"c";
    let left: &[u8] = b"a";
    let right: &[u8] = b"bc";
    let different = tagged("dcr:t", &[a, b]) != tagged("dcr:t", &[left, right]);
    // Empty parts must not be confusable with absent ones.
    let empty_part = tagged("dcr:t", &[b"", a]) != tagged("dcr:t", &[a]);
    with_cases(
        cert(
            "ctx",
            "tagged_parts_are_length_prefixed",
            "(`\"ab\"`, `\"c\"`) and (`\"a\"`, `\"bc\"`) digest differently, and an empty part is not the same as an absent one: the part boundaries survive into the hash.",
            f64::from(u8::from(!different)) + f64::from(u8::from(!empty_part)),
            0.0,
        ),
        2,
    )
}

/// The Merkle root is the container's identity, so it must be a function of the
/// *set* of objects: the store walks its directory in whatever order the
/// filesystem returns, and a root that depended on that order would change
/// between two machines holding identical content. It must also move when the
/// set changes, or it attests to nothing.
fn merkle_root_depends_on_the_set_not_the_order() -> Certificate {
    let entries: Vec<(String, peregrine_ctx::hash::Digest)> =
        (0..9).map(|i| (format!("obj/{i:03}"), sha256(&(i as u8).to_le_bytes()))).collect();
    let forward = MerkleTree::build(entries.clone());
    let mut shuffled = entries.clone();
    shuffled.reverse();
    let backward = MerkleTree::build(shuffled);
    let order_stable = forward.root() == backward.root();

    // A changed content digest must move the root, and a changed id must too.
    let mut altered = entries.clone();
    altered[4].1 = sha256(b"different");
    let content_moves = forward.root() != MerkleTree::build(altered).root();
    let mut renamed = entries.clone();
    renamed[4].0 = "obj/999".to_string();
    let id_moves = forward.root() != MerkleTree::build(renamed).root();

    // Duplicate ids collapse to the last one seen, so the root equals the root of
    // the de-duplicated set rather than of a set holding both.
    let mut dup = entries.clone();
    dup.push((entries[0].0.clone(), sha256(b"shadow")));
    let dedup = dup.len() - 1;
    let with_dup = MerkleTree::build(dup);
    let collapse = with_dup.len() == dedup && with_dup.root() == backward.root();
    with_cases(
        cert(
            "ctx",
            "merkle_root_is_a_function_of_the_set",
            "The Merkle root is identical for a set walked forwards and backwards, changes when any content digest or object id changes, and collapses duplicate ids to the last one seen rather than admitting both.",
            f64::from(u8::from(!order_stable))
                + f64::from(u8::from(!content_moves))
                + f64::from(u8::from(!id_moves))
                + f64::from(u8::from(!collapse)),
            0.0,
        ),
        4,
    )
}

/// The point of an inclusion proof: the right leaf verifies, and everything else
/// does not. A verifier that accepts a wrong content digest, a wrong id, a
/// flipped side bit, or a truncated proof is worse than no verifier — it is a
/// verifier that says yes.
fn merkle_proofs_verify_and_forgeries_fail() -> Certificate {
    let entries: Vec<(String, peregrine_ctx::hash::Digest)> =
        (0..11).map(|i| (format!("obj/{i:03}"), sha256(&(i as u8).to_le_bytes()))).collect();
    let tree = MerkleTree::build(entries.clone());
    let root = tree.root();
    let mut bad = 0.0f64;
    let mut cases = 0usize;

    for (id, content) in &entries {
        let Some(steps) = tree.proof(id) else {
            bad += 1.0;
            continue;
        };
        // 1. The honest proof verifies.
        if !verify_proof(id, content, &steps, &root) {
            bad += 1.0;
        }
        cases += 1;
        // 2. A different content digest does not.
        if verify_proof(id, &sha256(b"forged"), &steps, &root) {
            bad += 1.0;
        }
        // 3. A different id does not — the id is inside the leaf hash.
        if verify_proof("obj/zzz", content, &steps, &root) {
            bad += 1.0;
        }
        // 4. A truncated proof does not. Dropping the top of the chain leaves a
        //    running hash that is a node digest, never the root.
        if steps.len() > 1 && verify_proof(id, content, &steps[..steps.len() - 1], &root) {
            bad += 1.0;
        }
        // 5. A flipped side bit does not.
        let mut swapped: Vec<peregrine_ctx::merkle::Step> = steps.clone();
        if let Some(first) = swapped.first_mut() {
            first.side = match first.side {
                Side::Left => Side::Right,
                Side::Right => Side::Left,
            };
            if verify_proof(id, content, &swapped, &root) {
                bad += 1.0;
            }
        }
        cases += 4;
    }
    // An absent object has no proof at all — returning one would be the failure.
    if tree.proof("obj/absent").is_some() {
        bad += 1.0;
    }
    with_cases(
        cert(
            "ctx",
            "merkle_proofs_verify_and_forgeries_fail",
            "Every honest inclusion proof verifies against its root; the same proof with a forged content digest, a forged id, a truncated step list or a flipped side bit does not; an absent object has no proof.",
            bad,
            0.0,
        ),
        cases,
    )
}

/// "No objects" must not digest to "a digest nobody computed": an empty
/// container and a container whose root was never written have to be
/// distinguishable, or a store that lost its index would verify as empty.
fn merkle_empty_root_is_distinct() -> Certificate {
    let empty = MerkleTree::build(Vec::<(String, peregrine_ctx::hash::Digest)>::new());
    let one = MerkleTree::build(vec![("a".to_string(), sha256(b"a"))]);
    let distinct = empty.root() != one.root()
        && empty.root() == empty_root()
        && !empty.root().as_bytes().iter().all(|b| *b == 0)
        && empty.is_empty()
        && empty.is_empty();
    with_cases(
        cert(
            "ctx",
            "an_empty_container_has_a_distinct_root",
            "An empty Merkle tree has a well-defined non-zero root that differs from any one-object root, so \"no objects\" cannot be confused with \"root not computed\".",
            exact(distinct),
            0.0,
        ),
        2,
    )
}

/// The leaf digest is what binds an object id to its content; a leaf computed
/// any other way would let the same content appear under two ids. Checked
/// through the production function rather than by restating its construction.
fn leaf_digest_binds_the_id() -> Certificate {
    let c = sha256(b"content");
    let other = sha256(b"other");
    let distinct_ids = leaf_digest("obj/a", &c) != leaf_digest("obj/b", &c);
    let distinct_content = leaf_digest("obj/a", &c) != leaf_digest("obj/a", &other);
    let stable = leaf_digest("obj/a", &c) == leaf_digest("obj/a", &c);
    with_cases(
        cert(
            "ctx",
            "leaf_digest_binds_id_to_content",
            "`leaf_digest` differs when either the object id or the content digest differs, and is stable for the same pair — the same content cannot appear under two ids.",
            f64::from(u8::from(!distinct_ids)) + f64::from(u8::from(!distinct_content)) + f64::from(u8::from(!stable)),
            0.0,
        ),
        3,
    )
}