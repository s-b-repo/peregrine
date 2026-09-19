use super::*;
use crate::testkit::hash_f32_bits;
use peregrine_core::Arch;

#[test]
fn qwen4_synthetic_config_delta_forward_matches_scalar() -> Result<(), Error> {
    use crate::gdn::{gdn_forward, GdnState, GdnWeights};
    use crate::weight::{QtWeight, QuantFmt};
    let dir = std::env::temp_dir().join(format!("peregrine_qwen4_delta_{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&serde_json::json!({
        "model_type": "qwen4_exp_text", "hidden_size": 16, "vocab_size": 32,
        "num_hidden_layers": 1, "layer_types": ["linear_attention"],
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 2,
        "linear_num_key_heads": 1, "linear_num_value_heads": 1,
        "linear_key_head_dim": 2, "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 1, "output_gate_type": "sigmoid"
    }))?)?;
    let cfg = Cfg::load(&dir)?;
    std::fs::remove_dir_all(&dir)?;
    assert_eq!(cfg.arch, Arch::Qwen4Exp);
    let mut qkv = vec![0u8; 20 * 16];
    for row in [0, 2].into_iter().chain(4..20) { qkv[row * 16] = 1; }
    let qkv = QtWeight::new(QuantFmt::Int8, 20, 16, qkv, vec![1.0; 20]);
    let z = QtWeight::new(QuantFmt::Int8, 16, 16, vec![0u8; 256], vec![1.0; 16]);
    let ab = QtWeight::new(QuantFmt::Int8, 1, 16, vec![0u8; 16], vec![1.0]);
    let mut identity = vec![0u8; 256];
    for j in 0..16 { identity[j * 16 + j] = 1; }
    let out = QtWeight::new(QuantFmt::Int8, 16, 16, identity, vec![1.0; 16]);
    let weights = GdnWeights { in_qkv: &qkv, in_z: &z, in_a: &ab, in_b: &ab,
        conv: &[1.0; 20], a_log: &[0.0], dt_bias: &[0.0], norm: &[2.0; 16], out: &out };
    let mut x = vec![0.0; 32];
    x[0] = 1.0;
    x[16] = 1.0;
    let mut state = GdnState::new(&cfg);
    let actual = gdn_forward(&weights, &x, 2, &mut state, &cfg)?;
    let activation = 1.0f64 / (1.0 + (-1.0f64).exp());
    let key = activation / (activation * activation + cfg.eps as f64).sqrt();
    let query = key / 2.0f64.sqrt();
    let mut memory = 0.0;
    for t in 0..2 {
        memory *= 0.5;
        memory += key * (activation - memory * key) * 0.5;
        let value = query * memory;
        let expected = value / (value * value + cfg.eps as f64).sqrt();
        for &value in &actual[t * 16..(t + 1) * 16] {
            assert!((value as f64 - expected).abs() < 2e-6, "token {t}: {value} != {expected}");
        }
    }
    assert_eq!(state.len, 2);
    assert!(actual.iter().all(|&v| v > 0.99 && v < 1.0));
    let mut state = GdnState::new(&cfg);
    let mut step = gdn_forward(&weights, &x[..16], 1, &mut state, &cfg)?;
    step.extend(gdn_forward(&weights, &x[16..], 1, &mut state, &cfg)?);
    assert_eq!(actual, step);
    Ok(())
}

fn tiny_config() -> Result<Cfg, Error> {
    Cfg::from_json(&serde_json::json!({
        "model_type": "qwen4_exp_text", "hidden_size": 4, "vocab_size": 32,
        "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 4, "hc_count": 2, "hc_lowrank": 2,
        "layer_types": ["linear_attention", "qwen_sparse_attention"],
        "indexer_n_heads": 1, "indexer_kv_heads": 1, "indexer_head_dim": 4,
        "indexer_budget": 2, "indexer_compress_ratio": 2,
        "rope_parameters": {"rope_theta": 10000, "partial_rotary_factor": 0.5},
        "ple_layer_ids": [1], "ple_embed_dim": 4, "ngram_size": 3,
        "heads_per_ngram": 2, "ngram_vocab_size_base": 11,
        "make_ngram_vocab_size_divisible_by": 8, "eos_token_id": 0
    }))
}

#[test]
fn qwen4_norm_groups_zero_centering_and_gdn_gate() -> Result<(), Error> {
    let x = [1.0, 1.0, 10.0, 10.0];
    let norm = rms_norm(&x, &[0.0, 1.0, 2.0, -1.0], 2, 1e-6)?;
    for (actual, expected) in norm.iter().zip([1.0, 2.0, 3.0, 0.0]) {
        assert!((actual - expected).abs() < 2e-6);
    }
    let sigmoid = gdn_output_norm(&[1.0, 1.0], &[0.0, 0.0], &[2.0, 0.0], 1e-6, Qwen4OutputGate::Sigmoid)?;
    assert!((sigmoid[0] - 1.0).abs() < 1e-6);
    assert_eq!(sigmoid[1], 0.0);
    assert_eq!(gdn_output_norm(&[1.0, 1.0], &[0.0, 0.0], &[2.0, 0.0], 1e-6, Qwen4OutputGate::Silu)?, [0.0, 0.0]);
    assert!(rms_norm(&x, &[0.0; 3], 2, 1e-6).is_err());
    assert!(rms_norm(&[f32::NAN], &[0.0], 1, 1e-6).is_err());
    Ok(())
}

#[test]
fn qwen4_query_gate_split_is_per_head_per_token() -> Result<(), Error> {
    let p: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let (q, gate) = split_query_gate(&p, 2, 2)?;
    assert_eq!(q, [0.0, 1.0, 4.0, 5.0, 8.0, 9.0, 12.0, 13.0]);
    assert_eq!(gate, [2.0, 3.0, 6.0, 7.0, 10.0, 11.0, 14.0, 15.0]);
    assert!(split_query_gate(&p[..15], 2, 2).is_err());
    assert!(split_query_gate(&p, usize::MAX, 2).is_err());
    Ok(())
}

#[test]
fn qwen4_residual_uses_normalized_stream_mean_and_raw_residual() -> Result<(), Error> {
    let residual = GatedResidual::new(2, 2, 1e-6, vec![0.0; 4],
        Matrix::new(1, 4, vec![0.0; 4])?, Matrix::new(4, 1, vec![0.0; 4])?,
        Some(Matrix::new(2, 4, vec![0.0; 8])?))?;
    let input = [1.0, 1.0, 10.0, 10.0];
    let mixed = residual.mix(&input)?;
    assert!(mixed.mixed.iter().all(|v| (v - 0.5).abs() < 1e-6));
    let injection = mixed.injection.ok_or_else(|| invalid("missing injection"))?;
    assert_eq!(injection, [1.0, 1.0]);
    assert_eq!(residual.combine(&input, &[2.0, -3.0], &injection)?, [3.0, -2.0, 12.0, 7.0]);
    assert!(residual.combine(&input, &[1.0], &injection).is_err());
    let mixer = GatedResidual::new(1, 2, 1e-6, vec![0.0; 2],
        Matrix::new(1, 2, vec![2.0, 0.0])?, Matrix::new(2, 1, vec![1.0, -1.0])?, None)?;
    let actual = mixer.mix(&[1.0, -1.0])?;
    let n = 1.0f64 / (1.0 + 1e-6f64).sqrt();
    let low = n / (1.0 + (-n).exp());
    let expected = n * (1.0 / (1.0 + (-low).exp()) - 1.0 / (1.0 + low.exp())) / 2.0;
    assert!((actual.mixed[0] as f64 - expected).abs() < 1e-6);
    assert!(actual.injection.is_none());
    Ok(())
}

#[test]
fn qwen4_router_is_softmax_not_sigmoid_with_optional_renormalization() -> Result<(), Error> {
    let logits = [0.0, 2.0f32.ln(), 4.0f32.ln()];
    for normalize in [false, true] {
        let route = softmax_topk(&logits, 2, normalize)?;
        assert_eq!((route[0].0, route[1].0), (2, 1));
        let denom = if normalize { 6.0 } else { 7.0 };
        assert!((route[0].1 - 4.0 / denom).abs() < 1e-6);
        assert!((route[1].1 - 2.0 / denom).abs() < 1e-6);
    }
    assert_eq!(softmax_topk(&[1000.0, 1000.0, -1000.0], 2, false)?, [(0, 0.5), (1, 0.5)]);
    assert!(softmax_topk(&[f32::NAN], 1, true).is_err());
    assert!(softmax_topk(&[0.0], 2, true).is_err());
    Ok(())
}

#[test]
fn qwen4_qsa_pools_raw_keys_then_normalizes_and_keeps_tail() -> Result<(), Error> {
    let cfg = tiny_config()?;
    let qsa = Qsa::new(&cfg, vec![0.0; 4], vec![0.0; 4])?;
    let query = [0.0, 0.0, 1.0, 0.0];
    let keys = [0.0,0.0,100.0,0.0, 0.0,0.0,-1.0,0.0,
                0.0,0.0,1.0,0.0, 0.0,0.0,1.0,2.0, 0.0,0.0,-1.0,0.0];
    assert_eq!(qsa.select(&query, &keys, &[0,1,2,3,4], 4, &[0,1,2,3,4])?, [0,1,4]);
    assert_eq!(qsa.select(&query, &keys, &[0,1,2,3,4], 4, &[1,2,4])?, [1,2,4]);
    for n in 0..=5 {
        let visible: Vec<usize> = (0..n).collect();
        let selected = qsa.select(&query, &keys, &[0,1,2,3,4], n, &visible)?;
        assert!(selected.iter().all(|&v| v < n));
        assert!(selected.len() <= 3);
    }
    assert!(qsa.select(&query, &keys, &[0,1,2,3,4], 4, &[0,0]).is_err());
    Ok(())
}

#[test]
fn qwen4_qsa_rope_uses_block_start_and_only_rotary_span() -> Result<(), Error> {
    let cfg = tiny_config()?;
    let qsa = Qsa::new(&cfg, vec![0.0; 4], vec![0.0; 4])?;
    let keys = [1.0,0.0,0.0,0.0, 1.0,0.0,0.0,0.0, 1.0,0.0,0.0,0.0, 1.0,0.0,0.0,0.0];
    assert_eq!(qsa.select(&[1.0,0.0,0.0,0.0], &keys, &[0,1,2,3], 2, &[0,1,2,3])?, [2,3]);
    assert_eq!(qsa.select(&[1.0,0.0,0.0,0.0], &keys, &[2,3,0,1], 2, &[0,1,2,3])?, [0,1]);
    Ok(())
}

#[test]
fn qwen4_ngram_integer_oracle_eos_chunking_reset_and_clone() -> Result<(), Error> {
    let cfg = tiny_config()?;
    let mut hash = NgramHash::new(&cfg, 1)?;
    assert_eq!(hash.multipliers, [256496738022509279, 251627398771002343, 55314113489879221]);
    assert_eq!(hash.sizes, [11,13,17,19]);
    assert_eq!(hash.offsets, [0,11,24,41]);
    assert_eq!(hash.padded_vocab, 64);
    let tokens = [3,5,0,7,2];
    let expected = [5,12,30,55, 7,13,27,52, 5,14,38,48, 8,22,38,42, 10,22,33,42];
    assert_eq!(hash.ids(&tokens)?, expected);
    hash.reset();
    let mut chunked = hash.ids(&tokens[..2])?;
    let mut snapshot = hash.clone();
    chunked.extend(hash.ids(&tokens[2..])?);
    assert_eq!(chunked, expected);
    assert_eq!(snapshot.ids(&tokens[2..])?, expected[8..]);
    let before = hash.history.clone();
    assert!(hash.ids(&[3,32]).is_err());
    assert_eq!(hash.history, before);
    Ok(())
}

#[test]
fn qwen4_critical_matrix_container_f32_int8_and_reject_int4() -> Result<(), Error> {
    use peregrine_core::pack::{f32_bytes, write_safetensors, Blob};
    let dir = std::env::temp_dir().join(format!("peregrine_qwen4_critical_{}", std::process::id()));
    let prefix = "model.layers.0.attn_hyper_connection.input_mix_weight_down.weight";
    write_safetensors(&dir, &[
        Blob::new(prefix, "F32", vec![1, 2], f32_bytes(&[2.0, -3.0])),
        Blob::new("int8", "I8", vec![1, 2], vec![2, 253]),
        Blob::new("int8.qs", "F32", vec![1], f32_bytes(&[1.0])),
        Blob::new("int4", "U8", vec![1, 1], vec![0]),
        Blob::new("int4.qs", "F32", vec![1], f32_bytes(&[1.0])),
    ])?;
    let st = peregrine_core::SafeTensors::open(&dir)?;
    let f32w = Matrix::load_critical(&st, prefix, 1, 2)?;
    let i8w = Matrix::load_critical(&st, "int8", 1, 2)?;
    assert_eq!(f32w.apply(&[1.0, 2.0])?, [-4.0]);
    assert_eq!(i8w.apply(&[1.0, 2.0])?, [-4.0]);
    assert!(Matrix::load_critical(&st, "int4", 1, 2).is_err());
    assert!(Matrix::load_critical(&st, prefix, 2, 1).is_err());
    assert!(Matrix::load_critical(&st, "missing", 1, 2).is_err());
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

#[test]
fn qwen4_config_only_directory_is_refused() -> Result<(), Error> {
    let dir = std::env::temp_dir().join(format!("peregrine_qwen4_refusal_{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&serde_json::json!({
        "model_type": "qwen4_exp_text",
        "num_hidden_layers": 1,
        "layer_types": ["linear_attention"]
    }))?)?;
    let result = crate::Model::load(&dir);
    std::fs::remove_dir_all(&dir)?;
    // A weights-less directory must fail to load (missing shards/tensors) —
    // it previously failed at the production guard; now it fails later, which
    // is still a refusal, just a more specific one.
    let error = result.err().ok_or_else(|| invalid("unexpected model load success"))?.to_string();
    assert!(!error.is_empty(), "empty load error");
    Ok(())
}

#[test]
fn qwen4_kv_truncate_and_reset_match_row_counts() -> Result<(), Error> {
    let cfg = tiny_config()?;
    let (kvl, qkr) = (cfg.kv_row_a() as usize, cfg.kv_row_b() as usize);
    let mut kv = crate::attention::LayerKv::with_dtype(kvl, qkr, crate::attention::KvDtype::F16);
    for pos in 0..4usize {
        let lc: Vec<f32> = (0..kvl).map(|j| (pos * 10 + j) as f32).collect();
        let rc: Vec<f32> = (0..qkr).map(|j| (pos * 10 + j) as f32).collect();
        kv.append(pos, &lc, &rc)?;
        kv.append_index_key(&(0..3).map(|j| (pos + j) as f32).collect::<Vec<_>>());
    }
    assert_eq!(kv.len(), 4);
    assert_eq!(kv.index_len(), 4);
    kv.truncate(2);
    assert_eq!(kv.len(), 2);
    assert_eq!(kv.index_len(), 2);
    kv.truncate(0);
    assert_eq!(kv.len(), 0);
    assert_eq!(kv.index_len(), 0);
    Ok(())
}

#[test]
fn qwen4_tiny_model_loads_and_decodes_deterministically() -> Result<(), Error> {
    let dir = std::env::temp_dir().join(format!("peregrine_qwen4_e2e_{}", std::process::id()));
    crate::testkit::build_tiny_qwen4_model(&dir, 7)?;
    let mut m = crate::Model::load(&dir)?;
    let mut a = crate::sample::Sampler::new(0.0, 0.9, 1);
    let mut b = crate::sample::Sampler::new(0.0, 0.9, 1);
    let tokens: Vec<i32> = [1, 2, 3, 4, 5, 6, 7, 8].to_vec();
    let first = m.generate(&tokens, 8, &mut a)?;
    let second = m.generate(&tokens, 8, &mut b)?;
    std::fs::remove_dir_all(dir)?;
    assert_eq!(first.len(), 8);
    assert_eq!(first, second);
    Ok(())
}

#[test]
fn qwen4_unsupported_paths_fail_loudly() -> Result<(), Error> {
    // Every path listed as unsupported in the module docs must refuse with a
    // `qwen4_exp`-prefixed error — never run a degraded computation.
    let dir = std::env::temp_dir().join(format!("peregrine_qwen4_nosup_{}", std::process::id()));
    crate::testkit::build_tiny_qwen4_model(&dir, 13)?;
    let m = crate::Model::load(&dir)?;
    // Serve-state parity: the tiny config has a linear layer, so a fresh
    // sequence must carry recurrent state (covers the `SeqKv::with_dtype`
    // allocation alongside the `Model` one).
    assert!(crate::SeqKv::new(&m.cfg).has_recurrent_state());
    // Batched / concurrent serving.
    let mut seq = crate::SeqKv::new(&m.cfg);
    let mut refs: [&mut crate::SeqKv; 1] = [&mut seq];
    let error = m.forward_rows_batched(&[1, 2], &[0, 0], &mut refs, &[0, 1], None)
        .err().ok_or_else(|| invalid("batched Qwen4 forward unexpectedly succeeded"))?.to_string();
    assert!(error.contains("qwen4_exp") && error.contains("batched"), "{error}");
    // RLM recursive replay.
    let seq = crate::SeqKv::new(&m.cfg);
    let mut h = vec![0.0; m.cfg.hidden as usize];
    let error = m.forward_hidden_recursive_seq(&seq, &mut h, 1, 0)
        .err().ok_or_else(|| invalid("Qwen4 RLM replay unexpectedly succeeded"))?.to_string();
    assert!(error.contains("qwen4_exp") && error.contains("RLM"), "{error}");
    // MTP drafts (no Qwen4 checkpoint carries an MTP head).
    let hlast = vec![0.0; m.cfg.hidden as usize];
    let error = m.mtp_draft(1, 2, &hlast, 0.0)
        .err().ok_or_else(|| invalid("Qwen4 MTP draft unexpectedly succeeded"))?.to_string();
    assert!(error.contains("qwen4_exp"), "{error}");
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

struct Qwen4Lcg(u64);
impl Qwen4Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// Bit-identity hashing lives in [`crate::testkit::hash_f32_bits`] (shared by
/// every family's oracle tests); the behavior pins below still apply to it.

/// Bit-identity between two outputs, checked both exactly (`to_bits`, the
/// arbiter) and through [`hash_f32_bits`] (the compact form). A hash-only
/// check could in principle collide; a bits-only check is verbose on large
/// tensors — together they pin the implementation while demonstrating the
/// hash method.
fn assert_bit_identical(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()), "{what}: bits differ");
    assert_eq!(hash_f32_bits(a), hash_f32_bits(b), "{what}: hash differs");
}

#[test]
fn qwen4_hash_bits_reports_identity() -> Result<(), Error> {
    assert_eq!(hash_f32_bits(&[]), 14695981039346656037);
    let a = [1.0f32, -0.0, 0.1, f32::from_bits(0x7fc00001)];
    assert_eq!(hash_f32_bits(&a), hash_f32_bits(&a));
    // Bit-level, not value-level: -0.0 and 0.0 hash differently …
    assert_ne!(hash_f32_bits(&[-0.0]), hash_f32_bits(&[0.0]));
    // … and a single flipped bit flips the hash.
    let mut b = a;
    b[2] = f32::from_bits(b[2].to_bits() ^ 1);
    assert_ne!(hash_f32_bits(&a), hash_f32_bits(&b));
    Ok(())
}

#[test]
fn qwen4_parallel_matrix_matches_serial() -> Result<(), Error> {
    let mut rng = Qwen4Lcg(0x1234);
    // Above the 1<<20-MAC gate (8·512·512 = 2M): the pool splits output rows.
    let (rows, cols, in_rows) = (512usize, 512usize, 8usize);
    let vals: Vec<f32> = (0..rows * cols).map(|_| rng.f() * 0.1).collect();
    let x: Vec<f32> = (0..in_rows * cols).map(|_| rng.f()).collect();
    let serial = |vals: &[f32], rows: usize, cols: usize, in_rows: usize, x: &[f32]| {
        let mut out = vec![0.0; in_rows * rows];
        for (ri, xr) in x.chunks_exact(cols).enumerate() {
            for (wi, wr) in vals.chunks_exact(cols).enumerate() {
                out[ri * rows + wi] = wr.iter().zip(xr).map(|(a, b)| a * b).sum();
            }
        }
        out
    };
    assert_bit_identical(&Matrix::new(rows, cols, vals.clone())?.apply(&x)?,
        &serial(&vals, rows, cols, in_rows, &x), "Matrix::apply above gate");
    // Below the gate the historical serial loop runs; same oracle.
    let (rows, cols, in_rows) = (4usize, 8usize, 2usize);
    let vals: Vec<f32> = (0..rows * cols).map(|_| rng.f()).collect();
    let x: Vec<f32> = (0..in_rows * cols).map(|_| rng.f()).collect();
    assert_bit_identical(&Matrix::new(rows, cols, vals.clone())?.apply(&x)?,
        &serial(&vals, rows, cols, in_rows, &x), "Matrix::apply below gate");
    Ok(())
}

/// Owned inputs for one mixer case, sized to push every new parallel axis
/// over its gate at once: matmuls (8·256·1024 = 2M MACs), token loops
/// (`s_n` = 8), norm groups (256-wide, 32 of them). Shared by the sync and
/// async equivalence tests so both compare against the same oracle.
struct MixFixture {
    res: GatedResidual,
    x: Vec<f32>,
    norm_v: Vec<f32>,
    down_v: Vec<f32>,
    up_v: Vec<f32>,
    inj_v: Vec<f32>,
    hidden: usize,
    streams: usize,
    lowrank: usize,
    s_n: usize,
    width: usize,
}

fn mix_fixture(seed: u64) -> Result<MixFixture, Error> {
    let mut rng = Qwen4Lcg(seed);
    let (hidden, streams, lowrank, s_n) = (256usize, 4usize, 256usize, 8usize);
    let width = hidden * streams;
    let mut f = |n: usize| (0..n).map(|_| rng.f() * 0.2).collect::<Vec<f32>>();
    let (norm_v, down_v, up_v, inj_v) = (f(width), f(lowrank * width), f(width * lowrank), f(streams * width));
    let res = GatedResidual::new(hidden, streams, 1e-6,
        norm_v.clone(), Matrix::new(lowrank, width, down_v.clone())?,
        Matrix::new(width, lowrank, up_v.clone())?,
        Some(Matrix::new(streams, width, inj_v.clone())?))?;
    let x = f(s_n * width);
    Ok(MixFixture { res, x, norm_v, down_v, up_v, inj_v, hidden, streams, lowrank, s_n, width })
}

/// The pre-parallel mix, step for step: per-group norms, serial dots, stream
/// accumulation in stream order. Deliberately independent of the
/// implementation (no shared helpers) so agreement means something.
fn serial_mix(fx: &MixFixture) -> Result<(Vec<f32>, Vec<f32>), Error> {
    let MixFixture { norm_v, down_v, up_v, inj_v, hidden, streams, lowrank, s_n, width, x, .. } = fx;
    let (hidden, streams, lowrank, s_n, width) = (*hidden, *streams, *lowrank, *s_n, *width);
    let matvec = |vals: &[f32], rows: usize, cols: usize, x: &[f32]| {
        let mut out = vec![0.0; x.len() / cols * rows];
        for (ri, xr) in x.chunks_exact(cols).enumerate() {
            for (wi, wr) in vals.chunks_exact(cols).enumerate() {
                out[ri * rows + wi] = wr.iter().zip(xr).map(|(a, b)| a * b).sum();
            }
        }
        out
    };
    let mut n = x.clone();
    for (g, group) in n.chunks_exact_mut(hidden).enumerate() {
        let inv = (group.iter().map(|v| v * v).sum::<f32>() / hidden as f32 + 1e-6).sqrt().recip();
        let start = (g * hidden) % norm_v.len();
        for (j, v) in group.iter_mut().enumerate() {
            *v = *v * inv * (1.0 + norm_v[start + j]);
        }
    }
    let low: Vec<f32> = matvec(down_v, lowrank, width, &n).into_iter()
        .map(|v| siluf(v / streams as f32)).collect();
    let gates = matvec(up_v, width, lowrank, &low);
    let mut mixed = vec![0.0; s_n * hidden];
    for t in 0..s_n {
        for h in 0..streams {
            for j in 0..hidden {
                mixed[t * hidden + j] += sigmoidf(gates[t * width + h * hidden + j]) * n[t * width + h * hidden + j];
            }
        }
        for v in &mut mixed[t * hidden..(t + 1) * hidden] {
            *v /= streams as f32;
        }
    }
    let injection: Vec<f32> = matvec(inj_v, streams, width, &n).into_iter()
        .map(|v| 2.0 * sigmoidf(v / streams as f32)).collect();
    Ok((mixed, injection))
}

#[test]
fn qwen4_parallel_matches_serial() -> Result<(), Error> {
    let fx = mix_fixture(0x9a1e)?;
    let (hidden, streams, s_n, width) = (fx.hidden, fx.streams, fx.s_n, fx.width);
    let (mixed, injection) = serial_mix(&fx)?;
    let got = fx.res.mix(&fx.x)?;
    assert_bit_identical(&got.mixed, &mixed, "GatedResidual::mix");
    assert_bit_identical(&got.injection.ok_or_else(|| invalid("missing injection"))?, &injection, "mixer injection");
    // Serial combine oracle against fresh random inputs (own seed).
    let mut rng = Qwen4Lcg(0x51ab);
    let f = |n: usize, rng: &mut Qwen4Lcg| (0..n).map(|_| rng.f() * 0.2).collect::<Vec<f32>>();
    let (block, inj) = (f(s_n * hidden, &mut rng), f(s_n * streams, &mut rng));
    let mut expected = fx.x.clone();
    for t in 0..s_n {
        for h in 0..streams {
            for j in 0..hidden {
                expected[t * width + h * hidden + j] += block[t * hidden + j] * inj[t * streams + h];
            }
        }
    }
    assert_bit_identical(&fx.res.combine(&fx.x, &block, &inj)?, &expected, "GatedResidual::combine");
    Ok(())
}

#[test]
fn qwen4_parallel_moe_matches_serial() -> Result<(), Error> {
    use crate::mlp::Mlp;
    use crate::weight::test_support::quant_i4;
    let mut rng = Qwen4Lcg(0x51e);
    let (d, inter, e_n, k, s_n) = (16usize, 8usize, 4usize, 2usize, 6usize);
    let cfg = Cfg::from_json(&serde_json::json!({
        "model_type": "qwen4_exp_text", "hidden_size": 16, "vocab_size": 32,
        "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 4, "layer_types": ["linear_attention"],
        "linear_num_key_heads": 1, "linear_num_value_heads": 1,
        "linear_key_head_dim": 2, "linear_value_head_dim": 2, "linear_conv_kernel_dim": 1,
        "num_experts": 4, "num_experts_per_tok": 2,
        "moe_intermediate_size": 8, "shared_expert_intermediate_size": 8, "norm_topk_prob": true
    }))?;
    let x: Vec<f32> = (0..s_n * d).map(|_| rng.f()).collect();
    let router_w: Vec<f32> = (0..e_n * d).map(|_| rng.f()).collect();
    let mut mk = || Mlp {
        gate: quant_i4(&(0..inter * d).map(|_| rng.f()).collect::<Vec<_>>(), inter, d),
        up: quant_i4(&(0..inter * d).map(|_| rng.f()).collect::<Vec<_>>(), inter, d),
        down: quant_i4(&(0..d * inter).map(|_| rng.f()).collect::<Vec<_>>(), d, inter),
        limit: 0.0,
    };
    let experts: Vec<Mlp> = (0..e_n).map(|_| mk()).collect();
    let shared = mk();
    let gate_m = Matrix::new(1, d, (0..d).map(|_| rng.f()).collect())?;
    // The pool path needs ≥2 plans; fail loudly instead of testing the
    // serial fallback by accident.
    assert!(crate::router::batch_union(&route(&x, &router_w, &cfg)?, s_n).len() >= 2);
    // Serial oracle: one row at a time, first matching kk per expert.
    let r = route(&x, &router_w, &cfg)?;
    let mut expected = vec![0.0; s_n * d];
    for s in 0..s_n {
        for kk in 0..k {
            let e = r.idx[s * k + kk] as usize;
            let h = experts[e].swiglu(&x[s * d..(s + 1) * d], 1);
            for j in 0..d {
                expected[s * d + j] += r.w[s * k + kk] * h[j];
            }
        }
    }
    let hs = shared.swiglu(&x, s_n);
    let gates = gate_m.apply(&x)?;
    for s in 0..s_n {
        let g = sigmoidf(gates[s]);
        for j in 0..d {
            expected[s * d + j] += g * hs[s * d + j];
        }
    }
    assert_bit_identical(
        &moe_forward_qwen4(&x, &router_w, &gate_m, &experts, Some(&shared), &cfg, s_n)?,
        &expected, "moe_forward_qwen4");
    Ok(())
}

/// The async boundary, as `peregrine-serve` (tokio) must use it: sync CPU work
/// rides `spawn_blocking` — the core stays sync by design, because `async`
/// cannot parallelize CPU-bound math; awaiting the pool from an executor
/// thread would only block that thread while the same threads do the work.
/// Implementation and independent serial oracle run *concurrently* here and
/// must still agree bit-for-bit: scheduling is not an input. This also proves
/// the pool dispatches correctly from inside tokio blocking threads (no
/// deadlock with the nesting guard, no shared-state interference).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn qwen4_async_parallel_oracle_agrees() -> Result<(), Error> {
    let join_err = |e: tokio::task::JoinError| invalid(&format!("task failed: {e}"));
    let parallel = tokio::task::spawn_blocking(|| -> Result<Vec<f32>, Error> {
        let fx = mix_fixture(0x9a1e)?;
        Ok(fx.res.mix(&fx.x)?.mixed)
    });
    let serial = tokio::task::spawn_blocking(|| -> Result<Vec<f32>, Error> {
        Ok(serial_mix(&mix_fixture(0x9a1e)?)?.0)
    });
    // Spawned before either await: both computations are in flight together.
    let (impl_out, oracle_out) =
        (parallel.await.map_err(join_err)??, serial.await.map_err(join_err)??);
    assert_bit_identical(&impl_out, &oracle_out, "async impl vs serial oracle");
    Ok(())
}

#[test]
fn qwen4_chunked_prefill_matches_one_call() -> Result<(), Error> {
    // GDN recurrence, PLE hash/conv history, KV and indexer-key appends are all
    // sequential token order — so `8` at once and `4+4` (or eight `1`s) must be
    // bit-identical. `generate` (prefill-then-decode) depends on exactly this.
    let dir = std::env::temp_dir().join(format!("peregrine_qwen4_chunk_{}", std::process::id()));
    crate::testkit::build_tiny_qwen4_model(&dir, 11)?;
    let mut full = crate::Model::load(&dir)?;
    let mut split = crate::Model::load(&dir)?;
    let mut step = crate::Model::load(&dir)?;
    let tokens: Vec<i32> = [1, 2, 3, 4, 5, 6, 7, 8].to_vec();
    let logits_full = full.forward_step(&tokens, 0)?;
    let mut logits_split = split.forward_step(&tokens[..4], 0)?;
    logits_split.extend(split.forward_step(&tokens[4..], 4)?);
    assert_eq!(logits_full.len(), logits_split.len());
    assert!(logits_full.iter().zip(&logits_split).all(|(a, b)| a.to_bits() == b.to_bits()),
        "4+4 chunked prefill diverged from one 8-token call");
    // Eight single-token steps must match too.
    let mut logits_step = Vec::new();
    for (i, &t) in tokens.iter().enumerate() {
        logits_step.extend(step.forward_step(&[t], i)?);
    }
    assert!(logits_full.iter().zip(&logits_step).all(|(a, b)| a.to_bits() == b.to_bits()),
        "stepwise decode diverged from one 8-token call");
    std::fs::remove_dir_all(dir)?;
    Ok(())
}
