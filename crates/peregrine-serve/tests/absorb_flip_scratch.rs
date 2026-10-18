// SCRATCH: MLA-absorb end-to-end flip cost on the real container.
// Deleted after reporting. Separate test binary (= own process) so the
// COLI_MLA_ABSORB toggle cannot race other tests' loads.
use peregrine_model::Model;
use peregrine_token::GigaTokenizer;

const MODEL: &str = "/srv/m-sdd/GLM-5.3-Flash-peregrine-v2";
const PROSE: &str = "A mixture-of-experts layer routes each token to a small subset of its experts. \
    The router scores every expert, keeps the top-k, and renormalizes their weights. Only those experts \
    are read from disk and computed; the rest cost nothing for this token. Because routing repeats across \
    nearby tokens, a warm cache of recently used experts absorbs most reads. The union of routed experts \
    across a batch is what the disk must deliver, so batching amortizes reads.";

#[test]
#[ignore = "requires /srv/m-sdd/GLM-5.3-Flash-peregrine-v2 and at least 10.5 GB available RAM"]
fn scratch_absorb_flip() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../peregrine-token/tests/fixtures/gpt2_tokenizer.json");
    let bytes = std::fs::read(&path).unwrap();
    let mut tok = GigaTokenizer::from_hf_json_bytes(&bytes).unwrap();
    let prompt: Vec<i32> = tok.encode(PROSE).into_iter().map(|t| t as i32).collect();
    println!("SCRATCH prompt tokens: {}", prompt.len());

    std::env::remove_var("COLI_MLA_ABSORB");
    let mut dense = Model::load(std::path::Path::new(MODEL)).unwrap();
    let a = dense.teacher_forcing(&prompt).unwrap();
    drop(dense);

    std::env::set_var("COLI_MLA_ABSORB", "1");
    let mut absorb = Model::load(std::path::Path::new(MODEL)).unwrap();
    let b = absorb.teacher_forcing(&prompt).unwrap();
    std::env::remove_var("COLI_MLA_ABSORB");

    let flip = Model::prediction_flip_rate(&a, &b);
    println!("SCRATCH absorb flip_rate={flip:?} over {} positions", prompt.len());
}
