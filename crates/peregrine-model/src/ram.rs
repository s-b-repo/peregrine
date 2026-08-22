//! Pre-load RAM projection: decide whether this model can run on this machine
//! **before** allocating anything, and say so in one line if it cannot.
//!
//! The failure this exists to prevent is silent. Loading a 744B checkpoint on a
//! box that cannot hold its resident set, the engine allocated until the kernel
//! sent SIGKILL — no error, no log, no exit code worth reading, after minutes of
//! disk I/O. The bytes needed are all in the safetensors headers, so the verdict
//! costs one pass over metadata that [`Model::load`] already makes.
//!
//! Everything here is pure arithmetic over injected inputs, following
//! `cap_ecache_budget`'s precedent: the policy is unit-testable without needing
//! a machine of a particular size. [`project_load`] computes the footprint,
//! [`ram_verdict`] judges it.
//!
//! Ported in spirit from colibrì's `cap_for_ram` — specifically its itemised
//! slack model (activations + page-cache reserve + streaming working set + KV)
//! and its refuse-to-start guard with a `COLI_RAM_OVERCOMMIT` override. colibrì
//! runs that projection *after* its load; running it before is the improvement.

/// Flat allowance for activations, thread stacks, allocator slop and the rest of
/// the process that is not weights, cache or KV. colibrì carries 1.2 GB for the
/// same purpose.
pub const ACTIVATION_SLACK: u64 = 1_200_000_000;

/// Headroom left to the page cache on the buffered read path. Reclaiming it is
/// not free: colibrì measured buffered-pread throughput collapsing (~800 →
/// ~180 MB/s) when the budget squeezed the cache out. O_DIRECT bypasses the page
/// cache, so this is charged only when reads are buffered.
pub const PAGE_CACHE_RESERVE: u64 = 2_500_000_000;

/// Context length the KV projection assumes when the caller has no better
/// number. Serve's `--max-prompt-tokens` default is 8192; a single sequence at
/// 4096 is the conservative middle for a load-time estimate.
pub const DEFAULT_PROJECTED_CTX: u64 = 4096;

/// What the process is about to ask for, in bytes.
#[derive(Debug, Clone, Copy)]
pub struct ProjectInputs {
    /// On-disk bytes of everything that is not a routed expert — attention,
    /// shared experts, embeddings, norms. Resident for the whole run.
    pub dense_disk: u64,
    /// On-disk bytes of the routed experts.
    pub expert_disk: u64,
    /// Whether routed experts stream from disk (else they are resident too).
    pub stream_experts: bool,
    /// Requested warm-cache budget. Shrinkable — it is the first thing given up.
    pub ecache: u64,
    /// Landing and reconstruction buffers the streaming lanes hold at once.
    pub stream_transient: u64,
    /// KV cache for the projected context.
    pub kv_pool: u64,
    /// True when reads go through the page cache (i.e. `COLI_DIRECT` is off).
    pub buffered_reads: bool,
}

/// The projected footprint. `floor` is the peak with the warm cache given up
/// entirely — if that does not fit, no amount of shrinking rescues the run.
#[derive(Debug, Clone, Copy)]
pub struct LoadProjection {
    pub resident: u64,
    pub kv_pool: u64,
    pub stream_transient: u64,
    pub slack: u64,
    pub ecache: u64,
    /// Peak with the requested warm cache.
    pub peak: u64,
    /// Peak with a zero warm cache — the smallest this run can be made.
    pub floor: u64,
}

/// Project the peak resident footprint. Pure; no I/O, no environment.
pub fn project_load(i: &ProjectInputs) -> LoadProjection {
    let resident = if i.stream_experts { i.dense_disk } else { i.dense_disk.saturating_add(i.expert_disk) };
    // The flat allowances describe a 744B-class process. Charged verbatim they
    // would refuse *every* model on a box with less than ~3.7 GB free — including
    // the megabyte-sized synthetic ones the tests load, which need none of it. So
    // scale the allowance with the model (an eighth of its resident set, floored
    // at 64 MB for the smallest) and cap it at the flat figures, which stay the
    // ceiling for anything big enough to deserve them.
    let flat = ACTIVATION_SLACK.saturating_add(if i.buffered_reads { PAGE_CACHE_RESERVE } else { 0 });
    let slack = (resident / 8).max(64 << 20).min(flat);
    let floor = resident
        .saturating_add(i.kv_pool)
        .saturating_add(i.stream_transient)
        .saturating_add(slack);
    LoadProjection {
        resident,
        kv_pool: i.kv_pool,
        stream_transient: i.stream_transient,
        slack,
        ecache: i.ecache,
        peak: floor.saturating_add(i.ecache),
        floor,
    }
}

fn gb(n: u64) -> f64 {
    n as f64 / 1e9
}

/// A cgroup limit large enough to mean "no limit". cgroup v1 spells unlimited as
/// `PAGE_COUNTER_MAX * PAGE_SIZE` — a number near `i64::MAX` that differs by kernel
/// and page size, so it is recognized by magnitude rather than by an exact constant.
/// No real machine has an exabyte of RAM, so anything this large is a sentinel.
const CGROUP_UNLIMITED: u64 = 1 << 60;

/// Bytes still available inside a cgroup v2 memory controller, from the contents of
/// `memory.max` and `memory.current`. `None` when there is no limit (`max`) or the
/// files are not parseable — in both cases the cgroup has nothing to say and the
/// host's own figure stands.
pub fn cgroup_v2_available(max: &str, current: &str) -> Option<u64> {
    let max = max.trim();
    if max == "max" {
        return None;
    }
    let limit: u64 = max.parse().ok()?;
    if limit >= CGROUP_UNLIMITED {
        return None;
    }
    let used: u64 = current.trim().parse().unwrap_or(0);
    Some(limit.saturating_sub(used))
}

/// The cgroup v1 equivalent, from `memory.limit_in_bytes` and `memory.usage_in_bytes`.
pub fn cgroup_v1_available(limit: &str, usage: &str) -> Option<u64> {
    let limit: u64 = limit.trim().parse().ok()?;
    if limit >= CGROUP_UNLIMITED {
        return None;
    }
    let used: u64 = usage.trim().parse().unwrap_or(0);
    Some(limit.saturating_sub(used))
}

/// The memory this process may actually use: the smaller of what the host reports
/// and what its cgroup permits.
///
/// Inside a container `/proc/meminfo` is the **host's**, not the container's — it is
/// not namespaced. A 4 GB container on a 512 GB host reads 512 GB of `MemAvailable`,
/// sizes its caches for a machine it is not running on, and is OOM-killed by the
/// cgroup while every projection in the engine says it had room to spare. Both
/// reference implementations of this problem hit exactly that (WASTE's `LEARNED.md`
/// §40 is titled for it), which is a good sign it is not an exotic case.
///
/// `0` from either source means "unknown" and is treated as no constraint, matching
/// [`ram_verdict`]'s rule that an unmeasurable machine is not a failing one.
pub fn effective_available(host_available: u64, cgroup_available: Option<u64>) -> u64 {
    match cgroup_available {
        Some(c) if host_available == 0 => c,
        Some(c) => host_available.min(c),
        None => host_available,
    }
}

/// A warning when the granted warm cache is a large enough share of available memory
/// to risk the page-fault cliff, or when it had to be capped. `None` when the sizing
/// is uncontroversial.
///
/// The failure mode this names is counter-intuitive enough to be worth a line of
/// output. WASTE swept four cache sizes in one process on a 64 GB machine: at
/// 17.3 GB of cache, 36.2 % hit rate and 0.63 tok/s; at 23.3 GB, a *higher* 38.4 %
/// hit rate, *fewer* bytes read from disk — and 0.07 tok/s, an eightfold collapse.
/// The engine was inside its budget and the machine was not, so every cache hit had
/// become a page fault. More cache monotonically improves every metric the cache
/// itself reports, right past the point where it destroys throughput; the cache's own
/// telemetry cannot see the cliff, which is why the warning has to come from the
/// sizing arithmetic instead.
pub fn cache_cliff_warning(requested: u64, granted: u64, available: u64) -> Option<String> {
    if available == 0 || granted == 0 {
        return None;
    }
    if granted < requested {
        return Some(format!(
            "[ecache] requested {:.1} GB, granted {:.1} GB — capped to fit the streaming buffers and \
             safety margin inside {:.1} GB available",
            gb(requested),
            gb(granted),
            gb(available)
        ));
    }
    // Half of what is available is the point past which the cache is competing with
    // the resident set for the same pages rather than using memory nobody wants.
    if granted.saturating_mul(2) > available {
        return Some(format!(
            "[ecache] {:.1} GB of cache against {:.1} GB available — a cache this large can lower \
             throughput while *raising* its own hit rate, because a hit that has been paged out is a \
             page fault. If decode is slow, measure a smaller COLI_ECACHE_GB before a larger one.",
            gb(granted),
            gb(available)
        ));
    }
    None
}

/// One line describing the projection, always printed so a slow load is
/// explicable and a later OOM has a number to be compared against.
pub fn summary(p: &LoadProjection, mem_available: u64) -> String {
    format!(
        "peregrine: [ram] avail {:.1} GB | resident {:.1} + KV {:.1} + stream {:.1} + slack {:.1} + cache {:.1} -> peak {:.1} GB",
        gb(mem_available),
        gb(p.resident),
        gb(p.kv_pool),
        gb(p.stream_transient),
        gb(p.slack),
        gb(p.ecache),
        gb(p.peak),
    )
}

/// `Err(message)` when this model cannot fit even with the warm cache given up.
///
/// `mem_available` of 0 means "unknown" (unreadable `/proc/meminfo`) and never
/// refuses — an unmeasurable machine is not a failing one. `overcommit` is the
/// `COLI_RAM_OVERCOMMIT=1` escape hatch, for operators who know their swap or
/// cgroup limits better than `MemAvailable` does.
pub fn ram_verdict(p: &LoadProjection, mem_available: u64, overcommit: bool) -> Result<(), String> {
    if mem_available == 0 || overcommit || p.floor <= mem_available {
        return Ok(());
    }
    let short = p.floor.saturating_sub(mem_available);
    Err(format!(
        "this model needs about {:.1} GB resident at minimum ({:.1} GB weights + {:.1} GB KV + \
         {:.1} GB streaming buffers + {:.1} GB activations/page-cache reserve) but only {:.1} GB \
         is available — {:.1} GB short, so the kernel would OOM-kill this run part-way through \
         loading.\n\
         Options: free RAM or add swap; lower the context; set COLI_ECACHE_GB=0 to drop the warm \
         cache (already assumed here); COLI_STREAM=1 to stream experts from disk instead of \
         resident; or COLI_RAM_OVERCOMMIT=1 to attempt the load anyway.",
        gb(p.floor),
        gb(p.resident),
        gb(p.kv_pool),
        gb(p.stream_transient),
        gb(p.slack),
        gb(mem_available),
        gb(short),
    ))
}

/// KV bytes for one sequence of `ctx` tokens: MLA keeps a `kv_lora + qk_rope`
/// latent row per layer per position, as f32.
pub fn kv_pool_bytes(kv_lora: u64, qk_rope: u64, n_layers: u64, ctx: u64) -> u64 {
    kv_lora
        .saturating_add(qk_rope)
        .saturating_mul(4)
        .saturating_mul(n_layers)
        .saturating_mul(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The measured GLM-5.2 container on the box this was written for: 10.59 GB
    /// dense, 363.17 GB of routed experts, against 5.8 GB of free RAM+swap. Both
    /// engines were OOM-killed there; the point of this module is that peregrine
    /// says so in a second instead of dying mute several minutes in.
    fn glm52_on_a_small_box() -> ProjectInputs {
        ProjectInputs {
            dense_disk: 10_592_000_000,
            expert_disk: 363_174_000_000,
            stream_experts: true,
            ecache: 2 << 30,
            stream_transient: 300_000_000,
            kv_pool: kv_pool_bytes(512, 64, 78, DEFAULT_PROJECTED_CTX),
            buffered_reads: true,
        }
    }

    #[test]
    fn refuses_glm52_on_a_seven_gb_box() {
        let p = project_load(&glm52_on_a_small_box());
        assert!(p.resident >= 10_000_000_000, "dense set is the resident floor when streaming");
        // `if let Err` rather than `unwrap_err`/`unwrap_or_default`: the panic gate
        // rejects the first pair, clippy rejects the second. The assert above is
        // what makes the refusal mandatory; the block checks what it says.
        let verdict = ram_verdict(&p, 5_800_000_000, false);
        assert!(verdict.is_err(), "must refuse: this peak cannot fit 5.8 GB");
        if let Err(err) = verdict {
            assert!(err.contains("OOM-kill"), "explains the consequence: {err}");
            assert!(err.contains("COLI_RAM_OVERCOMMIT=1"), "names the override: {err}");
        }
    }

    #[test]
    fn the_same_model_is_fine_on_a_large_box() {
        let p = project_load(&glm52_on_a_small_box());
        assert!(ram_verdict(&p, 64_000_000_000, false).is_ok(), "64 GB box must load");
    }

    #[test]
    fn overcommit_and_unknown_memory_never_refuse() {
        let p = project_load(&glm52_on_a_small_box());
        assert!(ram_verdict(&p, 5_800_000_000, true).is_ok(), "override wins");
        assert!(ram_verdict(&p, 0, false).is_ok(), "unmeasurable machine is not a failing one");
    }

    #[test]
    fn the_warm_cache_is_not_charged_to_the_floor() {
        // The floor is what decides a refusal, so a large requested cache must
        // not be able to cause one — the cache is shrinkable, the weights aren't.
        let mut i = glm52_on_a_small_box();
        i.ecache = 40 << 30;
        let p = project_load(&i);
        assert_eq!(p.floor, project_load(&ProjectInputs { ecache: 0, ..i }).floor);
        assert!(p.peak > p.floor, "the request still shows up in the peak");
    }

    #[test]
    fn resident_experts_are_charged_but_streamed_ones_are_not() {
        let streamed = project_load(&glm52_on_a_small_box());
        let resident = project_load(&ProjectInputs { stream_experts: false, ..glm52_on_a_small_box() });
        assert!(resident.resident > streamed.resident + 300_000_000_000, "363 GB of experts show up");
    }

    #[test]
    fn direct_reads_are_not_charged_the_page_cache_reserve() {
        // Only visible on a model big enough for the flat cap to bind: the scaled
        // allowance is `resident/8`, so the buffered/direct difference shows up
        // once that exceeds the direct-path flat figure.
        let big = ProjectInputs { dense_disk: 200_000_000_000, ..glm52_on_a_small_box() };
        let buffered = project_load(&big);
        let direct = project_load(&ProjectInputs { buffered_reads: false, ..big });
        assert_eq!(buffered.slack, ACTIVATION_SLACK + PAGE_CACHE_RESERVE);
        assert_eq!(direct.slack, ACTIVATION_SLACK);
    }

    #[test]
    fn a_tiny_model_is_not_charged_a_large_models_overheads() {
        // Regression: the flat allowances describe a 744B process. Charged to a
        // megabyte-sized synthetic model they made every load on a RAM-tight box
        // refuse — including the test suite's own models, on the very box this
        // check was written to help.
        let tiny = ProjectInputs {
            dense_disk: 2_000_000,
            expert_disk: 4_000_000,
            stream_experts: false,
            ecache: 0,
            stream_transient: 1_000_000,
            kv_pool: kv_pool_bytes(8, 4, 3, 256),
            buffered_reads: true,
        };
        let p = project_load(&tiny);
        assert!(p.floor < 100 << 20, "tiny model must project under 100 MB, got {}", p.floor);
        assert!(ram_verdict(&p, 500 << 20, false).is_ok(), "must load with 500 MB free");
    }
}

/// Measured resident set of this process, in bytes; `0` when unreadable.
///
/// `/proc/self/statm` field 2 is resident pages. Cheap enough to call every few
/// tokens — one small read, no allocation beyond the line.
pub fn read_rss_bytes() -> u64 {
    let s = match peregrine_io::read_proc_string("/proc/self/statm") {
        Ok(s) => s,
        // A masked or absent `/proc` disables the guard rather than failing the
        // run. Named explicitly (not swallowed) because the two cases differ:
        // absence is a sandbox's normal shape, while any other error means the
        // guard is silently off on a machine that does have `/proc`.
        Err(e) if matches!(e.kind(), std::io::ErrorKind::NotFound | std::io::ErrorKind::PermissionDenied) => {
            return 0;
        }
        Err(e) => {
            peregrine_io::note_advisory_err("RSS guard: /proc/self/statm unreadable (guard off)", &e);
            return 0;
        }
    };
    let Some(res_pages) = s.split_whitespace().nth(1).and_then(|f| f.parse::<u64>().ok()) else {
        return 0;
    };
    // `statm` counts *kernel* pages, so the multiplier has to be the kernel's
    // page size and not a constant: an aarch64 or ppc64le box with 64 KB pages
    // would otherwise report 1/16th of its true RSS and the guard below would
    // never fire on the one class of machine most likely to need it.
    res_pages.saturating_mul(peregrine_io::page_size() as u64)
}

/// Tolerance before the guard acts: 2% of budget plus 300 MB.
///
/// Without slack the guard would fire on ordinary allocator noise and evict a
/// cache that was never the problem. colibrì uses the same shape of margin.
pub fn rss_overshoot_tolerance(budget: u64) -> u64 {
    budget / 50 + 300_000_000
}

/// New warm-cache budget when measured RSS has overshot, or `None` to do nothing.
///
/// Pure, so the policy is testable without a process of a particular size.
/// `budget_limit` of 0 means "no limit configured" and never fires.
pub fn rss_guard_decide(rss: u64, budget_limit: u64, cache_budget: usize) -> Option<usize> {
    if budget_limit == 0 || rss == 0 || cache_budget == 0 {
        return None;
    }
    if rss <= budget_limit.saturating_add(rss_overshoot_tolerance(budget_limit)) {
        return None;
    }
    // Give back what we are over, from the one pool that can be given back.
    // Floor at zero rather than going negative: if the overshoot exceeds the
    // whole cache, dropping all of it is the most this lever can do, and the
    // remainder is resident weights the guard cannot touch.
    // (`saturating_sub` on u64 already floors at zero; a trailing `.max(0)`
    // would be a no-op and clippy rejects it.)
    let over = rss.saturating_sub(budget_limit);
    Some((cache_budget as u64).saturating_sub(over) as usize)
}

#[cfg(test)]
mod guard_tests {
    use super::*;

    #[test]
    fn within_tolerance_does_nothing() {
        let budget = 10_000_000_000;
        assert_eq!(rss_guard_decide(budget, budget, 2 << 30), None, "exactly at budget");
        let just_over = budget + rss_overshoot_tolerance(budget) - 1;
        assert_eq!(rss_guard_decide(just_over, budget, 2 << 30), None, "inside the margin");
    }

    #[test]
    fn a_real_overshoot_shrinks_the_cache_by_the_overshoot() {
        let budget = 10_000_000_000u64;
        let cache = 4_000_000_000usize;
        let rss = budget + 1_000_000_000; // 1 GB over, well past the margin
        assert_eq!(rss_guard_decide(rss, budget, cache), Some(3_000_000_000));
    }

    #[test]
    fn an_overshoot_larger_than_the_cache_drops_all_of_it() {
        // The cache is the only lever; resident weights cannot be given back.
        // Asking for more than it holds must floor at zero, not wrap.
        let budget = 10_000_000_000u64;
        assert_eq!(rss_guard_decide(budget + 9_000_000_000, budget, 1 << 30), Some(0));
    }

    #[test]
    fn unconfigured_or_unmeasurable_never_fires() {
        assert_eq!(rss_guard_decide(99_000_000_000, 0, 1 << 30), None, "no limit configured");
        assert_eq!(rss_guard_decide(0, 10_000_000_000, 1 << 30), None, "RSS unreadable");
        assert_eq!(rss_guard_decide(99_000_000_000, 10_000_000_000, 0), None, "no cache to give");
    }

    #[test]
    fn cgroup_v2_reports_headroom_and_recognizes_unlimited() {
        // 4 GiB limit, 1 GiB used → 3 GiB left.
        assert_eq!(cgroup_v2_available("4294967296\n", "1073741824\n"), Some(3 << 30));
        // The literal `max`, and the v1-style huge sentinel, both mean "no limit" —
        // and "no limit" must be `None` (defer to the host) rather than a colossal
        // number that would sail through every comparison as if it were real.
        assert_eq!(cgroup_v2_available("max", "0"), None);
        assert_eq!(cgroup_v2_available("9223372036854771712", "0"), None);
        assert_eq!(cgroup_v2_available("garbage", "0"), None);
        // Usage above the limit is a real transient state; it must floor at zero.
        assert_eq!(cgroup_v2_available("1000", "4000"), Some(0));
    }

    #[test]
    fn cgroup_v1_reports_headroom_and_recognizes_unlimited() {
        assert_eq!(cgroup_v1_available("4294967296", "1073741824"), Some(3 << 30));
        assert_eq!(cgroup_v1_available("9223372036854771712", "0"), None);
        assert_eq!(cgroup_v1_available("", "0"), None);
    }

    #[test]
    fn the_container_limit_wins_over_the_hosts_meminfo() {
        // The case this exists for: a small container on a large host. `/proc/meminfo`
        // is not namespaced, so the host's 512 GB is what the process reads, and
        // sizing against it is how a 4 GB container gets OOM-killed while every
        // projection says it had room.
        let host = 512_000_000_000u64;
        assert_eq!(effective_available(host, Some(4_000_000_000)), 4_000_000_000);
        // And the reverse: a generous cgroup on a busy host does not conjure memory.
        assert_eq!(effective_available(2_000_000_000, Some(64_000_000_000)), 2_000_000_000);
        // Unknown from either side is not a constraint.
        assert_eq!(effective_available(host, None), host);
        assert_eq!(effective_available(0, Some(4_000_000_000)), 4_000_000_000);
        assert_eq!(effective_available(0, None), 0);
    }

    #[test]
    fn the_cliff_warning_fires_on_a_capped_or_oversized_cache() {
        let avail = 20_000_000_000u64;
        // Uncontroversial: a small cache against plenty of memory.
        assert_eq!(cache_cliff_warning(2 << 30, 2 << 30, avail), None);
        // Capped — the operator asked for more than they got and should know.
        assert!(
            cache_cliff_warning(16 << 30, 4 << 30, avail).is_some_and(|w| w.contains("capped")),
            "a capped request must say so"
        );
        // Oversized: granted in full, but more than half of everything available.
        assert!(
            cache_cliff_warning(15 << 30, 15 << 30, avail).is_some_and(|w| w.contains("page fault")),
            "an oversized cache must name the failure mode"
        );
        // Unmeasurable or disabled says nothing rather than guessing.
        assert_eq!(cache_cliff_warning(4 << 30, 4 << 30, 0), None);
        assert_eq!(cache_cliff_warning(0, 0, avail), None);
    }
}
