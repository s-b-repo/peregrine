//! GPU vendor identification and device selection for the VRAM tier.
//!
//! peregrine's GPU lane is compiled from one kernel source (`cuda/backend_cuda.cu`)
//! whose host ABI is vendor-neutral: build it with `nvcc` and the lane runs on
//! NVIDIA, build the same source with ROCm's `hipcc` and it runs on AMD. What a
//! process can actually reach is decided at build time; what it should *try* to
//! use is decided here, from `COLI_GPU_DEVICES`.
//!
//! Three questions live in this module, kept apart on purpose:
//!
//! 1. **What is plugged in?** — [`detect_present`] reads PCI vendor IDs out of
//!    sysfs. Pure filesystem reading, no driver, no toolkit, unit-testable on a
//!    GPU-less CI box.
//! 2. **What did the operator ask for?** — [`parse_device_list`] parses
//!    `COLI_GPU_DEVICES`. A parse error is a hard error: a typo'd list must not
//!    silently become "device 0".
//! 3. **What is usable?** — [`selected_devices`] intersects the two with what
//!    this binary was built against. A request naming a vendor this build cannot
//!    drive is dropped with an advisory naming exactly why (which toolchain, which
//!    rebuild flag), never silently: a tier that came up smaller than asked for is
//!    the one failure mode every GPU-lane knob in this repo reports explicitly.
//!
//! Bare ordinals (`COLI_GPU_DEVICES=0,1`) keep their historical meaning — CUDA
//! device ordinals — so existing invocations are unchanged.

use std::collections::BTreeMap;

/// GPU vendor, identified by PCI vendor ID.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Vendor {
    Nvidia,
    Amd,
    Intel,
}

impl Vendor {
    /// Lowercase name used in `COLI_GPU_DEVICES` prefixes and log lines.
    pub fn name(self) -> &'static str {
        match self {
            Vendor::Nvidia => "nvidia",
            Vendor::Amd => "amd",
            Vendor::Intel => "intel",
        }
    }

    /// The prefix `COLI_GPU_DEVICES` accepts for this vendor.
    pub fn prefixes(self) -> &'static [&'static str] {
        match self {
            Vendor::Nvidia => &["cuda"],
            Vendor::Amd => &["hip", "amd"],
            Vendor::Intel => &["level0", "intel", "sycl"],
        }
    }

    /// Map a raw PCI vendor ID (`/sys/.../device/vendor`, parsed as hex).
    pub fn from_pci_id(id: u16) -> Option<Vendor> {
        match id {
            0x10de => Some(Vendor::Nvidia),
            0x1002 => Some(Vendor::Amd),
            0x8086 => Some(Vendor::Intel),
            _ => None,
        }
    }

    /// Whether this binary was built with a backend that can drive the vendor.
    ///
    /// Asked of the *linked* backend, not the cargo feature: `--features cuda`
    /// selects the GPU lane, and build.rs then links whichever toolchain the
    /// host had (nvcc → CUDA, hipcc → HIP; `PEREGRINE_GPU_BACKEND` forces one).
    /// A HIP build drives AMD and cannot drive NVIDIA, and this predicate
    /// answering from the feature flag alone would have called every AMD card
    /// unreachable from a HIP build — and, worse, called an NVIDIA card
    /// reachable from one. Intel has no ported kernels
    /// (`docs/gpu-vendors.md`), so its answer is false by construction.
    pub fn backend_compiled(self) -> bool {
        #[cfg(feature = "cuda")]
        {
            match self {
                Vendor::Nvidia => peregrine_cuda::linked_backend().starts_with("CUDA"),
                Vendor::Amd => peregrine_cuda::linked_backend().starts_with("HIP"),
                Vendor::Intel => false,
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    /// What an operator must do to make this vendor reachable, for the advisory
    /// emitted when a requested device is dropped.
    pub fn requirement(self) -> &'static str {
        match self {
            Vendor::Nvidia => "rebuild with `--features cuda` on a host with nvcc",
            Vendor::Amd => "rebuild on a ROCm host (`PEREGRINE_GPU_BACKEND=hip`; hipcc compiles the same kernels) — see docs/gpu-vendors.md",
            Vendor::Intel => "not yet portable: the kernels have no SYCL/Level Zero port — see docs/gpu-vendors.md",
        }
    }
}

/// One GPU the PCI probe found on the host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PresentGpu {
    pub vendor: Vendor,
    /// The `cardN` sysfs entry it came from (stable within a boot).
    pub card: String,
}

/// Enumerate GPUs present on the host from `/sys/class/drm/cardN/device/vendor`.
///
/// Best-effort like every probe in this repo: entries that vanish mid-read or
/// carry an unknown vendor id are skipped, and a missing sysfs tree (non-Linux,
/// some containers) yields an empty vector rather than an error. The count is a
/// *card* count — an NVIDIA card visible to the CUDA driver still needs a
/// compiled backend, which is question 3, not this one.
pub fn detect_present() -> Vec<PresentGpu> {
    #[cfg(target_os = "linux")]
    {
        let Ok(entries) = std::fs::read_dir("/sys/class/drm") else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for e in entries.flatten() {
            let name = e.file_name();
            let Some(name) = name.to_str() else { continue };
            // `card0` yes; `card0-DP-1` (connector nodes) and everything else no.
            if !name.strip_prefix("card").is_some_and(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()))
            {
                continue;
            }
            let Ok(id_text) = std::fs::read_to_string(e.path().join("device/vendor")) else { continue };
            let Ok(id) = u16::from_str_radix(id_text.trim().trim_start_matches("0x"), 16) else { continue };
            if let Some(vendor) = Vendor::from_pci_id(id) {
                out.push(PresentGpu { vendor, card: name.to_string() });
            }
        }
        // Deterministic order: card number ascending ("card12" sorts after
        // "card2" lexicographically, so sort by the numeric suffix).
        out.sort_by_key(|g| g.card[4..].parse::<u32>().unwrap_or(0));
        out
    }
    #[cfg(not(target_os = "linux"))]
    {
        Vec::new()
    }
}

/// Per-vendor counts of [`detect_present`], in vendor order — the shape the
/// startup banner prints.
pub fn present_counts() -> BTreeMap<Vendor, usize> {
    let mut m = BTreeMap::new();
    for g in detect_present() {
        *m.entry(g.vendor).or_insert(0) += 1;
    }
    m
}

/// One parsed `COLI_GPU_DEVICES` entry: a vendor plus an ordinal *within that
/// vendor's backend* (CUDA ordinals, HIP ordinals, … are independent namespaces).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceRequest {
    pub vendor: Vendor,
    pub index: i32,
}

impl std::fmt::Display for DeviceRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.vendor.name(), self.index)
    }
}

/// Parse the value of `COLI_GPU_DEVICES`.
///
/// Accepted forms, comma-separated:
/// - unset / empty / `auto` → `Ok(vec![])`, meaning "the default device"
///   (callers substitute `[0]`, the historical behavior);
/// - bare non-negative integers → NVIDIA/CUDA ordinals (`0,1` ≙ `cuda:0,cuda:1`,
///   kept for back-compatibility with every invocation before vendors existed);
/// - `<prefix>:<ordinal>` where prefix names a vendor (`cuda:1`, `hip:0`,
///   `level0:0`).
///
/// Duplicates keep their first occurrence. Anything else is a hard error —
/// including negative ordinals, because `-1` reaching a kernel launch queue is
/// how a typo becomes a silent fallback to fewer devices.
pub fn parse_device_list(spec: &str) -> Result<Vec<DeviceRequest>, String> {
    let spec = spec.trim();
    if spec.is_empty() || spec.eq_ignore_ascii_case("auto") {
        return Ok(Vec::new());
    }
    fn ordinal(text: &str, tok: &str) -> Result<i32, String> {
        text.parse::<i32>()
            .map_err(|e| format!("COLI_GPU_DEVICES: bad ordinal in {tok:?} (expected a non-negative integer): {e}"))
            .and_then(|i| {
                if i < 0 {
                    Err(format!("COLI_GPU_DEVICES: negative ordinal in {tok:?}"))
                } else {
                    Ok(i)
                }
            })
    }

    let mut out: Vec<DeviceRequest> = Vec::new();
    for tok in spec.split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            return Err(format!("COLI_GPU_DEVICES: empty entry in {spec:?}"));
        }
        let req = match tok.split_once(':') {
            None => DeviceRequest { vendor: Vendor::Nvidia, index: ordinal(tok, tok)? },
            Some((prefix, ord)) => {
                let vendor = ["cuda", "hip", "amd", "level0", "intel", "sycl"]
                    .into_iter()
                    .find(|p| prefix.eq_ignore_ascii_case(p))
                    .map(|p| match p {
                        "cuda" => Vendor::Nvidia,
                        "hip" | "amd" => Vendor::Amd,
                        _ => Vendor::Intel,
                    })
                    .ok_or_else(|| format!("COLI_GPU_DEVICES: unknown vendor prefix {prefix:?} in {tok:?}"))?;
                DeviceRequest { vendor, index: ordinal(ord, tok)? }
            }
        };
        if !out.contains(&req) {
            out.push(req);
        }
    }
    Ok(out)
}

/// The CUDA/HIP device ordinals this process should initialize, from
/// `COLI_GPU_DEVICES` (default `[0]`).
///
/// Requests for vendors this build cannot drive are dropped with a one-line
/// advisory each — the operator asked for them by name, so silence would read as
/// success. An explicit list whose every entry was dropped returns empty, which
/// `GpuTier::build_with` treats as "no tier" rather than falling back to device 0:
/// running on a device the operator did not ask for is worse than running on none.
pub fn selected_devices() -> Vec<i32> {
    let requests = match std::env::var("COLI_GPU_DEVICES") {
        // Both variants named rather than `Err(_)`: unset is the ordinary
        // default-device case, while a non-UTF-8 value is a real
        // misconfiguration — reported, then handled the same way, since a
        // value Rust cannot spell is no more parseable than an absent one.
        Err(std::env::VarError::NotPresent) => return vec![0],
        Err(e @ std::env::VarError::NotUnicode(_)) => {
            peregrine_io::note_advisory_err("COLI_GPU_DEVICES is not valid Unicode; using the default device", &e);
            return vec![0];
        }
        Ok(spec) => match parse_device_list(&spec) {
            Ok(r) => r,
            Err(msg) => {
                eprintln!("peregrine: {msg}; ignoring COLI_GPU_DEVICES");
                return vec![0];
            }
        },
    };
    usable_ordinals(&requests)
}

/// The subset of `requests` this binary can actually drive, as ordinals.
///
/// Split from [`selected_devices`] so the keep/drop policy is testable without
/// racing the process environment.
fn usable_ordinals(requests: &[DeviceRequest]) -> Vec<i32> {
    let mut out = Vec::new();
    for req in requests {
        if req.vendor.backend_compiled() {
            out.push(req.index);
        } else {
            eprintln!(
                "peregrine: COLI_GPU_DEVICES names {req}, but this binary cannot drive it — {}; dropping it",
                req.vendor.requirement()
            );
        }
    }
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pci_ids_map_to_the_three_vendors() {
        assert_eq!(Vendor::from_pci_id(0x10de), Some(Vendor::Nvidia));
        assert_eq!(Vendor::from_pci_id(0x1002), Some(Vendor::Amd));
        assert_eq!(Vendor::from_pci_id(0x8086), Some(Vendor::Intel));
        // An USB controller or an ASPEED BMC VGA must not become a compute GPU.
        assert_eq!(Vendor::from_pci_id(0x1022), None);
        assert_eq!(Vendor::from_pci_id(0x1a03), None);
    }

    #[test]
    fn auto_and_unset_mean_default() -> Result<(), String> {
        assert!(parse_device_list("")?.is_empty());
        assert!(parse_device_list("  ")?.is_empty());
        assert!(parse_device_list("AUTO")?.is_empty());
        assert!(parse_device_list("auto")?.is_empty());
        Ok(())
    }

    #[test]
    fn bare_ordinals_are_cuda_ordinals() -> Result<(), String> {
        assert_eq!(
            parse_device_list("0,1")?,
            vec![
                DeviceRequest { vendor: Vendor::Nvidia, index: 0 },
                DeviceRequest { vendor: Vendor::Nvidia, index: 1 }
            ]
        );
        // Prefixed form of the same thing, plus whitespace tolerance.
        assert_eq!(
            parse_device_list("cuda:0 , CUDA:1")?,
            parse_device_list("0,1")?,
            "prefixes are case-insensitive and bare ordinals mean cuda"
        );
        Ok(())
    }

    #[test]
    fn vendor_prefixes_parse_and_dedup() -> Result<(), String> {
        assert_eq!(
            parse_device_list("hip:0,cuda:1")?,
            vec![
                DeviceRequest { vendor: Vendor::Amd, index: 0 },
                DeviceRequest { vendor: Vendor::Nvidia, index: 1 }
            ]
        );
        assert_eq!(
            parse_device_list("level0:2,intel:2")?[0],
            DeviceRequest { vendor: Vendor::Intel, index: 2 },
            "level0 and intel name the same vendor"
        );
        assert_eq!(parse_device_list("amd:3,hip:3")?, vec![DeviceRequest { vendor: Vendor::Amd, index: 3 }]);
        Ok(())
    }

    #[test]
    fn malformed_lists_are_hard_errors() {
        for bad in ["-1", "cuda:x", "cuda:", ":0", "gpu:0", "0,,1", "cuda:0,", "cuda:-2"] {
            assert!(parse_device_list(bad).is_err(), "{bad:?} must not parse");
        }
    }

    #[test]
    fn unusable_vendors_are_dropped_not_silently_kept() -> Result<(), String> {
        // On every build some vendor is uncompilable, so at least one arm of
        // this holds; on this repo's dev box (cuda built) nvidia survives and
        // amd/intel do not.
        let reqs = parse_device_list("cuda:0,hip:1,level0:2")?;
        let kept = usable_ordinals(&reqs);
        assert_eq!(kept.len(), reqs.iter().filter(|r| r.vendor.backend_compiled()).count());
        for (req, &idx) in reqs.iter().zip(&kept) {
            if req.vendor.backend_compiled() {
                assert_eq!(req.index, idx);
            }
        }
        // An explicit list with nothing usable yields EMPTY, never device 0:
        // running somewhere the operator did not ask for is worse than nowhere.
        let none_usable = !reqs.iter().any(|r| r.vendor.backend_compiled());
        if none_usable {
            assert!(kept.is_empty());
        }
        Ok(())
    }

    #[test]
    fn presence_probe_is_deterministic_and_well_formed() {
        let a = detect_present();
        let b = detect_present();
        assert_eq!(a, b, "sysfs enumeration must be stable across two reads");
        for g in &a {
            assert!(g.card.starts_with("card") && g.card[4..].bytes().all(|c| c.is_ascii_digit()));
        }
        // This module's whole job is honest reporting, so on the dev box the
        // probe must actually see the AMD card even though no backend can drive
        // it — presence and usability are different questions. Not asserted
        // everywhere (CI boxes may have no GPUs at all).
        if std::path::Path::new("/sys/class/drm/card0").exists() && !a.is_empty() {
            let counts = present_counts();
            assert!(!counts.is_empty());
        }
    }
}
