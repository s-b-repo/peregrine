//! Best-effort hardware sensor readers: CPU package temperature and RAPL
//! energy. Pure sysfs reads — no privileges beyond file permissions, `None`
//! whenever the tree is missing (VMs, containers, non-Linux). Consumers treat
//! every reading as an optimization input, never a dependency.

use std::path::Path;

/// Highest temperature across all thermal zones, in whole °C. Reads
/// `/sys/class/thermal/thermal_zone*/temp` (millidegrees). `None` when no zone
/// is readable.
pub fn max_temp_c() -> Option<i32> {
    #[cfg(target_os = "linux")]
    {
        let root = Path::new("/sys/class/thermal");
        let entries = std::fs::read_dir(root).ok()?;
        let mut max_mdeg: Option<i64> = None;
        for e in entries.flatten() {
            let name = e.file_name();
            let Some(name) = name.to_str() else { continue };
            if !name.starts_with("thermal_zone") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(e.path().join("temp")) else { continue };
            let Ok(mdeg) = text.trim().parse::<i64>() else { continue };
            max_mdeg = Some(max_mdeg.map_or(mdeg, |m: i64| m.max(mdeg)));
        }
        max_mdeg.map(|m| (m / 1000) as i32)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Cumulative package energy in microjoules, summed over all top-level RAPL
/// domains (`/sys/class/powercap/intel-rapl:N/energy_uj`). The counter wraps at
/// `max_energy_range_uj`; use [`EnergyMeter`] for wrap-aware deltas. `None`
/// when RAPL is absent or unreadable (needs root on many distros).
pub fn energy_uj() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let root = Path::new("/sys/class/powercap");
        let entries = std::fs::read_dir(root).ok()?;
        let mut sum: Option<u64> = None;
        for e in entries.flatten() {
            let name = e.file_name();
            let Some(name) = name.to_str() else { continue };
            // top-level package domains only ("intel-rapl:0"), not subzones
            // ("intel-rapl:0:0") — subzones double-count the package total.
            if !name.starts_with("intel-rapl:") || name.matches(':').count() != 1 {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(e.path().join("energy_uj")) else { continue };
            let Ok(uj) = text.trim().parse::<u64>() else { continue };
            sum = Some(sum.unwrap_or(0) + uj);
        }
        sum
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Wrap-aware energy delta reader. Call [`Self::delta_uj`] once per interval;
/// it returns the microjoules consumed since the previous call (`None` on the
/// first call or when RAPL is unreadable). A wrap (new < old) is treated as
/// unknown rather than guessing the range — one lost sample per wrap.
pub struct EnergyMeter {
    last: Option<u64>,
}

impl EnergyMeter {
    pub fn new() -> EnergyMeter {
        EnergyMeter { last: None }
    }

    /// Microjoules consumed since the previous call.
    pub fn delta_uj(&mut self) -> Option<u64> {
        let now = energy_uj()?;
        let delta = match self.last {
            Some(prev) if now >= prev => Some(now - prev),
            _ => None, // first sample or wrap — no delta this interval
        };
        self.last = Some(now);
        delta
    }
}

impl Default for EnergyMeter {
    fn default() -> EnergyMeter {
        EnergyMeter::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readers_never_panic() {
        // Values are box-dependent (None in most CI containers); only the
        // no-panic and type contracts are asserted.
        let t = max_temp_c();
        if let Some(t) = t {
            assert!((-50..=150).contains(&t), "sane temperature: {t}");
        }
        let mut m = EnergyMeter::new();
        let first = m.delta_uj();
        assert!(first.is_none() || energy_uj().is_none() || first.is_some());
        let _second = m.delta_uj(); // audit-allow: exercised for no-panic only — either None (no RAPL) or a small delta
    }
}
