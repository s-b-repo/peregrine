//! Device I/O geometry — the granularity a read is actually served at.
//!
//! The engine aligns O_DIRECT reads to [`crate::slab::ALIGN`] (4096), which is
//! what the *syscall* requires. That is not the granularity the **drive**
//! serves at, and the two are usually different by one to two orders of
//! magnitude: NVMe parts commonly report a 128 KB optimal transfer while their
//! logical block is 512 B or 4 KB.
//!
//! The distinction matters because an expert is read through **six** regions,
//! so a region whose start is arbitrary pays a straddled unit at each end, six
//! times per expert per routing. Aligning region starts in the offline layout
//! trades a little disk for strictly fewer units touched — but only above 4 KB,
//! because below that the existing alignment already covers it.
//!
//! # Why not the host page size
//!
//! The first version of this idea was "remap tensors to 2 MB hugepages at the
//! filesystem layer". That targets TLB pressure, and TLB pressure is not in
//! this engine's read path: nothing `mmap`s the checkpoint, reads go through
//! io_uring or `pread` into owned buffers. The quantity worth protecting is
//! **queue depth** — a read that straddles the drive's transfer unit becomes
//! two device operations, and the second one is what stalls the pipeline. So
//! the constant is a *device* property to be probed, not a host constant to be
//! chosen.

use std::path::Path;

/// Fallback when the device cannot be probed.
///
/// 128 KB is the commonly reported NVMe optimal transfer size. It is a
/// **guess** and is labelled as one by [`Geometry::probed`]: acting on an
/// unprobed constant is how the host-page-size version of this idea went wrong.
pub const ASSUMED_OPTIMAL: u64 = 128 * 1024;

/// What granularity reads to this device should be aligned to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    /// The drive's preferred transfer size in bytes.
    pub optimal: u64,
    /// Smallest addressable unit — what O_DIRECT alignment already satisfies.
    pub minimum: u64,
    /// `false` when [`ASSUMED_OPTIMAL`] was substituted because nothing could
    /// be read from sysfs. Every consumer must report this rather than quietly
    /// treating a guess as a measurement.
    pub probed: bool,
}

impl Default for Geometry {
    fn default() -> Geometry {
        Geometry { optimal: ASSUMED_OPTIMAL, minimum: crate::slab::ALIGN as u64, probed: false }
    }
}

impl Geometry {
    /// Whether aligning to [`Self::optimal`] could buy anything the existing
    /// 4096-byte O_DIRECT alignment does not already provide.
    ///
    /// If the drive's optimal transfer is at or below the alignment the engine
    /// already uses, the entire idea is closed for this device — and that is a
    /// result, not a failure to probe.
    pub fn worth_aligning(&self) -> bool {
        self.optimal > crate::slab::ALIGN as u64
    }

    /// Units of [`Self::optimal`] a read of `len` bytes at `offset` touches.
    ///
    /// This is the quantity alignment reduces: a region that starts mid-unit
    /// pays for the partial unit at each end.
    pub fn units_touched(&self, offset: u64, len: u64) -> u64 {
        if self.optimal == 0 || len == 0 {
            return 0;
        }
        let first = offset / self.optimal;
        // `offset + len - 1` is the last byte, not one past the end: using the
        // exclusive end would count an extra unit for a read that lands exactly
        // on a boundary, inflating the "before" side of every comparison this
        // module exists to make.
        let last = (offset + len - 1) / self.optimal;
        last - first + 1
    }

    /// Units touched if the region started on a boundary — the floor, and the
    /// best any alignment scheme can do.
    pub fn units_aligned(&self, len: u64) -> u64 {
        if self.optimal == 0 {
            return 0;
        }
        len.div_ceil(self.optimal)
    }

    /// Padding bytes needed to push `offset` up to the next boundary.
    pub fn pad_to_boundary(&self, offset: u64) -> u64 {
        if self.optimal == 0 {
            return 0;
        }
        let r = offset % self.optimal;
        if r == 0 {
            0
        } else {
            self.optimal - r
        }
    }
}

/// What aligning a set of regions would cost and save.
///
/// The item this answers is deliberately *sizing*, not implementation: padding
/// region starts inflates the container, which is the one quantity the
/// workload-reduction work exists to shrink, so the trade has to be priced
/// before the layout writer is touched.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AlignCost {
    pub regions: u64,
    /// Device units the regions touch as they are laid out today.
    pub units_now: u64,
    /// Device units they would touch with every start on a boundary — the
    /// floor, since a region can never touch fewer units than its own length
    /// requires.
    pub units_aligned: u64,
    /// Padding bytes the aligned layout would add to the container.
    pub pad_bytes: u64,
    /// Total region bytes, for expressing the padding as a share.
    pub data_bytes: u64,
}

impl AlignCost {
    /// Share of device units alignment would remove. `0.0` means the layout is
    /// already effectively aligned and there is nothing to buy.
    pub fn units_saved(&self) -> f64 {
        if self.units_now == 0 {
            return 0.0;
        }
        (self.units_now - self.units_aligned.min(self.units_now)) as f64 / self.units_now as f64
    }

    /// Container inflation the padding would cost, as a share of region bytes.
    pub fn disk_cost(&self) -> f64 {
        if self.data_bytes == 0 {
            0.0
        } else {
            self.pad_bytes as f64 / self.data_bytes as f64
        }
    }

    /// The verdict. Blunt in the negative case, because the negative case is
    /// the likely one: an expert region is far larger than any device transfer
    /// unit, so the straddled unit at each end is a small share of a big read.
    pub fn verdict(&self, g: &Geometry) -> String {
        let mut s = format!(
            "[align] {} regions, unit={} KB{}\n\
             [align] device units: {} now -> {} aligned ({:.2}% fewer)\n\
             [align] padding: {:.2} MB ({:.2}% container inflation)\n",
            self.regions,
            g.optimal / 1024,
            if g.probed { " (probed)" } else { " (ASSUMED — sysfs said nothing)" },
            self.units_now,
            self.units_aligned,
            100.0 * self.units_saved(),
            self.pad_bytes as f64 / 1e6,
            100.0 * self.disk_cost(),
        );
        if !g.worth_aligning() {
            s.push_str(
                "[align] VERDICT: closed for this device. Its optimal transfer is no coarser than \
                 the 4096-byte alignment O_DIRECT already uses, so there is nothing above the \
                 existing alignment to win.\n",
            );
            return s;
        }
        // Compare the two shares directly. Fewer units is only worth buying if
        // it beats what the padding costs, and on large regions it usually
        // will not — which is the answer, not a failed measurement.
        let (saved, cost) = (self.units_saved(), self.disk_cost());
        let mean_region = if self.regions > 0 { self.data_bytes / self.regions } else { 0 };
        // Two different situations both produce "no units saved", and the
        // explanation is opposite in each. Regions much LARGER than the unit
        // barely straddle; regions much SMALLER than it already fit inside one
        // and padding them to a boundary is ruinous. The first version of this
        // printed the large-region explanation for both, which on a fixture of
        // 53-byte tensors read as reassurance beside a 242,000 % inflation
        // figure. Caught by running it.
        if mean_region < g.optimal {
            s.push_str(&format!(
                "[align] VERDICT: nothing to win, and alignment would be ruinous here. The mean \
                 region is {mean_region} B against a {} KB unit, so regions already fit inside one \
                 unit and padding each to a boundary inflates the container {:.0}x. This shape is \
                 what a fixture or a heavily sharded container looks like, not a production \
                 expert layout.\n",
                g.optimal / 1024,
                self.disk_cost()
            ));
            return s;
        }
        s.push_str(if saved <= 0.005 {
            "[align] VERDICT: nothing to win. Regions are large relative to the device unit, so \
             the straddled unit at each end is already noise. Do not touch the layout writer.\n"
        } else if saved <= cost {
            "[align] VERDICT: priced out. The container grows by at least as much as the unit \
             count falls, and container bytes are the quantity the whole workload-reduction \
             effort exists to reduce.\n"
        } else {
            "[align] VERDICT: worth measuring for real. Fewer units AND a smaller share spent on \
             padding. Next step is still not the writer: confirm with `iobench` on the real \
             shards with a cold cache, since a unit count is not a latency.\n"
        });
        s
    }
}

/// Price aligning `regions` (`(offset, len)` pairs, in layout order).
///
/// The aligned arm re-lays the regions end to end with each start padded up to
/// a boundary, which is what a layout writer would actually do — comparing
/// against a hypothetical where every region is independently aligned in place
/// would understate the padding by ignoring that they share a file.
pub fn align_cost(g: &Geometry, regions: &[(u64, u64)]) -> AlignCost {
    let mut c = AlignCost { regions: regions.len() as u64, ..AlignCost::default() };
    let mut cursor = 0u64;
    for &(off, len) in regions {
        if len == 0 {
            continue;
        }
        c.data_bytes += len;
        c.units_now += g.units_touched(off, len);
        let pad = g.pad_to_boundary(cursor);
        c.pad_bytes += pad;
        cursor = cursor.saturating_add(pad).saturating_add(len);
        c.units_aligned += g.units_aligned(len);
    }
    c
}

/// Read one unsigned integer out of a sysfs queue attribute.
fn sysfs_u64(dev: &str, attr: &str) -> Option<u64> {
    let path = format!("/sys/block/{dev}/queue/{attr}");
    let text = std::fs::read_to_string(path).ok()?;
    text.trim().parse::<u64>().ok().filter(|v| *v > 0)
}

/// Strip a partition suffix to get the parent block device.
///
/// `nvme0n1p3` → `nvme0n1`, `sda2` → `sda`. Queue attributes live on the parent
/// device, so probing the partition name finds nothing and silently falls back
/// to the assumed constant — which would look identical to a device that
/// genuinely reports nothing.
fn parent_device(name: &str) -> String {
    // NVMe: the partition suffix is `p<N>` after the namespace.
    if let Some(idx) = name.rfind('p') {
        let (head, tail) = name.split_at(idx);
        if head.starts_with("nvme") && tail.len() > 1 && tail[1..].chars().all(|c| c.is_ascii_digit()) {
            return head.to_string();
        }
    }
    // SCSI/SATA/virtio: trailing digits are the partition — but ONLY for these
    // families. Stripping trailing digits generally is wrong and quietly so:
    // `dm-0` would become `dm-` and `md0` (a RAID array, not a partition of
    // "md") would become `md`, and both would then probe nothing and fall back
    // to the assumed constant, which is indistinguishable from a device that
    // advertises none. An allowlist fails in the safe direction: an unknown
    // name is left alone, and a name we cannot resolve reports `probed = false`.
    let trimmed = name.trim_end_matches(|c: char| c.is_ascii_digit());
    // `<family><drive-letters>`: sda, sdaa, vdb, xvdc. Anything whose stem is
    // not one of these families keeps its digits.
    let is_partitionable = ["xvd", "sd", "hd", "vd"].iter().any(|fam| {
        trimmed
            .strip_prefix(fam)
            .is_some_and(|rest| !rest.is_empty() && rest.chars().all(|c| c.is_ascii_lowercase()))
    });
    if trimmed != name && is_partitionable {
        return trimmed.to_string();
    }
    name.to_string()
}

/// Probe the block device backing `path`.
///
/// Returns a [`Geometry`] with `probed = false` when the device cannot be
/// identified or reports nothing usable — the caller is expected to say so
/// rather than quote the fallback as a measurement.
#[cfg(target_os = "linux")]
pub fn probe(path: &Path) -> Geometry {
    let mut g = Geometry::default();
    let Some(dev) = device_name(path) else { return g };
    let dev = parent_device(&dev);
    // `optimal_io_size` is the drive's preferred transfer; it is legitimately 0
    // on devices that decline to advertise one, which `sysfs_u64` filters out.
    let optimal = sysfs_u64(&dev, "optimal_io_size");
    let minimum = sysfs_u64(&dev, "minimum_io_size")
        .or_else(|| sysfs_u64(&dev, "physical_block_size"))
        .or_else(|| sysfs_u64(&dev, "logical_block_size"));
    if let Some(m) = minimum {
        g.minimum = m;
    }
    match optimal {
        Some(o) => {
            g.optimal = o;
            g.probed = true;
        }
        None => {
            // A device that advertises no optimal transfer is not the same as
            // an unprobeable one, but both leave the constant a guess, and the
            // consumer's obligation is identical.
            g.probed = false;
        }
    }
    g
}

#[cfg(not(target_os = "linux"))]
pub fn probe(_path: &Path) -> Geometry {
    Geometry::default()
}

/// Block device name (e.g. `nvme0n1p2`) backing `path`, via its `st_dev`.
#[cfg(target_os = "linux")]
fn device_name(path: &Path) -> Option<String> {
    use std::os::unix::fs::MetadataExt;
    let md = std::fs::metadata(path).ok()?;
    let dev = md.dev();
    // Kernel encoding: sysfs exposes `<major>:<minor>` under /sys/dev/block.
    let (major, minor) = (libc::major(dev), libc::minor(dev));
    let link = std::fs::read_link(format!("/sys/dev/block/{major}:{minor}")).ok()?;
    link.file_name().and_then(|n| n.to_str()).map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_straddling_read_touches_more_units_than_an_aligned_one() {
        // The entire premise, stated as arithmetic: the same length costs more
        // when it starts mid-unit. If this were not true there would be nothing
        // for the layout writer to buy.
        let g = Geometry { optimal: 128 * 1024, minimum: 4096, probed: true };
        let len = 128 * 1024;
        assert_eq!(g.units_touched(0, len), 1, "an aligned read is one unit");
        assert_eq!(g.units_touched(4096, len), 2, "a straddling read is two");
        assert_eq!(g.units_aligned(len), 1, "the floor is what alignment reaches");
    }

    #[test]
    fn a_read_ending_exactly_on_a_boundary_is_not_charged_an_extra_unit() {
        // Using the exclusive end would inflate the "before" side of every
        // comparison, which is the direction that would make alignment look
        // better than it is.
        let g = Geometry { optimal: 1024, minimum: 512, probed: true };
        assert_eq!(g.units_touched(0, 1024), 1);
        assert_eq!(g.units_touched(0, 1025), 2);
        assert_eq!(g.units_touched(1024, 1024), 1);
        assert_eq!(g.units_touched(0, 0), 0, "a zero-length read touches nothing");
    }

    #[test]
    fn padding_reaches_the_next_boundary_and_no_further() {
        let g = Geometry { optimal: 4096, minimum: 512, probed: true };
        assert_eq!(g.pad_to_boundary(0), 0, "an aligned offset needs no padding");
        assert_eq!(g.pad_to_boundary(1), 4095);
        assert_eq!(g.pad_to_boundary(4096), 0);
        assert_eq!(g.pad_to_boundary(4097), 4095);
    }

    #[test]
    fn a_device_no_coarser_than_the_existing_alignment_closes_the_idea() {
        // The negative result this module has to be able to produce: if the
        // drive's optimal transfer is at or below the 4096 O_DIRECT already
        // uses, there is nothing to win and the layout writer must not be
        // touched. That has to be as easy to read as a positive.
        let small = Geometry { optimal: 4096, minimum: 4096, probed: true };
        assert!(!small.worth_aligning());
        let big = Geometry { optimal: 128 * 1024, minimum: 4096, probed: true };
        assert!(big.worth_aligning());
    }

    #[test]
    fn a_real_sized_expert_region_still_straddles_enough_to_matter() {
        // I expected this to come back "pointless" and wrote the test asserting
        // it. It does not, and the arithmetic is why: a ~3 MB region spans ~23
        // device units, and starting mid-unit adds one — about 4 % more units
        // for about 0.5 % more disk. Small, but the padding is smaller still,
        // so the model calls it worth measuring for real rather than closing it.
        //
        // Recorded as a corrected expectation rather than tuned away: the point
        // of pricing the trade was to find out, and "my prior was wrong" is the
        // result.
        let g = Geometry { optimal: 128 * 1024, minimum: 4096, probed: true };
        let regions: Vec<(u64, u64)> = (0..600).map(|i| (i * 3_000_000 + 4096, 3_000_000)).collect();
        let c = align_cost(&g, &regions);
        assert!(c.units_now > c.units_aligned, "a straddling layout must cost more units");
        assert!(
            (0.02..0.10).contains(&c.units_saved()),
            "expected a few percent of units, got {}",
            c.units_saved()
        );
        assert!(c.disk_cost() < c.units_saved(), "the padding must be the cheaper side here");
        assert!(c.verdict(&g).contains("worth measuring for real"), "{}", c.verdict(&g));
    }

    #[test]
    fn small_regions_are_where_the_saving_would_be() {
        // The other side, so the tool is not merely always-positive: regions
        // near the unit size straddle badly, and the saving is a large share.
        let g = Geometry { optimal: 128 * 1024, minimum: 4096, probed: true };
        let regions: Vec<(u64, u64)> = (0..600).map(|i| (i * 140_000 + 64 * 1024, 130_000)).collect();
        let c = align_cost(&g, &regions);
        assert_eq!(c.units_aligned, 600, "each region fits in one unit when aligned");
        // 1197, not 1200: three of the 600 starts happen to land on a unit
        // boundary and cost only one. Asserting the exact multiple would be
        // asserting a coincidence of the fixture rather than the effect.
        assert!(c.units_now > 600, "most regions must straddle as laid out: {}", c.units_now);
        assert!(c.units_saved() > 0.3, "expected a large saving, got {}", c.units_saved());
    }

    #[test]
    fn a_layout_already_on_boundaries_has_nothing_to_win() {
        // The genuine no-op case: every region already starts on a unit.
        let g = Geometry { optimal: 128 * 1024, minimum: 4096, probed: true };
        let regions: Vec<(u64, u64)> = (0..64).map(|i| (i * 128 * 1024, 128 * 1024)).collect();
        let c = align_cost(&g, &regions);
        assert_eq!(c.units_now, c.units_aligned, "an aligned layout is its own floor");
        assert_eq!(c.pad_bytes, 0);
        assert!(c.verdict(&g).contains("nothing to win"), "{}", c.verdict(&g));
    }

    #[test]
    fn regions_smaller_than_the_unit_get_the_opposite_explanation() {
        // Both this and the large-region case report "no units saved", and the
        // reason is opposite. Printing the large-region explanation here — as
        // the first version did — reads as reassurance next to an inflation
        // figure in the thousands of percent.
        let g = Geometry { optimal: 128 * 1024, minimum: 4096, probed: true };
        let regions: Vec<(u64, u64)> = (0..72).map(|i| (i * 64, 53)).collect();
        let c = align_cost(&g, &regions);
        let v = c.verdict(&g);
        assert!(v.contains("ruinous"), "{v}");
        assert!(!v.contains("Regions are large"), "the wrong explanation must not appear: {v}");
        assert!(c.disk_cost() > 100.0, "padding dwarfs the data here: {}", c.disk_cost());
    }

    #[test]
    fn an_unprobed_geometry_is_labelled_in_the_verdict() {
        // Quoting the assumed constant as if it were the drive's is the exact
        // failure the host-page-size version of this idea made.
        let g = Geometry::default();
        let c = align_cost(&g, &[(4096, 1_000_000)]);
        assert!(c.verdict(&g).contains("ASSUMED"), "{}", c.verdict(&g));
        let probed = Geometry { probed: true, ..Geometry::default() };
        assert!(c.verdict(&probed).contains("(probed)"), "{}", c.verdict(&probed));
    }

    #[test]
    fn a_device_that_cannot_benefit_short_circuits_the_verdict() {
        let g = Geometry { optimal: 4096, minimum: 4096, probed: true };
        let c = align_cost(&g, &[(1234, 1_000_000)]);
        assert!(c.verdict(&g).contains("closed for this device"), "{}", c.verdict(&g));
    }

    #[test]
    fn zero_length_regions_do_not_inflate_the_count() {
        let g = Geometry { optimal: 4096, minimum: 512, probed: true };
        let c = align_cost(&g, &[(0, 0), (0, 4096)]);
        assert_eq!(c.units_now, 1, "an empty region touches nothing");
        assert_eq!(c.data_bytes, 4096);
    }

    #[test]
    fn the_fallback_is_marked_as_a_guess() {
        // Acting on an unprobed constant is exactly how the host-page-size
        // version of this idea went wrong, so the default must never look like
        // a measurement.
        let d = Geometry::default();
        assert!(!d.probed, "the default is an assumption, not a probe");
        assert_eq!(d.optimal, ASSUMED_OPTIMAL);
        assert_eq!(d.minimum, crate::slab::ALIGN as u64);
    }

    #[test]
    fn partition_names_resolve_to_their_parent_device() {
        // Queue attributes live on the parent. Probing the partition finds
        // nothing and falls back silently, which is indistinguishable from a
        // device that genuinely advertises nothing.
        assert_eq!(parent_device("nvme0n1p3"), "nvme0n1");
        assert_eq!(parent_device("nvme0n1"), "nvme0n1");
        assert_eq!(parent_device("sda2"), "sda");
        assert_eq!(parent_device("sda"), "sda");
        assert_eq!(parent_device("vda1"), "vda");
        assert_eq!(parent_device("xvdb2"), "xvdb");
        // Trailing digits are NOT always a partition, and getting this wrong
        // fails silently: the mangled name probes nothing and falls back to the
        // assumed constant, which looks exactly like a device that advertises
        // none. Both of these are whole devices.
        assert_eq!(parent_device("dm-0"), "dm-0", "device-mapper names end in a digit");
        assert_eq!(parent_device("md0"), "md0", "md0 is a RAID array, not a partition of `md`");
        assert_eq!(parent_device("loop3"), "loop3");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn probing_a_real_path_returns_something_usable() {
        // Whatever this box reports, the result has to be self-consistent:
        // a positive granularity and a minimum that does not exceed it.
        let g = probe(std::path::Path::new("."));
        assert!(g.optimal > 0, "a zero granularity would divide by zero downstream");
        assert!(g.minimum > 0);
        if g.probed {
            assert!(
                g.optimal >= g.minimum,
                "optimal {} below minimum {} is not a coherent geometry",
                g.optimal,
                g.minimum
            );
        }
    }
}
