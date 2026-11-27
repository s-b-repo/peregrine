use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::ops::Range;
use std::os::unix::io::RawFd;

use parking_lot::{Mutex, MutexGuard};
use peregrine_core::safetensors::{layer_tensor, TensorRanges};
use peregrine_core::{Compression, Context, Dtype, Error, SafeTensors};
use peregrine_io::{OwnedReadReq, Reactor, ALIGN};

const MERGE_BYTES: usize = 16 << 20;

pub(crate) fn window_from_env() -> Result<usize, Error> {
    match std::env::var("COLI_LAYER_STREAM").as_deref().map(str::trim) {
        Err(std::env::VarError::NotPresent) | Ok("" | "0" | "false") => Ok(0),
        Ok("true") => Ok(2),
        Ok(s) => s.parse::<usize>().map_err(|e| {
            Error::Format(format!(
                "COLI_LAYER_STREAM expects a layer count or true/false ({e})"
            ))
        }),
        Err(e) => Err(Error::Format(format!("COLI_LAYER_STREAM: {e}"))),
    }
}

struct Extent {
    layer: usize,
    shard: usize,
    device: u8,
    fd: RawFd,
    offset: u64,
    len: usize,
    direct: bool,
}

pub(crate) struct LayerPlan {
    layers: Vec<Range<usize>>,
    extents: Vec<Extent>,
    window: usize,
    pub(crate) resident_bytes: u64,
    pub(crate) working_bytes: u64,
}

impl LayerPlan {
    pub(crate) fn new(
        st: &SafeTensors,
        n_layers: usize,
        window: usize,
        stream_experts: bool,
        direct: bool,
    ) -> Result<Self, Error> {
        let mut tensors = vec![Vec::new(); n_layers];
        let mut resident_bytes = 0u64;
        let decoded_bytes = |t: &peregrine_core::TensorInfo| match t.dtype {
            Dtype::U8 => t.uncompressed_nbytes as u64,
            _ => (t.numel as u64).saturating_mul(4),
        };
        for t in st.tensors() {
            if stream_experts && t.name.contains(".mlp.experts.") {
                continue;
            }
            match layer_tensor(&t.name) {
                Some((layer, _)) if layer < n_layers => tensors[layer].push(t),
                _ => resident_bytes = resident_bytes.saturating_add(decoded_bytes(t)),
            }
        }

        let mut layers = Vec::with_capacity(n_layers);
        let mut extents: Vec<Extent> = Vec::new();
        let mut staged_bytes = Vec::with_capacity(n_layers);
        let mut decode_peak = 0u64;
        for (layer, mut tensors) in tensors.into_iter().enumerate() {
            if tensors.is_empty() {
                return Err(Error::Format(format!(
                    "no tensors found for transformer layer {layer}"
                )));
            }
            tensors.sort_unstable_by_key(|t| (t.file_idx, t.off));
            let start = extents.len();
            let mut decoded = 0u64;
            let mut scratch = 0u64;
            for t in tensors {
                let len = usize::try_from(t.nbytes).map_err(|e| {
                    Error::Format(format!(
                        "tensor '{}' exceeds the address space ({e})",
                        t.name
                    ))
                })?;
                decoded = decoded.saturating_add(decoded_bytes(t));
                // Compressed tensors and layout conversion may need one
                // tensor-sized scratch allocation at a time.
                if t.compression != Compression::None || t.layout.is_some() {
                    scratch = scratch.max(t.uncompressed_nbytes as u64);
                }
                if len == 0 {
                    continue;
                }
                if let Some(last) = extents.last_mut() {
                    if last.layer == layer
                        && last.shard == t.file_idx
                        && last.offset + last.len as u64 == t.off
                        && last.len.saturating_add(len) <= MERGE_BYTES
                    {
                        last.len += len;
                        continue;
                    }
                }
                let direct_region = direct.then(|| st.region_direct(&t.name)).flatten();
                let (fd, offset, len) = direct_region
                    .or_else(|| st.region(&t.name))
                    .ok_or_else(|| Error::Format(format!("no shard region for '{}'", t.name)))?;
                extents.push(Extent {
                    layer,
                    shard: t.file_idx,
                    device: st.device_of(t.file_idx),
                    fd,
                    offset,
                    len,
                    direct: direct_region.is_some(),
                });
            }
            let range = start..extents.len();
            let staged = extents[range.clone()].iter().fold(0u64, |sum, r| {
                let padding = if r.direct { 2 * ALIGN as u64 } else { 0 };
                sum.saturating_add(r.len as u64).saturating_add(padding)
            });
            layers.push(range);
            staged_bytes.push(staged);
            decode_peak = decode_peak.max(decoded.saturating_add(scratch));
        }
        let window = window.clamp(1, n_layers.max(1));
        let staging_peak = staged_bytes
            .windows(window)
            .map(|layers| layers.iter().copied().fold(0u64, u64::saturating_add))
            .max()
            .unwrap_or(0);
        Ok(Self {
            layers,
            extents,
            window,
            resident_bytes,
            working_bytes: staging_peak.saturating_add(decode_peak),
        })
    }
}

pub(crate) struct LayerStream {
    plan: LayerPlan,
    ring: Mutex<Option<Reactor>>,
}

impl LayerStream {
    pub(crate) fn new(plan: LayerPlan, st: &SafeTensors) -> Self {
        let ring = match Reactor::new_streaming(128) {
            Ok(mut ring) => {
                #[cfg(target_os = "linux")]
                ring.set_force_async(true);
                if let Err(e) = ring.register_files(&st.shard_fds()) {
                    peregrine_io::note_advisory_err("register layer shard fds", &e);
                }
                let devices = plan
                    .extents
                    .iter()
                    .map(|r| r.device)
                    .collect::<BTreeSet<_>>()
                    .len();
                eprintln!(
                    "peregrine: [layer-io] io_uring, window={} layers, devices={devices}",
                    plan.window
                );
                Some(ring)
            }
            Err(e) => {
                eprintln!(
                    "peregrine: [layer-io] using positioned reads (io_uring unavailable: {e})"
                );
                None
            }
        };
        Self {
            plan,
            ring: Mutex::new(ring),
        }
    }

    pub(crate) fn begin(&self, start: usize) -> Result<LayerReads<'_>, Error> {
        let mut reads = LayerReads {
            plan: &self.plan,
            ring: self.ring.lock(),
            scheduled: start,
            pending: BTreeMap::new(),
        };
        reads.queue(self.plan.window)?;
        Ok(reads)
    }

    #[cfg(test)]
    pub(crate) fn uses_ring(&self) -> bool {
        self.ring.lock().is_some()
    }
}

struct PendingLayer {
    remaining: usize,
    ranges: TensorRanges,
}

/// Owns this model's I/O window for one pass. Concurrent callers wait for the
/// same ring so they cannot multiply the planned memory footprint.
pub(crate) struct LayerReads<'a> {
    plan: &'a LayerPlan,
    ring: MutexGuard<'a, Option<Reactor>>,
    scheduled: usize,
    pending: BTreeMap<usize, PendingLayer>,
}

impl LayerReads<'_> {
    pub(crate) fn queue(&mut self, count: usize) -> Result<(), Error> {
        let Some(ring) = self.ring.as_mut() else {
            return Ok(());
        };
        let end = self
            .scheduled
            .saturating_add(count)
            .min(self.plan.layers.len());
        if end == self.scheduled {
            return Ok(());
        }
        let mut devices = BTreeMap::<u8, VecDeque<_>>::new();
        for layer in self.scheduled..end {
            let ids = self.plan.layers[layer].clone();
            self.pending.insert(
                layer,
                PendingLayer {
                    remaining: ids.len(),
                    ranges: TensorRanges::new(),
                },
            );
            for id in ids {
                let r = &self.plan.extents[id];
                let req = OwnedReadReq {
                    fd: r.fd,
                    offset: r.offset,
                    len: r.len,
                    tag: id as u64,
                };
                devices
                    .entry(r.device)
                    .or_default()
                    .push_back((req, r.direct));
            }
        }
        // Rotate at submission time so the first SQEs reach every active drive
        // even when one layer supplies more extents than fit in the ring.
        let mut devices: VecDeque<_> = devices.into_values().collect();
        let requests = std::iter::from_fn(|| {
            let mut drive = devices.pop_front()?;
            let next = drive.pop_front();
            if !drive.is_empty() {
                devices.push_back(drive);
            }
            next
        });
        ring.submit_owned_mixed(requests)
            .ctx(|| format!("submit SSD reads for layers {}..{end}", self.scheduled))?;
        self.scheduled = end;
        Ok(())
    }

    pub(crate) fn take(&mut self, layer: usize) -> Result<Option<TensorRanges>, Error> {
        let Some(ring) = self.ring.as_mut() else {
            return Ok(None);
        };
        if !self.pending.contains_key(&layer) {
            return Err(Error::Format(format!(
                "layer {layer} is outside the SSD read window"
            )));
        }
        let mut done = Vec::new();
        while self.pending[&layer].remaining != 0 {
            if ring.pending_owned() == 0 {
                return Err(Error::Format(format!(
                    "SSD read window ended before layer {layer} completed"
                )));
            }
            ring.reap_some(&mut done)
                .ctx(|| format!("read SSD window at layer {layer}"))?;
            for result in done.drain(..) {
                let r = &self.plan.extents[result.tag as usize];
                let slot = self.pending.get_mut(&r.layer).ok_or_else(|| {
                    Error::Format(format!("unexpected SSD completion for layer {}", r.layer))
                })?;
                slot.ranges.insert((r.shard, r.offset), result.bytes);
                slot.remaining -= 1;
            }
        }
        Ok(self.pending.remove(&layer).map(|slot| slot.ranges))
    }
}

impl Drop for LayerReads<'_> {
    fn drop(&mut self) {
        if let Some(ring) = self.ring.as_mut() {
            if ring.pending_owned() > 0 {
                if let Err(e) = ring.discard_owned() {
                    eprintln!("peregrine: [layer-io] switching to positioned reads after ring failure ({e})");
                    *self.ring = None;
                }
            }
        }
    }
}
