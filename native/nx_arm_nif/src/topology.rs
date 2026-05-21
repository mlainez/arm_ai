//! big.LITTLE CPU topology detection + perf-cluster pinning.
//!
//! On heterogeneous ARM chips (Snapdragon big.LITTLE, Tegra, Apple,
//! Cortex-X/A7x/A5x mixes), spreading parallel work across every
//! logical core hurts: the LITTLE cores have narrower NEON pipes
//! and run hot loops 1.5–2× slower than the big cores. Worse, the
//! kernel's load balancer keeps migrating threads off the big
//! cluster onto idle LITTLE cores.
//!
//! At NIF load time we read `/sys/devices/system/cpu/cpu*/...`,
//! identify the perf cluster, and (a) pin rayon worker threads to
//! it via `sched_setaffinity`, (b) size the pool to the perf
//! cluster's core count.
//!
//! Detection priority:
//!   1. `cpu_capacity` (kernel DT-derived, perf cluster = highest)
//!   2. `cpufreq/cpuinfo_max_freq` (highest max-freq cluster wins)
//!   3. `regs/identification/midr_el1` partnum table (A73 > A55 > A53,
//!      Kryo Gold > Kryo Silver, etc.)
//!   4. fall back: use all cores (no pinning)
//!
//! Caller can override via `init_thread_pool_op(n)` which always
//! builds with the explicit count and skips affinity.

use std::collections::BTreeMap;
use std::fs;

#[derive(Debug, Clone)]
pub struct Topology {
    /// CPU IDs belonging to the highest-capacity cluster.
    pub perf_cores: Vec<usize>,
    /// All known CPU IDs, lowest-first.
    pub all_cores: Vec<usize>,
    /// How we figured it out (for logging/diagnostics).
    pub source: &'static str,
}

impl Topology {
    /// Returns the topology with the perf cluster identified, or a
    /// fallback "all cores" topology if none of the heuristics fire.
    pub fn detect() -> Self {
        let all_cores = enumerate_cpus();

        if let Some(t) = detect_by_capacity(&all_cores) {
            return t;
        }
        if let Some(t) = detect_by_max_freq(&all_cores) {
            return t;
        }
        if let Some(t) = detect_by_midr(&all_cores) {
            return t;
        }

        Topology {
            perf_cores: all_cores.clone(),
            all_cores,
            source: "fallback (homogeneous or detection failed)",
        }
    }
}

fn enumerate_cpus() -> Vec<usize> {
    let mut out = Vec::new();
    if let Ok(entries) = fs::read_dir("/sys/devices/system/cpu") {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let s = name.to_string_lossy();
            if let Some(rest) = s.strip_prefix("cpu") {
                if let Ok(id) = rest.parse::<usize>() {
                    out.push(id);
                }
            }
        }
    }
    out.sort_unstable();
    out
}

fn read_u64(path: &str) -> Option<u64> {
    fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
}

fn read_hex_u64(path: &str) -> Option<u64> {
    let s = fs::read_to_string(path).ok()?;
    let s = s.trim();
    let s = s.strip_prefix("0x").unwrap_or(s);
    u64::from_str_radix(s, 16).ok()
}

fn group_max(values: &BTreeMap<usize, u64>) -> Option<Vec<usize>> {
    let max = values.values().copied().max()?;
    if values.values().all(|&v| v == max) {
        // Homogeneous: every CPU has the same value, no point pinning.
        return None;
    }
    Some(values.iter().filter(|(_, &v)| v == max).map(|(&k, _)| k).collect())
}

fn detect_by_capacity(cpus: &[usize]) -> Option<Topology> {
    let mut caps = BTreeMap::new();
    for &id in cpus {
        let v = read_u64(&format!("/sys/devices/system/cpu/cpu{}/cpu_capacity", id))?;
        caps.insert(id, v);
    }
    let perf = group_max(&caps)?;
    Some(Topology {
        perf_cores: perf,
        all_cores: cpus.to_vec(),
        source: "cpu_capacity",
    })
}

fn detect_by_max_freq(cpus: &[usize]) -> Option<Topology> {
    let mut freqs = BTreeMap::new();
    for &id in cpus {
        let v = read_u64(&format!(
            "/sys/devices/system/cpu/cpu{}/cpufreq/cpuinfo_max_freq",
            id
        ))?;
        freqs.insert(id, v);
    }
    let perf = group_max(&freqs)?;
    Some(Topology {
        perf_cores: perf,
        all_cores: cpus.to_vec(),
        source: "cpufreq.cpuinfo_max_freq",
    })
}

/// MIDR partnum → relative performance score. Higher is better.
/// Covers the common ARM standard cores + Qualcomm Kryo derivatives
/// we'd actually see on Linux SBCs and phones.
fn part_score(midr: u64) -> u64 {
    // PartNum is bits 15:4, Implementer is bits 31:24.
    let implementer = (midr >> 24) & 0xFF;
    let partnum = (midr >> 4) & 0xFFF;

    match (implementer, partnum) {
        // ARM Ltd. (0x41) standard cores. Score = approximate IPC × MHz peer
        // class, ordered conservatively. Newer/wider = higher.
        (0x41, 0xd4d) => 90, // Cortex-A715
        (0x41, 0xd4b) => 88, // Cortex-A510 — wider than A55
        (0x41, 0xd47) => 85, // Cortex-A710
        (0x41, 0xd46) => 82, // Cortex-A510 v2
        (0x41, 0xd44) => 95, // Cortex-X1
        (0x41, 0xd0e) => 80, // Cortex-A76AE
        (0x41, 0xd0d) => 78, // Cortex-A77
        (0x41, 0xd0c) => 75, // Cortex-A76
        (0x41, 0xd0b) => 72, // Cortex-A76 (variant)
        (0x41, 0xd0a) => 70, // Cortex-A75
        (0x41, 0xd09) => 65, // Cortex-A73 (FP3 perf cluster!)
        (0x41, 0xd08) => 60, // Cortex-A72
        (0x41, 0xd07) => 55, // Cortex-A57
        (0x41, 0xd05) => 40, // Cortex-A55
        (0x41, 0xd04) => 35, // Cortex-A35
        (0x41, 0xd03) => 30, // Cortex-A53 (FP3 LITTLE cluster!)
        (0x41, 0xc09) => 25, // Cortex-A9
        (0x41, 0xc07) => 20, // Cortex-A7
        // Qualcomm (0x51) Kryo silicon. Kryo Gold = A7x-class, Silver = A5x-class.
        (0x51, 0x800) => 65, // Kryo 260/385 Gold (A73 derivative)
        (0x51, 0x801) => 30, // Kryo 260/385 Silver (A53 derivative)
        (0x51, 0x802) => 75, // Kryo 485 Gold (A76 derivative)
        (0x51, 0x803) => 40, // Kryo 485 Silver (A55 derivative)
        (0x51, 0x804) => 80, // Kryo Gold (A77 derivative)
        (0x51, 0x805) => 42, // Kryo Silver (A55 v2)
        // Apple (0x61) — every cluster Apple ships counts as perf for our purposes.
        (0x61, _) => 90,
        // Unknown / generic.
        _ => 50,
    }
}

fn detect_by_midr(cpus: &[usize]) -> Option<Topology> {
    let mut scores = BTreeMap::new();
    for &id in cpus {
        let v = read_hex_u64(&format!(
            "/sys/devices/system/cpu/cpu{}/regs/identification/midr_el1",
            id
        ))?;
        scores.insert(id, part_score(v));
    }
    let perf = group_max(&scores)?;
    Some(Topology {
        perf_cores: perf,
        all_cores: cpus.to_vec(),
        source: "midr_el1 part-number table",
    })
}

/// Pin the *current* thread to `cores` via sched_setaffinity. No-op
/// on non-Linux. Returns Ok even if the syscall fails — best effort.
pub fn pin_current_thread_to(cores: &[usize]) {
    #[cfg(target_os = "linux")]
    {
        use std::mem;
        const CPU_SETSIZE: usize = 1024;
        const SET_BYTES: usize = CPU_SETSIZE / 8;
        let mut set = [0u8; SET_BYTES];
        for &id in cores {
            if id < CPU_SETSIZE {
                set[id / 8] |= 1 << (id % 8);
            }
        }

        // sched_setaffinity(pid=0 for current, cpusetsize, cpu_set_t*)
        unsafe {
            libc_sched_setaffinity(0, mem::size_of_val(&set), set.as_ptr());
        }
    }
    let _ = cores;
}

// ---------------------------------------------------------------
// On-demand per-thread perf-cluster pinning for BEAM dirty
// schedulers.
//
// rayon's pool we pin at build time. But the BEAM thread that calls
// into our NIF (a "dirty CPU scheduler") is owned by the Erlang VM
// and the kernel scheduler may park it on any core, including
// LITTLE ones. The synchronous half of every NIF (binary unpacking,
// shape arithmetic, OwnedBinary alloc) runs on whichever core that
// thread happens to be on. On big.LITTLE chips that's the main
// source of dispatch latency variance.
//
// We fix it by pinning the calling thread to the perf cluster the
// first time it enters our code. A thread_local flag avoids
// re-pinning on every NIF call.
// ---------------------------------------------------------------

use std::sync::OnceLock;
static PERF_CORES: OnceLock<Vec<usize>> = OnceLock::new();

pub fn set_perf_cluster_cache(cores: Vec<usize>) {
    let _ = PERF_CORES.set(cores);
}

thread_local! {
    static THREAD_PINNED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Pin the calling thread to the perf cluster (if known). Call at
/// the top of any hot NIF. After the first hit on a given OS thread
/// this is just a thread-local load + branch.
pub fn ensure_thread_pinned() {
    THREAD_PINNED.with(|f| {
        if !f.get() {
            if let Some(cores) = PERF_CORES.get() {
                if !cores.is_empty() {
                    pin_current_thread_to(cores);
                }
            }
            f.set(true);
        }
    });
}

#[cfg(target_os = "linux")]
extern "C" {
    #[link_name = "sched_setaffinity"]
    fn libc_sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u8) -> i32;
}
