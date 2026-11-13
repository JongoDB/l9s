"""Hardware detection for l9s.

Answers the questions the rest of l9s needs, on any Linux box that can run vLLM:
  * which accelerator vendor (NVIDIA / AMD / none) and CPU architecture (amd64 / arm64)
  * which GPUs, and whether memory is unified with the CPU (GB10 / DGX Spark, Jetson, AMD APUs)
    or dedicated VRAM per GPU (RTX, A/H/B/L-series, Instinct)
  * the memory pool vLLM's --gpu-memory-utilization applies to (system RAM if unified, else per-GPU VRAM)
  * how Docker gets access to the GPUs, and a sensible default vLLM image
  * live GPU stats for the TUI

`python hw.py` prints a summary; `python hw.py env` prints KEY=value defaults for ~/.l9s/env.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field

UNIFIED_NAMES = re.compile(r"GB10|Thor|Orin|Xavier|MI300A|Radeon 8060S|Strix Halo", re.I)
NVIDIA_IMAGE = "vllm/vllm-openai:v0.33.0"
AMD_IMAGE = "rocm/vllm:latest"  # experimental: set IMAGE in ~/.l9s/env to the ROCm vLLM image you use


def _run(cmd: list[str], timeout: float = 10) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""


def _num(x: str) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def system_mem() -> int:
    if platform.system() == "Darwin":
        return int(_num(_run(["sysctl", "-n", "hw.memsize"])) or 0)
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


@dataclass
class GPU:
    index: int
    name: str
    mem_total: int | None  # bytes of dedicated memory; None = unified / not reported


@dataclass
class Hardware:
    vendor: str                    # "nvidia" | "amd" | "apple" | "none"
    arch: str                      # "amd64" | "arm64" | other
    os: str = "linux"              # "linux" | "darwin"
    gpus: list[GPU] = field(default_factory=list)
    unified: bool = False
    system_mem: int = 0

    @property
    def n_gpus(self) -> int:
        return len(self.gpus)

    def pool_per_gpu(self) -> int:
        """Bytes that --gpu-memory-utilization is a fraction of, per GPU."""
        if self.unified or not self.gpus:
            return self.system_mem
        return min(g.mem_total or 0 for g in self.gpus) or self.system_mem

    def label(self) -> str:
        if not self.gpus:
            return f"no GPU detected ({self.arch})"
        names = {g.name for g in self.gpus}
        name = names.pop() if len(names) == 1 else " + ".join(sorted({g.name for g in self.gpus}))
        count = f"{self.n_gpus}× " if self.n_gpus > 1 else ""
        mem = (f"{self.system_mem / 1e9:.0f} GB unified" if self.unified
               else f"{self.pool_per_gpu() / 1e9:.0f} GB each" if self.n_gpus > 1 else f"{self.pool_per_gpu() / 1e9:.0f} GB")
        return f"{count}{name} · {mem} · {self.arch}"

    def card_tags(self) -> set[str]:
        """Words that mark a model-card section as written for this hardware."""
        tags: set[str] = set()
        for g in self.gpus:
            n = g.name.lower()
            if "gb10" in n:
                tags |= {"dgx spark", "spark", "gb10"}
            for tok in re.findall(r"\b([a-z]?\d{3,4}[a-z]*|[a-z]+\d+[a-z]*)\b", n):
                if tok not in ("nvidia", "geforce"):
                    tags.add(tok)
            if "jetson" in n or "thor" in n or "orin" in n:
                tags |= {"jetson", "thor", "orin"}
        if self.vendor == "amd":
            tags |= {"rocm", "mi300", "mi300x", "mi325x", "mi355x", "amd"}
        return {t for t in tags if len(t) >= 3}

    @property
    def backend(self) -> str:
        """Default deployment for this machine: vLLM in Docker with an NVIDIA / AMD GPU, mlx-lm on Apple Silicon, and
        "remote" (client mode: use other l9s servers, no local model server) on anything else."""
        return "mlx" if self.vendor == "apple" else "vllm" if self.vendor in ("nvidia", "amd") else "remote"

    def docker_gpu_args(self) -> list[str]:
        if self.vendor == "nvidia":
            return ["--gpus", "all"]
        if self.vendor == "amd":
            return ["--device=/dev/kfd", "--device=/dev/dri", "--group-add=video",
                    "--security-opt=seccomp=unconfined"]
        return []

    def default_image(self) -> str:
        return AMD_IMAGE if self.vendor == "amd" else NVIDIA_IMAGE


def detect() -> Hardware:
    arch = {"aarch64": "arm64", "arm64": "arm64", "x86_64": "amd64", "amd64": "amd64"}.get(
        platform.machine().lower(), platform.machine().lower())
    hw = Hardware(vendor="none", arch=arch, system_mem=system_mem(), os=platform.system().lower())
    if hw.os == "darwin":
        chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "Apple Silicon"
        if arch == "arm64":
            hw.vendor, hw.unified = "apple", True
            hw.gpus.append(GPU(0, chip, None))
        return hw
    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"])
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3 and parts[0].isdigit():
                mt = _num(parts[2])
                hw.gpus.append(GPU(int(parts[0]), parts[1], int(mt * 1024 * 1024) if mt else None))
        if hw.gpus:
            hw.vendor = "nvidia"
    if not hw.gpus and (shutil.which("amd-smi") or shutil.which("rocm-smi") or os.path.exists("/dev/kfd")):
        hw.vendor = "amd"
        try:
            data = json.loads(_run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"]) or "{}")
        except json.JSONDecodeError:
            data = {}
        for i, (card, d) in enumerate(sorted(data.items())):
            if not card.startswith("card"):
                continue
            name = d.get("Card Series") or d.get("Card series") or d.get("Card model") or "AMD GPU"
            vram = _num(d.get("VRAM Total Memory (B)"))
            hw.gpus.append(GPU(i, name, int(vram) if vram else None))
        if not hw.gpus:
            hw.gpus.append(GPU(0, "AMD GPU", None))
    hw.unified = bool(hw.gpus) and any(g.mem_total is None or UNIFIED_NAMES.search(g.name) for g in hw.gpus)
    return hw


def gpu_stats(hw: Hardware) -> list[dict]:
    """Per-GPU util %, temp °C, power W, memory used/total bytes (None where not reported)."""
    out: list[dict] = []
    if hw.vendor == "nvidia":
        q = _run(["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total",
                  "--format=csv,noheader,nounits"], timeout=5)
        for line in q.splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 5:
                mu, mt = _num(p[3]), _num(p[4])
                out.append({"util": _num(p[0]), "temp": _num(p[1]), "power": _num(p[2]),
                            "mem_used": int(mu * 1048576) if mu is not None else None,
                            "mem_total": int(mt * 1048576) if mt is not None else None})
    elif hw.vendor == "apple":
        # GPU utilization from the IOAccelerator registry (no sudo needed)
        reg = _run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], timeout=5)
        m = re.search(r'"Device Utilization %"=(\d+)', reg)
        used = re.search(r'"In use system memory"=(\d+)', reg)
        out.append({"util": float(m.group(1)) if m else None, "temp": None, "power": None,
                    "mem_used": int(used.group(1)) if used else None, "mem_total": None})
    elif hw.vendor == "amd":
        try:
            d = json.loads(_run(["rocm-smi", "--showuse", "--showtemp", "--showpower", "--showmeminfo", "vram", "--json"],
                                timeout=5) or "{}")
        except json.JSONDecodeError:
            d = {}
        for card, v in sorted(d.items()):
            if not card.startswith("card"):
                continue
            temp = next((_num(x) for k, x in v.items() if "Temperature" in k), None)
            power = next((_num(x) for k, x in v.items() if "Power" in k and "(W)" in k), None)
            out.append({"util": _num(v.get("GPU use (%)")), "temp": temp, "power": power,
                        "mem_used": int(_num(v.get("VRAM Total Used Memory (B)")) or 0) or None,
                        "mem_total": int(_num(v.get("VRAM Total Memory (B)")) or 0) or None})
    return out


def mem_snapshot() -> dict:
    """System memory in bytes: total, available (incl. reclaimable cache), cached."""
    if platform.system() == "Darwin":
        total = system_mem()
        vm = _run(["vm_stat"], timeout=5)
        page = int(_num((re.search(r"page size of (\d+)", vm) or [None, "16384"])[1]) or 16384)
        pages = {k.strip().lower(): int(v.strip(" .")) for k, v in re.findall(r"^(.+?):\s+(\d+)\.?$", vm, re.M)}
        free = pages.get("pages free", 0) + pages.get("pages speculative", 0)
        cached = pages.get("file-backed pages", 0) + pages.get("pages purgeable", 0)
        avail = (free + cached + pages.get("pages inactive", 0) * 0) * page
        return {"total": total, "available": min(total, avail), "cached": cached * page}
    info = {}
    try:
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            info[k] = int(v.split()[0]) * 1024
    except OSError:
        pass
    return {"total": info.get("MemTotal", 0), "available": info.get("MemAvailable", 0),
            "cached": info.get("Cached", 0) + info.get("Buffers", 0)}


def host_ip() -> str:
    if platform.system() == "Darwin":
        route = _run(["route", "-n", "get", "default"])
        m = re.search(r"interface:\s*(\S+)", route)
        for iface in ([m.group(1)] if m else []) + ["en0", "en1"]:
            ip = _run(["ipconfig", "getifaddr", iface])
            if ip:
                return ip
        ips = [ip for ip in re.findall(r"inet (\d+\.\d+\.\d+\.\d+)", _run(["ifconfig"]))
               if not ip.startswith(("127.", "169.254.", "192.0.0."))]  # loopback, link-local, NAT64/hotspot
        private = [ip for ip in ips if re.match(r"^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)", ip)]
        return (private or ips or ["localhost"])[0]
    ips = [ip for ip in _run(["hostname", "-I"]).split() if re.match(r"^\d+\.\d+\.\d+\.\d+$", ip)]
    lan = [ip for ip in ips if re.match(r"^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)", ip)
           and not ip.startswith("172.17.")]  # not Docker's bridge
    return (lan or ips or ["localhost"])[0]


CGNAT = re.compile(r"^100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.")  # 100.64.0.0/10: Tailscale, Headscale, NetBird, …
VPN_IFACE = (("tailscale", "Tailscale"), ("wg", "WireGuard"), ("zt", "ZeroTier"), ("nebula", "Nebula"),
             ("nb-", "NetBird"), ("tun", "VPN"), ("utun", "VPN"), ("ipsec", "VPN"), ("ppp", "VPN"))
NOT_VPN = ("lo", "docker", "br-", "veth", "virbr", "cni", "flannel", "vmnet", "bridge", "awdl", "llw", "anpi", "en",
           "eth", "wl", "ap")


def _iface_ips() -> list[tuple[str, str]]:
    """(interface, IPv4) for every interface that has one."""
    if platform.system() == "Darwin":
        out, cur = [], None
        for line in _run(["ifconfig"]).splitlines():
            m = re.match(r"^(\S+?):\s", line)
            if m:
                cur = m.group(1)
            m = re.search(r"\binet (\d+\.\d+\.\d+\.\d+)", line)
            if m and cur:
                out.append((cur, m.group(1)))
        return out
    return [(m.group(1), m.group(2)) for m in
            re.finditer(r"^\d+:\s+(\S+?)(?:@\S+)?\s+inet (\d+\.\d+\.\d+\.\d+)", _run(["ip", "-o", "-4", "addr", "show"]),
                        re.M)]


def vpn_addresses() -> list[dict]:
    """This machine's addresses on VPNs (Tailscale, WireGuard, ZeroTier, …): [{ip, kind, iface}], Tailscale first."""
    found = {}
    ts = _run(["tailscale", "ip", "-4"], timeout=3).split() if shutil.which("tailscale") else []
    for ip in ts:
        found[ip] = {"ip": ip, "kind": "Tailscale", "iface": "tailscale"}
    for iface, ip in _iface_ips():
        if ip in found or ip.startswith(("127.", "169.254.")):
            continue
        kind = next((k for pre, k in VPN_IFACE if iface.startswith(pre)), None)
        if CGNAT.match(ip) and (kind in (None, "VPN") or iface.startswith("utun")):
            kind = "Tailscale / NetBird (100.64/10)" if kind in (None, "VPN") else kind
        elif not kind or iface.startswith(NOT_VPN):
            continue
        found[ip] = {"ip": ip, "kind": kind, "iface": iface}
    return sorted(found.values(), key=lambda d: (d["kind"] != "Tailscale", d["kind"], d["ip"]))


def vllm_reserved(hw: Hardware) -> int:
    """Bytes of GPU memory held by compute processes (vLLM), where the driver reports it."""
    if hw.vendor == "nvidia":
        q = _run(["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"], timeout=5)
        return int(sum(_num(x) or 0 for x in q.split()) * 1048576)
    return 0


if __name__ == "__main__":
    import sys
    h = detect()
    if len(sys.argv) > 1 and sys.argv[1] == "env":
        import shlex
        print(f"BACKEND={h.backend}")
        if h.backend == "vllm":
            print(f"GPU_ARGS={shlex.quote(' '.join(h.docker_gpu_args()))}")
            print(f"IMAGE={shlex.quote(h.default_image())}")
    else:
        print(f"backend: {h.backend}\nvendor:  {h.vendor}\narch:    {h.arch}\ngpus:    {h.label()}\nunified: {h.unified}\n"
              f"pool:    {h.pool_per_gpu() / 1e9:.0f} GB per GPU (what GPU_MEM is a fraction of)\n"
              f"docker:  {' '.join(h.docker_gpu_args()) or '(no GPU args)'}\ntags:    {sorted(h.card_tags())}")
