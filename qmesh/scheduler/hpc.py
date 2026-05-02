"""qmesh.scheduler.hpc — HPC connectors (SLURM / PBS) for QPU dispatch.

Phase 4β. The 2026 HPC + QC deployment direction (RIKEN Fugaku ↔ Reimei,
NERSC QCAN, ORNL + Infleqtion + GB200 NVL72, JSC NVIDIA DGX Quantum) needs
a scheduler that can hand off a `qmesh.ir.Module` to a batch system — not
just a local thread pool. This file provides:

  - `HPCConnector` (abstract base) — render a script, submit, poll, fetch.
  - `SLURMConnector`              — sbatch / squeue.
  - `PBSConnector`                — qsub / qstat.
  - `MockHPCConnector`            — runs the IR locally, used by tests &
                                    demos (CI has no SLURM/PBS).
  - `HPCQPUPrimitive`             — DAG node that dispatches to a connector
                                    instead of calling `qmesh.api.submit`
                                    directly. Manifests get an `hpc` block
                                    in `manifest.execution`.

α scope: stdlib `subprocess` only — no PySLURM / drmaa dependency. Polling
is exponential-backoff with a configurable timeout. `MockHPCConnector`
short-circuits the script-render / submit / poll path so tests pass without
a cluster.

β follow-ups: real-time stream of `--export-mqc` style metrics from the
running batch job; OAR / LSF / Cobalt connectors; per-site queue auto-
discovery; checkpoint+restart on preemption.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import subprocess
import textwrap
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from qmesh.ir.module import Module
from qmesh.scheduler.dag import Node


# ---------- HPC primitive node ----------

@dataclass(slots=True)
class HPCQPUPrimitive(Node):
    """Run a `qmesh.ir.Module` on a backend through an HPC batch scheduler.

    Same shape as `QPUPrimitive` (module / module_factory / backend / shots
    / sign), but execution is dispatched through a `connector` that knows
    how to wrap the call in an sbatch/qsub script, submit it, poll for
    completion, and read back the result artefacts. The aggregate run
    manifest gets an `hpc` block in `manifest.execution` recording the
    scheduler name, job ID, partition, requested vs actual walltime.
    """

    module: Module | None = None
    module_factory: Callable[[dict[str, Any]], Module] | None = None
    backend: str = "qmesh.aer"
    shots: int = 1024
    sign: bool = True
    connector: HPCConnector | None = None
    # Submission knobs (passed into the connector at submit time).
    partition: str = "default"
    walltime: str = "00:30:00"
    nodes: int = 1
    cpus_per_task: int = 1
    extra_sbatch: list[str] = field(default_factory=list)

    def kind(self) -> str:
        return "hpc_qpu"

    def __post_init__(self) -> None:
        if self.module is None and self.module_factory is None:
            raise ValueError(
                "HPCQPUPrimitive requires either `module` or `module_factory`"
            )
        if self.connector is None:
            raise ValueError(
                "HPCQPUPrimitive requires a `connector` "
                "(SLURMConnector / PBSConnector / MockHPCConnector)"
            )


# ---------- shared types ----------

@dataclass(slots=True)
class HPCSubmission:
    """Bookkeeping for a single batch submission."""

    job_id: str
    scheduler: str
    submit_dir: Path
    out_dir: Path
    script_path: Path
    submitted_at: float
    walltime_request: str
    partition: str


@dataclass(slots=True)
class HPCResult:
    """What a connector returns to `HPCQPUPrimitive` at execution time."""

    counts: dict[str, int]
    shots: int
    backend_name: str
    job_id: str
    scheduler: str
    partition: str
    walltime_request: str
    walltime_actual: float
    submit_dir: Path
    out_dir: Path
    inner_manifest_hash: str | None = None


# ---------- abstract base ----------

class HPCConnector(ABC):
    """Abstract base for HPC schedulers. See SLURMConnector / PBSConnector."""

    name: str = "hpc"

    def __init__(
        self,
        *,
        work_dir: Path | str = "ledger/hpc",
        python_executable: str = "python3",
        poll_timeout_seconds: float = 1800.0,
        poll_initial_seconds: float = 1.0,
        poll_max_seconds: float = 30.0,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.python_executable = python_executable
        self.poll_timeout_seconds = poll_timeout_seconds
        self.poll_initial_seconds = poll_initial_seconds
        self.poll_max_seconds = poll_max_seconds

    # ----- shared plumbing -----

    def _make_submission_dir(self) -> tuple[Path, Path]:
        """Return (submit_dir, out_dir). out_dir is where the entrypoint
        writes counts.json / inner_manifest.json. Both are created."""
        sid = uuid.uuid4().hex[:12]
        submit_dir = self.work_dir / time.strftime("%Y-%m-%d") / sid
        out_dir = submit_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        return submit_dir, out_dir

    def _serialise_module(self, module: Module, submit_dir: Path) -> Path:
        """Pickle the IR Module to disk so the entrypoint can rehydrate it.

        Pickle is the α choice: qmesh.ir Modules contain Python-only
        attributes (callables, dataclass slots) that don't survive a
        round-trip through JSON or CBOR yet. β will switch to a typed
        cbor2 encoding once the IR settles.
        """
        path = submit_dir / "module.pkl"
        path.write_bytes(pickle.dumps(module))
        return path

    def _entrypoint_source(
        self,
        module_path: Path,
        out_dir: Path,
        backend: str,
        shots: int,
        sign: bool,
    ) -> str:
        """Emit a self-contained Python entrypoint for the batch job.

        The compute node imports qmesh, unpickles the IR Module, calls
        `qmesh.api.submit`, then drops counts.json + inner_manifest.json
        into out_dir for the connector to read back.
        """
        return textwrap.dedent(f"""
            import json, pickle, sys, time
            from pathlib import Path
            module = pickle.loads(Path({str(module_path)!r}).read_bytes())
            from qmesh.api import submit
            t0 = time.time()
            result, manifest = submit(
                module, backend={backend!r}, shots={shots},
                ledger_dir={str(out_dir)!r}, sign={bool(sign)!r},
            )
            (Path({str(out_dir)!r}) / "counts.json").write_text(
                json.dumps({{
                    "counts": result.counts,
                    "shots": result.shots,
                    "wall_seconds": result.wall_seconds,
                    "backend": {backend!r},
                    "manifest_hash": manifest.hash(),
                    "elapsed": time.time() - t0,
                }})
            )
            (Path({str(out_dir)!r}) / "inner_manifest.json").write_text(
                manifest.to_json()
            )
            print("OK", manifest.hash())
        """).strip() + "\n"

    # ----- concrete subclasses fill these in -----

    @abstractmethod
    def render_script(
        self,
        *,
        module: Module,
        backend: str,
        shots: int,
        sign: bool,
        partition: str,
        walltime: str,
        nodes: int,
        cpus_per_task: int,
        extra_directives: list[str],
        submit_dir: Path,
        out_dir: Path,
    ) -> str:
        """Produce the full sbatch/qsub script text. Pure / side-effect-free."""
        raise NotImplementedError

    @abstractmethod
    def _submit_cmd(self, script_path: Path) -> list[str]: ...

    @abstractmethod
    def _parse_job_id(self, stdout: str) -> str: ...

    @abstractmethod
    def _status_cmd(self, job_id: str) -> list[str]: ...

    @abstractmethod
    def _is_complete(self, status_output: str) -> bool:
        """Return True if the job has left the queue (success or fail)."""
        raise NotImplementedError

    # ----- driver -----

    def submit(
        self,
        *,
        module: Module,
        backend: str,
        shots: int,
        sign: bool,
        partition: str,
        walltime: str,
        nodes: int,
        cpus_per_task: int,
        extra_directives: list[str],
    ) -> HPCSubmission:
        submit_dir, out_dir = self._make_submission_dir()
        self._serialise_module(module, submit_dir)
        script_text = self.render_script(
            module=module, backend=backend, shots=shots, sign=sign,
            partition=partition, walltime=walltime, nodes=nodes,
            cpus_per_task=cpus_per_task, extra_directives=extra_directives,
            submit_dir=submit_dir, out_dir=out_dir,
        )
        script_path = submit_dir / f"submit.{self.name}.sh"
        script_path.write_text(script_text)
        script_path.chmod(0o755)

        proc = subprocess.run(
            self._submit_cmd(script_path),
            check=True, capture_output=True, text=True,
        )
        job_id = self._parse_job_id(proc.stdout)

        return HPCSubmission(
            job_id=job_id,
            scheduler=self.name,
            submit_dir=submit_dir,
            out_dir=out_dir,
            script_path=script_path,
            submitted_at=time.time(),
            walltime_request=walltime,
            partition=partition,
        )

    def poll(self, sub: HPCSubmission) -> None:
        """Block until the job leaves the queue (or timeout fires)."""
        deadline = sub.submitted_at + self.poll_timeout_seconds
        delay = self.poll_initial_seconds
        while time.time() < deadline:
            try:
                proc = subprocess.run(
                    self._status_cmd(sub.job_id),
                    capture_output=True, text=True, timeout=30,
                )
                if self._is_complete(proc.stdout + proc.stderr):
                    return
            except subprocess.TimeoutExpired:
                pass
            time.sleep(delay)
            delay = min(delay * 2, self.poll_max_seconds)
        raise TimeoutError(
            f"{self.name} job {sub.job_id} did not complete within "
            f"{self.poll_timeout_seconds:.0f}s"
        )

    def fetch(self, sub: HPCSubmission) -> HPCResult:
        """Read counts.json + inner_manifest.json from the agreed out_dir."""
        counts_path = sub.out_dir / "counts.json"
        if not counts_path.exists():
            raise FileNotFoundError(
                f"{self.name} job {sub.job_id}: counts.json missing at "
                f"{counts_path} — did the entrypoint complete?"
            )
        data = json.loads(counts_path.read_text())
        return HPCResult(
            counts=data["counts"],
            shots=data["shots"],
            backend_name=data["backend"],
            job_id=sub.job_id,
            scheduler=sub.scheduler,
            partition=sub.partition,
            walltime_request=sub.walltime_request,
            walltime_actual=time.time() - sub.submitted_at,
            submit_dir=sub.submit_dir,
            out_dir=sub.out_dir,
            inner_manifest_hash=data.get("manifest_hash"),
        )

    def run(
        self,
        *,
        module: Module,
        backend: str,
        shots: int,
        sign: bool,
        partition: str,
        walltime: str,
        nodes: int,
        cpus_per_task: int,
        extra_directives: list[str],
    ) -> HPCResult:
        sub = self.submit(
            module=module, backend=backend, shots=shots, sign=sign,
            partition=partition, walltime=walltime, nodes=nodes,
            cpus_per_task=cpus_per_task, extra_directives=extra_directives,
        )
        self.poll(sub)
        return self.fetch(sub)

    def status(self, job_id: str) -> str:
        """Lightweight CLI helper — return raw scheduler output for a job."""
        proc = subprocess.run(
            self._status_cmd(job_id), capture_output=True, text=True,
        )
        return (proc.stdout + proc.stderr).strip()


# ---------- SLURM ----------

class SLURMConnector(HPCConnector):
    """SLURM (RIKEN Fugaku, JSC, NERSC, many university clusters)."""

    name = "slurm"

    def render_script(
        self, *, module, backend, shots, sign, partition, walltime,
        nodes, cpus_per_task, extra_directives, submit_dir, out_dir,
    ) -> str:
        module_path = submit_dir / "module.pkl"
        body = self._entrypoint_source(module_path, out_dir, backend, shots, sign)
        directives = [
            f"#SBATCH --job-name=qmesh-{submit_dir.name}",
            f"#SBATCH --partition={partition}",
            f"#SBATCH --nodes={nodes}",
            f"#SBATCH --cpus-per-task={cpus_per_task}",
            f"#SBATCH --time={walltime}",
            f"#SBATCH --output={out_dir}/slurm-%j.out",
            f"#SBATCH --error={out_dir}/slurm-%j.err",
            *extra_directives,
        ]
        return (
            "#!/bin/bash\n"
            + "\n".join(directives) + "\n"
            + "set -euo pipefail\n"
            + f"export PYTHONPATH=${{PYTHONPATH:-}}:{os.getcwd()}\n"
            + f"cd {submit_dir}\n"
            + f"{self.python_executable} - <<'PYEOF'\n{body}PYEOF\n"
        )

    def _submit_cmd(self, script_path: Path) -> list[str]:
        return ["sbatch", str(script_path)]

    def _parse_job_id(self, stdout: str) -> str:
        # SLURM: "Submitted batch job 12345"
        m = re.search(r"Submitted batch job (\d+)", stdout)
        if not m:
            raise RuntimeError(f"could not parse SLURM job id from: {stdout!r}")
        return m.group(1)

    def _status_cmd(self, job_id: str) -> list[str]:
        return ["squeue", "-h", "-j", job_id, "-o", "%T"]

    def _is_complete(self, status_output: str) -> bool:
        # squeue prints nothing once the job has left the queue.
        s = status_output.strip()
        if not s:
            return True
        return any(t in s for t in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"))


# ---------- PBS / Torque ----------

class PBSConnector(HPCConnector):
    """PBS Pro / Torque / OpenPBS (ORNL Frontier, many DOE labs)."""

    name = "pbs"

    def render_script(
        self, *, module, backend, shots, sign, partition, walltime,
        nodes, cpus_per_task, extra_directives, submit_dir, out_dir,
    ) -> str:
        module_path = submit_dir / "module.pkl"
        body = self._entrypoint_source(module_path, out_dir, backend, shots, sign)
        directives = [
            f"#PBS -N qmesh-{submit_dir.name}",
            f"#PBS -q {partition}",
            f"#PBS -l select={nodes}:ncpus={cpus_per_task}",
            f"#PBS -l walltime={walltime}",
            f"#PBS -o {out_dir}/pbs.out",
            f"#PBS -e {out_dir}/pbs.err",
            *extra_directives,
        ]
        return (
            "#!/bin/bash\n"
            + "\n".join(directives) + "\n"
            + "set -euo pipefail\n"
            + f"export PYTHONPATH=${{PYTHONPATH:-}}:{os.getcwd()}\n"
            + f"cd {submit_dir}\n"
            + f"{self.python_executable} - <<'PYEOF'\n{body}PYEOF\n"
        )

    def _submit_cmd(self, script_path: Path) -> list[str]:
        return ["qsub", str(script_path)]

    def _parse_job_id(self, stdout: str) -> str:
        # PBS: "12345.cluster.example.com" or just "12345.head"
        s = stdout.strip().splitlines()
        if not s:
            raise RuntimeError(f"could not parse PBS job id from: {stdout!r}")
        return s[-1].strip()

    def _status_cmd(self, job_id: str) -> list[str]:
        return ["qstat", job_id]

    def _is_complete(self, status_output: str) -> bool:
        # qstat exits non-zero or prints "Unknown Job Id" once finished.
        s = status_output.strip()
        if not s:
            return True
        if "Unknown Job" in s:
            return True
        # Otherwise look at the job-state line: "C" (completed) or "F" (finished).
        for line in s.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0].split(".")[0].isdigit():
                state = parts[4]
                if state in ("C", "F"):
                    return True
        return False


# ---------- Mock (used by tests + demo) ----------

class MockHPCConnector(HPCConnector):
    """Pretend-cluster connector. Renders a script (so the SLURM/PBS code
    paths are exercised), but instead of `subprocess.run(["sbatch", …])`
    runs the IR locally via `qmesh.api.submit` and writes the same
    counts.json / inner_manifest.json the real entrypoint would produce.

    This is what tests + CI use, since neither has access to a real
    SLURM or PBS controller. The α-honest scope: it covers the
    `HPCQPUPrimitive` execution path, manifest.execution.hpc plumbing,
    and result fan-in — but does *not* exercise actual sbatch/qsub
    subprocesses or remote filesystem semantics.
    """

    name = "mock"

    def __init__(
        self,
        *,
        work_dir: Path | str = "ledger/hpc",
        simulated_walltime_seconds: float = 0.0,
        scheduler_emulated: str = "mock-slurm",
    ) -> None:
        super().__init__(work_dir=work_dir)
        self.simulated_walltime_seconds = simulated_walltime_seconds
        self.scheduler_emulated = scheduler_emulated

    def render_script(
        self, *, module, backend, shots, sign, partition, walltime,
        nodes, cpus_per_task, extra_directives, submit_dir, out_dir,
    ) -> str:
        module_path = submit_dir / "module.pkl"
        body = self._entrypoint_source(module_path, out_dir, backend, shots, sign)
        return (
            f"#!/bin/bash\n"
            f"# mock connector emulating {self.scheduler_emulated}\n"
            f"# partition={partition} nodes={nodes} walltime={walltime}\n"
            f"{self.python_executable} - <<'PYEOF'\n{body}PYEOF\n"
        )

    def _submit_cmd(self, script_path: Path) -> list[str]:
        # Never actually invoked — `submit()` is overridden below.
        return ["true"]

    def _parse_job_id(self, stdout: str) -> str:
        return f"mock-{uuid.uuid4().hex[:8]}"

    def _status_cmd(self, job_id: str) -> list[str]:
        return ["true"]

    def _is_complete(self, status_output: str) -> bool:
        return True

    def submit(
        self,
        *,
        module: Module,
        backend: str,
        shots: int,
        sign: bool,
        partition: str,
        walltime: str,
        nodes: int,
        cpus_per_task: int,
        extra_directives: list[str],
    ) -> HPCSubmission:
        # Render the script for transparency/audit, then run locally.
        submit_dir, out_dir = self._make_submission_dir()
        self._serialise_module(module, submit_dir)
        script_text = self.render_script(
            module=module, backend=backend, shots=shots, sign=sign,
            partition=partition, walltime=walltime, nodes=nodes,
            cpus_per_task=cpus_per_task, extra_directives=extra_directives,
            submit_dir=submit_dir, out_dir=out_dir,
        )
        script_path = submit_dir / "submit.mock.sh"
        script_path.write_text(script_text)

        # Local execution path — bypasses subprocess + scheduler entirely.
        from qmesh.api import submit as _submit
        if self.simulated_walltime_seconds > 0:
            time.sleep(self.simulated_walltime_seconds)
        result, inner_manifest = _submit(
            module, backend=backend, shots=shots,
            ledger_dir=out_dir, sign=sign,
        )
        (out_dir / "counts.json").write_text(json.dumps({
            "counts": result.counts,
            "shots": result.shots,
            "wall_seconds": result.wall_seconds,
            "backend": backend,
            "manifest_hash": inner_manifest.hash(),
            "elapsed": result.wall_seconds,
        }))
        (out_dir / "inner_manifest.json").write_text(inner_manifest.to_json())

        return HPCSubmission(
            job_id=self._parse_job_id(""),
            scheduler=self.name,
            submit_dir=submit_dir,
            out_dir=out_dir,
            script_path=script_path,
            submitted_at=time.time(),
            walltime_request=walltime,
            partition=partition,
        )

    def poll(self, sub: HPCSubmission) -> None:
        return  # already complete by the time submit() returns


__all__ = [
    "HPCQPUPrimitive",
    "HPCConnector",
    "SLURMConnector",
    "PBSConnector",
    "MockHPCConnector",
    "HPCSubmission",
    "HPCResult",
]
