from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .packing import generate_segregated_packing, write_lammps_data


TWO_PI = 2.0 * math.pi


def _steps(duration: float, dt: float) -> int:
    return max(1, int(round(duration / dt)))


def _cmd(lmp, text: str) -> None:
    lmp.command(text)


def _contact_model(cfg: dict[str, Any]) -> str:
    return (
        f"hooke {cfg['normal_stiffness']} {cfg['coefficient_restitution']} "
        f"tangential linear_history {cfg['tangential_stiffness']} 1.0 "
        f"{cfg['friction_coefficient']} damping coeff_restitution"
    )


def _setup_lammps(lmp, cfg: dict[str, Any], data_path: Path) -> None:
    model = _contact_model(cfg)
    _cmd(lmp, "clear")
    _cmd(lmp, "units si")
    _cmd(lmp, "dimension 2")
    _cmd(lmp, "boundary f f p")
    _cmd(lmp, "atom_style sphere")
    _cmd(lmp, "atom_modify map array")
    _cmd(lmp, f"read_data {data_path.resolve()}")
    _cmd(lmp, f"set group all density/disc {cfg['areal_density']}")
    _cmd(lmp, "pair_style granular")
    _cmd(lmp, f"pair_coeff * * {model}")
    _cmd(lmp, f"neighbor {cfg['neighbor_skin']} bin")
    _cmd(lmp, "neigh_modify delay 0 every 1 check yes")
    _cmd(lmp, f"timestep {cfg['timestep']}")
    _cmd(lmp, "fix int all nve/sphere disc")
    _cmd(lmp, f"fix grav all gravity {cfg['gravity']} vector 0 -1 0")
    _cmd(
        lmp,
        f"fix side all wall/gran granular {model} xplane 0.0 {cfg['box_width']}",
    )
    _cmd(lmp, f"fix bottom all wall/gran granular {model} yplane 0.0 NULL")
    _cmd(lmp, "compute keT all ke")
    # Exact 2D-disc rotational KE: I=(1/2)mR^2, so KE_rot=(1/4)mR^2*omega_z^2.
    _cmd(lmp, "variable _gw_keRot atom 0.25*rmass*radius*radius*omegaz*omegaz")
    _cmd(lmp, "compute keR all reduce sum v__gw_keRot")
    _cmd(lmp, "thermo 5000")
    _cmd(lmp, "thermo_style custom step time atoms c_keT c_keR")
    _cmd(lmp, "thermo_modify norm no")
    _cmd(lmp, "run 0")


def _kinetic_energy(lmp) -> float:
    # Import constants only when LAMMPS is available.
    from lammps import LMP_STYLE_GLOBAL, LMP_TYPE_SCALAR

    ket = float(lmp.numpy.extract_compute("keT", LMP_STYLE_GLOBAL, LMP_TYPE_SCALAR))
    ker = float(lmp.numpy.extract_compute("keR", LMP_STYLE_GLOBAL, LMP_TYPE_SCALAR))
    return ket + ker


def _relax(
    lmp,
    n_particles: int,
    dt: float,
    chunk_time: float,
    max_time: float,
    ke_tol_per_particle: float,
    stable_chunks_required: int,
) -> tuple[int, float, bool]:
    chunk_steps = _steps(chunk_time, dt)
    max_steps = _steps(max_time, dt)
    done_steps = 0
    stable = 0
    kepp = float("inf")

    while done_steps < max_steps:
        nrun = min(chunk_steps, max_steps - done_steps)
        _cmd(lmp, f"run {nrun} post no")
        done_steps += nrun
        kepp = _kinetic_energy(lmp) / n_particles
        if kepp <= ke_tol_per_particle:
            stable += 1
            if stable >= stable_chunks_required:
                return done_steps, kepp, True
        else:
            stable = 0
    return done_steps, kepp, False


def _gather_snapshot(lmp) -> dict[str, np.ndarray]:
    """Gather an ID-ordered snapshot, valid for serial or MPI LAMMPS."""
    n = int(lmp.get_natoms())

    def doubles(name: str, count: int) -> np.ndarray:
        buf = lmp.gather_atoms(name, 1, count)
        return np.ctypeslib.as_array(buf, shape=(n * count,)).copy().reshape(n, count)

    def ints(name: str) -> np.ndarray:
        buf = lmp.gather_atoms(name, 0, 1)
        return np.ctypeslib.as_array(buf, shape=(n,)).copy()

    x = doubles("x", 3)
    radius = doubles("radius", 1)[:, 0]
    typ = ints("type").astype(np.int32)
    return {"xy": x[:, :2], "radii": radius, "colors": typ}


def _save_snapshot(lmp, out_dir: Path, cycle: int) -> None:
    s = _gather_snapshot(lmp)
    np.savez_compressed(
        out_dir / "snapshots" / f"cycle_{cycle:05d}.npz",
        xy=s["xy"],
        radii=s["radii"],
        colors=s["colors"],
    )


def _drive_one_cycle(lmp, cfg: dict[str, Any], amplitude: float, frequency: float) -> float:
    """Drive bottom wall for one full sinusoidal period and return boundary work.

    Work = integral F_wall->particles dot v_wall dt.  The force is obtained from
    columns 2-4 of ``fix wall/gran ... contacts`` and summed with compute reduce.
    """
    from lammps import LMP_VAR_EQUAL

    dt = float(cfg["timestep"])
    period = 1.0 / float(frequency)
    nsteps = max(2, int(round(period / dt)))
    # Make the LAMMPS wall's requested period exactly compatible with integer steps.
    period_eff = nsteps * dt
    model = _contact_model(cfg)

    _cmd(lmp, "unfix bottom")
    # Save the global time as an immediate numeric constant for the velocity phase.
    _cmd(lmp, "variable _gw_now equal time")
    t0 = float(lmp.extract_variable("_gw_now", None, LMP_VAR_EQUAL))
    _cmd(lmp, "variable _gw_now delete")

    _cmd(
        lmp,
        f"fix bottom all wall/gran granular {model} yplane 0.0 NULL "
        f"wiggle y {amplitude} {period_eff} contacts",
    )
    _cmd(lmp, "compute _gw_Fy all reduce sum f_bottom[3]")
    _cmd(
        lmp,
        "variable _gw_vwall equal "
        f"{amplitude * TWO_PI / period_eff:.17g}*"
        f"sin({TWO_PI / period_eff:.17g}*(time-{t0:.17g}))",
    )
    _cmd(lmp, "variable _gw_power equal c__gw_Fy*v__gw_vwall")
    _cmd(lmp, f"fix _gw_phist all vector 1 v__gw_power nmax {nsteps + 2}")
    _cmd(lmp, f"run {nsteps} post no")
    _cmd(lmp, "variable _gw_work equal dt*trap(f__gw_phist)")
    work = float(lmp.extract_variable("_gw_work", None, LMP_VAR_EQUAL))

    _cmd(lmp, "unfix _gw_phist")
    _cmd(lmp, "uncompute _gw_Fy")
    _cmd(lmp, "variable _gw_vwall delete")
    _cmd(lmp, "variable _gw_power delete")
    _cmd(lmp, "variable _gw_work delete")
    _cmd(lmp, "unfix bottom")
    _cmd(lmp, f"fix bottom all wall/gran granular {model} yplane 0.0 NULL")
    return work


def run_experiment(config_path: str | Path, output_dir: str | Path) -> Path:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "snapshots").mkdir(exist_ok=True)

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    (output_dir / "config_used.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    packing = generate_segregated_packing(
        n=int(cfg["n_particles"]),
        width=float(cfg["box_width"]),
        fill_height=float(cfg["initial_fill_height"]),
        d_small=float(cfg["diameter_small"]),
        d_large=float(cfg["diameter_large"]),
        large_fraction=float(cfg["large_fraction"]),
        seed=int(cfg["seed"]),
    )
    data_path = write_lammps_data(
        packing,
        output_dir / "initial.data",
        box_width=float(cfg["box_width"]),
        box_height=float(cfg["box_height"]),
    )

    try:
        from lammps import lammps
    except ImportError as exc:
        raise RuntimeError(
            "The Python package 'lammps' is not importable. Install/build LAMMPS "
            "with the GRANULAR package and shared-library Python interface first."
        ) from exc

    log_path = output_dir / "lammps.log"
    lmp = lammps(cmdargs=["-log", str(log_path), "-screen", "none"])
    try:
        if not lmp.has_package("GRANULAR"):
            raise RuntimeError("This LAMMPS build does not include the GRANULAR package.")
        _setup_lammps(lmp, cfg, data_path)

        settle_steps, kepp, settled = _relax(
            lmp,
            n_particles=int(cfg["n_particles"]),
            dt=float(cfg["timestep"]),
            chunk_time=float(cfg["settle_chunk_time"]),
            max_time=float(cfg["settle_max_time"]),
            ke_tol_per_particle=float(cfg["ke_tolerance_per_particle"]),
            stable_chunks_required=int(cfg["stable_chunks_required"]),
        )
        _save_snapshot(lmp, output_dir, cycle=0)

        protocol = cfg["protocol"]
        cycles = int(protocol["cycles"])
        amp = float(protocol["amplitude"])
        freq = float(protocol["frequency"])
        cumulative_work = 0.0

        metrics_path = output_dir / "trajectory.csv"
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "cycle",
                    "work_cycle_J",
                    "work_cumulative_J",
                    "relax_steps",
                    "ke_per_particle_J",
                    "relaxed",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "cycle": 0,
                    "work_cycle_J": 0.0,
                    "work_cumulative_J": 0.0,
                    "relax_steps": settle_steps,
                    "ke_per_particle_J": kepp,
                    "relaxed": settled,
                }
            )

            for cycle in range(1, cycles + 1):
                work = _drive_one_cycle(lmp, cfg, amp, freq)
                cumulative_work += work
                relax_steps, kepp, relaxed = _relax(
                    lmp,
                    n_particles=int(cfg["n_particles"]),
                    dt=float(cfg["timestep"]),
                    chunk_time=float(cfg["relax_chunk_time"]),
                    max_time=float(cfg["relax_max_time"]),
                    ke_tol_per_particle=float(cfg["ke_tolerance_per_particle"]),
                    stable_chunks_required=int(cfg["stable_chunks_required"]),
                )
                _save_snapshot(lmp, output_dir, cycle)
                writer.writerow(
                    {
                        "cycle": cycle,
                        "work_cycle_J": work,
                        "work_cumulative_J": cumulative_work,
                        "relax_steps": relax_steps,
                        "ke_per_particle_J": kepp,
                        "relaxed": relaxed,
                    }
                )
                f.flush()

        _cmd(lmp, f"write_restart {str((output_dir / 'final.restart').resolve())}")
    finally:
        lmp.close()

    return output_dir
