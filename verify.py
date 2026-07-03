
import sys

import matplotlib
matplotlib.use("Agg")

from main import Simulation, SCENARIOS


def run(scenario, surprise=False):
    sim = Simulation(scenario, surprise=surprise)
    while not sim.done:
        sim.step()

    ratio = sim.travelled() / sim.path_len
    max_ratio = 2.2 if surprise else 1.6
    checks = {
        "goal reached": sim.reached,
        "no contacts": sim.collisions == 0,
        "clearance > 0.08 m": sim.min_clearance > 0.08,
        f"travel ratio < {max_ratio}": ratio < max_ratio,
    }
    ok = all(checks.values())
    label = f"scenario {scenario}" + (" +surprise" if surprise else "")
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: ratio {ratio:.2f}, "
          f"clearance {sim.min_clearance:.2f} m, contacts {sim.collisions}, "
          f"replans {sim.replans}, explored "
          f"{sim.slam.explored_fraction()*100:.0f} %")
    for name, passed in checks.items():
        if not passed:
            print(f"        failed check: {name}")
    print()
    return ok


def main():
    results = [run(sc) for sc in SCENARIOS]
    results += [run(sc, surprise=True) for sc in SCENARIOS]
    n_ok = sum(results)
    print(f"[AMRN] verify: {n_ok}/{len(results)} runs passed")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
