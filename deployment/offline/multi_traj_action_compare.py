#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path


def run_eval(eval_script, traj_dir, save_dir, model, args,
             obs_corruption="none",
             noise_std=0.0,
             occlusion_ratio=0.0):

    cmd = [
        "uv", "run", "python", eval_script,
        "--traj-dir", str(traj_dir),
        "--save-dir", str(save_dir),
        "--model", model,
        "--topomap-every", str(args.topomap_every),
        "--max-steps", str(args.max_steps),
        "--radius", str(args.radius),
        "--num-samples", str(args.num_samples),
        "--waypoint", str(args.waypoint),
        "--device", args.device,
        "--seed", str(args.seed),
        "--obs-corruption", obs_corruption,
        "--corrupt-frame-mode", args.corrupt_frame_mode,
        "--corruption-ratio", str(args.corruption_ratio),
        "--obs-noise-std", str(noise_std),
        "--obs-occlusion-ratio", str(occlusion_ratio),
        "--occlusion-fill", str(args.occlusion_fill),
    ]

    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def count_images(traj_dir):
    traj_dir = Path(traj_dir)

    return (
        len(list(traj_dir.glob("*.jpg")))
        + len(list(traj_dir.glob("*.png")))
    )


def find_trajectories(data_dir, max_trajs, min_frames=10):
    data_dir = Path(data_dir)
    trajs = []

    for p in sorted(data_dir.iterdir()):
        if not p.is_dir():
            continue

        num_imgs = count_images(p)

        if num_imgs < min_frames:
            print(f"Skip {p.name}: only {num_imgs} frames")
            continue

        trajs.append(p)

    if max_trajs > 0:
        trajs = trajs[:max_trajs]

    return trajs


def parse_list(s):
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def safe_name(x):
    return str(x).replace(".", "p")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval-script", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="nomad")

    parser.add_argument("--max-trajs", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--topomap-every", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--waypoint", type=int, default=2)
    parser.add_argument("--min-frames", type=int, default=10)
    # corruption schedule
    parser.add_argument("--corrupt-frame-mode", type=str, default="all",
                        choices=["none", "all", "random", "periodic"])
    parser.add_argument("--corruption-ratio", type=float, default=1.0)

    # sweep values
    parser.add_argument("--noise-std-list", type=str, default="0.05,0.1,0.2")
    parser.add_argument("--occlusion-ratio-list", type=str, default="0.1,0.2,0.3")
    parser.add_argument("--occlusion-fill", type=float, default=0.0)

    parser.add_argument("--skip-existing", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    noise_levels = parse_list(args.noise_std_list)
    occ_levels = parse_list(args.occlusion_ratio_list)

    #trajs = find_trajectories(args.data_dir, args.max_trajs)
    trajs = find_trajectories(
        args.data_dir,
        args.max_trajs,
        args.min_frames,
    )
    print(f"Found {len(trajs)} trajectories")

    for traj in trajs:
        traj_name = traj.name
        print(f"\n===== TRAJ: {traj_name} =====")

        # -------- clean --------
        clean_dir = out_dir / traj_name / "clean"
        if not (args.skip_existing and clean_dir.exists()):
            run_eval(
                args.eval_script,
                traj,
                clean_dir,
                args.model,
                args,
                obs_corruption="none",
            )

        # -------- noise sweep --------
        for std in noise_levels:
            run_dir = out_dir / traj_name / "noise" / f"std_{safe_name(std)}"
            if args.skip_existing and run_dir.exists():
                continue

            run_eval(
                args.eval_script,
                traj,
                run_dir,
                args.model,
                args,
                obs_corruption="noise",
                noise_std=std,
            )

        # -------- occlusion sweep --------
        for ratio in occ_levels:
            run_dir = out_dir / traj_name / "occlusion" / f"ratio_{safe_name(ratio)}"
            if args.skip_existing and run_dir.exists():
                continue

            run_eval(
                args.eval_script,
                traj,
                run_dir,
                args.model,
                args,
                obs_corruption="occlusion",
                occlusion_ratio=ratio,
            )


if __name__ == "__main__":
    main()