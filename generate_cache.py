#!/usr/bin/env python3
"""
generate_cache.py — unified forward+reverse pipeline for the
Diffusion Noise-Schedule Visualizer.

For every (image, schedule, seed) triple this:
  1. Loads the source PNG from images/.
  2. Resizes it to the model's native input size and maps to [-1, 1].
  3. Samples ε ~ 𝒩(0, I) with the seed.
  4. Forward-noises across the full [0, T] range (display strip).
  5. Computes the SDEdit bridge:  x_{t_start} = √ᾱ_{t_start}·x_0 + √(1-ᾱ_{t_start})·ε
     where t_start = round(strength · T).
  6. **Image-to-image denoising**: η-DDIM-samples x_{t_start} → x_0 with the
     pretrained denoiser. Because reverse starts from a partially-noised
     version of YOUR image (not pure noise), the denoised output preserves
     the source's structure rather than hallucinating from the model's
     training distribution.
  7. Writes 8 forward snapshots over [0, T] and 8 denoise snapshots over
     [0, t_start] into results/<img>/<schedule>_schedule_frames/seed_<n>/.

Default model: google/ddpm-cat-256 (256×256 unconditional Cat).
Sampling: η=0 deterministic DDIM, 100 steps, strength=0.6.
For lighter / faster runs swap in google/ddpm-cifar10-32 (32×32) via --model.

Output layout
-------------
results/
    manifest.json
    fig1/
        linear_schedule_frames/
            seed_0/frame_0_t1000.png
            seed_0/frame_1_t0860.png
            ...
            seed_4/frame_7_t0000.png
        cosine_schedule_frames/
            seed_0/...
            ...
        exponential_schedule_frames/
            ...
    fig2/
        ...

Schedule-mismatch caveat
------------------------
The default checkpoint (google/ddpm-cifar10-32) was trained on the
linear schedule; the cosine and exponential schedules feed it ᾱ_t
curves it never saw at training time. The trajectories still visualize
how the schedule shapes the reverse path, but they are not what you
would get from a denoiser natively trained for each schedule. Pass
--model to swap in a checkpoint trained for a different schedule.

Install / run
-------------
    pip install diffusers torch pillow numpy tqdm
    python generate_cache.py --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ─── Constants (mirror the JS) ───────────────────────────────────────────────
T = 1000
DENOISE_FRAMES = 8
DEFAULT_NUM_STEPS = 100     # evenly-spaced sample transitions in the denoise range
DEFAULT_ETA = 0.0           # 0.0 = deterministic DDIM (cleaner output);
                            # 1.0 = DDPM marginals (more variety, more noise)
DEFAULT_STRENGTHS = [0.3, 0.5, 0.7, 0.9]  # SDEdit img2img strengths (timestep-mode).
                            # All schedules share t_start = round(s · T), so the
                            # *resulting* ᾱ_{t_start} differs by schedule.

DEFAULT_TARGET_ALPHAS = [0.8, 0.6, 0.4, 0.2, 0.1]  # Target ᾱ values (alphabar-mode).
                            # Each schedule's t_start is found by inverting its
                            # ᾱ_t curve so all schedules denoise from the SAME
                            # noise level for fair side-by-side comparison.


# ─── Schedules ───────────────────────────────────────────────────────────────
def linear_schedule() -> np.ndarray:
    beta_min, beta_max = 1e-4, 0.02
    a = np.zeros(T + 1)
    a[0] = 1.0
    for t in range(1, T + 1):
        beta = beta_min + (beta_max - beta_min) * (t - 1) / (T - 1)
        a[t] = a[t - 1] * (1.0 - beta)
    return a


def cosine_schedule() -> np.ndarray:
    s = 0.008
    a = np.zeros(T + 1)
    f0 = math.cos((s / (1 + s)) * math.pi / 2) ** 2
    for t in range(T + 1):
        f_t = math.cos(((t / T + s) / (1 + s)) * math.pi / 2) ** 2
        a[t] = f_t / f0
    return a


def exponential_schedule() -> np.ndarray:
    lam, gamma = 12.0, 4.0
    a = np.zeros(T + 1)
    a[0] = 1.0
    for t in range(1, T + 1):
        progress = t / T
        a[t] = math.exp(-lam * progress ** gamma)
    return a


SCHEDULES: Dict[str, np.ndarray] = {
    "linear":      linear_schedule(),
    "cosine":      cosine_schedule(),
    "exponential": exponential_schedule(),
}


# ─── Sampling ────────────────────────────────────────────────────────────────
def stochastic_ddim_step(
    x: torch.Tensor,
    eps: torch.Tensor,
    aBar_t: float,
    aBar_p: float,
    z: torch.Tensor,
    eta: float = 1.0,
) -> torch.Tensor:
    """One η-stochastic DDIM step (η=1 ↔ DDPM marginals)."""
    sa_t = math.sqrt(max(aBar_t, 0.0))
    sn_t = math.sqrt(max(1.0 - aBar_t, 1e-12))
    sa_p = math.sqrt(max(aBar_p, 0.0))
    x0_pred = (x - sn_t * eps) / max(sa_t, 1e-12)

    if aBar_p < 1e-12:
        sigma_sq = 0.0
    else:
        sigma_sq = (1.0 - aBar_p) / max(1.0 - aBar_t, 1e-12) * (1.0 - aBar_t / aBar_p)
        sigma_sq = max(sigma_sq, 0.0)
    sigma = math.sqrt(sigma_sq) * eta
    dir_term = math.sqrt(max(1.0 - aBar_p - sigma ** 2, 0.0)) * eps

    return sa_p * x0_pred + dir_term + sigma * z


def get_sample_timesteps(num_steps: int, t_start: int = T) -> List[int]:
    """Evenly-spaced sampling timesteps from t_start down to 0 inclusive.
    Length = num_steps + 1. For full DDPM generation pass t_start=T; for
    SDEdit-style img2img denoising pass t_start = round(strength · T)."""
    return np.linspace(t_start, 0, num_steps + 1).round().astype(int).tolist()


def compute_forward_frame_ts() -> List[int]:
    """Forward strip always shows the full [0, T] noising progression so the
    schedule comparison is fair regardless of denoising strength."""
    return np.linspace(T, 0, DENOISE_FRAMES).round().astype(int).tolist()


def compute_denoise_frame_ts(num_steps: int, t_start: int) -> List[int]:
    """Denoise strip captures snapshots from t_start → 0 (image-to-image
    range). Picked off the actual sampling grid so file timestep labels and
    saved frames agree."""
    times = get_sample_timesteps(num_steps, t_start)
    n = len(times) - 1
    idx = sorted({round(n * f / (DENOISE_FRAMES - 1)) for f in range(DENOISE_FRAMES)})
    return [times[i] for i in idx]


def strength_tag(strength: float) -> str:
    """Strength → 's030', 's050', 's070', etc. Used as a directory-name
    component so each strength's denoise frames live in their own folder."""
    return "s" + str(int(round(strength * 100))).zfill(3)


def alpha_tag(target_alpha: float) -> str:
    """Target ᾱ → 'a080', 'a060', 'a010', etc. Used as the directory name for
    matched-noise-level (alphabar-mode) denoise frames."""
    return "a" + str(int(round(target_alpha * 100))).zfill(3)


def find_t_for_alpha(aBar: np.ndarray, target: float) -> int:
    """Invert ᾱ_t to find the t whose ᾱ is closest to `target`. ᾱ_t is
    monotonically decreasing in t (clean → noisy), so binary search works."""
    if target >= aBar[0]:           return 0
    if target <= aBar[len(aBar) - 1]: return len(aBar) - 1
    lo, hi = 0, len(aBar) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if aBar[mid] > target:
            lo = mid
        else:
            hi = mid
    return lo if abs(aBar[lo] - target) <= abs(aBar[hi] - target) else hi


def run_unified(
    model,
    aBar: np.ndarray,
    x0: torch.Tensor,
    seed: int,
    device: torch.device,
    num_steps: int = DEFAULT_NUM_STEPS,
    eta: float = DEFAULT_ETA,
    strength: float = 0.6,
    t_start: int = None,
) -> Tuple[List[Tuple[int, torch.Tensor]], List[Tuple[int, torch.Tensor]]]:
    """
    Unified forward + image-to-image (SDEdit) reverse pipeline.

        ε ~ 𝒩(0, I)               (one per seed, shared by forward & reverse)
        t_start = round(strength · T)
        forward:  x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε   for t ∈ [0, T]   (display)
        bridge:   x_{t_start} = √ᾱ_{t_start} · x_0 + √(1-ᾱ_{t_start}) · ε
        reverse:  η-DDIM-sample x_{t_start} → x_0   (the actual denoising)

    With strength < 1 the source's structure is preserved through reverse,
    so the trajectory reads as "denoise THIS image" rather than "imagine a
    new image from pure noise".

    Returns (forward_snaps, denoise_snaps) — each a list of (t, tensor) of
    length DENOISE_FRAMES. Forward spans [0, T]; denoise spans [0, t_start].

    If `t_start` is provided it takes precedence over `strength` (this is what
    the alphabar-mode generation pass uses, so each schedule lands at the same
    target ᾱ regardless of curve shape).
    """
    if t_start is None:
        t_start = max(1, min(T, int(round(strength * T))))
    else:
        t_start = max(1, min(T, int(t_start)))

    # ε for forward (and reverse start) — generator advanced once.
    g_eps = torch.Generator(device="cpu").manual_seed(int(seed))
    eps = torch.randn(x0.shape, generator=g_eps).to(device)

    # ── Forward task: closed-form x_t across the full [0, T] range ──────
    forward_ts = compute_forward_frame_ts()
    forward_snaps: List[Tuple[int, torch.Tensor]] = []
    for t in forward_ts:
        sa = math.sqrt(max(float(aBar[t]), 0.0))
        sn = math.sqrt(max(1.0 - float(aBar[t]), 1e-12))
        x_t = sa * x0 + sn * eps
        forward_snaps.append((t, x_t[0].detach().cpu().clone()))

    # ── Reverse task: img2img — start from x_{t_start}, NOT x_T ─────────
    aBar_s = float(aBar[t_start])
    sa_s = math.sqrt(max(aBar_s, 0.0))
    sn_s = math.sqrt(max(1.0 - aBar_s, 1e-12))
    x = sa_s * x0 + sn_s * eps   # ← x_{t_start}, the actual denoiser input

    # Sampling grid for reverse: t_start → 0 with `num_steps` transitions.
    times = get_sample_timesteps(num_steps, t_start)
    n = len(times) - 1
    snap_indices = sorted({round(n * f / (DENOISE_FRAMES - 1)) for f in range(DENOISE_FRAMES)})
    snap_idx_set = set(snap_indices)

    # z's for each reverse step — separate stream so it doesn't disturb ε.
    g_z = torch.Generator(device="cpu").manual_seed(int(seed) + 1_000_003)

    denoise_snaps: List[Tuple[int, torch.Tensor]] = []
    if 0 in snap_idx_set:
        denoise_snaps.append((times[0], x[0].detach().cpu().clone()))

    for s in range(n):
        t = times[s]
        t_prev = times[s + 1]
        with torch.no_grad():
            t_in = torch.tensor([max(t, 0)], device=device, dtype=torch.long)
            eps_pred = model(x, t_in).sample

        if t_prev > 0 and eta > 0.0:
            z = torch.randn(x.shape, generator=g_z).to(device)
        else:
            z = torch.zeros_like(x)

        x = stochastic_ddim_step(x, eps_pred, float(aBar[t]), float(aBar[t_prev]), z, eta=eta)

        if (s + 1) in snap_idx_set:
            denoise_snaps.append((t_prev, x[0].detach().cpu().clone()))

    return forward_snaps, denoise_snaps


# ─── Helpers ─────────────────────────────────────────────────────────────────
def tensor_to_image(t: torch.Tensor, target_size: int) -> Image.Image:
    arr = t.cpu().numpy()
    arr = (arr + 1.0) * 127.5
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))
    img = Image.fromarray(arr, mode="RGB")
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.NEAREST)
    return img


def load_x0(path: Path, model_size: int, device: torch.device) -> torch.Tensor:
    pil = Image.open(path).convert("RGB")
    # Cover-fit crop so non-square images aren't squashed.
    w, h = pil.size
    s = max(model_size / w, model_size / h)
    pil = pil.resize((int(round(w * s)), int(round(h * s))), Image.LANCZOS)
    left = (pil.width - model_size) // 2
    top = (pil.height - model_size) // 2
    pil = pil.crop((left, top, left + model_size, top + model_size))
    arr = np.array(pil, dtype=np.float32) / 127.5 - 1.0  # H × W × 3 in [-1, 1]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def pick_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")




# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--images-dir", default="images",
                        help="directory of input PNG/JPG source images (default: images/)")
    parser.add_argument("--results-dir", default="results",
                        help="root output directory (default: results/)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="list of seeds (default: 0 1 2 3 4)")
    parser.add_argument("--model", default="google/ddpm-cat-256",
                        help="HuggingFace UNet2DModel checkpoint. Defaults to a 256×256 "
                             "checkpoint for high-quality output. For a lighter/faster run "
                             "use google/ddpm-cifar10-32 (32×32, ~30MB) or any 256×256 "
                             "DDPM (celebahq, bedroom, church, cat).")
    parser.add_argument("--frame-size", type=int, default=0,
                        help="output PNG size in px. 0 (default) keeps the model's native "
                             "resolution (no resampling). Specify a value to NEAREST-resize "
                             "the saved frames.")
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS,
                        help=f"reverse-sampling steps from t_start→0 (default {DEFAULT_NUM_STEPS}). "
                             "More = smoother / cleaner output, linearly more compute.")
    parser.add_argument("--eta", type=float, default=DEFAULT_ETA,
                        help=f"DDIM stochasticity (default {DEFAULT_ETA}). 0=deterministic "
                             "(cleanest), 1=DDPM marginals (more variety, more residual noise).")
    parser.add_argument("--strengths", type=float, nargs="+", default=DEFAULT_STRENGTHS,
                        help=f"SDEdit img2img strengths (timestep-mode, default {DEFAULT_STRENGTHS}). "
                             "All schedules use the same t_start = round(s · T); the resulting "
                             "ᾱ_{t_start} differs by schedule. Pass [] to skip strength caches.")
    parser.add_argument("--target-alphas", type=float, nargs="+", default=DEFAULT_TARGET_ALPHAS,
                        help=f"Matched-noise-level targets (alphabar-mode, default "
                             f"{DEFAULT_TARGET_ALPHAS}). Each schedule's t_start is solved by "
                             f"inverting its ᾱ_t curve so all schedules denoise from the SAME "
                             f"ᾱ. Pass [] to skip target-alpha caches.")
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    parser.add_argument("--schedules", nargs="+", default=list(SCHEDULES.keys()),
                        help="subset of schedules to render")
    parser.add_argument("--exts", nargs="+", default=["png", "jpg", "jpeg", "webp"],
                        help="source image extensions to discover")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    results_dir = Path(args.results_dir)
    if not images_dir.exists():
        sys.exit(f"error: {images_dir}/ does not exist — create it and drop PNGs in it")
    image_paths = sorted(
        p for ext in args.exts for p in images_dir.glob(f"*.{ext}")
    )
    if not image_paths:
        sys.exit(f"error: no images in {images_dir}/ matching extensions {args.exts}")

    device = pick_device(args.device)
    print(f"Device: {device}")

    try:
        from diffusers import UNet2DModel
    except ImportError:
        sys.exit("error: install dependencies first — pip install diffusers torch pillow numpy tqdm")

    print(f"Loading model: {args.model}")
    model = UNet2DModel.from_pretrained(args.model).to(device).eval()
    model_size = int(model.config.sample_size)
    print(f"Model native resolution: {model_size}×{model_size}")
    print(f"Found {len(image_paths)} source image(s): {[p.name for p in image_paths]}")

    schedules_to_run = [s for s in args.schedules if s in SCHEDULES]
    if not schedules_to_run:
        sys.exit(f"error: no valid schedules in {args.schedules}; choose from {list(SCHEDULES)}")

    strengths = sorted(set(round(s, 3) for s in args.strengths if 0.0 < s <= 1.0))
    target_alphas = sorted(
        {round(a, 3) for a in args.target_alphas if 0.0 < a < 1.0},
        reverse=True,  # high ᾱ (clean) first → consistent slider ordering
    )
    if not strengths and not target_alphas:
        sys.exit("error: nothing to generate — provide --strengths and/or --target-alphas")

    results_dir.mkdir(parents=True, exist_ok=True)
    target_size = args.frame_size if args.frame_size > 0 else model_size

    image_ids: List[str] = []
    pbar = tqdm(
        total=len(image_paths) * len(schedules_to_run) * len(args.seeds)
              * (len(strengths) + len(target_alphas)),
        desc="trajectories", unit="traj",
    )
    # forward frames written once per (image, schedule, seed) — they don't
    # depend on strength or target-alpha (forward is closed-form x_t over [0, T]).
    forward_done: set = set()

    for img_path in image_paths:
        img_id = img_path.stem
        image_ids.append(img_id)
        x0 = load_x0(img_path, model_size, device)

        for sched_name in schedules_to_run:
            aBar = SCHEDULES[sched_name]
            sched_dir = results_dir / img_id / f"{sched_name}_schedule_frames"
            sched_dir.mkdir(parents=True, exist_ok=True)

            for seed in args.seeds:
                seed_dir = sched_dir / f"seed_{seed}"
                seed_dir.mkdir(exist_ok=True)

                # ── Strength-indexed caches (timestep mode) ────────────────
                for strength in strengths:
                    fwd_snaps, dn_snaps = run_unified(
                        model, aBar, x0,
                        seed=seed, device=device,
                        num_steps=args.num_steps, eta=args.eta, strength=strength,
                    )
                    fwd_key = (img_id, sched_name, seed)
                    if fwd_key not in forward_done:
                        for i, (t, x_t) in enumerate(fwd_snaps):
                            tensor_to_image(x_t, target_size).save(
                                seed_dir / f"forward_{i}_t{t:04d}.png"
                            )
                        forward_done.add(fwd_key)

                    s_dir = seed_dir / strength_tag(strength)
                    s_dir.mkdir(exist_ok=True)
                    for i, (t, x_t) in enumerate(dn_snaps):
                        tensor_to_image(x_t, target_size).save(
                            s_dir / f"denoise_{i}_t{t:04d}.png"
                        )
                    pbar.update(1)

                # ── Target-α caches (alphabar mode) ────────────────────────
                # For each target ᾱ we resolve t_start by inverting THIS
                # schedule's curve, so all schedules denoise from the same ᾱ.
                for target_alpha in target_alphas:
                    t_start_a = max(1, find_t_for_alpha(aBar, target_alpha))
                    fwd_snaps_a, dn_snaps_a = run_unified(
                        model, aBar, x0,
                        seed=seed, device=device,
                        num_steps=args.num_steps, eta=args.eta, t_start=t_start_a,
                    )
                    # Forward gets written once per (img, sched, seed) and is
                    # already done from the strength loop (or, if strengths
                    # list was empty, write it from the first alpha pass).
                    fwd_key = (img_id, sched_name, seed)
                    if fwd_key not in forward_done:
                        for i, (t, x_t) in enumerate(fwd_snaps_a):
                            tensor_to_image(x_t, target_size).save(
                                seed_dir / f"forward_{i}_t{t:04d}.png"
                            )
                        forward_done.add(fwd_key)

                    a_dir = seed_dir / alpha_tag(target_alpha)
                    a_dir.mkdir(exist_ok=True)
                    for i, (t, x_t) in enumerate(dn_snaps_a):
                        tensor_to_image(x_t, target_size).save(
                            a_dir / f"denoise_{i}_t{t:04d}.png"
                        )
                    pbar.update(1)
    pbar.close()

    denoise_ts_per_strength = {}
    t_start_per_strength = {}
    for strength in strengths:
        t_start = max(1, min(T, int(round(strength * T))))
        t_start_per_strength[strength_tag(strength)] = t_start
        denoise_ts_per_strength[strength_tag(strength)] = compute_denoise_frame_ts(
            args.num_steps, t_start
        )

    # Per-(schedule, alpha_tag) → t_start and denoise frame timesteps. The
    # frontend uses these in alphabar mode to construct PNG URLs and to
    # display each schedule's resolved t* under the same target ᾱ.
    t_start_per_alpha_per_sched: Dict[str, Dict[str, int]] = {}
    denoise_ts_per_alpha_per_sched: Dict[str, Dict[str, List[int]]] = {}
    for sched_name in schedules_to_run:
        aBar = SCHEDULES[sched_name]
        t_start_per_alpha_per_sched[sched_name] = {}
        denoise_ts_per_alpha_per_sched[sched_name] = {}
        for target_alpha in target_alphas:
            tag = alpha_tag(target_alpha)
            t_start_a = max(1, find_t_for_alpha(aBar, target_alpha))
            t_start_per_alpha_per_sched[sched_name][tag] = t_start_a
            denoise_ts_per_alpha_per_sched[sched_name][tag] = compute_denoise_frame_ts(
                args.num_steps, t_start_a
            )

    manifest = {
        "model": args.model,
        "model_native_size": model_size,
        "frame_size": target_size,
        "T": T,
        "num_steps": args.num_steps,
        "eta": args.eta,
        "strengths": strengths,
        "strength_tags": {strength_tag(s): s for s in strengths},
        "t_start_per_strength": t_start_per_strength,
        "target_alphas": target_alphas,
        "alpha_tags": {alpha_tag(a): a for a in target_alphas},
        "t_start_per_alpha_per_schedule": t_start_per_alpha_per_sched,
        "denoise_timesteps_per_alpha_per_schedule": denoise_ts_per_alpha_per_sched,
        "frames_per_trajectory": DENOISE_FRAMES,
        "forward_timesteps": compute_forward_frame_ts(),
        "denoise_timesteps_per_strength": denoise_ts_per_strength,
        "forward_filename": "forward_{i}_t{t:04d}.png",
        "denoise_filename_strength": "{strength_tag}/denoise_{i}_t{t:04d}.png",
        "denoise_filename_alpha":    "{alpha_tag}/denoise_{i}_t{t:04d}.png",
        "seeds": list(args.seeds),
        "schedules": schedules_to_run,
        "images": image_ids,
        "pipeline": (
            f"shared ε per (image, schedule, seed) across all strengths; "
            f"forward x_t closed-form for t ∈ [0, T]; "
            f"reverse (img2img) starts from x_{{round(s · T)}} for each "
            f"s ∈ {strengths} and runs η={args.eta} DDIM, "
            f"{args.num_steps} steps → 0"
        ),
    }
    manifest_path = results_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest → {manifest_path}")
    print(f"Done. Cache root: {results_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
