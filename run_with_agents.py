from __future__ import annotations
import os, time, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import imageio.v2 as imageio

import mini_fire_model as m
from agents import Agent, AgentType, step_agents

def spawn_agents_near_spark(
    state: np.ndarray,
    fuel_id: np.ndarray,
    spark_center: tuple[int,int],
    count: int,
    ring_min: int = 10,
    ring_max: int = 25,
    min_separation: int = 0,   # set >0 to keep agents apart by this many cells (Chebyshev dist)
    rng: np.random.Generator | None = None,
) -> list[tuple[int,int]]:
    """
    Return up to `count` (r,c) spawn cells:
      - not water / non-burnable
      - not currently burning
      - within a distance ring around spark_center
      - unique positions; optional min_separation between them
    Falls back by widening ring, then anywhere valid if ring has too few cells.
    """
    if rng is None:
        rng = np.random.default_rng()

    rows, cols = state.shape
    sy, sx = spark_center

    burnable = (fuel_id > 0) & (fuel_id != 4)  # exclude non-burn + water
    not_burning = (state != 2)

    y, x = np.ogrid[:rows, :cols]
    dist = np.hypot(y - sy, x - sx)

    def candidates_in_ring(rmin, rmax):
        ring = (dist >= rmin) & (dist <= rmax)
        return np.argwhere(burnable & not_burning & ring)

    # 1) try requested ring
    cand = candidates_in_ring(ring_min, ring_max)

    # 2) widen ring progressively if needed
    if cand.size < count:
        for grow in (10, 25, 50, 100):
            cand = candidates_in_ring(max(0, ring_min - grow), ring_max + grow)
            if cand.size >= count:
                break

    # 3) final fallback: anywhere valid
    if cand.size == 0:
        cand = np.argwhere(burnable & not_burning)

    # sample without replacement
    rng.shuffle(cand)
    chosen = []

    if min_separation <= 0:
        k = min(count, len(cand))
        chosen = [tuple(map(int, rc)) for rc in cand[:k]]
        return chosen

    # enforce min spacing (greedy)
    def too_close(r, c):
        for rr, cc in chosen:
            if max(abs(r-rr), abs(c-cc)) < min_separation:
                return True
        return False

    for r, c in cand:
        r, c = int(r), int(c)
        if not too_close(r, c):
            chosen.append((r, c))
            if len(chosen) == count:
                break

    return chosen

def main():
    if getattr(m, "SEED", None) is not None:
        random.seed(m.SEED)
        np.random.seed(m.SEED)

    # Init world from your existing model
    state, fuel_id, alt, fuel_load, spark_center = m.init_world(m.ROWS, m.COLS)
    wind_uv = m.unit_vec_from_deg(m.WIND_DIR_DEG) if m.USE_WIND else np.array([0.0, 0.0], dtype=np.float32)
    slope_weight = m.slope_factor(alt) if m.USE_ALTITUDE else np.ones((m.ROWS, m.COLS), dtype=np.float32)

    # Organic extras if enabled
    KERNEL_OFFSETS = m.build_radial_offsets(m.KERNEL_RADIUS, m.KERNEL_SIGMA) if getattr(m, "USE_RADIAL_KERNEL", False) else []
    P0_NOISE = m.make_smooth_noise(m.ROWS, m.COLS, m.NOISE_SMOOTH_PASSES, m.NOISE_STRENGTH) \
               if getattr(m, "USE_STATIC_NOISE", False) else np.ones((m.ROWS, m.COLS), dtype=np.float32)
    gust_std = m.GUST_STD if getattr(m, "USE_GUST_JITTER", False) else 0.0

    # --- Spawn multiple agents randomly in a ring around the spark ---
    NUM_CREW = 3
    SPAWN_RING_MIN = 200   # inner radius
    SPAWN_RING_MAX = 201  # outer radius
    MIN_SEP = 100           # min spacing between spawns;

    rng = np.random.default_rng(m.SEED) if getattr(m, "SEED", None) is not None else None
    crew_positions = spawn_agents_near_spark(
        state=state,
        fuel_id=fuel_id,
        spark_center=spark_center,
        count=NUM_CREW,
        ring_min=SPAWN_RING_MIN,
        ring_max=SPAWN_RING_MAX,
        min_separation=MIN_SEP,
        rng=rng,
    )

    agents = [Agent(kind=AgentType.CREW, r=r, c=c) for (r, c) in crew_positions]

    # Persistent retardant/cooldown
    cooldown_map = np.zeros((m.ROWS, m.COLS), dtype=np.float32)

    plt.figure("Wildfire + Agents", figsize=(6, 6))
    frames = []

    for t in range(m.STEPS):
        # Agents act → (instant p0 dampening, updated cooldown persistence)
        instant_p0_mult, cooldown_map = step_agents(agents, state, fuel_id, t, cooldown_map)

        # Total p0 multiplier = persistent cooldown + instant effect
        p0_multiplier = instant_p0_mult * (1.0 - cooldown_map)

        # Draw every 2 steps
        if t % 2 == 0:
            plt.clf()
            plt.title(f"t = {t}")
            img = m.make_rgb_img(state, fuel_id)
            plt.imshow(img, interpolation="nearest")

            # Draw agents
            ax = [a.c for a in agents]; ay = [a.r for a in agents]
            if ax:
                plt.scatter(ax, ay, s=18, marker='s', edgecolors='white', linewidths=0.6)

            legend_patches = [
                mpatches.Patch(color=m.FUEL_COLOR_MAP[0], label="Non-burnable"),
                mpatches.Patch(color=m.FUEL_COLOR_MAP[1], label="Grass"),
                mpatches.Patch(color=m.FUEL_COLOR_MAP[2], label="Shrub"),
                mpatches.Patch(color=m.FUEL_COLOR_MAP[3], label="Timber"),
                mpatches.Patch(color=m.FUEL_COLOR_MAP[4], label="Water"),
                mpatches.Patch(color=m.COLOR_BURNING, label="Burning"),
                mpatches.Patch(color=m.COLOR_BURNED, label="Burned"),
            ]
            plt.legend(handles=legend_patches, loc="lower center", ncol=4, frameon=False)
            plt.axis("off")
            plt.pause(0.001)

            # Capture figure (so agents appear)
            if hasattr(m, "capture_fig_rgb"):
                frames.append(m.capture_fig_rgb())
            else:
                frames.append((np.clip(img, 0.0, 1.0) * 255).astype(np.uint8))

            if t % 10 == 0:
                print(f"t={t:03d} burning={(state == 2).sum():4d}  burned={(state == 3).sum():4d}")

        # Advance CA with our combined p0 multiplier
        state, fuel_load = m.step(
            state, fuel_id, alt, wind_uv, slope_weight, fuel_load,
            p0_multiplier=p0_multiplier,
            kernel_offsets=KERNEL_OFFSETS,
            p0_noise=P0_NOISE,
            gust_std=gust_std,
        )

    # Final frame
    plt.clf()
    plt.title(f"t = {m.STEPS}")
    img = m.make_rgb_img(state, fuel_id)
    plt.imshow(img, interpolation="nearest")
    ax = [a.c for a in agents]; ay = [a.r for a in agents]
    if ax:
        plt.scatter(ax, ay, s=18, marker='s', edgecolors='white', linewidths=0.6)
    plt.axis("off")
    if hasattr(m, "capture_fig_rgb"):
        frames.append(m.capture_fig_rgb())
    else:
        frames.append((np.clip(img, 0.0, 1.0) * 255).astype(np.uint8))

    # Save GIF
    os.makedirs(m.GIF_DIR, exist_ok=True)
    out_path = os.path.join(m.GIF_DIR, f"wildfire_agents_{time.strftime('%Y%m%d_%H%M%S')}.gif")
    imageio.mimsave(out_path, frames, duration=1.0 / m.GIF_FPS)
    print(f"[INFO] Saved GIF with {len(frames)} frames to: {out_path}")

    plt.show()

if __name__ == "__main__":
    main()
