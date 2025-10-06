"""
Mini Wildfire Sim
- States: 0=EMPTY (no fuel), 1=FUEL, 2=BURNING, 3=BURNED
- Optional fuel types (1..3) change how easy it is to ignite and keep burning
- Optional wind
- Optional PNG maps for fuel_id and altitude (grayscale)

Run:
    pip install numpy matplotlib pillow
    python mini_fire_model.py
"""

from __future__ import annotations
import os
import math
import random
from dataclasses import dataclass
from typing import Optional
import rasterio
from rasterio.enums import Resampling

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
import time
import imageio.v2 as imageio



from matplotlib.colors import to_rgb

COLOR_BURNING = "#ff0000"   # red
COLOR_BURNED  = "#000000"   # black

FUEL_COLOR_MAP = {
    0: "#9e9e9e",  # non-burnable (gray)
    1: "#08ff38",  # grass
    2: "#03a323",  # shrub
    3: "#126923",  # timber
    4: "#2196f3", #water (blue)
}
# -----------------------------
# Config (change me freely)
# -----------------------------
ROWS, COLS = 800, 800         # grid size
STEPS = 400                   # simulation steps
SEED = 42                    # set None for randomness

# toggles
USE_WIND = True
USE_FUEL_TYPES = True
USE_ALTITUDE = False          # slope effect (needs altitude map or will be flat)
USE_FUELMAP_PNG = False       # read data/fuel_id.png (0..255 -> classes)
USE_ALTIMAP_PNG = False       # read data/altitude.png (grayscale -> meters-ish)

# base physics knobs
P0_IGNITE = 0.58           # base ignition contribution from burning neighbors
WIND_STRENGTH = 100.0           # 0..1 (how much wind biases spread)
WIND_DIR_DEG = 180           # where wind is blowing TOWARD (0=right/E, 90=down/S)
SLOPE_GAIN = 0.04             # how strongly upslope helps (only if USE_ALTITUDE)

# Require at least K burning neighbors before a fuel cell can ignite
K_NEIGHBORS = {1: 1, 2: 1, 3: 2}   # grass=1, shrub=1, timber=2

# initial ignition pattern
IGNITE_CENTER = True          # center spark
IGNITE_RANDOM_N = 0           # or N random sparks

# optional PNG paths (put files under ./data/)
FUELMAP_PATH = os.path.join("data", "fuel_id.png")
ALTIMAP_PATH = os.path.join("data", "altitude.png")


#gif settings
GIF_DIR = "gifs"
GIF_FPS = 8          # playback speed
GIF_EVERY_N = 2      # capture every Nth step (you currently draw every 2nd)

def capture_fig_rgb() -> np.ndarray:
    """Return the current Matplotlib figure as an RGB uint8 array (H,W,3)."""
    fig = plt.gcf()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    # works on Agg, InterAgg, QtAgg, etc.
    buf = fig.canvas.buffer_rgba()
    frame_rgba = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    return frame_rgba[..., :3].copy()



# -----------------------------
# Fuel model
# -----------------------------
@dataclass
class FuelModel:
    name: str
    ignite_mult: float   # scales how easy a fuel cell ignites
    sustain_p: float     # per-step keep-burning probability

# 1..3 id -> model
FUEL_MODELS = {
    1: FuelModel("Grass",   ignite_mult=1.30, sustain_p=0.50),
    2: FuelModel("Shrub",   ignite_mult=1.00, sustain_p=0.68),
    3: FuelModel("Timber",  ignite_mult=0.55, sustain_p=0.92),
}
DEFAULT_FUEL_ID = 2

# ---- Fuel load model ----
USE_FUEL_LOAD = True

# initial fuel (arbitrary units) per fuel type id; water(4) & nonburn(0) = 0
FUEL_LOAD_INIT = {0: 0.0, 1: 1.2, 2:1.8, 3:2.2, 4: 0.0}
# fuel consumption per burning step (units/step) per fuel type
CONSUME_PER_STEP = {1:0.7, 2:0.50, 3:0.35}

# how strongly local remaining fuel affects ignition (0..1)
LOCAL_FUEL_WEIGHT = 0.4


# Density multiplier (optional, keep simple = 1.0)
DENSITY_MULT = {1: 0.95, 2: 1.00, 3: 1.0}

# -----------------------------
# Helpers
# -----------------------------
def unit_vec_from_deg(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    # screen coords: +x right, +y down (to match array rows/cols)
    return np.array([math.cos(rad), math.sin(rad)], dtype=np.float32)

def load_grayscale_png(path: str, out_shape: tuple[int, int]) -> Optional[np.ndarray]:
    try:
        from PIL import Image
    except ImportError:
        return None
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert("L").resize((out_shape[1], out_shape[0]), resample=Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    return arr

def neighbor_burning_count(state: np.ndarray) -> np.ndarray:
    """Count burning neighbors (8-neighborhood)."""
    burning = (state == 2).astype(np.int16)
    s = np.zeros_like(burning, dtype=np.int16)
    # 8 shifts (no wrap)
    s[:-1, :-1] += burning[1:, 1:]
    s[:-1,  0:] += burning[1:, 0:]
    s[:-1, 1: ] += burning[1:, :-1]
    s[0: , :-1] += burning[0:, 1:]
    s[0: , 1: ] += burning[0:, :-1]
    s[1: , :-1] += burning[:-1, 1:]
    s[1: ,  0:] += burning[:-1, 0:]
    s[1: , 1: ] += burning[:-1, :-1]
    return s

def simple_wind_weight(nbr_dx: int, nbr_dy: int, wind_uv: np.ndarray, strength: float) -> float:
    if strength <= 0.0:
        return 1.0
    v = np.array([nbr_dx, nbr_dy], dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm == 0:
        return 1.0
    v /= norm
    cosang = float(np.clip(np.dot(v, wind_uv), -1.0, 1.0))
    w = 1.0 + strength * cosang
    return max(0.0, w)   # <-- clamp so wind never subtracts pressure below zero


def slope_factor(alt: np.ndarray) -> np.ndarray:
    """
    Very simple slope helper: returns a factor per cell that increases with slope.
    Here we use gradient magnitude as a proxy (no direction).
    If you want upslope-only wrt each neighbor, keep USE_ALTITUDE=False here for simplicity.
    """
    if alt is None:
        return np.ones((ROWS, COLS), dtype=np.float32)
    gy, gx = np.gradient(alt.astype(np.float32))  # rows (y), cols (x)
    slope_mag = np.hypot(gx, gy)
    return np.exp(SLOPE_GAIN * slope_mag).astype(np.float32)

def fuel_from_worldcover(rows: int, cols: int, tif_path: str = "data/worldcover.tif") -> np.ndarray:
    """
    Read an ESA WorldCover GeoTIFF tile and map its land cover classes to simple fuel IDs:
      1 = Grass, 2 = Shrub, 3 = Timber, 0 = non-burnable (water/urban/etc.)
    The raster is resampled to (rows, cols) with nearest-neighbor.
    """
    with rasterio.open(tif_path) as src:
        lc = src.read(1, out_shape=(rows, cols), resampling=Resampling.nearest).astype(np.int32)

    # WorldCover classes (common ones):
    # 10 Tree cover, 20 Shrubland, 30 Grassland, 40 Cropland,
    # 50 Built-up, 60 Bare, 70 Snow/Ice, 80 Water, 90 Herbaceous wetland, 95 Mangrove, 100 Moss/Lichen.
    fuel = np.zeros_like(lc, dtype=np.int16)  # default 0 = non-burnable

    fuel[lc == 30] = 1   # Grassland -> Grass
    fuel[lc == 20] = 2   # Shrubland -> Shrub
    fuel[lc == 10] = 3   # Tree cover -> Timber
    fuel[lc == 90] = 1   # Herbaceous wetland -> treat like Grass (optional)
    fuel[lc == 80] = 4  # Water -> WATER (non-burnable, but we still color it)  ← NEW

    return fuel


def init_world(rows: int, cols: int):
    # states
    state = np.ones((rows, cols), dtype=np.int16)  # 1=FUEL everywhere by default

    # optional: fuel ids
    if USE_FUEL_TYPES:
        fuel_id = fuel_from_worldcover(rows, cols)  # <- ensure this runs
        state[(fuel_id == 0) | (fuel_id == 4)] = 0  # block ignition on non-burn + water  ← NEW
    else:
        fuel_id = np.full((rows, cols), 2, dtype=np.int16)

    # per-cell fuel load
    if USE_FUEL_LOAD:
        fuel_load = np.zeros((rows, cols), dtype=np.float32)
        for fid, start in FUEL_LOAD_INIT.items():
            fuel_load[fuel_id == fid] = float(start)
    else:
        fuel_load = np.zeros((rows, cols), dtype=np.float32)

    print("fuel counts:",
          "grass=", int((fuel_id == 1).sum()),
          "shrub=", int((fuel_id == 2).sum()),
          "timber=", int((fuel_id == 3).sum()),
          "water=", int((fuel_id == 4).sum()),  # ← NEW
          "nonburn=", int((fuel_id == 0).sum()))

    # optional altitude
    if USE_ALTITUDE:
        if USE_ALTIMAP_PNG:
            alt = load_grayscale_png(ALTIMAP_PATH, (rows, cols))
            if alt is None:
                alt = np.zeros((rows, cols), dtype=np.float32)
            else:
                # normalize ~ meters
                alt = (alt - alt.min()) / max(1.0, (alt.max() - alt.min()))
                alt *= 500.0
        else:
            # gentle hill
            x = np.linspace(-1, 1, cols, dtype=np.float32)
            y = np.linspace(-1, 1, rows, dtype=np.float32)
            X, Y = np.meshgrid(x, y)
            alt = 200.0 * (X**2 + Y**2)
    else:
        alt = np.zeros((rows, cols), dtype=np.float32)

    # initial ignitions: so we dont start on a non buraneble
    if IGNITE_CENTER:
        r0, c0 = rows // 2, cols // 2
        if state[r0, c0] != 1:  # center isn't fuel → move to nearest fuel cell
            print(f"[WARN] Center ignition at ({r0},{c0}) is non-burnable "
                  f"(fuel_id={int(fuel_id[r0, c0])}). Snapping to nearest fuel...")
            rr, cc = np.where(state == 1)
            if rr.size > 0:
                k = np.argmin((rr - r0) ** 2 + (cc - c0) ** 2)
                r0, c0 = int(rr[k]), int(cc[k])
            else:
                print("No burnable cells found for ignition.")
        state[r0, c0] = 2

    # add N random ignitions, each forced onto a fuel cell
    for _ in range(IGNITE_RANDOM_N):
        placed = False
        for _ in range(200):  # try a few times to hit a fuel cell
            r = random.randrange(rows)
            c = random.randrange(cols)
            if state[r, c] == 1:  # only ignite where there's fuel
                state[r, c] = 2
                placed = True
                break
        if not placed:
            # fallback: if random sampling didn't find fuel, drop one at the nearest fuel to center
            rr, cc = np.where(state == 1)
            if rr.size > 0:
                k = np.argmin((rr - rows // 2) ** 2 + (cc - cols // 2) ** 2)
                state[int(rr[k]), int(cc[k])] = 2
            else:
                print("No burnable cells found for random ignition.")

    return state, fuel_id, alt, fuel_load


def box_mean_3x3(a: np.ndarray) -> np.ndarray:
    """Mean in a 3x3 neighborhood with proper edge counts (no SciPy)."""
    a = a.astype(np.float32)
    s = np.zeros_like(a, dtype=np.float32)
    c = np.zeros_like(a, dtype=np.float32)

    # center
    s += a; c += 1.0

    # neighbors: accumulate shifted values and counts
    # right / left
    s[:, 1:]  += a[:, :-1]; c[:, 1:]  += 1.0
    s[:, :-1] += a[:, 1:] ; c[:, :-1] += 1.0
    # down / up
    s[1:, :]  += a[:-1, :]; c[1:, :]  += 1.0
    s[:-1, :] += a[1:, :] ; c[:-1, :] += 1.0
    # diagonals
    s[1:, 1:]   += a[:-1, :-1]; c[1:, 1:]   += 1.0
    s[1:, :-1]  += a[:-1, 1:] ; c[1:, :-1]  += 1.0
    s[:-1, 1:]  += a[1:, :-1] ; c[:-1, 1:]  += 1.0
    s[:-1, :-1] += a[1:, 1:]  ; c[:-1, :-1] += 1.0

    return s / np.maximum(c, 1.0)


# -----------------------------
# Main step function
# -----------------------------
def step(state: np.ndarray, fuel_id: np.ndarray, alt: np.ndarray,
         wind_uv: np.ndarray, slope_weight: np.ndarray,
         fuel_load: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    """
    One CA step. Returns new state array.
    """
    rows, cols = state.shape
    new_state = state.copy()

    # compute how many burning neighbors each cell has
    nburn = neighbor_burning_count(state)
    # K-neighbor rule per fuel type
    k_req = np.zeros_like(fuel_id, dtype=np.int16)
    k_req[fuel_id == 1] = K_NEIGHBORS[1]
    k_req[fuel_id == 2] = K_NEIGHBORS[2]
    k_req[fuel_id == 3] = K_NEIGHBORS[3]
    mask_k = (nburn >= k_req)

    # base ignition probability from neighbors:
    # p = 1 - (1 - p0_eff)^nburn
    # we’ll fold wind/fuel/slope into p0_eff below.
    # Prepare p0_eff per cell:
    p0_eff = np.full((rows, cols), P0_IGNITE, dtype=np.float32)
    if USE_FUEL_LOAD:
        # normalize by max initial load to get ~0..1
        max_load = max(v for v in FUEL_LOAD_INIT.values())
        norm_load = np.divide(fuel_load, max_load if max_load > 0 else 1.0, dtype=np.float32)
        local_load = box_mean_3x3(norm_load)  # local remaining fuel fraction
        # mix in: low local fuel -> lower ignition
        p0_eff *= (1.0 - LOCAL_FUEL_WEIGHT) + LOCAL_FUEL_WEIGHT * local_load

    # Fuel multipliers
    if USE_FUEL_TYPES:
        p0_eff = p0_eff * np.where(fuel_id == 1, FUEL_MODELS[1].ignite_mult, 1.0)
        p0_eff = p0_eff * np.where(fuel_id == 2, FUEL_MODELS[2].ignite_mult, 1.0)
        p0_eff = p0_eff * np.where(fuel_id == 3, FUEL_MODELS[3].ignite_mult, 1.0)

    # Slope multiplier (directionless, very simple)
    if USE_ALTITUDE:
        p0_eff = p0_eff * slope_weight

    # Wind multiplier (directional — approximate by weighting neighbor influence)
    # Simpler approach: boost effective nburn by weighted sum of burning neighbor directions.
    if USE_WIND:
        # Build a weighted "burn pressure" from neighbors:
        burn = (state == 2).astype(np.float32)
        press = np.zeros_like(burn, dtype=np.float32)

        # neighbor vectors from neighbor->cell:
        # (dx,dy) relative to array indexing (+x right/+y down)
        shifts = [
            (+1, 0), (-1, 0), (0, +1), (0, -1),
            (+1, +1), (+1, -1), (-1, +1), (-1, -1)
        ]
        for dx, dy in shifts:
            w = simple_wind_weight(dx, dy, wind_uv, WIND_STRENGTH)
            # shift burn opposite direction to accumulate at current cell
            if dx == 1 and dy == 0:
                press[:, 1:] += w * burn[:, :-1]
            elif dx == -1 and dy == 0:
                press[:, :-1] += w * burn[:, 1:]
            elif dx == 0 and dy == 1:
                press[1:, :] += w * burn[:-1, :]
            elif dx == 0 and dy == -1:
                press[:-1, :] += w * burn[1:, :]
            elif dx == 1 and dy == 1:
                press[1:, 1:] += w * burn[:-1, :-1]
            elif dx == 1 and dy == -1:
                press[:-1, 1:] += w * burn[1:, :-1]
            elif dx == -1 and dy == 1:
                press[1:, :-1] += w * burn[:-1, 1:]
            elif dx == -1 and dy == -1:
                press[:-1, :-1] += w * burn[1:, 1:]

        # convert weighted pressure to an effective neighbor count boost
        # small scaling so it's comparable to nburn
        nburn_eff = nburn.astype(np.float32) + 0.25 * press
    else:
        nburn_eff = nburn.astype(np.float32)

    # ignition probability for FUEL cells
    with np.errstate(over='ignore'):
        p_ignite = 1.0 - np.power(1.0 - np.clip(p0_eff, 0.0, 1.0), np.clip(nburn_eff, 0.0, 8.0))
    p_ignite = np.clip(p_ignite, 0.0, 1.0)

    # apply ignition only to FUEL cells (state==1)
    fuel_mask = (state == 1)
    if USE_FUEL_LOAD:
        fuel_mask &= (fuel_load > 0.05)  # need a bit of fuel to ignite
    rnd = np.random.random(size=state.shape).astype(np.float32)
    ignite_mask = fuel_mask & mask_k & (rnd < p_ignite)
    new_state[ignite_mask] = 2  # becomes BURNING

    # burning cells may continue or become burned
    burning_mask = (state == 2)
    if USE_FUEL_TYPES:
        # per-cell sustain prob
        sustain = np.full(state.shape, 0.7, dtype=np.float32)
        sustain = np.where(fuel_id == 1, FUEL_MODELS[1].sustain_p, sustain)
        sustain = np.where(fuel_id == 2, FUEL_MODELS[2].sustain_p, sustain)
        sustain = np.where(fuel_id == 3, FUEL_MODELS[3].sustain_p, sustain)
    else:
        sustain = np.full(state.shape, 0.7, dtype=np.float32)

    keep_mask = (np.random.random(size=state.shape).astype(np.float32) < sustain)

    if USE_FUEL_LOAD:
        # build per-cell consumption rates
        cons = np.zeros_like(fuel_load, dtype=np.float32)
        cons[fuel_id == 1] = CONSUME_PER_STEP[1]
        cons[fuel_id == 2] = CONSUME_PER_STEP[2]
        cons[fuel_id == 3] = CONSUME_PER_STEP[3]

        # subtract fuel where burning
        fuel_load = np.maximum(0.0, fuel_load - cons * burning_mask.astype(np.float32))

        # any burning cell that ran out of fuel becomes burned
        out_mask = burning_mask & (fuel_load <= 1e-6)
        new_state[out_mask] = 3

    # cells that were burning and don't keep -> burned
    to_burned = (burning_mask & ~keep_mask)
    new_state[to_burned] = 3  # BURNED
    # cells that keep burning remain 2 (already 2 in copy)

    return new_state, fuel_load


# -----------------------------
# Main
# -----------------------------

def make_rgb_img(state: np.ndarray, fuel_id: np.ndarray) -> np.ndarray:
    """
    Build an RGB image from state and fuel_id.
    Priority: burning (red) > burned (black) > fuel color.
    """
    rows, cols = state.shape
    img = np.zeros((rows, cols, 3), dtype=np.float32)

    # base: fuel colors (by fuel_id 0..3)
    palette = np.array([to_rgb(FUEL_COLOR_MAP[i]) for i in range(0, 5)], dtype=np.float32)
    img[:] = palette[np.clip(fuel_id, 0, 4)]

    # override for burning and burned
    burning_mask = (state == 2)
    burned_mask  = (state == 3)
    if burning_mask.any():
        img[burning_mask] = to_rgb(COLOR_BURNING)
    if burned_mask.any():
        img[burned_mask] = to_rgb(COLOR_BURNED)

    return img


def main():
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)

    state, fuel_id, alt, fuel_load = init_world(ROWS, COLS)

    wind_uv = unit_vec_from_deg(WIND_DIR_DEG) if USE_WIND else np.array([0.0, 0.0], dtype=np.float32)
    slope_weight = slope_factor(alt) if USE_ALTITUDE else np.ones((ROWS, COLS), dtype=np.float32)

    plt.figure("Mini Wildfire CA", figsize=(6, 6))
    frames = []

    for t in range(STEPS):
        if t % 2 == 0:
            plt.clf()
            plt.title(f"t = {t}")
            img = make_rgb_img(state, fuel_id)      # float32 in [0,1], shape (H,W,3)
            plt.imshow(img, interpolation="nearest")
            legend_patches = [
                mpatches.Patch(color=FUEL_COLOR_MAP[0], label="Non-burnable"),
                mpatches.Patch(color=FUEL_COLOR_MAP[1], label="Grass"),
                mpatches.Patch(color=FUEL_COLOR_MAP[2], label="Shrub"),
                mpatches.Patch(color=FUEL_COLOR_MAP[3], label="Timber"),
                mpatches.Patch(color=FUEL_COLOR_MAP[4], label="Water"),
                mpatches.Patch(color=COLOR_BURNING, label="Burning"),
                mpatches.Patch(color=COLOR_BURNED, label="Burned"),
            ]
            plt.legend(handles=legend_patches, loc="lower center", ncol=4, frameon=False)
            plt.axis("off")
            plt.pause(0.001)

            if t % 10 == 0:
                print(f"t={t:03d} burning={(state == 2).sum():4d}  burned={(state == 3).sum():4d}")

            # --- capture directly from the RGB array (rock-solid, no blank frames) ---
            if t % GIF_EVERY_N == 0:
                frame = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)  # (H,W,3) uint8
                frames.append(frame)

        state, fuel_load = step(state, fuel_id, alt, wind_uv, slope_weight, fuel_load)

    # final frame (also capture)
    plt.clf()
    plt.title(f"t = {STEPS}")
    img = make_rgb_img(state, fuel_id)
    plt.imshow(img, interpolation="nearest")
    plt.axis("off")
    plt.pause(0.001)

    frames.append((np.clip(img, 0.0, 1.0) * 255).astype(np.uint8))

    # --- save GIF ---
    os.makedirs(GIF_DIR, exist_ok=True)
    out_path = os.path.join(GIF_DIR, f"wildfire_{time.strftime('%Y%m%d_%H%M%S')}.gif")
    imageio.mimsave(out_path, frames, duration=1.0 / GIF_FPS)
    print(f"[INFO] Saved GIF with {len(frames)} frames to: {out_path}")

    plt.show()



if __name__ == "__main__":
    main()
