"""
Mini Wildfire Sim
- States: 0=EMPTY (no fuel), 1=FUEL, 2=BURNING, 3=BURNED
- Optional fuel types (1..3) change how easy it is to ignite and keep burning
- Optional wind
- Optional PNG/remote maps for fuel_id and altitude (grayscale/DEM)
"""
from __future__ import annotations

import os
import math
import time
import random
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import json
import shutil
import requests
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import reproject
from rasterio.transform import from_bounds

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import imageio.v2 as imageio

from shapely.geometry import Polygon
from matplotlib.colors import to_rgb
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------------------
# Colors
# --------------------------------------------------------------------------------------
COLOR_BURNING = "#ff0000"   # red
COLOR_BURNED  = "#000000"   # black

FUEL_COLOR_MAP = {
    0: "#9e9e9e",  # non-burnable (gray)
    1: "#08ff38",  # grass
    2: "#03a323",  # shrub
    3: "#126923",  # timber
    4: "#2196f3",  # water (blue)
    5: "#800080",
}

# --------------------------------------------------------------------------------------
# Online fetch
# --------------------------------------------------------------------------------------
AUTO_FETCH_DATA = True       # turn on downloading
WORLDCOVER_YEAR = 2021       # 2020 or 2021
DEM_TYPE = "NASADEM"         # "COP30", "NASADEM", "SRTMGL1", etc.
WATER_FRACTION_SKIP_DEM = 0.90  # if AOI is >90% water, skip DEM

# Area of interest (center + size) in lat/lon
AOI_CENTER = (39.116, -120.274)  # (lat, lon)  ← set this to a LAND location
AOI_SIZE_KM = 15             # square half-size

# Credentials via env vars
OPENTOPO_API_KEY = os.getenv("OPENTOPO_API_KEY")  # required for OpenTopography
TERRASCOPE_USER = os.getenv("TERRASCOPE_USER")    # optional for Terrascope
TERRASCOPE_PASS = os.getenv("TERRASCOPE_PASS")

# --------------------------------------------------------------------------------------
# simulation config
# --------------------------------------------------------------------------------------
ROWS, COLS = 700, 700        # grid size
STEPS = 800                  # simulation steps
SEED = None                    # set None for randomness

# Toggles
USE_WIND = False
USE_FUEL_TYPES = True
USE_ALTITUDE = True          # slope effect (needs elevation)
USE_FUELMAP_PNG = False      # read ./data/fuel_id.png (0..255 -> classes)
USE_ALTIMAP_PNG = False      # read ./data/altitude.png (grayscale -> meters-ish)

# Base physics
P0_IGNITE = 0.68             # base ignition contribution from burning neighbors
WIND_STRENGTH = 1.00        # wind bias (>=0)
WIND_DIR_DEG = 0             # where wind is blowing TOWARD (0=E/right, 90=S/down)
SLOPE_GAIN = 0.5            # how strongly upslope helps (only if USE_ALTITUDE)

# Require at least K burning neighbors before a fuel cell can ignite
K_NEIGHBORS = {1: 1, 2: 1, 3: 2}   # grass=1, shrub=1, timber=2

# Initial ignition pattern
IGNITE_CENTER = True         # center spark
IGNITE_RANDOM_N = 0          # or N random sparks

# Optional PNG paths (put files under ./data/)
FUELMAP_PATH = os.path.join("data", "fuel_id.png")
ALTIMAP_PATH = os.path.join("data", "altitude.png")

# GIF settings
GIF_DIR = "gifs"
GIF_FPS = 8
GIF_EVERY_N = 2



# --- Spark boost  ---
SPARK_MODE = "boost"     # "boost" to enable; anything else/None to disable
SPARK_RADIUS_CELLS = 4   # spatial radius of the boost
SPARK_P0_MULT = 2      # multiplies P0_IGNITE near spark (2.0 = 2× hotter)
SPARK_SUSTAIN_MULT = 1.3 # boosts persistence near spark
SPARK_DECAY_STEPS = 30   # boost fades to 0 by this many steps

# --- o spread tweaks ---
USE_RADIAL_KERNEL = True
KERNEL_RADIUS = 1
KERNEL_SIGMA = 1.2
KERNEL_SCALE = 1.0
DISCRETE_NEIGHBOR_WEIGHT = 0.20

# gate ignition by circular heat (instead of K_NEIGHBORS)
HEAT_GATE = True
HEAT_THRESH = 0.1       # Too high => no growth.

USE_STATIC_NOISE = True
NOISE_STRENGTH = 0.3
NOISE_SMOOTH_PASSES = 2

USE_GUST_JITTER = True
GUST_STD = 0.07


# --------------------------------------------------------------------------------------
# Fuel model & load
# --------------------------------------------------------------------------------------
@dataclass
class FuelModel:
    name: str
    ignite_mult: float   # scales how easy a fuel cell ignites
    sustain_p: float     # per-step keep-burning probability

# 1..3 id -> model
FUEL_MODELS = {
    1: FuelModel("Grass",   ignite_mult=1.50, sustain_p=0.55),
    2: FuelModel("Shrub",   ignite_mult=1.30, sustain_p=0.72),
    3: FuelModel("Timber",  ignite_mult=0.55, sustain_p=0.85),
}
DEFAULT_FUEL_ID = 2

USE_FUEL_LOAD = True
FUEL_LOAD_INIT = {0: 0.0, 1: 1.2, 2: 1.8, 3: 2.2, 4: 0.0}   # water & nonburn = 0
CONSUME_PER_STEP = {1: 0.7, 2: 0.50, 3: 0.30}               # burn-rate per step
LOCAL_FUEL_WEIGHT = 0.5                                     # 0..1, how much local remaining fuel damps ignition

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _aoi_signature() -> dict:
    return {
        "lat": round(float(AOI_CENTER[0]), 6),
        "lon": round(float(AOI_CENTER[1]), 6),
        "size_km": float(AOI_SIZE_KM),
        "worldcover_year": int(WORLDCOVER_YEAR),
        "dem_type": str(DEM_TYPE),
    }

def _purge_cached_layers_if_stale(cache_json_path: str, files_to_remove: list[str]) -> bool:
    """
    If the saved signature differs from the current AOI/params, delete the
    given files and write the new signature. Returns True if a purge happened.
    """
    os.makedirs(os.path.dirname(cache_json_path), exist_ok=True)
    current = _aoi_signature()
    previous = None

    if os.path.exists(cache_json_path):
        try:
            with open(cache_json_path, "r", encoding="utf-8") as f:
                previous = json.load(f)
        except Exception:
            previous = None

    if previous != current:
        # Delete stale rasters
        for p in files_to_remove:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

        # Also drop WorldCover temp dir if it exists
        tmpdir = os.path.join("data", "_wc_tmp")
        if os.path.isdir(tmpdir):
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass

        # Save the new signature
        try:
            with open(cache_json_path, "w", encoding="utf-8") as f:
                json.dump(current, f, indent=2)
        except Exception:
            pass

        print("[CACHE] AOI/params changed → purged cached GeoTIFFs.")
        return True

    return False


def unit_vec_from_deg(deg: float) -> np.ndarray:
    rad = math.radians(deg)
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

def build_radial_offsets(radius: int, sigma: float) -> List[Tuple[int,int,float]]:
    """
    Return a list of (dx, dy, w) for all cells within the given radius
    using a Gaussian-like weight exp(-r^2/(2*sigma^2)), excluding (0,0).
    """
    offsets = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            r2 = dx*dx + dy*dy
            if r2 <= radius*radius:
                w = math.exp(-r2 / (2.0 * sigma * sigma))
                offsets.append((dx, dy, w))
    # normalize weights so sum ~= 1
    s = sum(w for _, _, w in offsets)
    if s > 0:
        offsets = [(dx, dy, w/s) for dx, dy, w in offsets]
    return offsets

def make_smooth_noise(rows: int, cols: int, passes: int, strength: float) -> np.ndarray:
    """
    Make a smooth multiplicative field in [1-strength, 1+strength].
    """
    n = np.random.rand(rows, cols).astype(np.float32)
    for _ in range(max(0, passes)):
        n = box_mean_3x3(n)
    n = (n - n.min()) / max(1e-6, (n.max() - n.min()))
    return 1.0 + (n - 0.5) * 2.0 * float(strength)

def simple_wind_weight(nbr_dx: int, nbr_dy: int, wind_uv: np.ndarray, strength: float) -> float:
    """Weight for a neighbor based on alignment with wind direction (clamped >= 0)."""
    if strength <= 0.0:
        return 1.0
    v = np.array([nbr_dx, nbr_dy], dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm == 0:
        return 1.0
    v /= norm
    cosang = float(np.clip(np.dot(v, wind_uv), -1.0, 1.0))
    w = 1.0 + strength * cosang
    return max(0.0, w)   # never negative

def aoi_bbox(center_lat: float, center_lon: float, half_size_km: float):
    """Compute bbox (west, south, east, north) from center + half-size in km."""
    deg_per_km = 1.0 / 111.32
    dlat = half_size_km * deg_per_km
    clat = max(math.cos(math.radians(center_lat)), 1e-6)
    dlon = half_size_km * (deg_per_km / clat)
    south = center_lat - dlat
    north = center_lat + dlat
    west  = center_lon - dlon
    east  = center_lon + dlon
    return (west, south, east, north)

def slope_factor(alt: np.ndarray) -> np.ndarray:
    """
    Meter-aware slope factor: gradient per meter using AOI bbox and grid size.
    Nodata (NaN) yields neutral weight (1.0). Output is clipped to avoid overflow.
    """
    if alt is None or alt.size == 0:
        return np.ones((ROWS, COLS), dtype=np.float32)

    west, south, east, north = aoi_bbox(AOI_CENTER[0], AOI_CENTER[1], AOI_SIZE_KM)
    dlat = (north - south) / ROWS
    dlon = (east - west) / COLS
    lat0 = math.radians((north + south) * 0.5)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * max(math.cos(lat0), 1e-6)
    dy_m = max(dlat * m_per_deg_lat, 1e-6)
    dx_m = max(dlon * m_per_deg_lon, 1e-6)

    alt = alt.astype(np.float32)
    gy, gx = np.gradient(alt)
    gy /= dy_m
    gx /= dx_m

    slope_mag = np.hypot(gx, gy)
    # prevent exp overflow
    slope_mag = np.nan_to_num(slope_mag, nan=0.0, posinf=0.0, neginf=0.0)
    slope_mag = np.clip(slope_mag, 0.0, 200.0)

    # clamp exponent argument
    expo = np.clip(SLOPE_GAIN * slope_mag, -20.0, 20.0)
    w = np.exp(expo).astype(np.float32)
    return w


def _looks_like_tiff(b: bytes) -> bool:
    # TIFF magic bytes: little-endian 'II*\x00' or big-endian 'MM\x00*'
    return len(b) >= 4 and (b[:4] == b"II*\x00" or b[:4] == b"MM\x00*")

def download_dem_opentopo(bbox, demtype, api_key, out_tif):
    """
    Fetch DEM from OpenTopography GlobalDEM for bbox.
    Robust handling for ZIP/TIFF/JSON/HTML and empty bodies.
    """
    if not api_key:
        raise RuntimeError("OPENTOPO_API_KEY not set in environment")
    os.makedirs(os.path.dirname(out_tif), exist_ok=True)

    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": demtype,
        "west": bbox[0], "south": bbox[1], "east": bbox[2], "north": bbox[3],
        "outputFormat": "GTiff",
        "API_Key": api_key
    }

    def _mask(v):
        return (v[:3] + "…" + v[-3:]) if isinstance(v, str) and len(v) > 7 else "****"
    dbg = {**params, "API_Key": _mask(params["API_Key"]) }
    print(f"[DEM] GET {url} with {dbg}")

    try:
        resp = requests.get(
            url, params=params,
            headers={"Accept": "application/octet-stream"},
            timeout=600, allow_redirects=True
        )
    except requests.RequestException as e:
        raise RuntimeError(f"OpenTopography request failed: {e}")

    status = resp.status_code
    ct = (resp.headers.get("Content-Type") or "").lower()
    data = resp.content or b""
    print(f"[DEM] HTTP {status}; Content-Type={resp.headers.get('Content-Type')}; "
          f"Content-Length={resp.headers.get('Content-Length')}")

    if status != 200:
        dbg_path = out_tif + ".txt"
        with open(dbg_path, "wb") as f:
            f.write(data)
        raise RuntimeError(f"OpenTopography returned HTTP {status}. Saved body to {dbg_path}")

    if len(data) == 0:
        dbg_path = out_tif + ".txt"
        with open(dbg_path, "wb") as f:
            pass
        raise RuntimeError(f"OpenTopography returned an EMPTY body. Saved placeholder to {dbg_path}")

    # ZIP?
    if "zip" in ct or data[:2] == b"PK":
        import io, zipfile
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            tifs = [n for n in z.namelist() if n.lower().endswith(".tif")]
            if not tifs:
                lst = out_tif + ".ziplist.txt"
                with open(lst, "w", encoding="utf-8") as f:
                    f.write("\n".join(z.namelist()))
                raise RuntimeError(f"ZIP contained no .tif (listing saved to {lst})")
            with z.open(tifs[0]) as zin, open(out_tif, "wb") as fout:
                fout.write(zin.read())

    # Direct GeoTIFF?
    elif "tif" in ct or "tiff" in ct or "geotiff" in ct or "image/" in ct or _looks_like_tiff(data):
        with open(out_tif, "wb") as f:
            f.write(data)

    # JSON error?
    elif "json" in ct:
        dbg_path = out_tif + ".json"
        with open(dbg_path, "wb") as f:
            f.write(data)
        try:
            msg = resp.json()
        except Exception:
            msg = data.decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenTopography JSON error: {msg} (saved to {dbg_path})")

    # HTML / other
    else:
        dbg_path = out_tif + ".txt"
        with open(dbg_path, "wb") as f:
            f.write(data)
        raise RuntimeError(f"Unexpected response Content-Type={ct}. Saved body to {dbg_path}")

    # Validate GeoTIFF
    try:
        with rasterio.open(out_tif) as ds:
            _ = ds.read(1, out_shape=(min(32, ds.height), min(32, ds.width)))
            print(f"[DEM] Valid GeoTIFF: shape={ds.width}x{ds.height}, crs={ds.crs}")
    except Exception as e:
        raise RuntimeError(f"Downloaded DEM is not a valid GeoTIFF: {e}")

def fetch_dem_with_fallback(bbox, demtypes, api_key, out_tif):
    last_err = None
    for dt in demtypes:
        try:
            print(f"[DEM] Trying {dt} ...")
            download_dem_opentopo(bbox, dt, api_key, out_tif)
            with rasterio.open(out_tif) as ds:
                arr = ds.read(1, out_shape=(min(32, ds.height), min(32, ds.width)))
                print(f"[DEM] OK {dt}: shape={ds.width}x{ds.height}, crs={ds.crs}, "
                      f"min={np.nanmin(arr):.2f} max={np.nanmax(arr):.2f}")
            return dt
        except Exception as e:
            print(f"[DEM] {dt} failed: {e}")
            last_err = e
            if os.path.exists(out_tif):
                try:
                    os.remove(out_tif)
                except:
                    pass
    raise last_err

def resample_to_grid(src_path, bbox, rows, cols, resampling):
    """
    Reproject & resample a source GeoTIFF to EPSG:4326 grid defined by bbox + (rows, cols).
    Preserves nodata as NaN to avoid bogus values (e.g., -32768).
    """
    with rasterio.open(src_path) as src:
        dst = np.full((rows, cols), np.nan, dtype=np.float32)
        dst_transform = from_bounds(*bbox, cols, rows)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=resampling,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )
        return dst


def worldcover_collection(year: int) -> str:
    if year >= 2021:
        return "urn:eop:VITO:ESA_WorldCover_10m_2021_V2"
    else:
        return "urn:eop:VITO:ESA_WorldCover_10m_2020_V1"

def download_worldcover_terrascope(bbox, out_tif, collection=None):
    """
    Downloads all WorldCover tiles intersecting bbox and mosaics them to out_tif.
    """
    from terracatalogueclient import Catalogue

    os.makedirs(os.path.dirname(out_tif), exist_ok=True)
    if collection is None:
        collection = worldcover_collection(WORLDCOVER_YEAR)

    cat = Catalogue()
    if TERRASCOPE_USER and TERRASCOPE_PASS:
        cat = cat.authenticate_non_interactive(TERRASCOPE_USER, TERRASCOPE_PASS)
    else:
        cat = cat.authenticate()  # interactive

    geom = Polygon.from_bounds(*bbox)
    products = cat.get_products(collection, geometry=geom)

    tmpdir = os.path.join("data", "_wc_tmp")
    os.makedirs(tmpdir, exist_ok=True)
    # this may ask to proceed (download size)
    cat.download_products(products, tmpdir)

    # find *Map.tif files and mosaic
    tifs = []
    for root, _, files in os.walk(tmpdir):
        for f in files:
            if f.endswith("_Map.tif"):
                tifs.append(os.path.join(root, f))
    if not tifs:
        raise RuntimeError("No WorldCover *Map.tif found in Terrascope downloads")

    srcs = [rasterio.open(p) for p in tifs]
    mosaic, out_transform = merge(srcs, bounds=bbox, method="first")
    meta = srcs[0].meta.copy()
    for s in srcs:
        s.close()
    meta.update(driver="GTiff", height=mosaic.shape[1], width=mosaic.shape[2],
                transform=out_transform, count=mosaic.shape[0])
    with rasterio.open(out_tif, "w", **meta) as dst:
        dst.write(mosaic)

def map_worldcover_to_fuel_ids(lc_codes: np.ndarray) -> np.ndarray:
    """
    WorldCover ⇒ your fuel IDs:
      1=Grass, 2=Shrub, 3=Timber, 4=Water, 0=non-burnable
    (Cropland=40 remains non-burnable by design.)
    """
    lc = lc_codes.astype(np.int32)
    fuel = np.zeros_like(lc, dtype=np.int16)
    fuel[lc == 30] = 1  # Grassland -> Grass
    fuel[lc == 20] = 2  # Shrubland -> Shrub
    fuel[lc == 10] = 3  # Tree cover -> Timber
    fuel[lc == 90] = 1  # Herbaceous wetland -> Grass-like
    fuel[lc == 80] = 4  # Water
    # everything else (e.g., built-up, bare, snow/ice, cropland=40) stays 0 (non-burnable)
    return fuel

def ensure_remote_layers(rows, cols):
    """
    Ensures data/worldcover_aoi.tif and data/dem_aoi.tif exist for the AOI,
    then returns (fuel_id, altitude) arrays resampled to (rows, cols).
    If AOI is mostly water, DEM is skipped and altitude = zeros.
    Automatically purges old rasters when AOI/year/DEM type change.
    """
    bbox = aoi_bbox(AOI_CENTER[0], AOI_CENTER[1], AOI_SIZE_KM)
    wc_path  = os.path.join("data", "worldcover_aoi.tif")
    dem_path = os.path.join("data", "dem_aoi.tif")
    sig_path = os.path.join("data", "aoi_signature.json")

    # Purge old files if the AOI signature changed
    _purge_cached_layers_if_stale(sig_path, [wc_path, dem_path])

    # ---- WorldCover first (so we can decide about DEM) ----
    if not os.path.exists(wc_path):
        download_worldcover_terrascope(bbox, wc_path)

    wc_codes = resample_to_grid(wc_path, bbox, rows, cols, Resampling.nearest)

    # Debug: which classes did we get?
    vals, cnts = np.unique(wc_codes[np.isfinite(wc_codes)].astype(np.int32), return_counts=True)
    print("[WorldCover] codes present:", dict(zip(vals.tolist(), cnts.tolist())))

    # Map to fuel ids
    fuel_id = map_worldcover_to_fuel_ids(wc_codes)

    # Water fraction in AOI (WorldCover class 80 -> our fuel_id 4)
    water_frac = float((fuel_id == 4).sum()) / float(fuel_id.size)
    print(f"[AOI] water fraction = {water_frac:.3f}")

    # ---- DEM: skip if mostly water, else fetch + resample ----
    if water_frac >= WATER_FRACTION_SKIP_DEM:
        print("[DEM] AOI is mostly water. Skipping DEM fetch; using flat terrain.")
        alt = np.zeros((rows, cols), dtype=np.float32)
    else:
        if not os.path.exists(dem_path):
            # Try your chosen DEM first, then fall back gracefully
            fetch_dem_with_fallback(bbox, [DEM_TYPE, "NASADEM", "SRTMGL1"], OPENTOPO_API_KEY, dem_path)
        alt = resample_to_grid(dem_path, bbox, rows, cols, Resampling.bilinear)
        # Clean up nodata / crazy values so slope calc is stable
        alt = alt.astype(np.float32)
        # If src.nodata wasn't carried through for any reason, catch common sentinels:
        alt[alt <= -10000] = np.nan

        print(f"[DEM] stats: min={np.nanmin(alt):.2f}, max={np.nanmax(alt):.2f}, mean={np.nanmean(alt):.2f}")

    return fuel_id, alt.astype(np.float32)


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

def make_radial_field(rows, cols, center, radius):
    """1.0 at center, linearly decays to 0 at radius (outside = 0)."""
    y, x = np.ogrid[:rows, :cols]
    cy, cx = center
    d = np.hypot(y - cy, x - cx)
    f = np.clip(1.0 - d / float(radius), 0.0, 1.0)
    return f.astype(np.float32)

# --------------------------------------------------------------------------------------
# World initialization
# --------------------------------------------------------------------------------------
def fuel_from_worldcover(rows: int, cols: int, tif_path: str = "data/worldcover.tif") -> np.ndarray:
    """Local fallback: read a WorldCover GeoTIFF and map classes to fuel IDs."""
    with rasterio.open(tif_path) as src:
        lc = src.read(1, out_shape=(rows, cols), resampling=Resampling.nearest).astype(np.int32)
    fuel = np.zeros_like(lc, dtype=np.int16)
    fuel[lc == 30] = 1
    fuel[lc == 20] = 2
    fuel[lc == 10] = 3
    fuel[lc == 90] = 1
    fuel[lc == 80] = 4
    return fuel

def init_world(rows: int, cols: int):
    # states
    state = np.ones((rows, cols), dtype=np.int16)  # 1=FUEL everywhere by default

    # ---- fuel & altitude (either fetched or local) ----
    if AUTO_FETCH_DATA:
        fuel_id, alt = ensure_remote_layers(rows, cols)
        if not USE_FUEL_TYPES:
            burnable = (fuel_id > 0) & (fuel_id != 4)
            fuel_id = np.where(burnable, DEFAULT_FUEL_ID, 0).astype(np.int16)
    else:
        if USE_FUEL_TYPES:
            fuel_id = fuel_from_worldcover(rows, cols)
        else:
            fuel_id = np.full((rows, cols), DEFAULT_FUEL_ID, dtype=np.int16)
        if USE_ALTITUDE and USE_ALTIMAP_PNG:
            alt = load_grayscale_png(ALTIMAP_PATH, (rows, cols))
            if alt is None:
                alt = np.zeros((rows, cols), dtype=np.float32)
            else:
                alt = (alt - alt.min()) / max(1.0, (alt.max() - alt.min()))
                alt *= 500.0
        elif USE_ALTITUDE:
            x = np.linspace(-1, 1, cols, dtype=np.float32)
            y = np.linspace(-1, 1, rows, dtype=np.float32)
            X, Y = np.meshgrid(x, y)
            alt = 200.0 * (X ** 2 + Y ** 2)
        else:
            alt = np.zeros((rows, cols), dtype=np.float32)

    # mark EMPTY where non-burnable or water
    state[(fuel_id == 0) | (fuel_id == 4)] = 0

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
          "water=", int((fuel_id == 4).sum()),
          "nonburn=", int((fuel_id == 0).sum()))

    # --- primary spark ---
    if IGNITE_CENTER:
        # keep your old “center + snap” behavior
        r0, c0 = rows // 2, cols // 2
        if state[r0, c0] != 1:  # center isn't fuel → move to nearest fuel cell
            print(f"[WARN] Center ignition at ({r0},{c0}) is non-burnable "
                  f"(fuel_id={int(fuel_id[r0, c0])}). Snapping to nearest fuel...")
            rr, cc = np.where(state == 1)
            if rr.size > 0:
                k = np.argmin((rr - r0) ** 2 + (cc - c0) ** 2)
                r0, c0 = int(rr[k]), int(cc[k])
            else:
                raise RuntimeError("No burnable cells found for ignition.")
        state[r0, c0] = 2
    else:
        # pick a random fuel cell (never water/non-burnable)
        fuel_cells = np.argwhere(state == 1)
        if fuel_cells.size == 0:
            raise RuntimeError("No burnable cells found in AOI.")
        r0, c0 = map(int, fuel_cells[np.random.randint(len(fuel_cells))])
        state[r0, c0] = 2

    spark_center = (r0, c0)

    # --- optional extra random ignitions (besides the primary spark) ---
    extra = IGNITE_RANDOM_N
    if extra > 0:
        # sample without replacement from remaining fuel cells
        fuel_cells = np.argwhere(state == 1)
        if fuel_cells.size > 0:
            idx = np.random.choice(len(fuel_cells), size=min(extra, len(fuel_cells)), replace=False)
            for r, c in fuel_cells[idx]:
                state[int(r), int(c)] = 2
        else:
            print("[WARN] No additional fuel cells available for extra ignitions.")

    return state, fuel_id, alt, fuel_load, spark_center


# --------------------------------------------------------------------------------------
# CA step
# --------------------------------------------------------------------------------------
def step(state: np.ndarray, fuel_id: np.ndarray, alt: np.ndarray,
         wind_uv: np.ndarray, slope_weight: np.ndarray,
         fuel_load: np.ndarray,
         p0_multiplier: Optional[np.ndarray] = None,
         sustain_multiplier: Optional[np.ndarray] = None,
         kernel_offsets: Optional[List[Tuple[int,int,float]]] = None,
         p0_noise: Optional[np.ndarray] = None,
         gust_std: float = 0.0
         ) -> tuple[np.ndarray, np.ndarray]:
    """One CA step. Returns (new_state, new_fuel_load)."""
    rows, cols = state.shape
    new_state = state.copy()


    # base ignition probability from neighbors:
    p0_eff = np.full((rows, cols), P0_IGNITE, dtype=np.float32)

    # local-fuel damping (weakens spread where fuel is low)
    if USE_FUEL_LOAD:
        max_load = max(v for v in FUEL_LOAD_INIT.values())
        norm_load = np.divide(fuel_load, max_load if max_load > 0 else 1.0, dtype=np.float32)
        local_load = box_mean_3x3(norm_load)
        p0_eff *= (1.0 - LOCAL_FUEL_WEIGHT) + LOCAL_FUEL_WEIGHT * local_load

    # Fuel multipliers
    if USE_FUEL_TYPES:
        p0_eff *= np.where(fuel_id == 1, FUEL_MODELS[1].ignite_mult, 1.0)
        p0_eff *= np.where(fuel_id == 2, FUEL_MODELS[2].ignite_mult, 1.0)
        p0_eff *= np.where(fuel_id == 3, FUEL_MODELS[3].ignite_mult, 1.0)

    # Slope multiplier
    if USE_ALTITUDE:
        p0_eff *= slope_weight

    # Spark ignition boost (Option B)
    if p0_multiplier is not None:
        p0_eff *= p0_multiplier

    # Static spatial heterogeneity
    if p0_noise is not None:
        p0_eff *= p0_noise

    # Tiny global gust jitter
    if gust_std > 0.0 and USE_GUST_JITTER:
        jitter = float(np.random.normal(loc=0.0, scale=gust_std))
        p0_eff *= max(0.1, 1.0 + jitter)  # clamp so it never goes negative

    # --- 8-neighbor count (kept only for optional blending) ---
    nburn = neighbor_burning_count(state).astype(np.float32)

    # --- circular "press" field from the radial kernel (isotropic) ---
    burn = (state == 2).astype(np.float32)
    press = np.zeros_like(burn, dtype=np.float32)
    if kernel_offsets:  # you pass this prebuilt list from main()
        H, W = burn.shape
        wind_strength = WIND_STRENGTH if USE_WIND else 0.0
        for dx, dy, w0 in kernel_offsets:
            # wind alignment factor (clamped >= 0); =1 when wind disabled
            w_wind = simple_wind_weight(dx, dy, wind_uv, wind_strength)
            w = w0 * w_wind
            if w <= 0.0:
                continue

            # src/dst slices for a shift of (dx,dy)
            y_src0 = max(0, -dy);
            y_src1 = min(H, H - dy)
            x_src0 = max(0, -dx);
            x_src1 = min(W, W - dx)
            y_dst0 = max(0, dy);
            y_dst1 = min(H, H + dy)
            x_dst0 = max(0, dx);
            x_dst1 = min(W, W + dx)
            if (y_src0 < y_src1) and (x_src0 < x_src1):
                press[y_dst0:y_dst1, x_dst0:x_dst1] += w * burn[y_src0:y_src1, x_src0:x_src1]

    # --- blend to get a circular "heat" field (this drives ignition) ---
    # Set DISCRETE_NEIGHBOR_WEIGHT = 0.0 to eliminate the square look.
    heat = DISCRETE_NEIGHBOR_WEIGHT * nburn + KERNEL_SCALE * press

    # --- gate ignition: by circular heat (preferred), or by K_NEIGHBORS ---
    if HEAT_GATE:
        mask_k = (heat >= HEAT_THRESH)
    else:
        k_req = np.zeros_like(fuel_id, dtype=np.int16)
        k_req[fuel_id == 1] = K_NEIGHBORS[1]
        k_req[fuel_id == 2] = K_NEIGHBORS[2]
        k_req[fuel_id == 3] = K_NEIGHBORS[3]
        mask_k = (nburn >= k_req)

    # --- effective neighbor/heat used in ignition probability ---
    nburn_eff = np.clip(heat, 0.0, 8.0)

    # ignition probability for FUEL cells
    with np.errstate(over='ignore'):
        p_ignite = 1.0 - np.power(1.0 - np.clip(p0_eff, 0.0, 1.0), np.clip(nburn_eff, 0.0, 8.0))
    p_ignite = np.clip(p_ignite, 0.0, 1.0)

    # apply ignition only to FUEL cells (state==1)
    fuel_mask = (state == 1)
    if USE_FUEL_LOAD:
        fuel_mask &= (fuel_load > 0.05)  # need a bit of fuel
    rnd = np.random.random(size=state.shape).astype(np.float32)
    ignite_mask = fuel_mask & mask_k & (rnd < p_ignite)
    new_state[ignite_mask] = 2  # becomes BURNING

    # burning cells may continue or become burned
    burning_mask = (state == 2)
    if USE_FUEL_TYPES:
        sustain = np.full(state.shape, 0.7, dtype=np.float32)
        sustain = np.where(fuel_id == 1, FUEL_MODELS[1].sustain_p, sustain)
        sustain = np.where(fuel_id == 2, FUEL_MODELS[2].sustain_p, sustain)
        sustain = np.where(fuel_id == 3, FUEL_MODELS[3].sustain_p, sustain)
    else:
        sustain = np.full(state.shape, 0.7, dtype=np.float32)

    # Spark sustain boost
    if sustain_multiplier is not None:
        sustain = np.clip(sustain * sustain_multiplier, 0.0, 1.0)

    keep_mask = (np.random.random(size=state.shape).astype(np.float32) < sustain)

    if USE_FUEL_LOAD:
        cons = np.zeros_like(fuel_load, dtype=np.float32)
        cons[fuel_id == 1] = CONSUME_PER_STEP[1]
        cons[fuel_id == 2] = CONSUME_PER_STEP[2]
        cons[fuel_id == 3] = CONSUME_PER_STEP[3]
        fuel_load = np.maximum(0.0, fuel_load - cons * burning_mask.astype(np.float32))
        out_mask = burning_mask & (fuel_load <= 1e-6)
        new_state[out_mask] = 3

    to_burned = (burning_mask & ~keep_mask)
    new_state[to_burned] = 3  # BURNED

    return new_state, fuel_load

# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------
def make_rgb_img(state: np.ndarray, fuel_id: np.ndarray) -> np.ndarray:
    """
    Build an RGB image from state and fuel_id.
    Priority: burning (red) > burned (black) > fuel color.
    """
    rows, cols = state.shape
    img = np.zeros((rows, cols, 3), dtype=np.float32)
    palette = np.array([to_rgb(FUEL_COLOR_MAP[i]) for i in range(0, 6)], dtype=np.float32)
    img[:] = palette[np.clip(fuel_id, 0, 5)]
    burning_mask = (state == 2)
    burned_mask  = (state == 3)
    if burning_mask.any():
        img[burning_mask] = to_rgb(COLOR_BURNING)
    if burned_mask.any():
        img[burned_mask] = to_rgb(COLOR_BURNED)
    return img

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)

    state, fuel_id, alt, fuel_load, spark_center = init_world(ROWS, COLS)

    wind_uv = unit_vec_from_deg(WIND_DIR_DEG) if USE_WIND else np.array([0.0, 0.0], dtype=np.float32)
    slope_weight = slope_factor(alt) if USE_ALTITUDE else np.ones((ROWS, COLS), dtype=np.float32)

    # --- Organic precomputes (for rounder, more natural spread) ---
    KERNEL_OFFSETS = build_radial_offsets(KERNEL_RADIUS, KERNEL_SIGMA) if USE_RADIAL_KERNEL else []
    P0_NOISE = make_smooth_noise(ROWS, COLS, NOISE_SMOOTH_PASSES, NOISE_STRENGTH) if USE_STATIC_NOISE else np.ones(
        (ROWS, COLS), dtype=np.float32)

    # Spark boost fields (Option B)
    spark_field_unit = make_radial_field(ROWS, COLS, spark_center, SPARK_RADIUS_CELLS) \
                       if SPARK_MODE == "boost" else None

    plt.figure("Mini Wildfire CA", figsize=(6, 6))
    frames = []

    p0_mult = None
    sustain_mult = None

    for t in range(STEPS):
        # Build decaying multipliers near spark
        if SPARK_MODE == "boost" and spark_field_unit is not None:
            decay = max(0.0, 1.0 - t / float(SPARK_DECAY_STEPS))  # 1 → 0 over time
            p0_mult = 1.0 + (SPARK_P0_MULT - 1.0) * decay * spark_field_unit
            sustain_mult = 1.0 + (SPARK_SUSTAIN_MULT - 1.0) * decay * spark_field_unit
        else:
            p0_mult = None
            sustain_mult = None

        # draw every 2nd step (match GIF_EVERY_N logic)
        if t % 2 == 0:
            plt.clf()
            plt.title(f"t = {t}")
            img = make_rgb_img(state, fuel_id)
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

            if t % GIF_EVERY_N == 0:
                frames.append((np.clip(img, 0.0, 1.0) * 255).astype(np.uint8))

        state, fuel_load = step(
            state, fuel_id, alt, wind_uv, slope_weight, fuel_load,
            p0_multiplier=p0_mult,
            sustain_multiplier=sustain_mult,
            kernel_offsets=KERNEL_OFFSETS,
            p0_noise=P0_NOISE,
            gust_std=(GUST_STD if USE_GUST_JITTER else 0.0),
        )

    # final frame
    plt.clf()
    plt.title(f"t = {STEPS}")
    img = make_rgb_img(state, fuel_id)
    plt.imshow(img, interpolation="nearest")
    plt.axis("off")
    plt.pause(0.001)
    frames.append((np.clip(img, 0.0, 1.0) * 255).astype(np.uint8))

    # save GIF
    os.makedirs(GIF_DIR, exist_ok=True)
    out_path = os.path.join(GIF_DIR, f"wildfire_{time.strftime('%Y%m%d_%H%M%S')}.gif")
    imageio.mimsave(out_path, frames, duration=1.0 / GIF_FPS)
    print(f"[INFO] Saved GIF with {len(frames)} frames to: {out_path}")

    plt.show()

if __name__ == "__main__":
    main()
