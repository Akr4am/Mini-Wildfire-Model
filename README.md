# Mini Fire Model 

A simple, customizable wildfire cellular automaton. Each grid cell is one of:
`0=EMPTY, 1=FUEL, 2=BURNING, 3=BURNED`.
Optionally, fuel is typed (Grass=1, Shrub=2, Timber=3, Water=4) with different ignition and burn behavior.
Terrain/wind/fuel-load can bias spread.

<img src="gifs/demo.gif" alt="Wildfire simulation" width="600">


## Quick start
`pip install numpy matplotlib pillow rasterio
python mini_fire_model.py`


Place your inputs in ./data/:

`worldcover.tif`: ESA WorldCover land cover → fuel types

`fuel_id.png`: custom fuel map (if using the PNG route)

`altitude.png` or `dem.tif`: elevation (if you enable altitude) WORK IN PROGRESS



## What to tweak 
### Grid & runtime

`ROWS, COLS`  grid size (bigger = more detail, slower).

`STEPS` number of simulation steps.

`SEED` None for randomness each run, or an int for repeatability.

### Feature toggles

`USE_FUEL_TYPES` use per-fuel behavior & map from land cover.

`USE_WIND` enable wind-biased spread.

`USE_ALTITUDE` enable slope effects (needs elevation).

`USE_FUEL_LOAD` finite fuel that gets consumed (fire weakens/stalls).

`USE_FUELMAP_PNG`, `USE_ALTIMAP_PNG` use PNG inputs instead of GeoTIFFs.

### Core spread (heat)

`P0_IGNITE` global aggressiveness.
_Higher → faster spread everywhere._

### Wind

`WIND_DIR_DEG` direction fire is blown TOWARD (_0=E/right, 90=S/down_).

`WIND_STRENGTH` strength of the wind (_Higher → stronger bias downwind_).

### Slope (if enabled) WORK IN PROGRESS

`SLOPE_GAIN` steeper areas ignite more easily.

### Fuel behavior (speed & duration)

`FUEL_MODELS = {id: FuelModel(name, ignite_mult, sustain_p)}`

`ignite_mult` speed: how easily that fuel ignites (_Grass > Shrub > Timber for “fast → slow”_).

`sustain_p` duration: chance a burning cell stays burning each step (_Timber > Shrub > Grass for “long → short”_).

Neighbor threshold `K_NEIGHBORS` = {1,2,3} minimum burning neighbors required to ignite by fuel type.

**Finite fuel (weakens fire as it spreads)**

`FUEL_LOAD_INIT = {0..4: float}` starting fuel per type (_Water=4 and Non-burn=0 should be 0.0_).

`CONSUME_PER_STEP = {1..3: float}` fuel consumed each step while burning (_Higher for grass, lower for timber → grass burns out quickest_).

`LOCAL_FUEL_WEIGHT (0..1)` how much local remaining fuel (3×3 mean) damps ignition (_Higher → fronts thin and stall sooner_).

### Colors / visualization

`FUEL_COLOR_MAP = {0..4: "#hex"}` non-burnable/grass/shrub/timber/water.

`COLOR_BURNING, COLOR_BURNED` overrides for state 2 (red) and 3 (black).

### Inputs & mapping

`fuel_from_worldcover(...)` maps ESA WorldCover codes → fuel ids:
30→Grass(1), 20→Shrub(2), 10→Timber(3), 90→Grass(1), 80→Water(4), others non-burn (0).
You can make 40 (Cropland) behave like grass by mapping fuel[lc==40]=1.




## How it works 

### Initialize grids

`fuel_id` (from WorldCover or PNG) marks the fuel type per cell.

state starts as FUEL (1) except non-burnables & Water → EMPTY (0).

Optionally create `fuel_load` per cell from `FUEL_LOAD_INIT`.

Ignite one or more cells; we “snap” the center spark to the nearest fuel if it lands on non-burn.

### Each step

Count burning neighbors (nburn) with an 8-neighborhood.

Build a base ignition `p0_eff` from `P0_IGNITE`, then modulate by:

Fuel type: multiply by `ignite_mult`.

Slope (optional): multiply by slope_weight. **WORK IN PROGRESS**

Local remaining fuel (optional): blend with the local 3×3 fuel fraction using `LOCAL_FUEL_WEIGHT`.

Compute wind-biased pressure by weighting burning neighbors in the wind direction; form an effective neighbor count
_nburn_eff = nburn + α * press_.

Convert to ignite probability:
_p_ignite = 1 - (1 - p0_eff)^(nburn_eff)_ (more heat → higher chance).

### Apply guards:

Fuel present (state==1) and K-neighbor rule by fuel type.

If using finite fuel, require some fuel left.

Sample randomness: cells that pass ignite become BURNING (2).

Burn persistence: burning cells stay burning with probability `sustain_p` (by fuel).
Those that don’t persist become BURNED (3).

If using finite fuel, subtract consumption; any burning cell with zero fuel becomes BURNED.


### Draw

Base colors from `FUEL_COLOR_MAP` by fuel type,

Override BURNING (red) and BURNED (black).




## Common recipes

**Grass fastest, timber longest**
Increase grass `ignite_mult`, decrease timber `ignite_mult`; set `sustain_p` low for grass, high for timber.
Use `K_NEIGHBORS={1:1, 2:1, 3:2}` to keep timber from flashing.

**Globally calmer**
Lower `P0_IGNITE` by ~0.02–0.04, or raise `K_NEIGHBORS[3]`.

**Weaken fronts over distance** (_WORK IN PROGRESS_)
Enable `USE_FUEL_LOAD`, raise `LOCAL_FUEL_WEIGHT`, and tune `CONSUME_PER_STEP`.




## Notes

_Water is a distinct type (4) for coloring/stats but remains non-burnable (state set to 0)._

_If the fire won’t grow, check: ignition isn’t on water, K_NEIGHBORS not too strict, WIND_STRENGTH reasonable, fuel-load damping/consumption not excessive, and map resolution is not too small._

