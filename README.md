 # Mini Fire Model + Agents
 
 A simple, customizable wildfire cellular automaton plus optional **agent** layer (crews, trucks, aircraft, drones).  
 Each grid cell is one of: `0=EMPTY, 1=FUEL, 2=BURNING, 3=BURNED`.  
 Optionally, fuel is typed (Grass=1, Shrub=2, Timber=3, Water=4) with different ignition/burn behavior.  
 Terrain, wind, and finite fuel can bias spread.
 
 <img src="gifs/demo.gif" alt="Wildfire simulation" width="300">
 
 
 
 ## Repository layout
 
 - `mini_fire_model.py`  core wildfire CA (map fetch, ignition, spread, GIF export).
 - `agents.py`  agent taxonomy and rules + the engine that moves agents and applies suppression (returns per-tick p0 multipliers and a persistent retardant map).
 - `run_with_agents.py`  runner that imports the core model, spawns agents (e.g., crews) around the initial spark, steps both systems, draws agents, and writes a GIF.
 
 
 
 ## Quick start
 
 ### 1) Install
 pip install numpy matplotlib pillow rasterio terracatalogueclient shapely requests python-dotenv imageio
 
 ### 2) (Optional) Put OPENTOPO_API_KEY in your environment or a .env file
 Needed only if AUTO_FETCH_DATA is True.
 
 ### 3a) Run the base model
 python mini_fire_model.py
 
 ### 3b) Run the model + agents
 python run_with_agents.py
 
 
 Place inputs in `./data/` (if you’re not letting the code fetch them):
 
 - `worldcover.tif`   ESA WorldCover land cover → fuel types  
 - `fuel_id.png`   custom fuel map (if using the PNG route)  
 - `altitude.png` or `dem.tif`   elevation (if you enable altitude)
 
 **Output:** animated GIFs are written to `./gifs/`.
 
 
 
 ## What to tweak
 
 ### Grid & runtime
 
 - `ROWS, COLS`   grid size (bigger = more detail, slower).
 - `STEPS`   number of simulation steps.
 - `SEED`   `None` for randomness each run, or an int for repeatability.
 
 ### Feature toggles (in `mini_fire_model.py`)
 
 - `USE_FUEL_TYPES`   per-fuel behavior (Grass/Shrub/Timber/Water).
 - `USE_WIND`   wind-biased spread.
 - `USE_ALTITUDE`   slope effect (needs elevation).
 - `USE_FUEL_LOAD`   finite fuel that gets consumed.
 - `USE_FUELMAP_PNG`, `USE_ALTIMAP_PNG`   use local PNGs instead of GeoTIFFs/remote fetch.
 
 ### Core spread
 
 - `P0_IGNITE`   global aggressiveness. _Higher → faster spread everywhere._
 - `K_NEIGHBORS = {1,2,3}`   minimum burning neighbors required by fuel type.
 
 ### Wind
 
 - `WIND_DIR_DEG`   direction fire is blown **toward** (`0=E/right`, `90=S/down`).
 - `WIND_STRENGTH`   bias strength (_higher → stronger push downwind_).
 
 ### Slope (if enabled)
 
 - `SLOPE_GAIN`   steeper areas ignite more easily.
 
 ### Fuel behavior (speed & duration)
 
 - `FUEL_MODELS = {id: FuelModel(name, ignite_mult, sustain_p)}`
   - `ignite_mult`   _speed_: how easily that fuel ignites (Grass > Shrub > Timber).
   - `sustain_p`   _duration_: chance a burning cell stays burning each step (Timber > Shrub > Grass).
 
 ### Finite fuel (weakens fronts)
 
 - `FUEL_LOAD_INIT = {0..4: float}`   starting fuel per type (Water=4 and Non-burn=0 should be `0.0`).
 - `CONSUME_PER_STEP = {1..3: float}`   per-step consumption while burning.
 - `LOCAL_FUEL_WEIGHT (0..1)`   how much local remaining fuel (3×3 mean) damps ignition.
 
 ### Colors / visualization
 
 - `FUEL_COLOR_MAP = {0..4: "#hex"}`   non-burnable/grass/shrub/timber/water.
 - `COLOR_BURNING`, `COLOR_BURNED`   overrides for states 2 (red) and 3 (black).
 
 ### Inputs & mapping
 
 `fuel_from_worldcover(...)` maps WorldCover codes → fuel ids:  
 `30→Grass(1)`, `20→Shrub(2)`, `10→Timber(3)`, `90→Grass(1)`, `80→Water(4)`, others → `0` (non-burn).  
 (You can choose to remap `40`/Cropland to grass if desired.)
 
 
 
 ## How the fire model works (high-level)
 
 ### Initialize grids
 
 - `fuel_id` per cell (from WorldCover/PNG).  
 - `state` starts as FUEL (`1`) except non-burnables & Water → EMPTY (`0`).  
 - Optional `fuel_load` per cell from `FUEL_LOAD_INIT`.  
 - Ignite one or more cells; the center spark **snaps to the nearest fuel** if it lands on non-burn.
 
 ### Each step
 
 - Count burning neighbors (`nburn`) in an 8-neighborhood.  
 - Build base `p0_eff` from `P0_IGNITE`, then modulate by:
   - fuel type (× `ignite_mult`)
   - slope (if enabled)
   - local remaining fuel (if finite fuel is on)
 - Add wind-biased “pressure” from burning neighbors to form `nburn_eff`.  
 - Convert to ignition probability:
 
   ```
   p_ignite = 1 - (1 - p0_eff)^(nburn_eff)
   ```
 
 - Cells that pass ignite → **BURNING (2)**.  
 - Burning cells persist with probability `sustain_p`; otherwise → **BURNED (3)**.  
 - Finite fuel subtracts load; zero fuel forces **BURNED**.
 
 
 
 ## Agents (optional)
 
 Agents are pixels that move and act each tick. Each has:
 - a **type** (`AgentType`) and a rule profile (`RULES`) that sets *speed, action range, suppression, extinguish chance, cooldown deposition, decay, payload, cadence*, etc.  
 - a simple “seek nearest fire” movement by default (you can plug in your own routing later).
 
 ### Files
 
 - `agents.py`   defines `AgentType`, default `RULES` per type, the `Agent` dataclass (per-agent overrides allowed), and `step_agents(...)` which moves agents and returns:
   - `instant_p0_mult`   per-cell multiplier for **immediate cooling** this tick,
   - `cooldown_map`   a persistent retardant layer that decays globally each tick.
 - `run_with_agents.py`   bootstraps the fire model, **spawns** agents near the initial spark (`spawn_agents_near_spark`), steps both systems, draws agents (small squares), and saves a GIF.
 
 ### Spawning & running
 
 In `run_with_agents.py`, adjust:
 
 - `NUM_CREW`, `SPAWN_RING_MIN`, `SPAWN_RING_MAX`, `MIN_SEP`   how many agents spawn and how far from the spark.  
 - Edit the `agents = [...]` 
 

 
 ### Add a new agent type
 
 Open `agents.py`:
 
 1. Add an enum entry in `AgentType`.  
 2. Add a default entry in the `RULES` dict with speed/range/suppression/etc.  
 3. (Optional) Use per-agent overrides in `run_with_agents.py` when instantiating `Agent(...)`.
 

 
 ## Common recipes
 
 **Grass fastest, timber longest**  
 Increase grass `ignite_mult`, reduce timber `ignite_mult`. Set grass `sustain_p` low, timber high.  
 Use `K_NEIGHBORS={1:1, 2:1, 3:2}` to keep timber from “flashing”.
 
 **Globally calmer**  
 Lower `P0_IGNITE` by ~0.02 0.04, or raise `K_NEIGHBORS[3]`.
 
 **Weaken fronts over distance**  
 Enable `USE_FUEL_LOAD`, raise `LOCAL_FUEL_WEIGHT`, and tune `CONSUME_PER_STEP`.

 
 ## Notes
 
 - Water is a distinct type (`4`) for coloring/stats but remains non-burnable (state set to `0`).  
 - If the fire won’t grow, check: ignition isn’t on water, `K_NEIGHBORS` not too strict, reasonable `WIND_STRENGTH`, fuel-load damping/consumption not excessive, and map resolution large enough.
