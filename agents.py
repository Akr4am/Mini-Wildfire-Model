from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple, Optional
import numpy as np

# -------------------------
# Agent
# -------------------------
class AgentType(IntEnum):
    CREW = 1
    TRUCK = 2
    HELI = 3
    PLANE = 4
    DRONE = 5

@dataclass
class AgentRules:
    # Movement/targeting
    speed: int = 1                 # cells moved per tick
    seek_nearest_burn: bool = True # if False, you’d plug in your own routing later

    # Action footprint + immediate suppression
    range: int = 1                 # Chebyshev radius (square patch)
    suppression: float = 0.5       # multiply p0 by (1 - suppression) in patch this tick

    # Extinguish behavior (immediate)
    extinguish_chance: float = 0.0 # chance to flip burning->fuel for each burning cell in patch

    # Persistent retardant ("cooldown") deposit
    cooldown_strength: float = 0.0 # added to cooldown_map in patch (then clipped to [0..1])
    cooldown_decay: float = 0.10   # global decay fraction per tick (applied to entire map)

    # Payload and reloading
    payload_capacity: float = 0.0  # max tank
    payload_per_tick: float = 0.0  # consumed per tick when acting
    reload_on_water: bool = False  # refill to capacity when standing on water (fuel_id==4)

    # Cadence (e.g., aircraft drops)
    drop_every: int = 1            # act every N ticks (1 = every tick)

# Default rules per type (tune freely)
RULES: Dict[AgentType, AgentRules] = {
    AgentType.CREW:  AgentRules(speed=2, range=5, suppression=0.30,
                                extinguish_chance=0.75,
                                cooldown_strength=0.15, cooldown_decay=0.08,
                                payload_capacity=1.0, payload_per_tick=0.0,
                                reload_on_water=False, drop_every=1),
    AgentType.TRUCK: AgentRules(speed=1, range=2, suppression=0.6,
                                extinguish_chance=0.35,
                                cooldown_strength=0.25, cooldown_decay=0.06,
                                payload_capacity=400.0, payload_per_tick=2.0,
                                reload_on_water=True, drop_every=1),
    AgentType.HELI:  AgentRules(speed=2, range=3, suppression=0.7,
                                extinguish_chance=0.45,
                                cooldown_strength=0.35, cooldown_decay=0.05,
                                payload_capacity=300.0, payload_per_tick=5.0,
                                reload_on_water=True, drop_every=2),
    AgentType.PLANE: AgentRules(speed=10, range=4, suppression=1.0,
                                extinguish_chance=0.90,
                                cooldown_strength=0.45, cooldown_decay=0.05,
                                payload_capacity=1200.0, payload_per_tick=20.0,
                                reload_on_water=False, drop_every=4),
    AgentType.DRONE: AgentRules(speed=3, range=1, suppression=0.20,
                                extinguish_chance=0.0,
                                cooldown_strength=0.10, cooldown_decay=0.10,
                                payload_capacity=0.0, payload_per_tick=0.0,
                                reload_on_water=False, drop_every=1),
}

@dataclass
class Agent:
    kind: AgentType
    r: int
    c: int
    # Overridable per-agent if you want to deviate from RULES:
    speed: Optional[int] = None
    range: Optional[int] = None
    suppression: Optional[float] = None
    extinguish_chance: Optional[float] = None
    cooldown_strength: Optional[float] = None
    cooldown_decay: Optional[float] = None
    payload_capacity: Optional[float] = None
    payload_per_tick: Optional[float] = None
    reload_on_water: Optional[bool] = None
    drop_every: Optional[int] = None
    seek_nearest_burn: Optional[bool] = None
    # internal state
    payload: float = None  # initialized at first step

def _nearest_burning(state: np.ndarray, r0: int, c0: int) -> tuple[int, int] | None:
    rr, cc = np.where(state == 2)  # burning
    if rr.size == 0:
        return None
    k = np.argmin((rr - r0) ** 2 + (cc - c0) ** 2)
    return int(rr[k]), int(cc[k])

def _get_rule(a: Agent) -> AgentRules:
    base = RULES[a.kind]
    # merge overrides if provided
    return AgentRules(
        speed=a.speed if a.speed is not None else base.speed,
        seek_nearest_burn=a.seek_nearest_burn if a.seek_nearest_burn is not None else base.seek_nearest_burn,
        range=a.range if a.range is not None else base.range,
        suppression=a.suppression if a.suppression is not None else base.suppression,
        extinguish_chance=a.extinguish_chance if a.extinguish_chance is not None else base.extinguish_chance,
        cooldown_strength=a.cooldown_strength if a.cooldown_strength is not None else base.cooldown_strength,
        cooldown_decay=a.cooldown_decay if a.cooldown_decay is not None else base.cooldown_decay,
        payload_capacity=a.payload_capacity if a.payload_capacity is not None else base.payload_capacity,
        payload_per_tick=a.payload_per_tick if a.payload_per_tick is not None else base.payload_per_tick,
        reload_on_water=a.reload_on_water if a.reload_on_water is not None else base.reload_on_water,
        drop_every=a.drop_every if a.drop_every is not None else base.drop_every,
    )

def step_agents(
    agents: List[Agent],
    state: np.ndarray,
    fuel_id: np.ndarray,
    tick: int,
    cooldown_map: np.ndarray,     # persistent map in [0..1] (decays)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Move agents and apply suppression.
    Returns (instant_p0_mult, updated_cooldown_map).

    - instant_p0_mult: multiply into model.step's p0_multiplier for this tick (immediate cooling).
    - cooldown_map: persistent retardant; decay applied globally each call.
    """
    h, w = state.shape
    instant = np.ones((h, w), dtype=np.float32)

    # global cooldown decay
    # (decay rate taken as the max of agents’ cooldown_decay, so "strongest decay" wins)
    global_decay = 0.0
    for a in agents:
        rules = _get_rule(a)
        global_decay = max(global_decay, rules.cooldown_decay)
    if global_decay > 0:
        cooldown_map *= (1.0 - global_decay)
        np.clip(cooldown_map, 0.0, 1.0, out=cooldown_map)

    for a in agents:
        rules = _get_rule(a)

        # init payload
        if a.payload is None:
            a.payload = rules.payload_capacity

        # 1) Movement toward nearest burn
        if rules.seek_nearest_burn:
            target = _nearest_burning(state, a.r, a.c)
            if target is not None:
                tr, tc = target
                for _ in range(max(1, rules.speed)):
                    dr = int(np.sign(tr - a.r))
                    dc = int(np.sign(tc - a.c))
                    nr, nc = a.r + dr, a.c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        a.r, a.c = nr, nc
                    else:
                        break

        # 2) Reload if on water
        if rules.reload_on_water and fuel_id[a.r, a.c] == 4:
            a.payload = rules.payload_capacity

        # 3) Act (cadence + payload gating)
        can_act = (tick % max(1, rules.drop_every) == 0)
        if not can_act:
            continue

        if rules.payload_capacity > 0 and a.payload <= 0:
            continue  # out of payload

        rmin = max(0, a.r - rules.range)
        rmax = min(h, a.r + rules.range + 1)
        cmin = max(0, a.c - rules.range)
        cmax = min(w, a.c + rules.range + 1)

        # Immediate p0 cooling this tick
        if rules.suppression > 0:
            instant[rmin:rmax, cmin:cmax] *= (1.0 - rules.suppression)

        # Extinguish some actively burning cells (stochastic)
        if rules.extinguish_chance > 0:
            patch = state[rmin:rmax, cmin:cmax]
            burn_mask = (patch == 2)
            if burn_mask.any():
                rnd = (np.random.random(size=burn_mask.shape) < rules.extinguish_chance)
                patch[burn_mask & rnd] = 1  # back to fuel (instead of burned)

        # Persistent retardant deposition
        if rules.cooldown_strength > 0:
            cooldown_map[rmin:rmax, cmin:cmax] = np.clip(
                cooldown_map[rmin:rmax, cmin:cmax] + rules.cooldown_strength, 0.0, 1.0
            )

        # Payload use
        if rules.payload_capacity > 0 and rules.payload_per_tick > 0:
            a.payload = max(0.0, a.payload - rules.payload_per_tick)

    return instant, cooldown_map
