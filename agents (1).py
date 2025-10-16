from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Tuple, Optional
import numpy as np
import math


# -------------------------
# Agent Enums and Dataclasses
# -------------------------
class AgentType(IntEnum):
    """Defines the types of agents available in the simulation."""
    CREW = 1
    AIR_TANKER = 2


class AgentState(IntEnum):
    """Defines the behavioral state of an agent."""
    IDLE = 0
    MOVING_TO_TARGET = 1
    WORKING = 2  # e.g., Crew suppressing, Tanker dropping
    RETURNING_TO_BASE = 3


@dataclass
class AgentRules:
    """Defines the base capabilities of an agent type."""
    speed: int = 1
    range: int = 1
    suppression: float = 0.0
    extinguish_chance: float = 0.0
    cooldown_strength: float = 0.0
    cooldown_decay: float = 0.10
    payload_capacity: float = 0.0
    payload_per_tick: float = 0.0
    reload_on_water: bool = False
    drop_every: int = 1


@dataclass
class Agent:
    """Represents a single agent instance in the simulation."""
    kind: AgentType
    r: int
    c: int
    base_pos: Tuple[int, int] = (0, 0)  # Home base for refueling/resupply

    # Internal state machine
    state: AgentState = AgentState.IDLE
    target_pos: Optional[Tuple[int, int]] = None
    target_end_pos: Optional[Tuple[int, int]] = None  # For line-based actions

    # Internal resource tracking
    payload: float = field(init=False)  # Will be initialized based on rules

    def __post_init__(self):
        # Initialize payload from rules
        rules = RULES[self.kind]
        self.payload = rules.payload_capacity


# -------------------------
# Agent Ruleset
# -------------------------
RULES: Dict[AgentType, AgentRules] = {
    AgentType.CREW: AgentRules(
        speed=2, range=5, suppression=0.30,
        extinguish_chance=0.75, cooldown_strength=0.15,
        cooldown_decay=0.08, drop_every=1
    ),
    AgentType.AIR_TANKER: AgentRules(
        speed=15,
        range=2,
        payload_capacity=1.0,
        payload_per_tick=1.0,
        cooldown_strength=0.0,
        cooldown_decay=0.0
    ),
}


# -------------------------
# Targeting Logic
# -------------------------
def _nearest_burning(state: np.ndarray, r0: int, c0: int) -> tuple[int, int] | None:
    """Finds the coordinates of the nearest burning cell (for CREW)."""
    rr, cc = np.where(state == 2)
    if rr.size == 0: return None
    k = np.argmin((rr - r0) ** 2 + (cc - c0) ** 2)
    return int(rr[k]), int(cc[k])


def _find_drop_line_target(
        state: np.ndarray, wind_uv: np.ndarray, h: int, w: int, lead_dist: int = 40, line_len: int = 50
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Calculates a retardant drop line ahead of the fire's leading edge.
    This version uses a direct vector method to ensure correct line orientation.
    """
    burn_r, burn_c = np.where(state == 2)
    if burn_r.size < 10: return None

    effective_wind = wind_uv
    if np.linalg.norm(effective_wind) < 1e-6:
        map_center_r, map_center_c = h // 2, w // 2
        fire_center_r, fire_center_c = burn_r.mean(), burn_c.mean()
        pseudo_wind_c = fire_center_c - map_center_c
        pseudo_wind_r = fire_center_r - map_center_r
        effective_wind = np.array([pseudo_wind_c, pseudo_wind_r])
        norm = np.linalg.norm(effective_wind)
        if norm > 0:
            effective_wind /= norm
        else:
            effective_wind = np.array([0, 1.0])

    fire_center_r, fire_center_c = burn_r.mean(), burn_c.mean()
    max_dot = -np.inf
    leading_edge_point = None
    for r, c in zip(burn_r, burn_c):
        vec_to_cell = np.array([c - fire_center_c, r - fire_center_r])
        dot_product = np.dot(vec_to_cell, effective_wind)
        if dot_product > max_dot:
            max_dot = dot_product
            leading_edge_point = (r, c)

    if leading_edge_point is None:
        leading_edge_point = (fire_center_r, fire_center_c)

    wind_c, wind_r = effective_wind[0], effective_wind[1]
    drop_center_r = leading_edge_point[0] + wind_r * lead_dist
    drop_center_c = leading_edge_point[1] + wind_c * lead_dist

    # --- NEW & FIXED: Simpler Perpendicular Vector Calculation ---
    # A vector perpendicular to (c, r) is simply (-r, c). This is more robust.
    perp_c = -wind_r
    perp_r = wind_c

    # Normalize the perpendicular vector to ensure the line length is accurate.
    perp_norm = math.sqrt(perp_c ** 2 + perp_r ** 2)
    if perp_norm > 0:
        perp_c /= perp_norm
        perp_r /= perp_norm

    half_line = line_len // 2

    # Calculate start and end points by moving from the center along the perpendicular vector.
    start_c = int(drop_center_c - perp_c * half_line)
    start_r = int(drop_center_r - perp_r * half_line)
    end_c = int(drop_center_c + perp_c * half_line)
    end_r = int(drop_center_r + perp_r * half_line)

    return (start_r, start_c), (end_r, end_c)
# -------------------------
# Agent Behavior Handlers
# -------------------------
def _move_agent_towards(agent: Agent, target_r: int, target_c: int, speed: int, h: int, w: int):
    # ... (This function is unchanged)
    for _ in range(speed):
        dr, dc = np.sign(target_r - agent.r), np.sign(target_c - agent.c)
        if agent.r == target_r and agent.c == target_c: break
        agent.r, agent.c = np.clip(agent.r + dr, 0, h - 1), np.clip(agent.c + dc, 0, w - 1)


def _draw_line(r0, c0, r1, c1, state, fuel_id, width_range):
    # ... (This function is unchanged)
    h, w = state.shape
    dr, dc = r1 - r0, c1 - c0
    dist = max(abs(dr), abs(dc), 1)
    for i in range(int(dist)):
        t = i / dist
        curr_r, curr_c = int(r0 + t * dr), int(c0 + t * dc)
        rmin, rmax = max(0, curr_r - width_range), min(h, curr_r + width_range + 1)
        cmin, cmax = max(0, curr_c - width_range), min(w, curr_c + width_range + 1)
        fuel_id[rmin:rmax, cmin:cmax], state[rmin:rmax, cmin:cmax] = 5, 0


def _handle_crew(agent: Agent, state: np.ndarray, instant_map: np.ndarray, cooldown_map: np.ndarray, h: int, w: int):
    # ... (This function is unchanged)
    rules = RULES[agent.kind]
    target = _nearest_burning(state, agent.r, agent.c)
    if target:
        _move_agent_towards(agent, target[0], target[1], rules.speed, h, w)
        rmin, rmax = max(0, agent.r - rules.range), min(h, agent.r + rules.range + 1)
        cmin, cmax = max(0, agent.c - rules.range), min(w, agent.c + rules.range + 1)
        if rules.extinguish_chance > 0:
            patch = state[rmin:rmax, cmin:cmax]
            burn_mask = (patch == 2)
            if burn_mask.any():
                rnd = (np.random.random(size=burn_mask.shape) < rules.extinguish_chance)
                patch[burn_mask & rnd] = 1
        if rules.suppression > 0:
            instant_map[rmin:rmax, cmin:cmax] *= (1.0 - rules.suppression)
        if rules.cooldown_strength > 0:
            cooldown_map[rmin:rmax, cmin:cmax] = np.clip(
                cooldown_map[rmin:rmax, cmin:cmax] + rules.cooldown_strength, 0.0, 1.0
            )


def _handle_air_tanker(agent: Agent, state: np.ndarray, fuel_id: np.ndarray, wind_uv: np.ndarray, h: int, w: int):
    """State machine logic for an AIR_TANKER agent."""
    rules = RULES[agent.kind]
    if agent.state == AgentState.IDLE:
        if agent.payload > 0:
            target_line = _find_drop_line_target(state, wind_uv, h, w)
            if target_line:
                agent.target_pos, agent.target_end_pos = target_line
                agent.state = AgentState.MOVING_TO_TARGET
                print(f"Tanker dispatched. Target: {agent.target_pos} -> {agent.target_end_pos}")

    elif agent.state == AgentState.MOVING_TO_TARGET:
        _move_agent_towards(agent, agent.target_pos[0], agent.target_pos[1], rules.speed, h, w)
        if agent.r == agent.target_pos[0] and agent.c == agent.target_pos[1]:
            agent.state = AgentState.WORKING

    elif agent.state == AgentState.WORKING:
        start_r, start_c = agent.r, agent.c
        _move_agent_towards(agent, agent.target_end_pos[0], agent.target_end_pos[1], rules.speed, h, w)
        end_r, end_c = agent.r, agent.c
        _draw_line(start_r, start_c, end_r, end_c, state, fuel_id, rules.range)
        if agent.r == agent.target_end_pos[0] and agent.c == agent.target_end_pos[1]:
            agent.payload = 0
            agent.state = AgentState.RETURNING_TO_BASE
            print("Tanker drop complete. Returning to base.")

    elif agent.state == AgentState.RETURNING_TO_BASE:
        _move_agent_towards(agent, agent.base_pos[0], agent.base_pos[1], rules.speed, h, w)
        if agent.r == agent.base_pos[0] and agent.c == agent.base_pos[1]:
            agent.payload = rules.payload_capacity
            agent.state = AgentState.IDLE
            print("Tanker refueled and ready.")


# -------------------------
# Main Step Function
# -------------------------
def step_agents(
        agents: List[Agent],
        state: np.ndarray,
        fuel_id: np.ndarray,
        wind_uv: np.ndarray,
        cooldown_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Moves agents and applies their effects."""
    h, w = state.shape
    instant_map = np.ones((h, w), dtype=np.float32)
    crew_agents = [a for a in agents if a.kind == AgentType.CREW]
    if crew_agents:
        global_decay = max(RULES[a.kind].cooldown_decay for a in crew_agents)
        if global_decay > 0:
            cooldown_map *= (1.0 - global_decay)

    for a in agents:
        if a.kind == AgentType.CREW:
            _handle_crew(a, state, instant_map, cooldown_map, h, w)
        elif a.kind == AgentType.AIR_TANKER:
            _handle_air_tanker(a, state, fuel_id, wind_uv, h, w)

    return instant_map, cooldown_map