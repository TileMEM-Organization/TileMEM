from __future__ import annotations

from enum import Enum

from tilepo.mir import TileId


class TileResidencyState(str, Enum):
    COLD_STORAGE = "COLD_STORAGE"
    CPU_RESIDENT = "CPU_RESIDENT"
    PINNED_READY = "PINNED_READY"
    GPU_PREFETCHING = "GPU_PREFETCHING"
    GPU_RESIDENT = "GPU_RESIDENT"
    GPU_EXECUTING = "GPU_EXECUTING"
    EVICTING = "EVICTING"
    KT_FALLBACK = "KT_FALLBACK"


_ALLOWED: dict[TileResidencyState, set[TileResidencyState]] = {
    TileResidencyState.COLD_STORAGE: {TileResidencyState.CPU_RESIDENT, TileResidencyState.KT_FALLBACK},
    TileResidencyState.CPU_RESIDENT: {TileResidencyState.PINNED_READY, TileResidencyState.EVICTING, TileResidencyState.KT_FALLBACK},
    TileResidencyState.PINNED_READY: {TileResidencyState.GPU_PREFETCHING, TileResidencyState.EVICTING, TileResidencyState.KT_FALLBACK},
    TileResidencyState.GPU_PREFETCHING: {TileResidencyState.GPU_RESIDENT, TileResidencyState.KT_FALLBACK},
    TileResidencyState.GPU_RESIDENT: {TileResidencyState.GPU_EXECUTING, TileResidencyState.EVICTING},
    TileResidencyState.GPU_EXECUTING: {TileResidencyState.GPU_RESIDENT, TileResidencyState.EVICTING},
    TileResidencyState.EVICTING: {TileResidencyState.CPU_RESIDENT, TileResidencyState.COLD_STORAGE},
    TileResidencyState.KT_FALLBACK: {TileResidencyState.COLD_STORAGE, TileResidencyState.CPU_RESIDENT},
}


class TileState:
    def __init__(self) -> None:
        self._state: dict[str, TileResidencyState] = {}

    def get(self, tile: TileId | str) -> TileResidencyState:
        key = tile.stable_key() if isinstance(tile, TileId) else tile
        return self._state.get(key, TileResidencyState.COLD_STORAGE)

    def transition(self, tile: TileId | str, next_state: TileResidencyState) -> None:
        key = tile.stable_key() if isinstance(tile, TileId) else tile
        current = self.get(key)
        if next_state not in _ALLOWED[current] and next_state != current:
            raise ValueError(f"invalid tile state transition {current.value} -> {next_state.value}")
        self._state[key] = next_state

    def mark_gpu_resident(self, tile_key: str) -> None:
        state = self.get(tile_key)
        if state == TileResidencyState.COLD_STORAGE:
            self.transition(tile_key, TileResidencyState.CPU_RESIDENT)
            self.transition(tile_key, TileResidencyState.PINNED_READY)
            self.transition(tile_key, TileResidencyState.GPU_PREFETCHING)
            self.transition(tile_key, TileResidencyState.GPU_RESIDENT)
        elif state == TileResidencyState.CPU_RESIDENT:
            self.transition(tile_key, TileResidencyState.PINNED_READY)
            self.transition(tile_key, TileResidencyState.GPU_PREFETCHING)
            self.transition(tile_key, TileResidencyState.GPU_RESIDENT)
        elif state == TileResidencyState.PINNED_READY:
            self.transition(tile_key, TileResidencyState.GPU_PREFETCHING)
            self.transition(tile_key, TileResidencyState.GPU_RESIDENT)
        elif state == TileResidencyState.GPU_PREFETCHING:
            self.transition(tile_key, TileResidencyState.GPU_RESIDENT)

