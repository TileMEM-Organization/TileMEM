#include "tilemem/types.h"

#include <stdexcept>

namespace tilemem {

std::string to_string(MatrixKind matrix) {
  switch (matrix) {
  case MatrixKind::Gate:
    return "gate";
  case MatrixKind::Up:
    return "up";
  case MatrixKind::Down:
    return "down";
  }
  return "unknown";
}

std::string to_string(Phase phase) {
  switch (phase) {
  case Phase::Prefill:
    return "prefill";
  case Phase::Decode:
    return "decode";
  }
  return "unknown";
}

std::string to_string(Policy policy) {
  switch (policy) {
  case Policy::OnDemand:
    return "on_demand";
  case Policy::Lru:
    return "lru";
  case Policy::LayerwisePrefetch:
    return "layerwise_prefetch";
  case Policy::Oracle:
    return "oracle";
  case Policy::Tilemem:
    return "tilemem";
  case Policy::TilememNoConflict:
    return "tilemem_no_conflict";
  case Policy::TilememNoDramCache:
    return "tilemem_no_dram_cache";
  case Policy::TilememNoHbmPrefetch:
    return "tilemem_no_hbm_prefetch";
  case Policy::TilememNoEvictionRegret:
    return "tilemem_no_eviction_regret";
  case Policy::TilememNoDeadline:
    return "tilemem_no_deadline";
  }
  return "unknown";
}

MatrixKind parse_matrix_kind(const std::string &value) {
  if (value == "gate") {
    return MatrixKind::Gate;
  }
  if (value == "up") {
    return MatrixKind::Up;
  }
  if (value == "down") {
    return MatrixKind::Down;
  }
  throw std::runtime_error("unknown matrix kind: " + value);
}

Phase parse_phase(const std::string &value) {
  if (value == "prefill") {
    return Phase::Prefill;
  }
  if (value == "decode") {
    return Phase::Decode;
  }
  throw std::runtime_error("unknown phase: " + value);
}

Policy parse_policy(const std::string &value) {
  if (value == "on_demand") {
    return Policy::OnDemand;
  }
  if (value == "lru") {
    return Policy::Lru;
  }
  if (value == "layerwise_prefetch") {
    return Policy::LayerwisePrefetch;
  }
  if (value == "oracle") {
    return Policy::Oracle;
  }
  if (value == "tilemem") {
    return Policy::Tilemem;
  }
  if (value == "tilemem_no_conflict") {
    return Policy::TilememNoConflict;
  }
  if (value == "tilemem_no_dram_cache") {
    return Policy::TilememNoDramCache;
  }
  if (value == "tilemem_no_hbm_prefetch") {
    return Policy::TilememNoHbmPrefetch;
  }
  if (value == "tilemem_no_eviction_regret") {
    return Policy::TilememNoEvictionRegret;
  }
  if (value == "tilemem_no_deadline") {
    return Policy::TilememNoDeadline;
  }
  throw std::runtime_error("unknown policy: " + value);
}

} // namespace tilemem
