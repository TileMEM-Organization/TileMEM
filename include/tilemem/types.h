#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace tilemem {

enum class MatrixKind { Gate, Up, Down };
enum class Tier { Ssd, Dram, Hbm };
enum class Phase { Prefill, Decode };
enum class Policy {
  OnDemand,
  Lru,
  LayerwisePrefetch,
  Oracle,
  Tilemem,
  TilememNoConflict,
  TilememNoDramCache,
  TilememNoHbmPrefetch,
  TilememNoEvictionRegret,
  TilememNoDeadline
};

struct StorageTileId {
  int layer = 0;
  int expert = 0;
  MatrixKind matrix = MatrixKind::Gate;
  int n_start = 0;
  int n_end = 0;

  bool operator==(const StorageTileId &other) const {
    return layer == other.layer && expert == other.expert &&
           matrix == other.matrix && n_start == other.n_start &&
           n_end == other.n_end;
  }
};

struct StorageTile {
  StorageTileId id;
  std::size_t bytes = 0;
  std::uint64_t file_offset = 0;
  int k_dim = 0;
  int n_dim = 0;
  Tier tier = Tier::Ssd;
};

struct TraceEvent {
  std::uint64_t step = 0;
  Phase phase = Phase::Decode;
  int request_id = 0;
  int layer = 0;
  int token_count = 0;
  std::vector<int> experts;
};

std::string to_string(MatrixKind matrix);
std::string to_string(Phase phase);
std::string to_string(Policy policy);
MatrixKind parse_matrix_kind(const std::string &value);
Phase parse_phase(const std::string &value);
Policy parse_policy(const std::string &value);

} // namespace tilemem

template <> struct std::hash<tilemem::StorageTileId> {
  std::size_t operator()(const tilemem::StorageTileId &id) const noexcept {
    std::size_t h = static_cast<std::size_t>(id.layer);
    h = h * 1315423911u + static_cast<std::size_t>(id.expert);
    h = h * 1315423911u + static_cast<std::size_t>(id.n_start);
    h = h * 1315423911u + static_cast<std::size_t>(id.n_end);
    h = h * 1315423911u + static_cast<std::size_t>(id.matrix);
    return h;
  }
};
