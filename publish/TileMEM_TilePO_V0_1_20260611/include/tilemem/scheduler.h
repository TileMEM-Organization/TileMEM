#pragma once

#include <unordered_map>
#include <vector>

#include "tilemem/types.h"

namespace tilemem {

struct SchedulePlan {
  std::vector<StorageTileId> required;
  std::vector<StorageTileId> prefetch;
};

class Scheduler {
public:
  Scheduler(std::vector<StorageTile> manifest, std::vector<TraceEvent> trace);

  SchedulePlan plan(std::size_t event_index, Policy policy, int lookahead) const;
  const StorageTile &tile(const StorageTileId &id) const;

private:
  std::vector<StorageTileId> required_for_event(std::size_t event_index) const;
  std::vector<StorageTileId> future_tiles(std::size_t event_index,
                                          int lookahead) const;

  std::vector<StorageTile> manifest_;
  std::vector<TraceEvent> trace_;
  std::unordered_map<StorageTileId, std::size_t> tile_index_;
};

} // namespace tilemem
