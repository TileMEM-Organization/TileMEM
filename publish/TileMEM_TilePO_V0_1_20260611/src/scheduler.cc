#include "tilemem/scheduler.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace tilemem {

Scheduler::Scheduler(std::vector<StorageTile> manifest,
                     std::vector<TraceEvent> trace)
    : manifest_(std::move(manifest)), trace_(std::move(trace)) {
  for (std::size_t i = 0; i < manifest_.size(); ++i) {
    tile_index_[manifest_[i].id] = i;
  }
}

const StorageTile &Scheduler::tile(const StorageTileId &id) const {
  auto it = tile_index_.find(id);
  if (it == tile_index_.end()) {
    throw std::runtime_error("unknown storage tile");
  }
  return manifest_[it->second];
}

std::vector<StorageTileId>
Scheduler::required_for_event(std::size_t event_index) const {
  const auto &event = trace_.at(event_index);
  std::unordered_set<int> experts(event.experts.begin(), event.experts.end());
  std::vector<StorageTileId> result;
  for (const auto &tile : manifest_) {
    if (tile.id.layer == event.layer &&
        experts.find(tile.id.expert) != experts.end()) {
      result.push_back(tile.id);
    }
  }
  std::sort(result.begin(), result.end(), [](const auto &a, const auto &b) {
    if (a.layer != b.layer) {
      return a.layer < b.layer;
    }
    if (a.expert != b.expert) {
      return a.expert < b.expert;
    }
    if (a.matrix != b.matrix) {
      return static_cast<int>(a.matrix) < static_cast<int>(b.matrix);
    }
    return a.n_start < b.n_start;
  });
  return result;
}

std::vector<StorageTileId> Scheduler::future_tiles(std::size_t event_index,
                                                   int lookahead) const {
  std::vector<StorageTileId> result;
  std::unordered_set<StorageTileId> seen;
  auto end = std::min(trace_.size(), event_index + 1 +
                                      static_cast<std::size_t>(lookahead));
  for (std::size_t i = event_index + 1; i < end; ++i) {
    for (const auto &id : required_for_event(i)) {
      if (seen.insert(id).second) {
        result.push_back(id);
      }
    }
  }
  return result;
}

SchedulePlan Scheduler::plan(std::size_t event_index, Policy policy,
                             int lookahead) const {
  SchedulePlan plan;
  plan.required = required_for_event(event_index);

  if (policy == Policy::OnDemand || policy == Policy::Lru ||
      policy == Policy::TilememNoHbmPrefetch || lookahead <= 0) {
    return plan;
  }

  auto future = future_tiles(event_index, lookahead);
  std::unordered_set<StorageTileId> required(plan.required.begin(),
                                             plan.required.end());
  future.erase(std::remove_if(future.begin(), future.end(),
                              [&](const auto &id) {
                                return required.find(id) != required.end();
                              }),
               future.end());

  if (policy == Policy::LayerwisePrefetch || policy == Policy::Oracle ||
      policy == Policy::TilememNoConflict) {
    plan.prefetch = std::move(future);
    return plan;
  }

  std::vector<std::pair<double, StorageTileId>> scored;
  for (const auto &id : future) {
    const auto &t = tile(id);
    double distance_penalty = 1000.0;
    double reuse_bonus = 0.0;
    for (std::size_t i = event_index + 1; i < trace_.size(); ++i) {
      auto required_ids = required_for_event(i);
      if (std::find(required_ids.begin(), required_ids.end(), id) !=
          required_ids.end()) {
        if (distance_penalty == 1000.0) {
          distance_penalty = static_cast<double>(i - event_index);
        }
        reuse_bonus += static_cast<double>(trace_[i].token_count);
      }
    }
    double size_mb = static_cast<double>(t.bytes) / (1024.0 * 1024.0);
    double deadline_bonus =
        policy == Policy::TilememNoDeadline ? 0.0 : 700.0 / distance_penalty;
    double regret_bonus =
        policy == Policy::TilememNoEvictionRegret ? 0.0 : reuse_bonus * 30.0;
    double contention_penalty =
        policy == Policy::TilememNoConflict ? 0.0 : size_mb * 25.0;
    double score = deadline_bonus + regret_bonus - contention_penalty;
    scored.emplace_back(score, id);
  }
  std::sort(scored.begin(), scored.end(), [](const auto &a, const auto &b) {
    return a.first > b.first;
  });
  for (const auto &[_, id] : scored) {
    plan.prefetch.push_back(id);
  }
  return plan;
}

} // namespace tilemem
