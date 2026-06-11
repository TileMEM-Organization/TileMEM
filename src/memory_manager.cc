#include "tilemem/memory_manager.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace tilemem {
namespace {

void check_cuda(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(status));
  }
}

} // namespace

PinnedDramPool::PinnedDramPool(std::size_t budget_bytes)
    : budget_bytes_(budget_bytes) {}

PinnedDramPool::~PinnedDramPool() {
  for (auto &[ptr, _] : allocations_) {
    cudaFreeHost(ptr);
  }
}

void *PinnedDramPool::allocate(std::size_t bytes) {
  if (used_bytes_ + bytes > budget_bytes_) {
    throw std::runtime_error("pinned DRAM pool budget exceeded");
  }
  void *ptr = nullptr;
  check_cuda(cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault),
             "cudaHostAlloc");
  allocations_[ptr] = bytes;
  used_bytes_ += bytes;
  return ptr;
}

void PinnedDramPool::release(void *ptr) {
  if (!ptr) {
    return;
  }
  auto it = allocations_.find(ptr);
  if (it == allocations_.end()) {
    return;
  }
  used_bytes_ -= it->second;
  check_cuda(cudaFreeHost(ptr), "cudaFreeHost");
  allocations_.erase(it);
}

HbmPool::HbmPool(std::size_t budget_bytes) : budget_bytes_(budget_bytes) {}

HbmPool::~HbmPool() {
  for (auto &[ptr, _] : allocations_) {
    cudaFree(ptr);
  }
}

void *HbmPool::allocate(std::size_t bytes) {
  if (used_bytes_ + bytes > budget_bytes_) {
    throw std::runtime_error("HBM pool budget exceeded");
  }
  void *ptr = nullptr;
  check_cuda(cudaMalloc(&ptr, bytes), "cudaMalloc");
  allocations_[ptr] = bytes;
  used_bytes_ += bytes;
  return ptr;
}

void HbmPool::release(void *ptr) {
  if (!ptr) {
    return;
  }
  auto it = allocations_.find(ptr);
  if (it == allocations_.end()) {
    return;
  }
  used_bytes_ -= it->second;
  check_cuda(cudaFree(ptr), "cudaFree");
  allocations_.erase(it);
}

MemoryManager::MemoryManager(std::size_t dram_budget_bytes,
                             std::size_t hbm_budget_bytes)
    : dram_pool_(dram_budget_bytes), hbm_pool_(hbm_budget_bytes) {}

MemoryManager::Entry &MemoryManager::entry_for(const StorageTile &tile) {
  auto it = entries_.find(tile.id);
  if (it == entries_.end()) {
    Entry entry;
    entry.tile = tile;
    auto [inserted, _] = entries_.emplace(tile.id, entry);
    it = inserted;
  }
  return it->second;
}

bool MemoryManager::is_in_hbm(const StorageTileId &id) const {
  auto it = entries_.find(id);
  return it != entries_.end() && it->second.hbm;
}

bool MemoryManager::is_in_dram(const StorageTileId &id) const {
  auto it = entries_.find(id);
  return it != entries_.end() && it->second.dram;
}

void MemoryManager::evict_hbm_until(std::size_t required_bytes,
                                    const StorageTileId *keep) {
  while (hbm_pool_.used_bytes() + required_bytes > hbm_pool_.budget_bytes()) {
    auto victim = entries_.end();
    std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      if (!it->second.hbm) {
        continue;
      }
      if (keep && it->first == *keep) {
        continue;
      }
      if (it->second.last_used < oldest) {
        oldest = it->second.last_used;
        victim = it;
      }
    }
    if (victim == entries_.end()) {
      throw std::runtime_error("no HBM eviction victim available");
    }
    if (victim->second.hbm) {
      // Count only real residency changes. Test-only entries do not own memory
      // but still model eviction behavior.
    }
    if (victim->second.owns_memory) {
      hbm_pool_.release(victim->second.hbm_ptr);
    }
    victim->second.hbm_ptr = nullptr;
    victim->second.hbm = false;
  }
}

void MemoryManager::evict_dram_until(std::size_t required_bytes,
                                     const StorageTileId *keep) {
  while (dram_pool_.used_bytes() + required_bytes > dram_pool_.budget_bytes()) {
    auto victim = entries_.end();
    std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      if (!it->second.dram) {
        continue;
      }
      if (keep && it->first == *keep) {
        continue;
      }
      if (it->second.last_used < oldest) {
        oldest = it->second.last_used;
        victim = it;
      }
    }
    if (victim == entries_.end()) {
      throw std::runtime_error("no DRAM eviction victim available");
    }
    if (victim->second.owns_memory) {
      dram_pool_.release(victim->second.dram_ptr);
    }
    victim->second.dram_ptr = nullptr;
    victim->second.dram = false;
    if (victim->second.hbm) {
      if (victim->second.owns_memory) {
        hbm_pool_.release(victim->second.hbm_ptr);
      }
      victim->second.hbm_ptr = nullptr;
      victim->second.hbm = false;
    }
  }
}

void *MemoryManager::ensure_hbm(const StorageTile &tile, int fd,
                                UringIoEngine &io, cudaStream_t stream,
                                Metrics &metrics) {
  IoQueue io_queue(io);
  H2DQueue h2d_queue(stream);
  return ensure_hbm(tile, fd, io_queue, h2d_queue, metrics, false);
}

void *MemoryManager::ensure_hbm(const StorageTile &tile, int fd,
                                IoQueue &io_queue, H2DQueue &h2d_queue,
                                Metrics &metrics, bool disable_dram_cache) {
  if (tile.bytes > dram_pool_.budget_bytes() ||
      tile.bytes > hbm_pool_.budget_bytes()) {
    throw std::runtime_error("tile exceeds configured memory budget");
  }
  ++clock_;
  auto &entry = entry_for(tile);
  entry.last_used = clock_;

  if (entry.hbm) {
    ++metrics.hbm_hits;
    return entry.hbm_ptr;
  }
  ++metrics.hbm_misses;

  if (entry.dram) {
    ++metrics.dram_hits;
  } else {
    ++metrics.dram_misses;
    ++metrics.reloads;
    evict_dram_until(tile.bytes, &tile.id);
    entry.dram_ptr = dram_pool_.allocate(tile.bytes);
    entry.owns_memory = true;
    auto bytes = io_queue.read(fd, entry.dram_ptr, tile.bytes, tile.file_offset,
                               metrics);
    metrics.io_bytes += bytes;
    entry.dram = true;
  }

  evict_hbm_until(tile.bytes, &tile.id);
  entry.hbm_ptr = hbm_pool_.allocate(tile.bytes);
  entry.owns_memory = true;
  h2d_queue.copy(entry.hbm_ptr, entry.dram_ptr, tile.bytes, metrics);
  metrics.h2d_bytes += tile.bytes;
  entry.hbm = true;
  metrics.dram_occupancy_bytes =
      std::max(metrics.dram_occupancy_bytes,
               static_cast<std::uint64_t>(dram_pool_.used_bytes()));
  metrics.hbm_occupancy_bytes =
      std::max(metrics.hbm_occupancy_bytes,
               static_cast<std::uint64_t>(hbm_pool_.used_bytes()));
  if (disable_dram_cache && entry.dram) {
    dram_pool_.release(entry.dram_ptr);
    entry.dram_ptr = nullptr;
    entry.dram = false;
  }
  return entry.hbm_ptr;
}

void MemoryManager::mark_loaded_for_test(const StorageTile &tile,
                                         std::uint64_t tick) {
  auto &entry = entry_for(tile);
  entry.dram = true;
  entry.hbm = true;
  entry.owns_memory = false;
  entry.last_used = tick;
}

void MemoryManager::reserve_for_test(const StorageTile &tile,
                                     std::uint64_t tick) {
  if (tile.bytes > hbm_pool_.budget_bytes()) {
    throw std::runtime_error("test tile exceeds HBM budget");
  }
  while (true) {
    std::size_t used = 0;
    for (const auto &[_, entry] : entries_) {
      if (entry.hbm) {
        used += entry.tile.bytes;
      }
    }
    if (used + tile.bytes <= hbm_pool_.budget_bytes()) {
      break;
    }
    auto victim = entries_.end();
    std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      if (it->second.hbm && it->second.last_used < oldest) {
        oldest = it->second.last_used;
        victim = it;
      }
    }
    if (victim == entries_.end()) {
      throw std::runtime_error("no test eviction victim");
    }
    victim->second.hbm = false;
  }
  auto &entry = entry_for(tile);
  entry.hbm = true;
  entry.dram = true;
  entry.owns_memory = false;
  entry.last_used = tick;
}

} // namespace tilemem
