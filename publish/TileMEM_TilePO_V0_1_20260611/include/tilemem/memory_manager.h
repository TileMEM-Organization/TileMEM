#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>

#include <cuda_runtime.h>

#include "tilemem/io_engine.h"
#include "tilemem/metrics.h"
#include "tilemem/queues.h"
#include "tilemem/types.h"

namespace tilemem {

class PinnedDramPool {
public:
  explicit PinnedDramPool(std::size_t budget_bytes);
  ~PinnedDramPool();
  PinnedDramPool(const PinnedDramPool &) = delete;
  PinnedDramPool &operator=(const PinnedDramPool &) = delete;

  void *allocate(std::size_t bytes);
  void release(void *ptr);
  std::size_t used_bytes() const { return used_bytes_; }
  std::size_t budget_bytes() const { return budget_bytes_; }

private:
  std::size_t budget_bytes_ = 0;
  std::size_t used_bytes_ = 0;
  std::unordered_map<void *, std::size_t> allocations_;
};

class HbmPool {
public:
  explicit HbmPool(std::size_t budget_bytes);
  ~HbmPool();
  HbmPool(const HbmPool &) = delete;
  HbmPool &operator=(const HbmPool &) = delete;

  void *allocate(std::size_t bytes);
  void release(void *ptr);
  std::size_t used_bytes() const { return used_bytes_; }
  std::size_t budget_bytes() const { return budget_bytes_; }

private:
  std::size_t budget_bytes_ = 0;
  std::size_t used_bytes_ = 0;
  std::unordered_map<void *, std::size_t> allocations_;
};

class MemoryManager {
public:
  MemoryManager(std::size_t dram_budget_bytes, std::size_t hbm_budget_bytes);

  void *ensure_hbm(const StorageTile &tile, int fd, UringIoEngine &io,
                   cudaStream_t stream, Metrics &metrics);
  void *ensure_hbm(const StorageTile &tile, int fd, IoQueue &io_queue,
                   H2DQueue &h2d_queue, Metrics &metrics,
                   bool disable_dram_cache = false);

  bool is_in_hbm(const StorageTileId &id) const;
  bool is_in_dram(const StorageTileId &id) const;
  std::size_t dram_used_bytes() const { return dram_pool_.used_bytes(); }
  std::size_t hbm_used_bytes() const { return hbm_pool_.used_bytes(); }

  void mark_loaded_for_test(const StorageTile &tile, std::uint64_t tick);
  void reserve_for_test(const StorageTile &tile, std::uint64_t tick);

private:
  struct Entry {
    StorageTile tile;
    void *dram_ptr = nullptr;
    void *hbm_ptr = nullptr;
    bool dram = false;
    bool hbm = false;
    bool owns_memory = false;
    std::uint64_t last_used = 0;
  };

  void evict_dram_until(std::size_t required_bytes, const StorageTileId *keep);
  void evict_hbm_until(std::size_t required_bytes, const StorageTileId *keep);
  Entry &entry_for(const StorageTile &tile);

  PinnedDramPool dram_pool_;
  HbmPool hbm_pool_;
  std::unordered_map<StorageTileId, Entry> entries_;
  std::uint64_t clock_ = 0;
};

} // namespace tilemem
