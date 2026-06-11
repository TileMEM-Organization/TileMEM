#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace tilemem {

class FileDescriptor {
public:
  explicit FileDescriptor(int fd = -1) : fd_(fd) {}
  ~FileDescriptor();
  FileDescriptor(const FileDescriptor &) = delete;
  FileDescriptor &operator=(const FileDescriptor &) = delete;
  FileDescriptor(FileDescriptor &&other) noexcept;
  FileDescriptor &operator=(FileDescriptor &&other) noexcept;

  int get() const { return fd_; }

private:
  int fd_ = -1;
};

FileDescriptor open_readonly(const std::string &path);

class UringIoEngine {
public:
  explicit UringIoEngine(unsigned entries);
  ~UringIoEngine();
  UringIoEngine(const UringIoEngine &) = delete;
  UringIoEngine &operator=(const UringIoEngine &) = delete;

  std::size_t read_exact(int fd, void *buffer, std::size_t bytes,
                         std::uint64_t offset);

private:
  int ring_fd_ = -1;
  unsigned entries_ = 0;
  void *sq_ring_ = nullptr;
  void *cq_ring_ = nullptr;
  void *sqes_ = nullptr;
  std::size_t sq_ring_size_ = 0;
  std::size_t cq_ring_size_ = 0;
  std::size_t sqes_size_ = 0;
  std::size_t cqes_offset_ = 0;
  unsigned cq_entries_ = 0;

  unsigned *sq_head_ = nullptr;
  unsigned *sq_tail_ = nullptr;
  unsigned *sq_ring_mask_ = nullptr;
  unsigned *sq_array_ = nullptr;
  unsigned *cq_head_ = nullptr;
  unsigned *cq_tail_ = nullptr;
  unsigned *cq_ring_mask_ = nullptr;
};

} // namespace tilemem
