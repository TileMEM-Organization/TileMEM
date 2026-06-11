#include "tilemem/io_engine.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <linux/io_uring.h>

namespace tilemem {
namespace {

std::runtime_error syscall_error(const std::string &what) {
  return std::runtime_error(what + ": " + std::strerror(errno));
}

int io_uring_setup(unsigned entries, io_uring_params *params) {
  return static_cast<int>(syscall(__NR_io_uring_setup, entries, params));
}

int io_uring_enter(int ring_fd, unsigned to_submit, unsigned min_complete,
                   unsigned flags) {
  return static_cast<int>(syscall(__NR_io_uring_enter, ring_fd, to_submit,
                                  min_complete, flags, nullptr, 0));
}

} // namespace

FileDescriptor::~FileDescriptor() {
  if (fd_ >= 0) {
    close(fd_);
  }
}

FileDescriptor::FileDescriptor(FileDescriptor &&other) noexcept : fd_(other.fd_) {
  other.fd_ = -1;
}

FileDescriptor &FileDescriptor::operator=(FileDescriptor &&other) noexcept {
  if (this != &other) {
    if (fd_ >= 0) {
      close(fd_);
    }
    fd_ = other.fd_;
    other.fd_ = -1;
  }
  return *this;
}

FileDescriptor open_readonly(const std::string &path) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw syscall_error("open " + path);
  }
  return FileDescriptor(fd);
}

UringIoEngine::UringIoEngine(unsigned entries) : entries_(entries) {
  io_uring_params params{};
  ring_fd_ = io_uring_setup(entries, &params);
  if (ring_fd_ < 0) {
    throw syscall_error("io_uring_setup");
  }

  sq_ring_size_ = params.sq_off.array + params.sq_entries * sizeof(unsigned);
  cq_ring_size_ =
      params.cq_off.cqes + params.cq_entries * sizeof(io_uring_cqe);
  sqes_size_ = params.sq_entries * sizeof(io_uring_sqe);
  cqes_offset_ = params.cq_off.cqes;
  cq_entries_ = params.cq_entries;

  sq_ring_ = mmap(nullptr, sq_ring_size_, PROT_READ | PROT_WRITE,
                  MAP_SHARED | MAP_POPULATE, ring_fd_, IORING_OFF_SQ_RING);
  if (sq_ring_ == MAP_FAILED) {
    throw syscall_error("mmap SQ ring");
  }
  cq_ring_ = mmap(nullptr, cq_ring_size_, PROT_READ | PROT_WRITE,
                  MAP_SHARED | MAP_POPULATE, ring_fd_, IORING_OFF_CQ_RING);
  if (cq_ring_ == MAP_FAILED) {
    throw syscall_error("mmap CQ ring");
  }
  sqes_ = mmap(nullptr, sqes_size_, PROT_READ | PROT_WRITE,
               MAP_SHARED | MAP_POPULATE, ring_fd_, IORING_OFF_SQES);
  if (sqes_ == MAP_FAILED) {
    throw syscall_error("mmap SQEs");
  }

  auto *sq = static_cast<char *>(sq_ring_);
  sq_head_ = reinterpret_cast<unsigned *>(sq + params.sq_off.head);
  sq_tail_ = reinterpret_cast<unsigned *>(sq + params.sq_off.tail);
  sq_ring_mask_ = reinterpret_cast<unsigned *>(sq + params.sq_off.ring_mask);
  sq_array_ = reinterpret_cast<unsigned *>(sq + params.sq_off.array);

  auto *cq = static_cast<char *>(cq_ring_);
  cq_head_ = reinterpret_cast<unsigned *>(cq + params.cq_off.head);
  cq_tail_ = reinterpret_cast<unsigned *>(cq + params.cq_off.tail);
  cq_ring_mask_ = reinterpret_cast<unsigned *>(cq + params.cq_off.ring_mask);
}

UringIoEngine::~UringIoEngine() {
  if (sqes_ && sqes_ != MAP_FAILED) {
    munmap(sqes_, sqes_size_);
  }
  if (cq_ring_ && cq_ring_ != MAP_FAILED) {
    munmap(cq_ring_, cq_ring_size_);
  }
  if (sq_ring_ && sq_ring_ != MAP_FAILED) {
    munmap(sq_ring_, sq_ring_size_);
  }
  if (ring_fd_ >= 0) {
    close(ring_fd_);
  }
}

std::size_t UringIoEngine::read_exact(int fd, void *buffer, std::size_t bytes,
                                      std::uint64_t offset) {
  unsigned tail = __atomic_load_n(sq_tail_, __ATOMIC_ACQUIRE);
  unsigned head = __atomic_load_n(sq_head_, __ATOMIC_ACQUIRE);
  if (tail - head >= entries_) {
    throw std::runtime_error("io_uring SQ is full");
  }

  unsigned index = tail & *sq_ring_mask_;
  auto *sqes = static_cast<io_uring_sqe *>(sqes_);
  auto &sqe = sqes[index];
  std::memset(&sqe, 0, sizeof(sqe));
  sqe.opcode = IORING_OP_READ;
  sqe.fd = fd;
  sqe.off = offset;
  sqe.addr = reinterpret_cast<std::uint64_t>(buffer);
  sqe.len = static_cast<unsigned>(bytes);
  sqe.user_data = 1;
  sq_array_[index] = index;
  __atomic_store_n(sq_tail_, tail + 1, __ATOMIC_RELEASE);

  if (io_uring_enter(ring_fd_, 1, 1, IORING_ENTER_GETEVENTS) < 0) {
    throw syscall_error("io_uring_enter");
  }

  unsigned cq_head;
  unsigned cq_tail;
  do {
    cq_head = __atomic_load_n(cq_head_, __ATOMIC_ACQUIRE);
    cq_tail = __atomic_load_n(cq_tail_, __ATOMIC_ACQUIRE);
  } while (cq_head == cq_tail);

  auto *cqes_start =
      reinterpret_cast<io_uring_cqe *>(static_cast<char *>(cq_ring_) +
                                       cqes_offset_);
  auto &ready = cqes_start[cq_head & *cq_ring_mask_];
  int res = ready.res;
  __atomic_store_n(cq_head_, cq_head + 1, __ATOMIC_RELEASE);
  if (res < 0) {
    errno = -res;
    throw syscall_error("io_uring read");
  }
  if (static_cast<std::size_t>(res) != bytes) {
    throw std::runtime_error("short io_uring read");
  }
  return static_cast<std::size_t>(res);
}

} // namespace tilemem
