#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "tilemem/config.h"
#include "tilemem/types.h"

namespace fs = std::filesystem;

namespace {

struct Args {
  std::string config;
  std::string out;
  std::string workload = "synthetic";
};

Args parse_args(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string flag = argv[i];
    if (flag == "--config" && i + 1 < argc) {
      args.config = argv[++i];
    } else if (flag == "--out" && i + 1 < argc) {
      args.out = argv[++i];
    } else if (flag == "--workload" && i + 1 < argc) {
      args.workload = argv[++i];
    } else {
      throw std::runtime_error("unknown or incomplete argument: " + flag);
    }
  }
  if (args.config.empty() || args.out.empty()) {
    throw std::runtime_error(
        "usage: tilemem_generate_demo_data --config PATH --out DIR");
  }
  return args;
}

} // namespace

int main(int argc, char **argv) {
  try {
    auto args = parse_args(argc, argv);
    auto config = tilemem::ReplayConfig::from_file(args.config);
    fs::create_directories(args.out);

    const int layers = 2;
    const int experts = 4;
    const int k_dim = config.default_k;
    const int n_total = 64;
    const int shard_n = config.default_n;
    const std::string matrices[] = {"gate", "up", "down"};

    fs::path expert_file = fs::path(args.out) / "experts.bin";
    fs::path manifest_file = fs::path(args.out) / "manifest.jsonl";
    fs::path trace_file = fs::path(args.out) / "trace.jsonl";

    std::ofstream expert_out(expert_file, std::ios::binary);
    std::ofstream manifest_out(manifest_file);
    if (!expert_out || !manifest_out) {
      throw std::runtime_error("failed to open demo data outputs");
    }

    std::uint64_t offset = 0;
    for (int layer = 0; layer < layers; ++layer) {
      for (int expert = 0; expert < experts; ++expert) {
        for (const auto &matrix : matrices) {
          for (int n_start = 0; n_start < n_total; n_start += shard_n) {
            int n_end = n_start + shard_n;
            std::size_t bytes = static_cast<std::size_t>(k_dim) * shard_n / 2;
            for (std::size_t i = 0; i < bytes; ++i) {
              unsigned char value =
                  static_cast<unsigned char>(((offset + i) % 15) + 1);
              expert_out.write(reinterpret_cast<const char *>(&value), 1);
            }
            manifest_out << "{\"layer\":" << layer << ",\"expert\":" << expert
                         << ",\"matrix\":\"" << matrix
                         << "\",\"n_start\":" << n_start
                         << ",\"n_end\":" << n_end << ",\"bytes\":" << bytes
                         << ",\"file_offset\":" << offset
                         << ",\"k_dim\":" << k_dim
                         << ",\"n_dim\":" << shard_n << "}\n";
            offset += bytes;
          }
        }
      }
    }

    std::ofstream trace_out(trace_file);
    if (args.workload == "real") {
      trace_out << "{\"step\":0,\"phase\":\"prefill\",\"request_id\":11,"
                   "\"layer\":0,\"token_count\":16,\"experts\":[0,1]}\n";
      trace_out << "{\"step\":1,\"phase\":\"decode\",\"request_id\":11,"
                   "\"layer\":1,\"token_count\":1,\"experts\":[1,2]}\n";
      trace_out << "{\"step\":2,\"phase\":\"decode\",\"request_id\":11,"
                   "\"layer\":0,\"token_count\":1,\"experts\":[0,1]}\n";
      trace_out << "{\"step\":3,\"phase\":\"prefill\",\"request_id\":12,"
                   "\"layer\":1,\"token_count\":10,\"experts\":[2,3]}\n";
      trace_out << "{\"step\":4,\"phase\":\"decode\",\"request_id\":12,"
                   "\"layer\":0,\"token_count\":1,\"experts\":[0,3]}\n";
      trace_out << "{\"step\":5,\"phase\":\"decode\",\"request_id\":12,"
                   "\"layer\":1,\"token_count\":1,\"experts\":[1,2]}\n";
    } else {
      trace_out << "{\"step\":0,\"phase\":\"prefill\",\"request_id\":1,"
                   "\"layer\":0,\"token_count\":8,\"experts\":[0,1]}\n";
      trace_out << "{\"step\":1,\"phase\":\"decode\",\"request_id\":1,"
                   "\"layer\":1,\"token_count\":1,\"experts\":[1,2]}\n";
      trace_out << "{\"step\":2,\"phase\":\"prefill\",\"request_id\":2,"
                   "\"layer\":0,\"token_count\":6,\"experts\":[2,3]}\n";
      trace_out << "{\"step\":3,\"phase\":\"decode\",\"request_id\":2,"
                   "\"layer\":1,\"token_count\":1,\"experts\":[0,3]}\n";
    }

    std::cout << "wrote demo data to " << args.out << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "tilemem_generate_demo_data: " << e.what() << "\n";
    return 1;
  }
}
