#include "tilemem/manifest.h"

#include <fstream>
#include <stdexcept>

#include "parsing.h"

namespace tilemem {

std::vector<StorageTile> load_manifest(const std::string &path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open manifest: " + path);
  }

  std::vector<StorageTile> tiles;
  std::string line;
  while (std::getline(input, line)) {
    line = parsing::trim(line);
    if (line.empty()) {
      continue;
    }
    StorageTile tile;
    tile.id.layer = parsing::json_integer<int>(line, "layer");
    tile.id.expert = parsing::json_integer<int>(line, "expert");
    tile.id.matrix = parse_matrix_kind(parsing::json_string(line, "matrix"));
    tile.id.n_start = parsing::json_integer<int>(line, "n_start");
    tile.id.n_end = parsing::json_integer<int>(line, "n_end");
    tile.bytes = parsing::json_integer<std::size_t>(line, "bytes");
    tile.file_offset =
        parsing::json_integer<std::uint64_t>(line, "file_offset");
    tile.k_dim = parsing::json_integer<int>(line, "k_dim");
    tile.n_dim = parsing::json_integer<int>(line, "n_dim");
    tile.tier = Tier::Ssd;
    tiles.push_back(tile);
  }
  return tiles;
}

} // namespace tilemem
