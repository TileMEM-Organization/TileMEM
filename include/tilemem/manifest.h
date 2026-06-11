#pragma once

#include <string>
#include <vector>

#include "tilemem/types.h"

namespace tilemem {

std::vector<StorageTile> load_manifest(const std::string &path);

} // namespace tilemem
