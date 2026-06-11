#pragma once

#include <string>
#include <vector>

#include "tilemem/types.h"

namespace tilemem {

std::vector<TraceEvent> load_trace(const std::string &path);

} // namespace tilemem
