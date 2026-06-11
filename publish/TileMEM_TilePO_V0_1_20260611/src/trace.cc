#include "tilemem/trace.h"

#include <fstream>
#include <stdexcept>

#include "parsing.h"

namespace tilemem {

std::vector<TraceEvent> load_trace(const std::string &path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open trace: " + path);
  }

  std::vector<TraceEvent> events;
  std::string line;
  while (std::getline(input, line)) {
    line = parsing::trim(line);
    if (line.empty()) {
      continue;
    }
    TraceEvent event;
    event.step = parsing::json_integer<std::uint64_t>(line, "step");
    event.phase = parse_phase(parsing::json_string(line, "phase"));
    event.request_id = parsing::json_integer<int>(line, "request_id");
    event.layer = parsing::json_integer<int>(line, "layer");
    event.token_count = parsing::json_integer<int>(line, "token_count");
    event.experts = parsing::json_int_array(line, "experts");
    events.push_back(event);
  }
  return events;
}

} // namespace tilemem
