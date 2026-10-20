#pragma once

#include <functional>
#include <random>
#include <string>

namespace gsolver {

  //! How a node replaces its outgoing messages. Both default to off, so messages are simply
  //! overwritten.
  //!  damping: share of the previous message each new factor-to-variable message keeps (see
  //!           storeMessage). Against oscillation in loopy graphs; the fixed point is unchanged.
  //!  dropout: chance that a new message is not sent at all and the previous one stays, drawn
  //!           for every message. A node's first message is never dropped, nor a message
  //!           carrying nothing (a disabled factor must send exactly zero precision).
  //! Draws come from a generator per node seeded from its id (prepare), so a run is reproducible
  //! whatever thread each node runs on.
  class MessageUpdate {
  public:
    double damping = 0.0;
    double dropout = 0.0;

    void prepare(const std::string& node_id) {
      if (dropout > 0.0 && !seeded_) {
        rng_.seed(static_cast<std::minstd_rand::result_type>(std::hash<std::string>{}(node_id) % 2147483646u + 1u));
        seeded_ = true;
      }
    }
    bool drop() {
      return dropout > 0.0 && std::uniform_real_distribution<double>(0.0, 1.0)(rng_) < dropout;
    }

  private:
    std::minstd_rand rng_;
    bool seeded_ = false;
  };

} // namespace gsolver
