#pragma once

#include <stdexcept>

#include <gsolver/graph/edge_base.h>

namespace gsolver {

  template <typename VariableType_>
  struct Message {
    using VariableType = VariableType_;
    using EstimateType = typename VariableType::EstimateType;
    using MatrixLieDim = typename VariableType::MatrixLieDim;

    bool processed_ = false; // Flag indicating if the message has been processed
    MatrixLieDim Lambda_; // Lambda matrix (precision matrix of the factor)
    EstimateType mu_;     // Estimate of the variable
  };

  template <typename VariableType>
  class Edge : public EdgeBase {
  public:
    using VariableType_ = VariableType;
    using MessageType   = Message<VariableType>;

    static constexpr int LieDim      = VariableType::LieDim;
    static constexpr int PayloadSize = VariableType::EncodedSize + LieDim * (LieDim + 1) / 2;

    MessageType variable_to_factor_message_;
    MessageType factor_to_variable_message_;

    int payloadSize() const override {
      return PayloadSize;
    }

    void writeOutgoingPayload(double* payload) const override {
      const MessageType& message = outgoing();
      VariableType::encode(message.mu_, payload);
      double* lambda = payload + VariableType::EncodedSize;
      for (int r = 0; r < LieDim; ++r) {
        for (int c = r; c < LieDim; ++c) {
          *lambda++ = message.Lambda_(r, c);
        }
      }
    }

    void readIncomingPayload(const double* payload) override {
      MessageType& message = incoming();
      message.mu_          = VariableType::decode(payload);
      const double* lambda = payload + VariableType::EncodedSize;
      for (int r = 0; r < LieDim; ++r) {
        for (int c = r; c < LieDim; ++c) {
          message.Lambda_(r, c) = message.Lambda_(c, r) = *lambda++;
        }
      }
    }

  private:
    const MessageType& outgoing() const {
      switch (side_) {
        case EdgeSide::FACTOR_SIDE:
          return factor_to_variable_message_;
        case EdgeSide::VARIABLE_SIDE:
          return variable_to_factor_message_;
        default:
          throw std::logic_error("Edge payloads exist only for boundary edges");
      }
    }

    MessageType& incoming() {
      switch (side_) {
        case EdgeSide::FACTOR_SIDE:
          return variable_to_factor_message_;
        case EdgeSide::VARIABLE_SIDE:
          return factor_to_variable_message_;
        default:
          throw std::logic_error("Edge payloads exist only for boundary edges");
      }
    }
  };

} // namespace gsolver
