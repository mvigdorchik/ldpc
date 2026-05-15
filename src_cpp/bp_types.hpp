#pragma once

#include <vector>

#include "sparse_matrix_base.hpp"
#include "gf2sparse.hpp"

namespace ldpc {
    namespace bp {

        enum BpMethod {
            PRODUCT_SUM = 0,
            MINIMUM_SUM = 1
        };

        enum BpSchedule {
            SERIAL = 0,
            PARALLEL = 1,
            SERIAL_RELATIVE = 2,
            PARALLEL_PACKED = 3,
            PARALLEL_PACKED_FLOAT = 4,
            PARALLEL_PACKED_AVX = 5
        };

        enum BpInputType {
            SYNDROME = 0,
            RECEIVED_VECTOR = 1,
            AUTO = 2
        };

        const std::vector<int> NULL_INT_VECTOR = {};

        class BpEntry : public ldpc::sparse_matrix_base::EntryBase<BpEntry> {
        public:
            double bit_to_check_msg = 0.0;
            double check_to_bit_msg = 0.0;

            ~BpEntry() = default;
        };

        using BpSparse = ldpc::gf2sparse::GF2Sparse<BpEntry>;

    }
}
