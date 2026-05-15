#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "bp_types.hpp"

namespace ldpc {
    namespace bp {

        template<class Real>
        class PackedBpDecoderImpl {
        public:
            explicit PackedBpDecoderImpl(BpSparse &pcm) : check_count(pcm.m), bit_count(pcm.n) {
                this->build(pcm);
            }

            std::vector<uint8_t> &decode_minimum_sum(
                    std::vector<uint8_t> &syndrome,
                    const std::vector<double> &channel_probabilities,
                    const int maximum_iterations,
                    const double ms_scaling_factor,
                    std::vector<uint8_t> &decoding,
                    std::vector<uint8_t> &candidate_syndrome,
                    std::vector<double> &log_prob_ratios,
                    std::vector<double> &initial_log_prob_ratios,
                    int &iterations,
                    bool &converge) {
                converge = false;
                this->initialise_log_domain(channel_probabilities, initial_log_prob_ratios);

                for (int it = 1; it <= maximum_iterations; it++) {
                    Real alpha;
                    if(ms_scaling_factor == 0.0) {
                        alpha = static_cast<Real>(1.0 - std::pow(2.0, -1.0*it));
                    }
                    else {
                        alpha = static_cast<Real>(ms_scaling_factor);
                    }

                    this->update_check_to_bit_messages(syndrome, candidate_syndrome, alpha);
                    this->update_log_prob_ratios_and_decoding(
                            decoding,
                            candidate_syndrome,
                            log_prob_ratios,
                            initial_log_prob_ratios);

                    if (std::equal(candidate_syndrome.begin(), candidate_syndrome.end(), syndrome.begin())) {
                        converge = true;
                    }

                    iterations = it;

                    if (converge) {
                        this->sync_messages_to_pcm();
                        return decoding;
                    }

                    this->update_bit_to_check_messages();
                }

                this->sync_messages_to_pcm();
                return decoding;
            }

        private:
            int check_count;
            int bit_count;
            std::vector<int> row_offsets;
            std::vector<int> col_offsets;
            std::vector<int> col_edges;
            std::vector<int> edge_rows;
            std::vector<int> edge_cols;
            std::vector<BpEntry *> edges;
            std::vector<Real> bit_to_check_msg;
            std::vector<Real> check_to_bit_msg;

            void build(BpSparse &pcm) {
                this->row_offsets.assign(this->check_count + 1, 0);
                this->col_offsets.assign(this->bit_count + 1, 0);
                this->col_edges.clear();
                this->edge_rows.clear();
                this->edge_cols.clear();
                this->edges.clear();

                std::unordered_map<BpEntry *, int> edge_ids;
                edge_ids.reserve(static_cast<size_t>(pcm.entry_count()));

                for (int row = 0; row < this->check_count; row++) {
                    this->row_offsets[row] = static_cast<int>(this->edge_rows.size());
                    for (auto &e: pcm.iterate_row(row)) {
                        const int edge = static_cast<int>(this->edge_rows.size());
                        this->edge_rows.push_back(row);
                        this->edge_cols.push_back(e.col_index);
                        this->edges.push_back(&e);
                        edge_ids.emplace(&e, edge);
                    }
                }
                this->row_offsets[this->check_count] = static_cast<int>(this->edge_rows.size());

                for (int col = 0; col < this->bit_count; col++) {
                    this->col_offsets[col] = static_cast<int>(this->col_edges.size());
                    for (auto &e: pcm.iterate_column(col)) {
                        auto edge_it = edge_ids.find(&e);
                        if (edge_it == edge_ids.end()) {
                            throw std::runtime_error("Failed to pack BP matrix edge");
                        }
                        this->col_edges.push_back(edge_it->second);
                    }
                }
                this->col_offsets[this->bit_count] = static_cast<int>(this->col_edges.size());

                this->bit_to_check_msg.resize(this->edge_rows.size());
                this->check_to_bit_msg.resize(this->edge_rows.size());
            }

            void initialise_log_domain(
                    const std::vector<double> &channel_probabilities,
                    std::vector<double> &initial_log_prob_ratios) {
                for (int i = 0; i < this->bit_count; i++) {
                    const Real initial_log_prob_ratio = static_cast<Real>(std::log(
                            (1 - channel_probabilities[i]) / channel_probabilities[i]));
                    initial_log_prob_ratios[i] = static_cast<double>(initial_log_prob_ratio);

                    for (int edge_offset = this->col_offsets[i];
                         edge_offset < this->col_offsets[i + 1];
                         edge_offset++) {
                        const int edge = this->col_edges[edge_offset];
                        this->bit_to_check_msg[edge] = initial_log_prob_ratio;
                    }
                }
            }

            void update_check_to_bit_messages(
                    std::vector<uint8_t> &syndrome,
                    std::vector<uint8_t> &candidate_syndrome,
                    Real alpha) {
                for (int i = 0; i < this->check_count; i++) {
                    candidate_syndrome[i] = 0;
                    int total_sgn = syndrome[i];
                    int sgn = 0;
                    Real temp = std::numeric_limits<Real>::max();

                    for (int edge = this->row_offsets[i];
                         edge < this->row_offsets[i + 1];
                         edge++) {
                        if (this->bit_to_check_msg[edge] <= 0) {
                            total_sgn += 1;
                        }
                        this->check_to_bit_msg[edge] = temp;
                        Real abs_bit_to_check_msg = std::abs(this->bit_to_check_msg[edge]);
                        if (abs_bit_to_check_msg < temp) {
                            temp = abs_bit_to_check_msg;
                        }
                    }

                    temp = std::numeric_limits<Real>::max();
                    for (int edge_offset = this->row_offsets[i + 1];
                         edge_offset > this->row_offsets[i];) {
                        edge_offset--;
                        sgn = total_sgn;
                        if (this->bit_to_check_msg[edge_offset] <= 0) {
                            sgn += 1;
                        }
                        if (temp < this->check_to_bit_msg[edge_offset]) {
                            this->check_to_bit_msg[edge_offset] = temp;
                        }

                        Real message_sign = (sgn % 2 == 0) ? static_cast<Real>(1.0) : static_cast<Real>(-1.0);
                        this->check_to_bit_msg[edge_offset] *= message_sign * alpha;

                        Real abs_bit_to_check_msg = std::abs(this->bit_to_check_msg[edge_offset]);
                        if (abs_bit_to_check_msg < temp) {
                            temp = abs_bit_to_check_msg;
                        }
                    }
                }
            }

            void update_log_prob_ratios_and_decoding(
                    std::vector<uint8_t> &decoding,
                    std::vector<uint8_t> &candidate_syndrome,
                    std::vector<double> &log_prob_ratios,
                    const std::vector<double> &initial_log_prob_ratios) {
                for (int i = 0; i < this->bit_count; i++) {
                    Real temp = static_cast<Real>(initial_log_prob_ratios[i]);
                    for (int edge_offset = this->col_offsets[i];
                         edge_offset < this->col_offsets[i + 1];
                         edge_offset++) {
                        const int edge = this->col_edges[edge_offset];
                        this->bit_to_check_msg[edge] = temp;
                        temp += this->check_to_bit_msg[edge];
                    }

                    log_prob_ratios[i] = static_cast<double>(temp);
                    if (temp <= 0) {
                        decoding[i] = 1;
                        for (int edge_offset = this->col_offsets[i];
                             edge_offset < this->col_offsets[i + 1];
                             edge_offset++) {
                            const int edge = this->col_edges[edge_offset];
                            candidate_syndrome[this->edge_rows[edge]] ^= 1;
                        }
                    } else {
                        decoding[i] = 0;
                    }
                }
            }

            void update_bit_to_check_messages() {
                for (int i = 0; i < this->bit_count; i++) {
                    Real temp = 0;
                    for (int edge_offset = this->col_offsets[i + 1];
                         edge_offset > this->col_offsets[i];) {
                        edge_offset--;
                        const int edge = this->col_edges[edge_offset];
                        this->bit_to_check_msg[edge] += temp;
                        temp += this->check_to_bit_msg[edge];
                    }
                }
            }

            void sync_messages_to_pcm() {
                for (int edge = 0; edge < static_cast<int>(this->edges.size()); edge++) {
                    this->edges[edge]->bit_to_check_msg = static_cast<double>(this->bit_to_check_msg[edge]);
                    this->edges[edge]->check_to_bit_msg = static_cast<double>(this->check_to_bit_msg[edge]);
                }
            }
        };

        using PackedBpDecoder = PackedBpDecoderImpl<double>;
        using FloatPackedBpDecoder = PackedBpDecoderImpl<float>;

    }
}
