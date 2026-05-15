#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "bp_types.hpp"

namespace ldpc {
    namespace bp {

        class AvxPackedBpDecoder {
        public:
            explicit AvxPackedBpDecoder(BpSparse &pcm) : check_count(pcm.m), bit_count(pcm.n) {
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
                    float alpha;
                    if(ms_scaling_factor == 0.0) {
                        alpha = static_cast<float>(1.0 - std::pow(2.0, -1.0*it));
                    }
                    else {
                        alpha = static_cast<float>(ms_scaling_factor);
                    }

                    this->update_check_to_bit_messages_avx(syndrome, candidate_syndrome, alpha);
                    this->update_log_prob_ratios_and_decoding(
                            decoding,
                            candidate_syndrome,
                            log_prob_ratios);

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
            std::vector<BpEntry *> edges;
            std::vector<float> bit_to_check_msg;
            std::vector<float> check_to_bit_msg;
            std::vector<float> initial_log_prob_ratios_float;

            static int round_up_to_avx512_lanes(int value) {
                constexpr int lane_count = 16;
                return ((value + lane_count - 1) / lane_count) * lane_count;
            }

            void build(BpSparse &pcm) {
                this->row_offsets.assign(this->check_count + 1, 0);
                this->col_offsets.assign(this->bit_count + 1, 0);
                this->col_edges.clear();
                this->edge_rows.clear();
                this->edges.clear();

                std::unordered_map<BpEntry *, int> edge_ids;
                edge_ids.reserve(static_cast<size_t>(pcm.entry_count()));

                for (int row = 0; row < this->check_count; row++) {
                    this->row_offsets[row] = static_cast<int>(this->edge_rows.size());
                    int row_degree = 0;
                    for (auto &e: pcm.iterate_row(row)) {
                        const int edge = static_cast<int>(this->edge_rows.size());
                        this->edge_rows.push_back(row);
                        this->edges.push_back(&e);
                        edge_ids.emplace(&e, edge);
                        row_degree++;
                    }

                    const int padded_row_end =
                            this->row_offsets[row] + round_up_to_avx512_lanes(row_degree);
                    while (static_cast<int>(this->edge_rows.size()) < padded_row_end) {
                        this->edge_rows.push_back(row);
                        this->edges.push_back(nullptr);
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

                this->bit_to_check_msg.assign(
                        this->edge_rows.size(),
                        std::numeric_limits<float>::max());
                this->check_to_bit_msg.assign(this->edge_rows.size(), 0.0f);
                this->initial_log_prob_ratios_float.resize(this->bit_count);
            }

            void initialise_log_domain(
                    const std::vector<double> &channel_probabilities,
                    std::vector<double> &initial_log_prob_ratios) {
                for (int i = 0; i < this->bit_count; i++) {
                    const float initial_log_prob_ratio = static_cast<float>(std::log(
                            (1 - channel_probabilities[i]) / channel_probabilities[i]));
                    this->initial_log_prob_ratios_float[i] = initial_log_prob_ratio;
                    initial_log_prob_ratios[i] = static_cast<double>(initial_log_prob_ratio);

                    for (int edge_offset = this->col_offsets[i];
                         edge_offset < this->col_offsets[i + 1];
                         edge_offset++) {
                        const int edge = this->col_edges[edge_offset];
                        this->bit_to_check_msg[edge] = initial_log_prob_ratio;
                    }
                }
            }

            void update_check_to_bit_messages_avx(
                    std::vector<uint8_t> &syndrome,
                    std::vector<uint8_t> &candidate_syndrome,
                    float alpha) {
#if defined(__x86_64__) || defined(__i386__)
                __builtin_cpu_init();
                if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
                    __builtin_cpu_supports("avx512dq")) {
                    this->update_check_to_bit_messages_avx512(syndrome, candidate_syndrome, alpha);
                    return;
                }
                if (__builtin_cpu_supports("avx2")) {
                    this->update_check_to_bit_messages_avx2(syndrome, candidate_syndrome, alpha);
                    return;
                }
#endif
                throw std::runtime_error("PARALLEL_PACKED_AVX requires AVX2 or AVX-512 support");
            }

            __attribute__((target("avx512f,avx512bw,avx512dq")))
            void update_check_to_bit_messages_avx512(
                    std::vector<uint8_t> &syndrome,
                    std::vector<uint8_t> &candidate_syndrome,
                    float alpha) {
                const __m512 sign_mask = _mm512_set1_ps(-0.0f);
                const __m512 zero = _mm512_setzero_ps();
                const __m512 max_values = _mm512_set1_ps(std::numeric_limits<float>::max());
                const __m512 alpha_values = _mm512_set1_ps(alpha);

                for (int row = 0; row < this->check_count; row++) {
                    candidate_syndrome[row] = 0;
                    const int row_begin = this->row_offsets[row];
                    const int row_end = this->row_offsets[row + 1];
                    int total_sgn = syndrome[row];

                    __m512 min_values = max_values;
                    int edge = row_begin;
                    for (; edge + 16 <= row_end; edge += 16) {
                        const __m512 messages = _mm512_loadu_ps(&this->bit_to_check_msg[edge]);
                        const __m512 abs_messages = _mm512_andnot_ps(sign_mask, messages);
                        min_values = _mm512_min_ps(min_values, abs_messages);
                        const __mmask16 negative_or_zero = _mm512_cmp_ps_mask(messages, zero, _CMP_LE_OQ);
                        total_sgn += __builtin_popcount(static_cast<unsigned>(negative_or_zero));
                    }

                    alignas(64) float lanes[16];
                    _mm512_store_ps(lanes, min_values);
                    float min1 = std::numeric_limits<float>::max();
                    for (float lane_min: lanes) {
                        min1 = std::min(min1, lane_min);
                    }

                    const __m512 min1_values = _mm512_set1_ps(min1);
                    __m512 min2_values = max_values;
                    int min1_count = 0;

                    edge = row_begin;
                    for (; edge + 16 <= row_end; edge += 16) {
                        const __m512 messages = _mm512_loadu_ps(&this->bit_to_check_msg[edge]);
                        const __m512 abs_messages = _mm512_andnot_ps(sign_mask, messages);
                        const __mmask16 min1_mask = _mm512_cmp_ps_mask(abs_messages, min1_values, _CMP_EQ_OQ);
                        min1_count += __builtin_popcount(static_cast<unsigned>(min1_mask));
                        const __m512 min2_candidates = _mm512_mask_mov_ps(abs_messages, min1_mask, max_values);
                        min2_values = _mm512_min_ps(min2_values, min2_candidates);
                    }

                    _mm512_store_ps(lanes, min2_values);
                    float min2 = std::numeric_limits<float>::max();
                    for (float lane_min: lanes) {
                        min2 = std::min(min2, lane_min);
                    }

                    const __m512 min2_values_broadcast = _mm512_set1_ps(min2);
                    const __m512 positive_sign = _mm512_set1_ps((total_sgn % 2 == 0) ? 1.0f : -1.0f);
                    const __m512 negative_sign = _mm512_set1_ps((total_sgn % 2 == 0) ? -1.0f : 1.0f);

                    edge = row_begin;
                    for (; edge + 16 <= row_end; edge += 16) {
                        const __m512 messages = _mm512_loadu_ps(&this->bit_to_check_msg[edge]);
                        const __m512 abs_messages = _mm512_andnot_ps(sign_mask, messages);
                        const __mmask16 unique_min_mask =
                                (min1_count == 1) ? _mm512_cmp_ps_mask(abs_messages, min1_values, _CMP_EQ_OQ) : 0;
                        const __mmask16 negative_or_zero = _mm512_cmp_ps_mask(messages, zero, _CMP_LE_OQ);
                        const __m512 min_except_edge =
                                _mm512_mask_mov_ps(min1_values, unique_min_mask, min2_values_broadcast);
                        const __m512 message_sign =
                                _mm512_mask_mov_ps(positive_sign, negative_or_zero, negative_sign);
                        const __m512 check_messages =
                                _mm512_mul_ps(_mm512_mul_ps(min_except_edge, message_sign), alpha_values);
                        _mm512_storeu_ps(&this->check_to_bit_msg[edge], check_messages);
                    }

                }
            }

            __attribute__((target("avx2")))
            void update_check_to_bit_messages_avx2(
                    std::vector<uint8_t> &syndrome,
                    std::vector<uint8_t> &candidate_syndrome,
                    float alpha) {
                const __m256 sign_mask = _mm256_set1_ps(-0.0f);
                const __m256 zero = _mm256_setzero_ps();
                const __m256 max_values = _mm256_set1_ps(std::numeric_limits<float>::max());
                const __m256 alpha_values = _mm256_set1_ps(alpha);

                for (int row = 0; row < this->check_count; row++) {
                    candidate_syndrome[row] = 0;
                    const int row_begin = this->row_offsets[row];
                    const int row_end = this->row_offsets[row + 1];
                    int total_sgn = syndrome[row];

                    __m256 min_values = max_values;
                    int edge = row_begin;
                    for (; edge + 8 <= row_end; edge += 8) {
                        const __m256 messages = _mm256_loadu_ps(&this->bit_to_check_msg[edge]);
                        const __m256 abs_messages = _mm256_andnot_ps(sign_mask, messages);
                        min_values = _mm256_min_ps(min_values, abs_messages);
                        const __m256 negative_or_zero = _mm256_cmp_ps(messages, zero, _CMP_LE_OQ);
                        total_sgn += __builtin_popcount(static_cast<unsigned>(_mm256_movemask_ps(negative_or_zero)));
                    }

                    alignas(32) float min_lanes[8];
                    _mm256_store_ps(min_lanes, min_values);
                    float min1 = std::numeric_limits<float>::max();
                    for (float lane_min: min_lanes) {
                        min1 = std::min(min1, lane_min);
                    }

                    const __m256 min1_values = _mm256_set1_ps(min1);
                    __m256 min2_values = max_values;
                    int min1_count = 0;

                    for (edge = row_begin; edge + 8 <= row_end; edge += 8) {
                        const __m256 messages = _mm256_loadu_ps(&this->bit_to_check_msg[edge]);
                        const __m256 abs_messages = _mm256_andnot_ps(sign_mask, messages);
                        const __m256 min1_mask = _mm256_cmp_ps(abs_messages, min1_values, _CMP_EQ_OQ);
                        min1_count += __builtin_popcount(static_cast<unsigned>(_mm256_movemask_ps(min1_mask)));
                        const __m256 min2_candidates = _mm256_blendv_ps(abs_messages, max_values, min1_mask);
                        min2_values = _mm256_min_ps(min2_values, min2_candidates);
                    }

                    _mm256_store_ps(min_lanes, min2_values);
                    float min2 = std::numeric_limits<float>::max();
                    for (float lane_min: min_lanes) {
                        min2 = std::min(min2, lane_min);
                    }

                    const __m256 min2_values_broadcast = _mm256_set1_ps(min2);
                    const __m256 positive_sign = _mm256_set1_ps((total_sgn % 2 == 0) ? 1.0f : -1.0f);
                    const __m256 negative_sign = _mm256_set1_ps((total_sgn % 2 == 0) ? -1.0f : 1.0f);

                    for (edge = row_begin; edge + 8 <= row_end; edge += 8) {
                        const __m256 messages = _mm256_loadu_ps(&this->bit_to_check_msg[edge]);
                        const __m256 abs_messages = _mm256_andnot_ps(sign_mask, messages);
                        const __m256 min1_mask = _mm256_cmp_ps(abs_messages, min1_values, _CMP_EQ_OQ);
                        const __m256 unique_min_mask =
                                (min1_count == 1) ? min1_mask : _mm256_setzero_ps();
                        const __m256 negative_or_zero = _mm256_cmp_ps(messages, zero, _CMP_LE_OQ);
                        const __m256 min_except_edge =
                                _mm256_blendv_ps(min1_values, min2_values_broadcast, unique_min_mask);
                        const __m256 message_sign =
                                _mm256_blendv_ps(positive_sign, negative_sign, negative_or_zero);
                        const __m256 check_messages =
                                _mm256_mul_ps(_mm256_mul_ps(min_except_edge, message_sign), alpha_values);
                        _mm256_storeu_ps(&this->check_to_bit_msg[edge], check_messages);
                    }
                }
            }

            void update_log_prob_ratios_and_decoding(
                    std::vector<uint8_t> &decoding,
                    std::vector<uint8_t> &candidate_syndrome,
                    std::vector<double> &log_prob_ratios) {
                for (int i = 0; i < this->bit_count; i++) {
                    float temp = this->initial_log_prob_ratios_float[i];
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
                    float temp = 0;
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
                    if (this->edges[edge] == nullptr) {
                        continue;
                    }
                    this->edges[edge]->bit_to_check_msg = static_cast<double>(this->bit_to_check_msg[edge]);
                    this->edges[edge]->check_to_bit_msg = static_cast<double>(this->check_to_bit_msg[edge]);
                }
            }
        };

    }
}
