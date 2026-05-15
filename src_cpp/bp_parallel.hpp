#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "bp_types.hpp"

namespace ldpc {
    namespace bp {

        class LinkedListParallelBpDecoder {
        public:
            template<class Decoder>
            static std::vector<uint8_t> &decode_product_sum(
                    Decoder &decoder,
                    std::vector<uint8_t> &syndrome) {
                decoder.converge = 0;
                initialise_log_domain(decoder);

                for (int it = 1; it <= decoder.maximum_iterations; it++) {
                    for (int i = 0; i < decoder.check_count; i++) {
                        decoder.candidate_syndrome[i] = 0;

                        double temp = 1.0;
                        for (auto &e: decoder.pcm.iterate_row(i)) {
                            e.check_to_bit_msg = temp;
                            temp *= std::tanh(e.bit_to_check_msg / 2);
                        }

                        temp = 1;
                        for (auto &e: decoder.pcm.reverse_iterate_row(i)) {
                            e.check_to_bit_msg *= temp;
                            int message_sign = syndrome[i] != 0u ? -1.0 : 1.0;
                            e.check_to_bit_msg =
                                    message_sign * std::log((1 + e.check_to_bit_msg) / (1 - e.check_to_bit_msg));
                            temp *= std::tanh(e.bit_to_check_msg / 2);
                        }
                    }

                    update_log_prob_ratios_and_decoding(decoder, syndrome);
                    decoder.iterations = it;

                    if (decoder.converge) {
                        return decoder.decoding;
                    }

                    update_bit_to_check_messages(decoder);
                }

                return decoder.decoding;
            }

            template<class Decoder>
            static std::vector<uint8_t> &decode_minimum_sum(
                    Decoder &decoder,
                    std::vector<uint8_t> &syndrome) {
                decoder.converge = 0;
                initialise_log_domain(decoder);

                for (int it = 1; it <= decoder.maximum_iterations; it++) {
                    double alpha;
                    if(decoder.ms_scaling_factor == 0.0) {
                        alpha = 1.0 - std::pow(2.0, -1.0*it);
                    }
                    else {
                        alpha = decoder.ms_scaling_factor;
                    }

                    for (int i = 0; i < decoder.check_count; i++) {
                        decoder.candidate_syndrome[i] = 0;
                        int total_sgn = 0;
                        int sgn = 0;
                        total_sgn = syndrome[i];
                        double temp = std::numeric_limits<double>::max();

                        for (auto &e: decoder.pcm.iterate_row(i)) {
                            if (e.bit_to_check_msg <= 0) {
                                total_sgn += 1;
                            }
                            e.check_to_bit_msg = temp;
                            double abs_bit_to_check_msg = std::abs(e.bit_to_check_msg);
                            if (abs_bit_to_check_msg < temp) {
                                temp = abs_bit_to_check_msg;
                            }
                        }

                        temp = std::numeric_limits<double>::max();
                        for (auto &e: decoder.pcm.reverse_iterate_row(i)) {
                            sgn = total_sgn;
                            if (e.bit_to_check_msg <= 0) {
                                sgn += 1;
                            }
                            if (temp < e.check_to_bit_msg) {
                                e.check_to_bit_msg = temp;
                            }

                            int message_sign = (sgn % 2 == 0) ? 1.0 : -1.0;

                            e.check_to_bit_msg *= message_sign * alpha;

                            double abs_bit_to_check_msg = std::abs(e.bit_to_check_msg);
                            if (abs_bit_to_check_msg < temp) {
                                temp = abs_bit_to_check_msg;
                            }
                        }
                    }

                    update_log_prob_ratios_and_decoding(decoder, syndrome);
                    decoder.iterations = it;

                    if (decoder.converge) {
                        return decoder.decoding;
                    }

                    update_bit_to_check_messages(decoder);
                }

                return decoder.decoding;
            }

        private:
            template<class Decoder>
            static void initialise_log_domain(Decoder &decoder) {
                for (int i = 0; i < decoder.bit_count; i++) {
                    decoder.initial_log_prob_ratios[i] = std::log(
                            (1 - decoder.channel_probabilities[i]) / decoder.channel_probabilities[i]);

                    for (auto &e: decoder.pcm.iterate_column(i)) {
                        e.bit_to_check_msg = decoder.initial_log_prob_ratios[i];
                    }
                }
            }

            template<class Decoder>
            static void update_log_prob_ratios_and_decoding(
                    Decoder &decoder,
                    std::vector<uint8_t> &syndrome) {
                for (int i = 0; i < decoder.bit_count; i++) {
                    double temp = decoder.initial_log_prob_ratios[i];
                    for (auto &e: decoder.pcm.iterate_column(i)) {
                        e.bit_to_check_msg = temp;
                        temp += e.check_to_bit_msg;
                    }

                    decoder.log_prob_ratios[i] = temp;
                    if (temp <= 0) {
                        decoder.decoding[i] = 1;
                        for (auto &e: decoder.pcm.iterate_column(i)) {
                            decoder.candidate_syndrome[e.row_index] ^= 1;
                        }
                    } else {
                        decoder.decoding[i] = 0;
                    }
                }

                if (std::equal(decoder.candidate_syndrome.begin(), decoder.candidate_syndrome.end(), syndrome.begin())) {
                    decoder.converge = true;
                }
            }

            template<class Decoder>
            static void update_bit_to_check_messages(Decoder &decoder) {
                for (int i = 0; i < decoder.bit_count; i++) {
                    double temp = 0;
                    for (auto &e: decoder.pcm.reverse_iterate_column(i)) {
                        e.bit_to_check_msg += temp;
                        temp += e.check_to_bit_msg;
                    }
                }
            }
        };

    }
}
