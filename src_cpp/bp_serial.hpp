#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <set>
#include <vector>

#include "bp_types.hpp"

namespace ldpc {
    namespace bp {

        class LinkedListSerialBpDecoder {
        public:
            template<class Decoder>
            static std::vector<uint8_t> &decode_product_sum(Decoder &decoder, std::vector<uint8_t> &syndrome) {
                int check_index = 0;
                decoder.converge = false;
                initialise_log_domain(decoder);

                for (int it = 1; it <= decoder.maximum_iterations; it++) {
                    update_schedule(decoder, it);

                    for (int bit_index: decoder.serial_schedule_order) {
                        double temp = NAN;
                        decoder.log_prob_ratios[bit_index] = std::log(
                                (1 - decoder.channel_probabilities[bit_index]) /
                                decoder.channel_probabilities[bit_index]);
                        for (auto &e: decoder.pcm.iterate_column(bit_index)) {
                            check_index = e.row_index;
                            e.check_to_bit_msg = 1.0;
                            for (auto &g: decoder.pcm.iterate_row(check_index)) {
                                if (&g != &e) {
                                    e.check_to_bit_msg *= tanh(g.bit_to_check_msg / 2);
                                }
                            }
                            e.check_to_bit_msg = pow(-1, syndrome[check_index]) *
                                                 std::log((1 + e.check_to_bit_msg) / (1 - e.check_to_bit_msg));
                            e.bit_to_check_msg = decoder.log_prob_ratios[bit_index];
                            decoder.log_prob_ratios[bit_index] += e.check_to_bit_msg;
                        }
                        if (decoder.log_prob_ratios[bit_index] <= 0) {
                            decoder.decoding[bit_index] = 1;
                        } else {
                            decoder.decoding[bit_index] = 0;
                        }
                        temp = 0;
                        for (auto &e: decoder.pcm.reverse_iterate_column(bit_index)) {
                            e.bit_to_check_msg += temp;
                            temp += e.check_to_bit_msg;
                        }
                    }

                    decoder.candidate_syndrome = decoder.pcm.mulvec(decoder.decoding, decoder.candidate_syndrome);
                    decoder.iterations = it;
                    if (std::equal(decoder.candidate_syndrome.begin(), decoder.candidate_syndrome.end(), syndrome.begin())) {
                        decoder.converge = true;
                        return decoder.decoding;
                    }
                }
                return decoder.decoding;
            }

            template<class Decoder>
            static std::vector<uint8_t> &decode_minimum_sum(Decoder &decoder, std::vector<uint8_t> &syndrome) {
                int check_index = 0;
                decoder.converge = false;
                initialise_log_domain(decoder);

                for (int it = 1; it <= decoder.maximum_iterations; it++) {
                    double alpha;
                    if(decoder.ms_scaling_factor == 0.0) {
                        alpha = 1.0 - std::pow(2.0, -1.0*it);
                    }
                    else {
                        alpha = decoder.ms_scaling_factor;
                    }

                    update_schedule(decoder, it);

                    for (int bit_index: decoder.serial_schedule_order) {
                        double temp = NAN;
                        decoder.log_prob_ratios[bit_index] = std::log(
                                (1 - decoder.channel_probabilities[bit_index]) /
                                decoder.channel_probabilities[bit_index]);
                        for (auto &e: decoder.pcm.iterate_column(bit_index)) {
                            check_index = e.row_index;
                            int sgn = syndrome[check_index];
                            temp = std::numeric_limits<double>::max();
                            for (auto &g: decoder.pcm.iterate_row(check_index)) {
                                if (&g != &e) {
                                    double abs_bit_to_check_msg = std::abs(g.bit_to_check_msg);
                                    temp = std::min(abs_bit_to_check_msg, temp);
                                    if (g.bit_to_check_msg <= 0) {
                                        sgn += 1;
                                    }
                                }
                            }
                            double message_sign = (sgn % 2 == 0) ? 1.0 : -1.0;
                            e.check_to_bit_msg = alpha * message_sign * temp;
                            e.bit_to_check_msg = decoder.log_prob_ratios[bit_index];
                            decoder.log_prob_ratios[bit_index] += e.check_to_bit_msg;
                        }
                        if (decoder.log_prob_ratios[bit_index] <= 0) {
                            decoder.decoding[bit_index] = 1;
                        } else {
                            decoder.decoding[bit_index] = 0;
                        }
                        temp = 0;
                        for (auto &e: decoder.pcm.reverse_iterate_column(bit_index)) {
                            e.bit_to_check_msg += temp;
                            temp += e.check_to_bit_msg;
                        }
                    }

                    decoder.candidate_syndrome = decoder.pcm.mulvec(decoder.decoding, decoder.candidate_syndrome);
                    decoder.iterations = it;
                    if (std::equal(decoder.candidate_syndrome.begin(), decoder.candidate_syndrome.end(), syndrome.begin())) {
                        decoder.converge = true;
                        return decoder.decoding;
                    }
                }
                return decoder.decoding;
            }

            template<class Decoder>
            static std::vector<uint8_t> &decode_single_scan(Decoder &decoder, std::vector<uint8_t> &syndrome) {
                decoder.converge = 0;
                int CONVERGED = 0;

                std::vector<double> log_prob_ratios_old;
                log_prob_ratios_old.resize(decoder.bit_count);

                for (int i = 0; i < decoder.bit_count; i++) {
                    decoder.initial_log_prob_ratios[i] = std::log(
                            (1 - decoder.channel_probabilities[i]) / decoder.channel_probabilities[i]);
                    decoder.log_prob_ratios[i] = decoder.initial_log_prob_ratios[i];
                }

                for (int it = 1; it <= decoder.maximum_iterations; it++) {
                    if (CONVERGED != 0) {
                        continue;
                    }

                    log_prob_ratios_old = decoder.log_prob_ratios;

                    if (it != 1) {
                        decoder.log_prob_ratios = decoder.initial_log_prob_ratios;
                    }

                    for (int i = 0; i < decoder.check_count; i++) {
                        decoder.candidate_syndrome[i] = 0;

                        int total_sgn = 0;
                        int sgn = 0;
                        total_sgn = syndrome[i];
                        double temp = std::numeric_limits<double>::max();

                        double bit_to_check_msg = NAN;

                        for (auto &e: decoder.pcm.iterate_row(i)) {
                            if (it == 1) {
                                e.check_to_bit_msg = 0;
                            }
                            bit_to_check_msg = log_prob_ratios_old[e.col_index] - e.check_to_bit_msg;
                            if (bit_to_check_msg <= 0) {
                                total_sgn += 1;
                            }
                            e.bit_to_check_msg = temp;
                            double abs_bit_to_check_msg = std::abs(bit_to_check_msg);
                            if (abs_bit_to_check_msg < temp) {
                                temp = abs_bit_to_check_msg;
                            }
                        }

                        temp = std::numeric_limits<double>::max();
                        for (auto &e: decoder.pcm.reverse_iterate_row(i)) {
                            sgn = total_sgn;
                            if (it == 1) {
                                e.check_to_bit_msg = 0;
                            }
                            bit_to_check_msg = log_prob_ratios_old[e.col_index] - e.check_to_bit_msg;
                            if (bit_to_check_msg <= 0) {
                                sgn += 1;
                            }
                            if (temp < e.bit_to_check_msg) {
                                e.bit_to_check_msg = temp;
                            }

                            int message_sign = (sgn % 2 == 0) ? 1.0 : -1.0;
                            e.check_to_bit_msg = message_sign * decoder.ms_scaling_factor * e.bit_to_check_msg;
                            decoder.log_prob_ratios[e.col_index] += e.check_to_bit_msg;

                            double abs_bit_to_check_msg = std::abs(bit_to_check_msg);
                            if (abs_bit_to_check_msg < temp) {
                                temp = abs_bit_to_check_msg;
                            }
                        }
                    }

                    for (int i = 0; i < decoder.bit_count; i++) {
                        if (decoder.log_prob_ratios[i] <= 0) {
                            decoder.decoding[i] = 1;
                            for (auto &e: decoder.pcm.iterate_column(i)) {
                                decoder.candidate_syndrome[e.row_index] ^= 1;
                            }
                        } else {
                            decoder.decoding[i] = 0;
                        }
                    }

                    CONVERGED = 0;

                    if (std::equal(decoder.candidate_syndrome.begin(), decoder.candidate_syndrome.end(), syndrome.begin())) {
                        CONVERGED = 1;
                    }

                    decoder.iterations = it;

                    if (CONVERGED != 0) {
                        decoder.converge = (CONVERGED != 0);
                        return decoder.decoding;
                    }
                }

                decoder.converge = (CONVERGED != 0);
                return decoder.decoding;
            }

            template<class Decoder>
            static std::vector<uint8_t> &soft_info_decode(
                    Decoder &decoder,
                    std::vector<double> &soft_info_syndrome,
                    double cutoff,
                    double sigma) {
                std::vector<uint8_t> syndrome;
                decoder.soft_syndrome = soft_info_syndrome;
                for (int i = 0; i < decoder.check_count; i++) {
                    decoder.soft_syndrome[i] = 2 * decoder.soft_syndrome[i] / (sigma * sigma);
                    if (decoder.soft_syndrome[i] <= 0) {
                        syndrome.push_back(1);
                    } else {
                        syndrome.push_back(0);
                    }
                }

                int check_index = 0;
                decoder.converge = false;
                bool CONVERGED = false;
                bool loop_break = false;
                initialise_log_domain(decoder);
                std::set<int> check_indices_updated;

                for (int it = 1; it <= decoder.maximum_iterations; it++) {
                    if (CONVERGED) {
                        continue;
                    }
                    if (decoder.random_serial_schedule && decoder.omp_thread_count == 1) {
                        shuffle(decoder.serial_schedule_order.begin(), decoder.serial_schedule_order.end(),
                                std::default_random_engine(decoder.random_schedule_seed));
                    }

                    check_indices_updated.clear();
                    for (auto bit_index: decoder.serial_schedule_order) {
                        double temp = NAN;
                        decoder.log_prob_ratios[bit_index] = std::log(
                                (1 - decoder.channel_probabilities[bit_index]) /
                                decoder.channel_probabilities[bit_index]);
                        for (auto &check_nbr: decoder.pcm.iterate_column(bit_index)) {
                            check_index = check_nbr.row_index;
                            int sgn = 0;
                            temp = std::numeric_limits<double>::max();
                            for (auto &g: decoder.pcm.iterate_row(check_index)) {
                                if (&g != &check_nbr) {
                                    if (std::abs(g.bit_to_check_msg) < temp) {
                                        temp = std::abs(g.bit_to_check_msg);
                                    }
                                    if (g.bit_to_check_msg <= 0) {
                                        sgn ^= 1;
                                    }
                                }
                            }
                            double min_bit_to_check_msg = temp;
                            double propagated_msg = min_bit_to_check_msg;
                            double soft_syndrome_magnitude = std::abs(decoder.soft_syndrome[check_index]);

                            if (soft_syndrome_magnitude < cutoff) {
                                if (soft_syndrome_magnitude < std::abs(min_bit_to_check_msg)) {
                                    propagated_msg = soft_syndrome_magnitude;
                                    int check_node_sgn = sgn;
                                    if (check_nbr.bit_to_check_msg <= 0) {
                                        check_node_sgn ^= 1;
                                    }
                                    if (check_node_sgn == syndrome[check_index]) {
                                        if (std::abs(check_nbr.bit_to_check_msg) < min_bit_to_check_msg) {
                                            decoder.soft_syndrome[check_index] =
                                                    pow(-1, syndrome[check_index]) *
                                                    std::abs(check_nbr.bit_to_check_msg);
                                        } else {
                                            decoder.soft_syndrome[check_index] =
                                                    pow(-1, syndrome[check_index]) * min_bit_to_check_msg;
                                        }
                                    } else {
                                        syndrome[check_index] ^= 1;
                                        decoder.soft_syndrome[check_index] *= -1;
                                    }
                                }
                            }
                            sgn ^= syndrome[check_index];
                            check_nbr.check_to_bit_msg =
                                    decoder.ms_scaling_factor * pow(-1, sgn) * propagated_msg;
                            check_nbr.bit_to_check_msg = decoder.log_prob_ratios[bit_index];
                            decoder.log_prob_ratios[bit_index] += check_nbr.check_to_bit_msg;
                        }
                        if (decoder.log_prob_ratios[bit_index] <= 0) {
                            decoder.decoding[bit_index] = 1;
                        } else {
                            decoder.decoding[bit_index] = 0;
                        }
                        temp = 0;
                        for (auto &e: decoder.pcm.reverse_iterate_column(bit_index)) {
                            e.bit_to_check_msg += temp;
                            temp += e.check_to_bit_msg;
                        }
                    }
                    loop_break = false;
                    CONVERGED = true;
                    for (auto i = 0; i < soft_info_syndrome.size(); i++) {
                        if (soft_info_syndrome[i] <= 0) {
                            decoder.candidate_syndrome[i] = 1;
                        } else {
                            decoder.candidate_syndrome[i] = 0;
                        }
                    }
                    decoder.candidate_syndrome =
                            decoder.pcm.mulvec(decoder.decoding, decoder.candidate_syndrome);
                    for (auto i = 0; i < decoder.check_count && !loop_break; i++) {
                        if (decoder.candidate_syndrome[i] != syndrome[i]) {
                            CONVERGED = false;
                            loop_break = true;
                        }
                    }
                    decoder.iterations = it;
                    if (CONVERGED) {
                        decoder.converge = true;
                        return decoder.decoding;
                    }
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
            static void update_schedule(Decoder &decoder, int iteration) {
                if (decoder.random_serial_schedule) {
                    decoder.rng_list_shuffle.shuffle(decoder.serial_schedule_order);
                } else if (decoder.schedule == BpSchedule::SERIAL_RELATIVE) {
                    std::sort(decoder.serial_schedule_order.begin(), decoder.serial_schedule_order.end(),
                              [&decoder, iteration](int bit1, int bit2) {
                                  if (iteration != 1) {
                                      return decoder.log_prob_ratios[bit1] > decoder.log_prob_ratios[bit2];
                                  } else {
                                      return std::log(
                                              (1 - decoder.channel_probabilities[bit1]) /
                                              decoder.channel_probabilities[bit1]) >
                                             std::log((1 - decoder.channel_probabilities[bit2]) /
                                                      decoder.channel_probabilities[bit2]);
                                  }
                              });
                }
            }
        };

    }
}
