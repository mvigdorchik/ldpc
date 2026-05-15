#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "math.h"
#include "bp_types.hpp"
#include "rng.hpp"

#include "bp_parallel.hpp"
#include "bp_packed.hpp"
#include "bp_packed_avx.hpp"
#include "bp_serial.hpp"

namespace ldpc {
    namespace bp {

        class BpDecoder {
            // TODO properties should be private and only accessible via getters and setters
        public:
            BpSparse &pcm;
            std::vector<double> channel_probabilities;
            int check_count;
            int bit_count;
            int maximum_iterations;
            BpMethod bp_method;
            BpSchedule schedule;
            BpInputType bp_input_type;
            double ms_scaling_factor;
            std::vector<uint8_t> decoding;
            std::vector<uint8_t> candidate_syndrome;

            std::vector<double> log_prob_ratios;
            std::vector<double> initial_log_prob_ratios;
            std::vector<double> soft_syndrome;
            std::vector<int> serial_schedule_order;
            int iterations;
            int omp_thread_count;
            bool converge;
            int random_schedule_seed;
            bool random_serial_schedule;
            ldpc::rng::RandomListShuffle<int> rng_list_shuffle;
            PackedBpDecoder packed_decoder;
            FloatPackedBpDecoder float_packed_decoder;
            AvxPackedBpDecoder avx_packed_decoder;

            BpDecoder(
                    BpSparse &parity_check_matrix,
                    std::vector<double> channel_probabilities,
                    int maximum_iterations = 0,
                    BpMethod bp_method = PRODUCT_SUM,
                    BpSchedule schedule = PARALLEL,
                    double min_sum_scaling_factor = 0.625,
                    int omp_threads = 1,
                    const std::vector<int> &serial_schedule = NULL_INT_VECTOR,
                    int random_schedule_seed = 0,
                    bool random_serial_schedule = false,
                    BpInputType bp_input_type = AUTO) :
                    pcm(parity_check_matrix), channel_probabilities(std::move(channel_probabilities)),
                    check_count(pcm.m), bit_count(pcm.n), maximum_iterations(maximum_iterations), bp_method(bp_method),
                    schedule(schedule), bp_input_type(bp_input_type), ms_scaling_factor(min_sum_scaling_factor),
                    iterations(0), //the parity check matrix is passed in by reference
                    omp_thread_count(omp_threads), converge(false),
                    random_schedule_seed(random_schedule_seed), random_serial_schedule(random_serial_schedule),
                    packed_decoder(parity_check_matrix),
                    float_packed_decoder(parity_check_matrix),
                    avx_packed_decoder(parity_check_matrix)
            {

                this->initial_log_prob_ratios.resize(bit_count);
                this->log_prob_ratios.resize(bit_count);
                this->candidate_syndrome.resize(check_count);
                this->decoding.resize(bit_count);

                if (this->channel_probabilities.size() != this->bit_count) {
                    throw std::runtime_error(
                            "Channel probabilities vector must have length equal to the number of bits");
                }
                if (serial_schedule != NULL_INT_VECTOR) {
                    this->serial_schedule_order = serial_schedule;
                    if (this->random_serial_schedule) {
                        throw std::runtime_error("Random schedule cannot be used with a fixed serial schedule. Set `random_serial_schedule` input parameter to false.");
                    }
                } else if (this->random_serial_schedule) {
                    this->serial_schedule_order.resize(bit_count);
                    for (int i = 0; i < bit_count; i++) {
                        this->serial_schedule_order[i] = i;
                    }
                    this->rng_list_shuffle.seed(this->random_schedule_seed);
                }
                else {
                    this->serial_schedule_order.resize(bit_count);
                    for (int i = 0; i < bit_count; i++) {
                        this->serial_schedule_order[i] = i;
                    }
                }

                //Initialise OMP thread pool
                // this->omp_thread_count = omp_threads;
                // this->set_omp_thread_count(this->omp_thread_count);
            }

            ~BpDecoder() = default;

            void set_omp_thread_count(int count) {
                this->omp_thread_count = count;
                // omp_set_num_threads(this->omp_thread_count);
                // NotImplemented
            }

            void set_random_schedule_seed(int seed) {
                this->random_schedule_seed = seed;
                this->rng_list_shuffle.seed(seed);
            }

            void initialise_log_domain_bp() {
                for (int i = 0; i < this->bit_count; i++) {
                    this->initial_log_prob_ratios[i] = std::log(
                            (1 - this->channel_probabilities[i]) / this->channel_probabilities[i]);

                    for (auto &e: this->pcm.iterate_column(i)) {
                        e.bit_to_check_msg = this->initial_log_prob_ratios[i];
                    }
                }
            }

            std::vector<uint8_t> decode(std::vector<uint8_t> &input_vector) {
                if ((this->bp_input_type == AUTO && input_vector.size() == this->bit_count) ||
                    this->bp_input_type == RECEIVED_VECTOR) {
                    auto syndrome = pcm.mulvec(input_vector);
                    std::vector<uint8_t> rv_decoding = this->decode_syndrome(syndrome);

                    for (int i = 0; i < this->bit_count; i++) {
                        this->decoding[i] = rv_decoding[i] ^ input_vector[i];
                    }

                    return this->decoding;
                }

                return this->decode_syndrome(input_vector);
            }

            std::vector<uint8_t> &bp_decode_parallel_packed(std::vector<uint8_t> &syndrome) {
                if (this->bp_method != MINIMUM_SUM) {
                    throw std::runtime_error("PARALLEL_PACKED currently supports only MINIMUM_SUM");
                }

                return this->packed_decoder.decode_minimum_sum(
                        syndrome,
                        this->channel_probabilities,
                        this->maximum_iterations,
                        this->ms_scaling_factor,
                        this->decoding,
                        this->candidate_syndrome,
                        this->log_prob_ratios,
                        this->initial_log_prob_ratios,
                        this->iterations,
                        this->converge);
            }

            std::vector<uint8_t> &bp_decode_parallel_packed_float(std::vector<uint8_t> &syndrome) {
                if (this->bp_method != MINIMUM_SUM) {
                    throw std::runtime_error("PARALLEL_PACKED_FLOAT currently supports only MINIMUM_SUM");
                }

                return this->float_packed_decoder.decode_minimum_sum(
                        syndrome,
                        this->channel_probabilities,
                        this->maximum_iterations,
                        this->ms_scaling_factor,
                        this->decoding,
                        this->candidate_syndrome,
                        this->log_prob_ratios,
                        this->initial_log_prob_ratios,
                        this->iterations,
                        this->converge);
            }

            std::vector<uint8_t> &bp_decode_parallel_packed_avx(std::vector<uint8_t> &syndrome) {
                if (this->bp_method != MINIMUM_SUM) {
                    throw std::runtime_error("PARALLEL_PACKED_AVX currently supports only MINIMUM_SUM");
                }

                return this->avx_packed_decoder.decode_minimum_sum(
                        syndrome,
                        this->channel_probabilities,
                        this->maximum_iterations,
                        this->ms_scaling_factor,
                        this->decoding,
                        this->candidate_syndrome,
                        this->log_prob_ratios,
                        this->initial_log_prob_ratios,
                        this->iterations,
                        this->converge);
            }

            std::vector<uint8_t> &bp_decode_parallel(std::vector<uint8_t> &syndrome) {
                switch (this->bp_method) {
                    case PRODUCT_SUM:
                        return LinkedListParallelBpDecoder::decode_product_sum(*this, syndrome);
                    case MINIMUM_SUM:
                        return LinkedListParallelBpDecoder::decode_minimum_sum(*this, syndrome);
                    default:
                        throw std::runtime_error("Invalid BP method");
                }
            }

            std::vector<uint8_t> &bp_decode_single_scan(std::vector<uint8_t> &syndrome) {
                return LinkedListSerialBpDecoder::decode_single_scan(*this, syndrome);
            }

            std::vector<uint8_t> &bp_decode_serial(std::vector<uint8_t> &syndrome) {
                switch (this->bp_method) {
                    case PRODUCT_SUM:
                        return LinkedListSerialBpDecoder::decode_product_sum(*this, syndrome);
                    case MINIMUM_SUM:
                        return LinkedListSerialBpDecoder::decode_minimum_sum(*this, syndrome);
                    default:
                        throw std::runtime_error("Invalid BP method");
                }
            }

            std::vector<uint8_t> &
            soft_info_decode_serial(std::vector<double> &soft_info_syndrome, double cutoff, double sigma) {
                return LinkedListSerialBpDecoder::soft_info_decode(*this, soft_info_syndrome, cutoff, sigma);
            }

        private:
            std::vector<uint8_t> &decode_syndrome(std::vector<uint8_t> &syndrome) {
                switch (this->schedule) {
                    case PARALLEL:
                        return this->bp_decode_parallel(syndrome);
                    case PARALLEL_PACKED:
                        return this->bp_decode_parallel_packed(syndrome);
                    case PARALLEL_PACKED_FLOAT:
                        return this->bp_decode_parallel_packed_float(syndrome);
                    case PARALLEL_PACKED_AVX:
                        return this->bp_decode_parallel_packed_avx(syndrome);
                    case SERIAL:
                    case SERIAL_RELATIVE:
                        return this->bp_decode_serial(syndrome);
                    default:
                        throw std::runtime_error("Invalid BP schedule");
                }
            }
        };
    }
}  // namespace ldpc::bp
