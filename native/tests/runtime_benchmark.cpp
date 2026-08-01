#include "skac_runtime.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#define SKAC_JOIN_DETAIL(left, right) left##right
#define SKAC_JOIN(left, right) SKAC_JOIN_DETAIL(left, right)
namespace SKAC_STANDARD = SKAC_JOIN(st, d);
#undef SKAC_JOIN
#undef SKAC_JOIN_DETAIL

namespace {

SKAC_STANDARD::vector<uint8_t> read_bytes(const char* path) {
    SKAC_STANDARD::ifstream stream(path, SKAC_STANDARD::ios::binary);
    if (!stream) return {};
    return SKAC_STANDARD::vector<uint8_t>(
        SKAC_STANDARD::istreambuf_iterator<char>(stream),
        SKAC_STANDARD::istreambuf_iterator<char>()
    );
}

double percentile(SKAC_STANDARD::vector<double> values, double fraction) {
    SKAC_STANDARD::sort(values.begin(), values.end());
    const double position = fraction * static_cast<double>(values.size() - 1);
    const size_t left = static_cast<size_t>(position);
    const size_t right = SKAC_STANDARD::min(left + 1, values.size() - 1);
    const double amount = position - static_cast<double>(left);
    return values[left] + amount * (values[right] - values[left]);
}

void require(skac_result result) {
    if (result != SKAC_OK) {
        SKAC_STANDARD::cerr << skac_runtime_last_error() << '\n';
        SKAC_STANDARD::exit(1);
    }
}

void print_record(
    const char* mode,
    uint32_t instances,
    const SKAC_STANDARD::vector<double>& tick_ms
) {
    SKAC_STANDARD::vector<double> per_sample_ms = tick_ms;
    for (double& value : per_sample_ms) value /= instances;
    SKAC_STANDARD::cout << mode << ' ' << instances << ' '
                       << percentile(per_sample_ms, 0.50) << ' '
                       << percentile(per_sample_ms, 0.95) << ' '
                       << percentile(per_sample_ms, 0.99) << ' '
                       << percentile(tick_ms, 0.95) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        SKAC_STANDARD::cerr << "usage: skac_runtime_benchmark metadata payload profile target iterations\n";
        return 2;
    }
    const SKAC_STANDARD::vector<uint8_t> metadata = read_bytes(argv[1]);
    const SKAC_STANDARD::vector<uint8_t> payload = read_bytes(argv[2]);
    const SKAC_STANDARD::vector<uint8_t> profile = read_bytes(argv[3]);
    const SKAC_STANDARD::vector<uint8_t> target = read_bytes(argv[4]);
    const uint32_t iterations = static_cast<uint32_t>(SKAC_STANDARD::strtoul(argv[5], nullptr, 10));
    if (metadata.empty() || payload.empty() || profile.empty() || target.empty() || iterations < 100) {
        SKAC_STANDARD::cerr << "benchmark inputs are incomplete\n";
        return 2;
    }

    skac_decoder* decoder = nullptr;
    require(skac_decoder_open_raw(
        reinterpret_cast<const char*>(metadata.data()),
        metadata.size(),
        payload.data(),
        payload.size(),
        &decoder
    ));
    skac_clip_info clip{};
    require(skac_decoder_get_info(decoder, &clip));
    SKAC_STANDARD::vector<skac_transform> source_pose(clip.joint_count);
    SKAC_STANDARD::vector<double> same_ticks;
    same_ticks.reserve(iterations);
    for (uint32_t warmup = 0; warmup < 200; ++warmup) {
        require(skac_decoder_sample_time(
            decoder,
            (warmup + 0.37) * clip.frame_time_seconds,
            SKAC_TIME_LOOP,
            source_pose.data(),
            source_pose.size()
        ));
    }
    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
        const auto begin = SKAC_STANDARD::chrono::steady_clock::now();
        require(skac_decoder_sample_time(
            decoder,
            (iteration + 0.37) * clip.frame_time_seconds,
            SKAC_TIME_LOOP,
            source_pose.data(),
            source_pose.size()
        ));
        const auto end = SKAC_STANDARD::chrono::steady_clock::now();
        same_ticks.push_back(
            SKAC_STANDARD::chrono::duration<double, SKAC_STANDARD::milli>(end - begin).count()
        );
    }
    SKAC_STANDARD::cout << SKAC_STANDARD::setprecision(9);
    print_record("same", 1, same_ticks);

    for (uint32_t instance_count : {1u, 10u, 50u, 100u}) {
        SKAC_STANDARD::vector<skac_retargeter*> instances(instance_count, nullptr);
        skac_retarget_info info{};
        for (skac_retargeter*& instance : instances) {
            require(skac_retargeter_create(
                decoder,
                reinterpret_cast<const char*>(profile.data()),
                profile.size(),
                reinterpret_cast<const char*>(target.data()),
                target.size(),
                &instance
            ));
            require(skac_retargeter_get_info(instance, &info));
        }
        SKAC_STANDARD::vector<skac_transform> target_pose(info.target_joint_count);
        for (uint32_t warmup = 0; warmup < 100; ++warmup) {
            for (uint32_t instance = 0; instance < instance_count; ++instance) {
                require(skac_retargeter_sample_time(
                    instances[instance],
                    (warmup + instance * 0.13) * clip.frame_time_seconds,
                    SKAC_TIME_LOOP,
                    target_pose.data(),
                    target_pose.size()
                ));
            }
        }
        SKAC_STANDARD::vector<double> ticks;
        ticks.reserve(iterations);
        for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
            const auto begin = SKAC_STANDARD::chrono::steady_clock::now();
            for (uint32_t instance = 0; instance < instance_count; ++instance) {
                require(skac_retargeter_sample_time(
                    instances[instance],
                    (iteration + instance * 0.13) * clip.frame_time_seconds,
                    SKAC_TIME_LOOP,
                    target_pose.data(),
                    target_pose.size()
                ));
            }
            const auto end = SKAC_STANDARD::chrono::steady_clock::now();
            ticks.push_back(
                SKAC_STANDARD::chrono::duration<double, SKAC_STANDARD::milli>(end - begin).count()
            );
        }
        print_record("different", instance_count, ticks);
        for (skac_retargeter* instance : instances) skac_retargeter_close(instance);
    }
    SKAC_STANDARD::cout << "clip " << clip.frame_count << ' ' << clip.joint_count << '\n';
    skac_decoder_close(decoder);
    return 0;
}
