#include "skac_runtime.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <vector>

#define SKAC_JOIN_DETAIL(left, right) left##right
#define SKAC_JOIN(left, right) SKAC_JOIN_DETAIL(left, right)
namespace SKAC_STANDARD = SKAC_JOIN(st, d);
#undef SKAC_JOIN
#undef SKAC_JOIN_DETAIL

SKAC_STANDARD::vector<uint8_t> read_bytes(const char* path) {
    SKAC_STANDARD::ifstream stream(path, SKAC_STANDARD::ios::binary);
    if (!stream) return {};
    return SKAC_STANDARD::vector<uint8_t>(
        SKAC_STANDARD::istreambuf_iterator<char>(stream),
        SKAC_STANDARD::istreambuf_iterator<char>()
    );
}

struct InflationSource {
    const SKAC_STANDARD::vector<uint8_t>* bytes = nullptr;
    size_t offset = 0;
};

skac_result copy_inflated_chunk(
    const uint8_t*,
    size_t,
    uint8_t* destination,
    size_t destination_size,
    void* user_data
) {
    auto* source = static_cast<InflationSource*>(user_data);
    if (source == nullptr || source->bytes == nullptr ||
        destination_size > source->bytes->size() - source->offset) {
        return SKAC_DECOMPRESSION_FAILED;
    }
    SKAC_STANDARD::memcpy(
        destination, source->bytes->data() + source->offset, destination_size
    );
    source->offset += destination_size;
    return SKAC_OK;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        SKAC_STANDARD::cerr << "usage: skac_runtime_v2_probe container.skac raw_chunks.bin\n";
        return 2;
    }
    const auto container = read_bytes(argv[1]);
    const auto raw_chunks = read_bytes(argv[2]);
    if (container.empty() || raw_chunks.empty()) {
        SKAC_STANDARD::cerr << "probe input is empty\n";
        return 2;
    }
    InflationSource source{&raw_chunks, 0};
    skac_decoder* decoder = nullptr;
    const skac_result opened = skac_decoder_open_container(
        container.data(), container.size(), copy_inflated_chunk, &source, &decoder
    );
    if (opened != SKAC_OK) {
        SKAC_STANDARD::cerr << skac_runtime_last_error() << '\n';
        return 1;
    }
    if (source.offset != raw_chunks.size()) {
        SKAC_STANDARD::cerr << "native decoder did not request every required chunk\n";
        skac_decoder_close(decoder);
        return 1;
    }
    skac_clip_info info{};
    if (skac_decoder_get_info(decoder, &info) != SKAC_OK) {
        SKAC_STANDARD::cerr << skac_runtime_last_error() << '\n';
        skac_decoder_close(decoder);
        return 1;
    }
    SKAC_STANDARD::cout << SKAC_STANDARD::setprecision(17);
    SKAC_STANDARD::cout << "info " << info.frame_count << ' ' << info.joint_count << ' '
                       << info.frame_time_seconds << '\n';
    SKAC_STANDARD::vector<skac_transform> pose(info.joint_count);
    for (uint32_t frame = 0; frame < info.frame_count; ++frame) {
        if (skac_decoder_sample_frame(decoder, frame, pose.data(), pose.size()) != SKAC_OK) {
            SKAC_STANDARD::cerr << skac_runtime_last_error() << '\n';
            skac_decoder_close(decoder);
            return 1;
        }
        for (uint32_t joint = 0; joint < info.joint_count; ++joint) {
            const skac_transform& value = pose[joint];
            SKAC_STANDARD::cout << "pose " << frame << ' ' << joint << ' '
                               << value.rotation_x << ' ' << value.rotation_y << ' '
                               << value.rotation_z << ' ' << value.rotation_w << ' '
                               << value.translation_x << ' ' << value.translation_y << ' '
                               << value.translation_z << '\n';
        }
    }
    skac_decoder_close(decoder);
    return 0;
}
