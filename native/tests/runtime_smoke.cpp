#include "skac_runtime.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#define SKAC_JOIN_DETAIL(left, right) left##right
#define SKAC_JOIN(left, right) SKAC_JOIN_DETAIL(left, right)
namespace SKAC_STANDARD = SKAC_JOIN(st, d);
#undef SKAC_JOIN
#undef SKAC_JOIN_DETAIL

namespace {

void append_u32(SKAC_STANDARD::vector<uint8_t>& output, uint32_t value) {
    output.push_back(static_cast<uint8_t>(value));
    output.push_back(static_cast<uint8_t>(value >> 8));
    output.push_back(static_cast<uint8_t>(value >> 16));
    output.push_back(static_cast<uint8_t>(value >> 24));
}

SKAC_STANDARD::vector<uint8_t> pack_values(const SKAC_STANDARD::vector<SKAC_STANDARD::pair<uint32_t, uint32_t>>& fields) {
    SKAC_STANDARD::vector<uint8_t> result;
    uint64_t accumulator = 0;
    uint32_t count = 0;
    for (const auto& field : fields) {
        accumulator |= static_cast<uint64_t>(field.first) << count;
        count += field.second;
        while (count >= 8) {
            result.push_back(static_cast<uint8_t>(accumulator));
            accumulator >>= 8;
            count -= 8;
        }
    }
    if (count != 0) result.push_back(static_cast<uint8_t>(accumulator));
    return result;
}

}  // namespace

int main() {
    const SKAC_STANDARD::string metadata = R"json({
        "schema":"skac.animation",
        "schema_version":"1.0.0",
        "frame_count":1,
        "frame_time":0.03333333333333333,
        "skeleton":{
            "names":["Root"],
            "parents":[-1],
            "offsets":[[0,0,0]],
            "channels":[["Xposition","Yposition","Zposition","Zrotation","Xrotation","Yrotation"]],
            "end_site_offsets":[[0,1,0]],
            "has_end_sites":[true]
        },
        "codec":{
            "name":"key_reduced_smallest_three_v1",
            "rotation_bits":8,
            "translation_bits":8,
            "rotation_joints":[0],
            "rotation_key_counts":[1],
            "translation_components":[[0,0],[0,1],[0,2]],
            "translation_key_counts":[1,1,1],
            "translation_minima":[0,0,0],
            "translation_maxima":[0,0,0],
            "rotation_payload_bytes":9,
            "translation_payload_bytes":18
        }
    })json";

    SKAC_STANDARD::vector<uint8_t> raw;
    append_u32(raw, 1);
    raw.push_back(0);
    const auto quaternion = pack_values({{0, 2}, {128, 8}, {128, 8}, {128, 8}});
    raw.insert(raw.end(), quaternion.begin(), quaternion.end());
    for (int axis = 0; axis < 3; ++axis) {
        append_u32(raw, 1);
        raw.push_back(0);
        raw.push_back(0);
    }

    skac_decoder* decoder = nullptr;
    if (skac_decoder_open_raw(
            metadata.data(), metadata.size(), raw.data(), raw.size(), &decoder
        ) != SKAC_OK) {
        SKAC_STANDARD::cerr << skac_runtime_last_error() << '\n';
        return 1;
    }
    skac_clip_info info{};
    if (skac_decoder_get_info(decoder, &info) != SKAC_OK ||
        info.abi_version != SKAC_RUNTIME_ABI_VERSION || info.frame_count != 1 ||
        info.joint_count != 1) {
        skac_decoder_close(decoder);
        return 2;
    }
    skac_transform pose{};
    if (skac_decoder_sample_frame(decoder, 0, &pose, 1) != SKAC_OK ||
        SKAC_STANDARD::abs(pose.rotation_w) < 0.99f || SKAC_STANDARD::abs(pose.translation_x) > 1e-6f) {
        skac_decoder_close(decoder);
        return 3;
    }
    if (skac_decoder_sample_time(decoder, 10.0, SKAC_TIME_LOOP, &pose, 1) != SKAC_OK) {
        skac_decoder_close(decoder);
        return 4;
    }
    skac_decoder_close(decoder);
    return 0;
}
