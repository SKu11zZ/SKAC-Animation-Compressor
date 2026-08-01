#include "skac_runtime.h"

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
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
        SKAC_STANDARD::istreambuf_iterator<char>(stream), SKAC_STANDARD::istreambuf_iterator<char>()
    );
}

int main(int argc, char** argv) {
    if (argc != 3 && argc != 5) {
        SKAC_STANDARD::cerr << "usage: skac_runtime_probe metadata.json raw_payload.bin [profile.json target.json]\n";
        return 2;
    }
    const SKAC_STANDARD::vector<uint8_t> metadata = read_bytes(argv[1]);
    const SKAC_STANDARD::vector<uint8_t> payload = read_bytes(argv[2]);
    if (metadata.empty()) {
        SKAC_STANDARD::cerr << "metadata input is empty\n";
        return 2;
    }
    skac_decoder* decoder = nullptr;
    const skac_result opened = skac_decoder_open_raw(
        reinterpret_cast<const char*>(metadata.data()),
        metadata.size(),
        payload.data(),
        payload.size(),
        &decoder
    );
    if (opened != SKAC_OK) {
        SKAC_STANDARD::cerr << skac_runtime_last_error() << '\n';
        return 1;
    }
    skac_clip_info info{};
    if (skac_decoder_get_info(decoder, &info) != SKAC_OK) {
        SKAC_STANDARD::cerr << skac_runtime_last_error() << '\n';
        skac_decoder_close(decoder);
        return 1;
    }
    if (argc == 5) {
        const SKAC_STANDARD::vector<uint8_t> profile = read_bytes(argv[3]);
        const SKAC_STANDARD::vector<uint8_t> target = read_bytes(argv[4]);
        skac_retargeter* retargeter = nullptr;
        const skac_result created = skac_retargeter_create(
            decoder,
            reinterpret_cast<const char*>(profile.data()),
            profile.size(),
            reinterpret_cast<const char*>(target.data()),
            target.size(),
            &retargeter
        );
        if (created != SKAC_OK) {
            SKAC_STANDARD::cerr << skac_runtime_last_error() << '\n';
            skac_decoder_close(decoder);
            return 1;
        }
        skac_retarget_info retarget_info{};
        if (skac_retargeter_get_info(retargeter, &retarget_info) != SKAC_OK) {
            SKAC_STANDARD::cerr << skac_runtime_last_error() << '\n';
            skac_retargeter_close(retargeter);
            skac_decoder_close(decoder);
            return 1;
        }
        SKAC_STANDARD::cout << SKAC_STANDARD::setprecision(17);
        SKAC_STANDARD::cout << "info " << info.frame_count << ' '
                           << retarget_info.target_joint_count << ' '
                           << info.frame_time_seconds << '\n';
        SKAC_STANDARD::cout << "retarget " << retarget_info.mapped_joint_count << ' '
                           << retarget_info.root_translation_scale << '\n';
        for (uint32_t joint = 0; joint < retarget_info.target_joint_count; ++joint) {
            const char* name = nullptr;
            int32_t parent = 0;
            if (skac_retargeter_get_joint_name(retargeter, joint, &name) != SKAC_OK ||
                skac_retargeter_get_joint_parent(retargeter, joint, &parent) != SKAC_OK) {
                SKAC_STANDARD::cerr << skac_runtime_last_error() << '\n';
                skac_retargeter_close(retargeter);
                skac_decoder_close(decoder);
                return 1;
            }
            SKAC_STANDARD::cout << "joint " << joint << ' ' << parent << ' ' << name << '\n';
        }
        SKAC_STANDARD::vector<skac_transform> pose(retarget_info.target_joint_count);
        for (uint32_t frame = 0; frame < info.frame_count; ++frame) {
            if (skac_retargeter_sample_frame(retargeter, frame, pose.data(), pose.size()) != SKAC_OK) {
                SKAC_STANDARD::cerr << skac_runtime_last_error() << '\n';
                skac_retargeter_close(retargeter);
                skac_decoder_close(decoder);
                return 1;
            }
            for (uint32_t joint = 0; joint < retarget_info.target_joint_count; ++joint) {
                const skac_transform& value = pose[joint];
                SKAC_STANDARD::cout << "pose " << frame << ' ' << joint << ' '
                                   << value.rotation_x << ' ' << value.rotation_y << ' '
                                   << value.rotation_z << ' ' << value.rotation_w << ' '
                                   << value.translation_x << ' ' << value.translation_y << ' '
                                   << value.translation_z << '\n';
            }
        }
        skac_retargeter_close(retargeter);
        skac_decoder_close(decoder);
        return 0;
    }
    SKAC_STANDARD::cout << SKAC_STANDARD::setprecision(17);
    SKAC_STANDARD::cout << "info " << info.frame_count << ' ' << info.joint_count << ' '
              << info.frame_time_seconds << '\n';
    for (uint32_t joint = 0; joint < info.joint_count; ++joint) {
        const char* name = nullptr;
        int32_t parent = 0;
        if (skac_decoder_get_joint_name(decoder, joint, &name) != SKAC_OK ||
            skac_decoder_get_joint_parent(decoder, joint, &parent) != SKAC_OK) {
            SKAC_STANDARD::cerr << skac_runtime_last_error() << '\n';
            skac_decoder_close(decoder);
            return 1;
        }
        SKAC_STANDARD::cout << "joint " << joint << ' ' << parent << ' ' << name << '\n';
    }
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
