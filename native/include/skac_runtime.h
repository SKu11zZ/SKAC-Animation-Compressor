#ifndef SKAC_RUNTIME_H
#define SKAC_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && !defined(SKAC_RUNTIME_STATIC)
#  if defined(SKAC_RUNTIME_BUILD)
#    define SKAC_RUNTIME_API __declspec(dllexport)
#  else
#    define SKAC_RUNTIME_API __declspec(dllimport)
#  endif
#else
#  define SKAC_RUNTIME_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SKAC_RUNTIME_ABI_VERSION 1u

typedef struct skac_decoder skac_decoder;
typedef struct skac_retargeter skac_retargeter;

typedef enum skac_result {
    SKAC_OK = 0,
    SKAC_INVALID_ARGUMENT = 1,
    SKAC_INVALID_FORMAT = 2,
    SKAC_UNSUPPORTED = 3,
    SKAC_BUFFER_TOO_SMALL = 4,
    SKAC_OUT_OF_MEMORY = 5,
    SKAC_DECOMPRESSION_FAILED = 6,
    SKAC_INTERNAL_ERROR = 7
} skac_result;

typedef enum skac_time_mode {
    SKAC_TIME_CLAMP = 0,
    SKAC_TIME_LOOP = 1
} skac_time_mode;

typedef struct skac_clip_info {
    uint32_t abi_version;
    uint32_t frame_count;
    uint32_t joint_count;
    double frame_time_seconds;
    double duration_seconds;
} skac_clip_info;

typedef struct skac_retarget_info {
    uint32_t abi_version;
    uint32_t target_joint_count;
    uint32_t mapped_joint_count;
    double root_translation_scale;
} skac_retarget_info;

/* Engine-ready layout: quaternion xyzw followed by translation xyz. */
typedef struct skac_transform {
    float rotation_x;
    float rotation_y;
    float rotation_z;
    float rotation_w;
    float translation_x;
    float translation_y;
    float translation_z;
} skac_transform;

/*
 * Host decompression callback used by engines that already provide zlib.
 * The callback must write exactly destination_size bytes.
 */
typedef skac_result (*skac_inflate_fn)(
    const uint8_t* compressed,
    size_t compressed_size,
    uint8_t* destination,
    size_t destination_size,
    void* user_data
);

/* Decode canonical metadata plus an already inflated format-1.0 payload. */
SKAC_RUNTIME_API skac_result skac_decoder_open_raw(
    const char* metadata_json,
    size_t metadata_size,
    const uint8_t* raw_payload,
    size_t raw_payload_size,
    skac_decoder** out_decoder
);

/* Parse a complete SKAC v1 or v2 container and delegate zlib inflation to the host. */
SKAC_RUNTIME_API skac_result skac_decoder_open_container(
    const uint8_t* container,
    size_t container_size,
    skac_inflate_fn inflate,
    void* user_data,
    skac_decoder** out_decoder
);

/*
 * Parse a complete SKAC v1 or v2 container with the optional built-in zlib path.
 * Returns SKAC_UNSUPPORTED when the library was built without zlib.
 */
SKAC_RUNTIME_API skac_result skac_decoder_open_memory(
    const uint8_t* container,
    size_t container_size,
    skac_decoder** out_decoder
);

SKAC_RUNTIME_API void skac_decoder_close(skac_decoder* decoder);

SKAC_RUNTIME_API skac_result skac_decoder_get_info(
    const skac_decoder* decoder,
    skac_clip_info* out_info
);

SKAC_RUNTIME_API skac_result skac_decoder_get_joint_parent(
    const skac_decoder* decoder,
    uint32_t joint_index,
    int32_t* out_parent_index
);

/* Pointer remains valid until skac_decoder_close. */
SKAC_RUNTIME_API skac_result skac_decoder_get_joint_name(
    const skac_decoder* decoder,
    uint32_t joint_index,
    const char** out_utf8_name
);

SKAC_RUNTIME_API skac_result skac_decoder_get_joint_offset(
    const skac_decoder* decoder,
    uint32_t joint_index,
    float out_xyz[3]
);

SKAC_RUNTIME_API skac_result skac_decoder_sample_frame(
    const skac_decoder* decoder,
    uint32_t frame_index,
    skac_transform* out_transforms,
    size_t transform_capacity
);

SKAC_RUNTIME_API skac_result skac_decoder_sample_time(
    const skac_decoder* decoder,
    double time_seconds,
    skac_time_mode mode,
    skac_transform* out_transforms,
    size_t transform_capacity
);

/*
 * Compile a frozen Profile 2.0 plan for one playback instance. The target skeleton
 * document uses schema skac.runtime_skeleton 1.0. The decoder must remain alive until
 * the retargeter is closed. A retargeter owns reusable scratch buffers and must not be
 * sampled concurrently; separate instances may share one immutable decoder.
 */
SKAC_RUNTIME_API skac_result skac_retargeter_create(
    const skac_decoder* decoder,
    const char* profile_json,
    size_t profile_size,
    const char* target_skeleton_json,
    size_t target_skeleton_size,
    skac_retargeter** out_retargeter
);

SKAC_RUNTIME_API void skac_retargeter_close(skac_retargeter* retargeter);

SKAC_RUNTIME_API skac_result skac_retargeter_get_info(
    const skac_retargeter* retargeter,
    skac_retarget_info* out_info
);

SKAC_RUNTIME_API skac_result skac_retargeter_get_joint_parent(
    const skac_retargeter* retargeter,
    uint32_t joint_index,
    int32_t* out_parent_index
);

SKAC_RUNTIME_API skac_result skac_retargeter_get_joint_name(
    const skac_retargeter* retargeter,
    uint32_t joint_index,
    const char** out_utf8_name
);

SKAC_RUNTIME_API skac_result skac_retargeter_sample_frame(
    skac_retargeter* retargeter,
    uint32_t frame_index,
    skac_transform* out_transforms,
    size_t transform_capacity
);

SKAC_RUNTIME_API skac_result skac_retargeter_sample_time(
    skac_retargeter* retargeter,
    double time_seconds,
    skac_time_mode mode,
    skac_transform* out_transforms,
    size_t transform_capacity
);

/* Thread-local diagnostic for the most recent failed call on this thread. */
SKAC_RUNTIME_API const char* skac_runtime_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
