#include "skac_runtime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(SKAC_RUNTIME_WITH_ZLIB)
#include <zlib.h>
#endif

#define SKAC_JOIN_DETAIL(left, right) left##right
#define SKAC_JOIN(left, right) SKAC_JOIN_DETAIL(left, right)
namespace SKAC_STANDARD = SKAC_JOIN(st, d);
#undef SKAC_JOIN
#undef SKAC_JOIN_DETAIL

namespace {

constexpr size_t kPrefixSize = 44;
constexpr uint32_t kFlagZlib = 1;
constexpr uint16_t kFormatMajor = 1;
constexpr uint16_t kFormatMinor = 0;
constexpr uint64_t kMaxMetadataBytes = 8ull * 1024ull * 1024ull;
constexpr uint64_t kMaxRawPayloadBytes = 512ull * 1024ull * 1024ull;
constexpr double kSmallestThreeLimit = 0.70710678118654752440;

thread_local SKAC_STANDARD::string g_last_error;

struct RuntimeError : SKAC_STANDARD::runtime_error {
    RuntimeError(skac_result result_value, const SKAC_STANDARD::string& message)
        : SKAC_STANDARD::runtime_error(message), result(result_value) {}
    skac_result result;
};

[[noreturn]] void fail(skac_result result, const SKAC_STANDARD::string& message) {
    throw RuntimeError(result, message);
}

struct Json {
    enum class Type { Null, Boolean, Number, String, Array, Object };
    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    SKAC_STANDARD::string string;
    SKAC_STANDARD::vector<Json> array;
    SKAC_STANDARD::map<SKAC_STANDARD::string, Json> object;
};

class JsonParser {
public:
    JsonParser(const char* data, size_t size) : data_(data), size_(size) {}

    Json parse() {
        skip_space();
        Json value = parse_value();
        skip_space();
        if (position_ != size_) {
            fail(SKAC_INVALID_FORMAT, "metadata JSON contains trailing bytes");
        }
        return value;
    }

private:
    const char* data_;
    size_t size_;
    size_t position_ = 0;

    void skip_space() {
        while (position_ < size_) {
            const char value = data_[position_];
            if (value != ' ' && value != '\t' && value != '\r' && value != '\n') {
                break;
            }
            ++position_;
        }
    }

    char take() {
        if (position_ >= size_) {
            fail(SKAC_INVALID_FORMAT, "metadata JSON ended unexpectedly");
        }
        return data_[position_++];
    }

    bool consume(char expected) {
        if (position_ < size_ && data_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    void require_literal(const char* literal) {
        while (*literal != '\0') {
            if (take() != *literal++) {
                fail(SKAC_INVALID_FORMAT, "metadata JSON contains an invalid literal");
            }
        }
    }

    static void append_utf8(SKAC_STANDARD::string& output, uint32_t codepoint) {
        if (codepoint <= 0x7f) {
            output.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7ff) {
            output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else if (codepoint <= 0xffff) {
            output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
            output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else if (codepoint <= 0x10ffff) {
            output.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
            output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else {
            fail(SKAC_INVALID_FORMAT, "metadata JSON contains an invalid Unicode value");
        }
    }

    static uint32_t hex_digit(char value) {
        if (value >= '0' && value <= '9') return static_cast<uint32_t>(value - '0');
        if (value >= 'a' && value <= 'f') return 10u + static_cast<uint32_t>(value - 'a');
        if (value >= 'A' && value <= 'F') return 10u + static_cast<uint32_t>(value - 'A');
        fail(SKAC_INVALID_FORMAT, "metadata JSON contains an invalid Unicode escape");
    }

    uint32_t parse_hex4() {
        uint32_t result = 0;
        for (int index = 0; index < 4; ++index) {
            result = (result << 4) | hex_digit(take());
        }
        return result;
    }

    SKAC_STANDARD::string parse_string() {
        if (!consume('"')) {
            fail(SKAC_INVALID_FORMAT, "metadata JSON expected a string");
        }
        SKAC_STANDARD::string result;
        while (true) {
            const unsigned char value = static_cast<unsigned char>(take());
            if (value == '"') break;
            if (value < 0x20) {
                fail(SKAC_INVALID_FORMAT, "metadata JSON string contains a control byte");
            }
            if (value != '\\') {
                result.push_back(static_cast<char>(value));
                continue;
            }
            const char escaped = take();
            switch (escaped) {
                case '"': result.push_back('"'); break;
                case '\\': result.push_back('\\'); break;
                case '/': result.push_back('/'); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                case 'u': {
                    uint32_t codepoint = parse_hex4();
                    if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                        if (take() != '\\' || take() != 'u') {
                            fail(SKAC_INVALID_FORMAT, "metadata JSON has an incomplete surrogate pair");
                        }
                        const uint32_t low = parse_hex4();
                        if (low < 0xdc00 || low > 0xdfff) {
                            fail(SKAC_INVALID_FORMAT, "metadata JSON has an invalid surrogate pair");
                        }
                        codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
                    } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
                        fail(SKAC_INVALID_FORMAT, "metadata JSON has an unpaired surrogate");
                    }
                    append_utf8(result, codepoint);
                    break;
                }
                default:
                    fail(SKAC_INVALID_FORMAT, "metadata JSON contains an invalid escape");
            }
        }
        return result;
    }

    Json parse_number() {
        const size_t start = position_;
        if (consume('-')) {}
        if (consume('0')) {
        } else {
            if (position_ >= size_ || data_[position_] < '1' || data_[position_] > '9') {
                fail(SKAC_INVALID_FORMAT, "metadata JSON contains an invalid number");
            }
            while (position_ < size_ && data_[position_] >= '0' && data_[position_] <= '9') {
                ++position_;
            }
        }
        if (consume('.')) {
            if (position_ >= size_ || data_[position_] < '0' || data_[position_] > '9') {
                fail(SKAC_INVALID_FORMAT, "metadata JSON contains an invalid fraction");
            }
            while (position_ < size_ && data_[position_] >= '0' && data_[position_] <= '9') {
                ++position_;
            }
        }
        if (position_ < size_ && (data_[position_] == 'e' || data_[position_] == 'E')) {
            ++position_;
            if (position_ < size_ && (data_[position_] == '+' || data_[position_] == '-')) ++position_;
            if (position_ >= size_ || data_[position_] < '0' || data_[position_] > '9') {
                fail(SKAC_INVALID_FORMAT, "metadata JSON contains an invalid exponent");
            }
            while (position_ < size_ && data_[position_] >= '0' && data_[position_] <= '9') {
                ++position_;
            }
        }
        const SKAC_STANDARD::string encoded(data_ + start, position_ - start);
        char* end = nullptr;
        const double number = SKAC_STANDARD::strtod(encoded.c_str(), &end);
        if (end == nullptr || *end != '\0' || !SKAC_STANDARD::isfinite(number)) {
            fail(SKAC_INVALID_FORMAT, "metadata JSON number is not finite");
        }
        Json result;
        result.type = Json::Type::Number;
        result.number = number;
        return result;
    }

    Json parse_array() {
        consume('[');
        Json result;
        result.type = Json::Type::Array;
        skip_space();
        if (consume(']')) return result;
        while (true) {
            skip_space();
            result.array.push_back(parse_value());
            skip_space();
            if (consume(']')) return result;
            if (!consume(',')) fail(SKAC_INVALID_FORMAT, "metadata JSON expected an array comma");
        }
    }

    Json parse_object() {
        consume('{');
        Json result;
        result.type = Json::Type::Object;
        skip_space();
        if (consume('}')) return result;
        while (true) {
            skip_space();
            SKAC_STANDARD::string key = parse_string();
            skip_space();
            if (!consume(':')) fail(SKAC_INVALID_FORMAT, "metadata JSON expected an object colon");
            skip_space();
            if (!result.object.emplace(SKAC_STANDARD::move(key), parse_value()).second) {
                fail(SKAC_INVALID_FORMAT, "metadata JSON repeats an object key");
            }
            skip_space();
            if (consume('}')) return result;
            if (!consume(',')) fail(SKAC_INVALID_FORMAT, "metadata JSON expected an object comma");
        }
    }

    Json parse_value() {
        if (position_ >= size_) fail(SKAC_INVALID_FORMAT, "metadata JSON ended unexpectedly");
        const char value = data_[position_];
        if (value == '{') return parse_object();
        if (value == '[') return parse_array();
        if (value == '"') {
            Json result;
            result.type = Json::Type::String;
            result.string = parse_string();
            return result;
        }
        if (value == 't') {
            require_literal("true");
            Json result;
            result.type = Json::Type::Boolean;
            result.boolean = true;
            return result;
        }
        if (value == 'f') {
            require_literal("false");
            Json result;
            result.type = Json::Type::Boolean;
            return result;
        }
        if (value == 'n') {
            require_literal("null");
            return Json{};
        }
        return parse_number();
    }
};

const Json& member(const Json& value, const char* name) {
    if (value.type != Json::Type::Object) fail(SKAC_INVALID_FORMAT, "metadata value is not an object");
    const auto found = value.object.find(name);
    if (found == value.object.end()) fail(SKAC_INVALID_FORMAT, SKAC_STANDARD::string("metadata is missing ") + name);
    return found->second;
}

const SKAC_STANDARD::vector<Json>& array_value(const Json& value, const char* label) {
    if (value.type != Json::Type::Array) fail(SKAC_INVALID_FORMAT, SKAC_STANDARD::string(label) + " is not an array");
    return value.array;
}

SKAC_STANDARD::string string_value(const Json& value, const char* label) {
    if (value.type != Json::Type::String) fail(SKAC_INVALID_FORMAT, SKAC_STANDARD::string(label) + " is not a string");
    return value.string;
}

double number_value(const Json& value, const char* label) {
    if (value.type != Json::Type::Number || !SKAC_STANDARD::isfinite(value.number)) {
        fail(SKAC_INVALID_FORMAT, SKAC_STANDARD::string(label) + " is not a finite number");
    }
    return value.number;
}

int64_t integer_value(const Json& value, const char* label) {
    const double number = number_value(value, label);
    if (SKAC_STANDARD::floor(number) != number || number < static_cast<double>(SKAC_STANDARD::numeric_limits<int64_t>::min()) ||
        number > static_cast<double>(SKAC_STANDARD::numeric_limits<int64_t>::max())) {
        fail(SKAC_INVALID_FORMAT, SKAC_STANDARD::string(label) + " is not an integer");
    }
    return static_cast<int64_t>(number);
}

uint32_t u32_value(const Json& value, const char* label) {
    const int64_t number = integer_value(value, label);
    if (number < 0 || number > SKAC_STANDARD::numeric_limits<uint32_t>::max()) {
        fail(SKAC_INVALID_FORMAT, SKAC_STANDARD::string(label) + " is outside uint32");
    }
    return static_cast<uint32_t>(number);
}

SKAC_STANDARD::vector<uint32_t> u32_array(const Json& value, const char* label) {
    SKAC_STANDARD::vector<uint32_t> result;
    for (const Json& item : array_value(value, label)) result.push_back(u32_value(item, label));
    return result;
}

SKAC_STANDARD::vector<double> number_array(const Json& value, const char* label) {
    SKAC_STANDARD::vector<double> result;
    for (const Json& item : array_value(value, label)) result.push_back(number_value(item, label));
    return result;
}

uint16_t read_u16(const uint8_t* value) {
    return static_cast<uint16_t>(value[0]) | (static_cast<uint16_t>(value[1]) << 8);
}

uint32_t read_u32(const uint8_t* value) {
    return static_cast<uint32_t>(value[0]) |
        (static_cast<uint32_t>(value[1]) << 8) |
        (static_cast<uint32_t>(value[2]) << 16) |
        (static_cast<uint32_t>(value[3]) << 24);
}

uint64_t read_u64(const uint8_t* value) {
    uint64_t result = 0;
    for (int index = 7; index >= 0; --index) result = (result << 8) | value[index];
    return result;
}

uint32_t crc32_bytes(const uint8_t* data, size_t size) {
    uint32_t crc = 0xffffffffu;
    for (size_t index = 0; index < size; ++index) {
        crc ^= data[index];
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
        }
    }
    return ~crc;
}

struct Quaternion {
    double w = 1.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

Quaternion normalize(Quaternion value) {
    const double length = SKAC_STANDARD::sqrt(value.w * value.w + value.x * value.x + value.y * value.y + value.z * value.z);
    if (!(length > 1e-15) || !SKAC_STANDARD::isfinite(length)) fail(SKAC_INVALID_FORMAT, "decoded a zero quaternion");
    value.w /= length;
    value.x /= length;
    value.y /= length;
    value.z /= length;
    return value;
}

Quaternion slerp(Quaternion left, Quaternion right, double amount) {
    double dot = left.w * right.w + left.x * right.x + left.y * right.y + left.z * right.z;
    if (dot < 0.0) {
        right.w = -right.w;
        right.x = -right.x;
        right.y = -right.y;
        right.z = -right.z;
        dot = -dot;
    }
    dot = SKAC_STANDARD::max(-1.0, SKAC_STANDARD::min(1.0, dot));
    if (dot > 0.9995) {
        return normalize({
            left.w + amount * (right.w - left.w),
            left.x + amount * (right.x - left.x),
            left.y + amount * (right.y - left.y),
            left.z + amount * (right.z - left.z),
        });
    }
    const double theta = SKAC_STANDARD::acos(dot);
    const double sine = SKAC_STANDARD::sin(theta);
    const double a = SKAC_STANDARD::sin((1.0 - amount) * theta) / sine;
    const double b = SKAC_STANDARD::sin(amount * theta) / sine;
    return normalize({
        a * left.w + b * right.w,
        a * left.x + b * right.x,
        a * left.y + b * right.y,
        a * left.z + b * right.z,
    });
}

class Cursor {
public:
    Cursor(const uint8_t* data, size_t size) : data_(data), size_(size) {}

    uint32_t u32() {
        require(4);
        const uint32_t result = read_u32(data_ + position_);
        position_ += 4;
        return result;
    }

    uint32_t varuint() {
        uint32_t result = 0;
        uint32_t shift = 0;
        for (int index = 0; index < 5; ++index) {
            require(1);
            const uint8_t value = data_[position_++];
            result |= static_cast<uint32_t>(value & 0x7f) << shift;
            if ((value & 0x80) == 0) return result;
            shift += 7;
        }
        fail(SKAC_INVALID_FORMAT, "track varuint exceeds 32 bits");
    }

    const uint8_t* take(size_t count) {
        require(count);
        const uint8_t* result = data_ + position_;
        position_ += count;
        return result;
    }

    void finished() const {
        if (position_ != size_) fail(SKAC_INVALID_FORMAT, "track payload contains trailing bytes");
    }

private:
    const uint8_t* data_;
    size_t size_;
    size_t position_ = 0;

    void require(size_t count) const {
        if (count > size_ - position_) fail(SKAC_INVALID_FORMAT, "track payload ended unexpectedly");
    }
};

class BitReader {
public:
    BitReader(const uint8_t* data, size_t size) : data_(data), size_(size) {}

    uint32_t read(uint32_t bits) {
        while (count_ < bits) {
            if (position_ >= size_) fail(SKAC_INVALID_FORMAT, "packed payload ended unexpectedly");
            accumulator_ |= static_cast<uint64_t>(data_[position_++]) << count_;
            count_ += 8;
        }
        const uint64_t mask = (uint64_t{1} << bits) - 1;
        const uint32_t result = static_cast<uint32_t>(accumulator_ & mask);
        accumulator_ >>= bits;
        count_ -= bits;
        return result;
    }

private:
    const uint8_t* data_;
    size_t size_;
    size_t position_ = 0;
    uint64_t accumulator_ = 0;
    uint32_t count_ = 0;
};

size_t packed_bytes(size_t count, uint32_t bits) {
    if (bits == 0 || count > (SKAC_STANDARD::numeric_limits<size_t>::max() - 7) / bits) {
        fail(SKAC_INVALID_FORMAT, "packed payload size overflows");
    }
    return (count * bits + 7) / 8;
}

SKAC_STANDARD::vector<uint32_t> decode_indices(Cursor& cursor, uint32_t count, uint32_t frame_count) {
    if (count == 0 || count > frame_count) fail(SKAC_INVALID_FORMAT, "track key count is outside frame range");
    SKAC_STANDARD::vector<uint32_t> result(count);
    uint32_t previous = 0;
    for (uint32_t index = 0; index < count; ++index) {
        const uint32_t delta = cursor.varuint();
        if (index > 0 && delta > SKAC_STANDARD::numeric_limits<uint32_t>::max() - previous) {
            fail(SKAC_INVALID_FORMAT, "track key index overflows");
        }
        const uint32_t frame = index == 0 ? delta : previous + delta;
        if (frame >= frame_count || (index > 0 && frame <= previous)) {
            fail(SKAC_INVALID_FORMAT, "track indices are not strictly increasing");
        }
        result[index] = frame;
        previous = frame;
    }
    if (result.front() != 0) fail(SKAC_INVALID_FORMAT, "every track must begin at frame zero");
    return result;
}

Quaternion decode_quaternion(BitReader& reader, uint32_t bits) {
    const uint32_t omitted = reader.read(2);
    if (omitted > 3) fail(SKAC_INVALID_FORMAT, "quaternion omitted index is invalid");
    const double maximum = static_cast<double>((uint64_t{1} << bits) - 1);
    SKAC_STANDARD::array<double, 4> values{};
    double squared = 0.0;
    for (uint32_t component = 0; component < 4; ++component) {
        if (component == omitted) continue;
        const double quantized = static_cast<double>(reader.read(bits));
        const double value = (quantized / maximum) * (2.0 * kSmallestThreeLimit) - kSmallestThreeLimit;
        values[component] = value;
        squared += value * value;
    }
    values[omitted] = SKAC_STANDARD::sqrt(SKAC_STANDARD::max(0.0, 1.0 - squared));
    return normalize({values[0], values[1], values[2], values[3]});
}

struct SkeletonData {
    SKAC_STANDARD::vector<SKAC_STANDARD::string> names;
    SKAC_STANDARD::vector<int32_t> parents;
    SKAC_STANDARD::vector<SKAC_STANDARD::array<double, 3>> offsets;
    SKAC_STANDARD::vector<SKAC_STANDARD::vector<SKAC_STANDARD::string>> channels;
};

struct skac_decoder_impl {
    uint32_t frame_count = 0;
    uint32_t joint_count = 0;
    double frame_time = 0.0;
    SkeletonData skeleton;
    SKAC_STANDARD::vector<Quaternion> rotations;
    SKAC_STANDARD::vector<SKAC_STANDARD::array<double, 3>> translations;
};

size_t pose_index(const skac_decoder_impl& decoder, uint32_t frame, uint32_t joint) {
    return static_cast<size_t>(frame) * decoder.joint_count + joint;
}

SkeletonData parse_skeleton(const Json& value) {
    SkeletonData skeleton;
    for (const Json& item : array_value(member(value, "names"), "skeleton.names")) {
        skeleton.names.push_back(string_value(item, "skeleton name"));
    }
    if (skeleton.names.empty()) fail(SKAC_INVALID_FORMAT, "skeleton has no joints");
    for (const Json& item : array_value(member(value, "parents"), "skeleton.parents")) {
        const int64_t parent = integer_value(item, "skeleton parent");
        if (parent < SKAC_STANDARD::numeric_limits<int32_t>::min() || parent > SKAC_STANDARD::numeric_limits<int32_t>::max()) {
            fail(SKAC_INVALID_FORMAT, "skeleton parent is outside int32");
        }
        skeleton.parents.push_back(static_cast<int32_t>(parent));
    }
    for (const Json& item : array_value(member(value, "offsets"), "skeleton.offsets")) {
        const auto& components = array_value(item, "skeleton offset");
        if (components.size() != 3) fail(SKAC_INVALID_FORMAT, "skeleton offset must contain three numbers");
        skeleton.offsets.push_back({
            number_value(components[0], "offset x"),
            number_value(components[1], "offset y"),
            number_value(components[2], "offset z"),
        });
    }
    for (const Json& item : array_value(member(value, "channels"), "skeleton.channels")) {
        SKAC_STANDARD::vector<SKAC_STANDARD::string> channels;
        for (const Json& channel : array_value(item, "joint channels")) {
            channels.push_back(string_value(channel, "joint channel"));
        }
        skeleton.channels.push_back(SKAC_STANDARD::move(channels));
    }
    const size_t count = skeleton.names.size();
    if (skeleton.parents.size() != count || skeleton.offsets.size() != count || skeleton.channels.size() != count) {
        fail(SKAC_INVALID_FORMAT, "skeleton arrays do not match joint count");
    }
    if (skeleton.parents[0] != -1) fail(SKAC_INVALID_FORMAT, "root parent must be -1");
    for (size_t joint = 1; joint < count; ++joint) {
        if (skeleton.parents[joint] < 0 || static_cast<size_t>(skeleton.parents[joint]) >= joint) {
            fail(SKAC_INVALID_FORMAT, "skeleton parents must precede their children");
        }
    }
    return skeleton;
}

SKAC_STANDARD::vector<uint32_t> expected_rotation_joints(const SkeletonData& skeleton) {
    SKAC_STANDARD::vector<uint32_t> result;
    for (uint32_t joint = 0; joint < skeleton.channels.size(); ++joint) {
        const auto& channels = skeleton.channels[joint];
        if (SKAC_STANDARD::any_of(channels.begin(), channels.end(), [](const SKAC_STANDARD::string& item) {
                return item.size() >= 8 && item.compare(item.size() - 8, 8, "rotation") == 0;
            })) {
            result.push_back(joint);
        }
    }
    return result;
}

SKAC_STANDARD::vector<SKAC_STANDARD::pair<uint32_t, uint32_t>> expected_translation_components(const SkeletonData& skeleton) {
    SKAC_STANDARD::vector<SKAC_STANDARD::pair<uint32_t, uint32_t>> result;
    for (uint32_t joint = 0; joint < skeleton.channels.size(); ++joint) {
        for (const SKAC_STANDARD::string& channel : skeleton.channels[joint]) {
            if (channel.size() < 8 || channel.compare(channel.size() - 8, 8, "position") != 0) continue;
            uint32_t axis = 0;
            if (channel[0] == 'X') axis = 0;
            else if (channel[0] == 'Y') axis = 1;
            else if (channel[0] == 'Z') axis = 2;
            else fail(SKAC_INVALID_FORMAT, "position channel has an invalid axis");
            result.emplace_back(joint, axis);
        }
    }
    return result;
}

SKAC_STANDARD::unique_ptr<skac_decoder_impl> decode_raw(
    const char* metadata_json,
    size_t metadata_size,
    const uint8_t* raw_payload,
    size_t raw_payload_size
) {
    if (metadata_json == nullptr || (raw_payload == nullptr && raw_payload_size != 0)) {
        fail(SKAC_INVALID_ARGUMENT, "raw decoder arguments are null");
    }
    if (metadata_size == 0 || metadata_size > kMaxMetadataBytes || raw_payload_size > kMaxRawPayloadBytes) {
        fail(SKAC_INVALID_FORMAT, "metadata or raw payload exceeds format limits");
    }
    const Json root = JsonParser(metadata_json, metadata_size).parse();
    if (string_value(member(root, "schema"), "schema") != "skac.animation" ||
        string_value(member(root, "schema_version"), "schema_version") != "1.0.0") {
        fail(SKAC_UNSUPPORTED, "unsupported animation metadata schema");
    }
    auto decoder = SKAC_STANDARD::make_unique<skac_decoder_impl>();
    decoder->frame_count = u32_value(member(root, "frame_count"), "frame_count");
    decoder->frame_time = number_value(member(root, "frame_time"), "frame_time");
    if (decoder->frame_count == 0 || decoder->frame_time <= 0.0) {
        fail(SKAC_INVALID_FORMAT, "animation timing is invalid");
    }
    decoder->skeleton = parse_skeleton(member(root, "skeleton"));
    if (decoder->skeleton.names.size() > SKAC_STANDARD::numeric_limits<uint32_t>::max()) {
        fail(SKAC_INVALID_FORMAT, "joint count exceeds uint32");
    }
    decoder->joint_count = static_cast<uint32_t>(decoder->skeleton.names.size());
    if (decoder->frame_count > SKAC_STANDARD::numeric_limits<size_t>::max() / decoder->joint_count) {
        fail(SKAC_INVALID_FORMAT, "pose count overflows memory size");
    }
    const size_t pose_count = static_cast<size_t>(decoder->frame_count) * decoder->joint_count;
    decoder->rotations.assign(pose_count, Quaternion{});
    decoder->translations.resize(pose_count);
    for (uint32_t frame = 0; frame < decoder->frame_count; ++frame) {
        for (uint32_t joint = 0; joint < decoder->joint_count; ++joint) {
            decoder->translations[pose_index(*decoder, frame, joint)] = decoder->skeleton.offsets[joint];
        }
    }

    const Json& codec = member(root, "codec");
    if (string_value(member(codec, "name"), "codec.name") != "key_reduced_smallest_three_v1") {
        fail(SKAC_UNSUPPORTED, "unsupported codec payload");
    }
    const uint32_t rotation_bits = u32_value(member(codec, "rotation_bits"), "rotation_bits");
    const uint32_t translation_bits = u32_value(member(codec, "translation_bits"), "translation_bits");
    if (rotation_bits < 8 || rotation_bits > 20 || translation_bits < 8 || translation_bits > 24) {
        fail(SKAC_INVALID_FORMAT, "codec bit width is outside the supported range");
    }
    const SKAC_STANDARD::vector<uint32_t> rotation_joints = u32_array(member(codec, "rotation_joints"), "rotation_joints");
    const SKAC_STANDARD::vector<uint32_t> rotation_key_counts = u32_array(member(codec, "rotation_key_counts"), "rotation_key_counts");
    if (rotation_joints != expected_rotation_joints(decoder->skeleton) || rotation_key_counts.size() != rotation_joints.size()) {
        fail(SKAC_INVALID_FORMAT, "rotation track table does not match skeleton");
    }
    SKAC_STANDARD::vector<SKAC_STANDARD::pair<uint32_t, uint32_t>> translation_components;
    for (const Json& item : array_value(member(codec, "translation_components"), "translation_components")) {
        const auto& pair = array_value(item, "translation component");
        if (pair.size() != 2) fail(SKAC_INVALID_FORMAT, "translation component must contain joint and axis");
        translation_components.emplace_back(
            u32_value(pair[0], "translation joint"),
            u32_value(pair[1], "translation axis")
        );
    }
    const SKAC_STANDARD::vector<uint32_t> translation_key_counts = u32_array(member(codec, "translation_key_counts"), "translation_key_counts");
    const SKAC_STANDARD::vector<double> minima = number_array(member(codec, "translation_minima"), "translation_minima");
    const SKAC_STANDARD::vector<double> maxima = number_array(member(codec, "translation_maxima"), "translation_maxima");
    if (translation_components != expected_translation_components(decoder->skeleton) ||
        translation_key_counts.size() != translation_components.size() ||
        minima.size() != translation_components.size() || maxima.size() != translation_components.size()) {
        fail(SKAC_INVALID_FORMAT, "translation track table does not match skeleton");
    }
    const uint32_t rotation_payload_size = u32_value(member(codec, "rotation_payload_bytes"), "rotation_payload_bytes");
    const uint32_t translation_payload_size = u32_value(member(codec, "translation_payload_bytes"), "translation_payload_bytes");
    if (static_cast<uint64_t>(rotation_payload_size) + translation_payload_size != raw_payload_size) {
        fail(SKAC_INVALID_FORMAT, "payload section lengths are inconsistent");
    }

    Cursor rotation_cursor(raw_payload, rotation_payload_size);
    for (size_t track = 0; track < rotation_joints.size(); ++track) {
        const uint32_t key_count = rotation_cursor.u32();
        if (key_count != rotation_key_counts[track]) fail(SKAC_INVALID_FORMAT, "rotation key count differs from metadata");
        const SKAC_STANDARD::vector<uint32_t> indices = decode_indices(rotation_cursor, key_count, decoder->frame_count);
        const size_t byte_count = packed_bytes(key_count, 2 + 3 * rotation_bits);
        BitReader bits(rotation_cursor.take(byte_count), byte_count);
        SKAC_STANDARD::vector<Quaternion> values;
        values.reserve(key_count);
        for (uint32_t index = 0; index < key_count; ++index) values.push_back(decode_quaternion(bits, rotation_bits));
        size_t segment = 0;
        for (uint32_t frame = 0; frame < decoder->frame_count; ++frame) {
            Quaternion value = values.front();
            if (values.size() > 1) {
                while (segment + 1 < indices.size() - 1 && frame > indices[segment + 1]) ++segment;
                const uint32_t start = indices[segment];
                const uint32_t end = indices[segment + 1];
                const double amount = static_cast<double>(frame - start) / static_cast<double>(end - start);
                value = slerp(values[segment], values[segment + 1], amount);
            }
            decoder->rotations[pose_index(*decoder, frame, rotation_joints[track])] = value;
        }
    }
    rotation_cursor.finished();

    Cursor translation_cursor(raw_payload + rotation_payload_size, translation_payload_size);
    const double max_quantized = static_cast<double>((uint64_t{1} << translation_bits) - 1);
    for (size_t track = 0; track < translation_components.size(); ++track) {
        const uint32_t key_count = translation_cursor.u32();
        if (key_count != translation_key_counts[track]) fail(SKAC_INVALID_FORMAT, "translation key count differs from metadata");
        const SKAC_STANDARD::vector<uint32_t> indices = decode_indices(translation_cursor, key_count, decoder->frame_count);
        const size_t byte_count = packed_bytes(key_count, translation_bits);
        BitReader bits(translation_cursor.take(byte_count), byte_count);
        SKAC_STANDARD::vector<double> values;
        values.reserve(key_count);
        for (uint32_t index = 0; index < key_count; ++index) {
            const double normalized = static_cast<double>(bits.read(translation_bits)) / max_quantized;
            values.push_back(minima[track] + normalized * (maxima[track] - minima[track]));
        }
        size_t segment = 0;
        for (uint32_t frame = 0; frame < decoder->frame_count; ++frame) {
            double value = values.front();
            if (values.size() > 1) {
                while (segment + 1 < indices.size() - 1 && frame > indices[segment + 1]) ++segment;
                const uint32_t start = indices[segment];
                const uint32_t end = indices[segment + 1];
                const double amount = static_cast<double>(frame - start) / static_cast<double>(end - start);
                value = values[segment] + amount * (values[segment + 1] - values[segment]);
            }
            const auto component = translation_components[track];
            decoder->translations[pose_index(*decoder, frame, component.first)][component.second] = value;
        }
    }
    translation_cursor.finished();
    return decoder;
}

void write_transform(const Quaternion& rotation, const SKAC_STANDARD::array<double, 3>& translation, skac_transform& output) {
    output.rotation_x = static_cast<float>(rotation.x);
    output.rotation_y = static_cast<float>(rotation.y);
    output.rotation_z = static_cast<float>(rotation.z);
    output.rotation_w = static_cast<float>(rotation.w);
    output.translation_x = static_cast<float>(translation[0]);
    output.translation_y = static_cast<float>(translation[1]);
    output.translation_z = static_cast<float>(translation[2]);
}

template <typename Function>
skac_result guarded(Function&& function) noexcept {
    try {
        g_last_error.clear();
        function();
        return SKAC_OK;
    } catch (const RuntimeError& error) {
        g_last_error = error.what();
        return error.result;
    } catch (const SKAC_STANDARD::bad_alloc&) {
        g_last_error = "native runtime allocation failed";
        return SKAC_OUT_OF_MEMORY;
    } catch (const SKAC_STANDARD::exception& error) {
        g_last_error = error.what();
        return SKAC_INTERNAL_ERROR;
    } catch (...) {
        g_last_error = "unknown native runtime failure";
        return SKAC_INTERNAL_ERROR;
    }
}

}  // namespace

struct skac_decoder {
    SKAC_STANDARD::unique_ptr<skac_decoder_impl> value;
};

extern "C" {

skac_result skac_decoder_open_raw(
    const char* metadata_json,
    size_t metadata_size,
    const uint8_t* raw_payload,
    size_t raw_payload_size,
    skac_decoder** out_decoder
) {
    return guarded([&]() {
        if (out_decoder == nullptr) fail(SKAC_INVALID_ARGUMENT, "out_decoder is null");
        *out_decoder = nullptr;
        auto result = SKAC_STANDARD::make_unique<skac_decoder>();
        result->value = decode_raw(metadata_json, metadata_size, raw_payload, raw_payload_size);
        *out_decoder = result.release();
    });
}

skac_result skac_decoder_open_container(
    const uint8_t* container,
    size_t container_size,
    skac_inflate_fn inflate,
    void* user_data,
    skac_decoder** out_decoder
) {
    return guarded([&]() {
        if (container == nullptr || inflate == nullptr || out_decoder == nullptr) {
            fail(SKAC_INVALID_ARGUMENT, "container open argument is null");
        }
        *out_decoder = nullptr;
        if (container_size < kPrefixSize) fail(SKAC_INVALID_FORMAT, "container is shorter than prefix");
        if (SKAC_STANDARD::memcmp(container, "SKACANIM", 8) != 0) fail(SKAC_INVALID_FORMAT, "container magic is invalid");
        const uint16_t major = read_u16(container + 8);
        const uint16_t minor = read_u16(container + 10);
        const uint32_t flags = read_u32(container + 12);
        const uint32_t metadata_size = read_u32(container + 16);
        const uint64_t payload_size = read_u64(container + 20);
        const uint64_t raw_size = read_u64(container + 28);
        const uint32_t metadata_crc = read_u32(container + 36);
        const uint32_t payload_crc = read_u32(container + 40);
        if (major != kFormatMajor || minor > kFormatMinor) fail(SKAC_UNSUPPORTED, "container version is unsupported");
        if (flags != kFlagZlib) fail(SKAC_UNSUPPORTED, "container flags are unsupported");
        if (metadata_size > kMaxMetadataBytes || raw_size > kMaxRawPayloadBytes) fail(SKAC_INVALID_FORMAT, "container declarations exceed limits");
        if (payload_size > SKAC_STANDARD::numeric_limits<size_t>::max() || raw_size > SKAC_STANDARD::numeric_limits<size_t>::max()) {
            fail(SKAC_INVALID_FORMAT, "container size exceeds this platform");
        }
        const uint64_t expected = kPrefixSize + static_cast<uint64_t>(metadata_size) + payload_size;
        if (expected != container_size) fail(SKAC_INVALID_FORMAT, "container length does not match declarations");
        const uint8_t* metadata = container + kPrefixSize;
        const uint8_t* payload = metadata + metadata_size;
        if (crc32_bytes(metadata, metadata_size) != metadata_crc) fail(SKAC_INVALID_FORMAT, "metadata CRC does not match");
        if (crc32_bytes(payload, static_cast<size_t>(payload_size)) != payload_crc) fail(SKAC_INVALID_FORMAT, "payload CRC does not match");
        SKAC_STANDARD::vector<uint8_t> raw(static_cast<size_t>(raw_size));
        const skac_result inflate_result = inflate(
            payload,
            static_cast<size_t>(payload_size),
            raw.data(),
            raw.size(),
            user_data
        );
        if (inflate_result != SKAC_OK) fail(inflate_result, "host zlib inflation failed");
        auto result = SKAC_STANDARD::make_unique<skac_decoder>();
        result->value = decode_raw(
            reinterpret_cast<const char*>(metadata), metadata_size, raw.data(), raw.size()
        );
        *out_decoder = result.release();
    });
}

skac_result skac_decoder_open_memory(
    const uint8_t* container,
    size_t container_size,
    skac_decoder** out_decoder
) {
#if defined(SKAC_RUNTIME_WITH_ZLIB)
    const auto callback = [](const uint8_t* compressed, size_t compressed_size, uint8_t* destination,
                             size_t destination_size, void*) -> skac_result {
        uLongf output_size = static_cast<uLongf>(destination_size);
        const int result = uncompress(
            destination,
            &output_size,
            compressed,
            static_cast<uLong>(compressed_size)
        );
        return result == Z_OK && output_size == destination_size ? SKAC_OK : SKAC_DECOMPRESSION_FAILED;
    };
    return skac_decoder_open_container(container, container_size, callback, nullptr, out_decoder);
#else
    (void)container;
    (void)container_size;
    if (out_decoder != nullptr) *out_decoder = nullptr;
    g_last_error = "runtime was built without zlib; use skac_decoder_open_container with a host callback";
    return SKAC_UNSUPPORTED;
#endif
}

void skac_decoder_close(skac_decoder* decoder) {
    delete decoder;
}

skac_result skac_decoder_get_info(const skac_decoder* decoder, skac_clip_info* out_info) {
    return guarded([&]() {
        if (decoder == nullptr || decoder->value == nullptr || out_info == nullptr) {
            fail(SKAC_INVALID_ARGUMENT, "decoder or info output is null");
        }
        out_info->abi_version = SKAC_RUNTIME_ABI_VERSION;
        out_info->frame_count = decoder->value->frame_count;
        out_info->joint_count = decoder->value->joint_count;
        out_info->frame_time_seconds = decoder->value->frame_time;
        out_info->duration_seconds = (decoder->value->frame_count - 1) * decoder->value->frame_time;
    });
}

skac_result skac_decoder_get_joint_parent(const skac_decoder* decoder, uint32_t joint_index, int32_t* out_parent_index) {
    return guarded([&]() {
        if (decoder == nullptr || decoder->value == nullptr || out_parent_index == nullptr) {
            fail(SKAC_INVALID_ARGUMENT, "decoder or parent output is null");
        }
        if (joint_index >= decoder->value->joint_count) fail(SKAC_INVALID_ARGUMENT, "joint index is outside range");
        *out_parent_index = decoder->value->skeleton.parents[joint_index];
    });
}

skac_result skac_decoder_get_joint_name(const skac_decoder* decoder, uint32_t joint_index, const char** out_utf8_name) {
    return guarded([&]() {
        if (decoder == nullptr || decoder->value == nullptr || out_utf8_name == nullptr) {
            fail(SKAC_INVALID_ARGUMENT, "decoder or name output is null");
        }
        if (joint_index >= decoder->value->joint_count) fail(SKAC_INVALID_ARGUMENT, "joint index is outside range");
        *out_utf8_name = decoder->value->skeleton.names[joint_index].c_str();
    });
}

skac_result skac_decoder_sample_frame(
    const skac_decoder* decoder,
    uint32_t frame_index,
    skac_transform* out_transforms,
    size_t transform_capacity
) {
    return guarded([&]() {
        if (decoder == nullptr || decoder->value == nullptr || out_transforms == nullptr) {
            fail(SKAC_INVALID_ARGUMENT, "decoder or transform output is null");
        }
        if (frame_index >= decoder->value->frame_count) fail(SKAC_INVALID_ARGUMENT, "frame index is outside range");
        if (transform_capacity < decoder->value->joint_count) fail(SKAC_BUFFER_TOO_SMALL, "transform buffer is smaller than joint count");
        for (uint32_t joint = 0; joint < decoder->value->joint_count; ++joint) {
            const size_t index = pose_index(*decoder->value, frame_index, joint);
            write_transform(decoder->value->rotations[index], decoder->value->translations[index], out_transforms[joint]);
        }
    });
}

skac_result skac_decoder_sample_time(
    const skac_decoder* decoder,
    double time_seconds,
    skac_time_mode mode,
    skac_transform* out_transforms,
    size_t transform_capacity
) {
    return guarded([&]() {
        if (decoder == nullptr || decoder->value == nullptr || out_transforms == nullptr || !SKAC_STANDARD::isfinite(time_seconds)) {
            fail(SKAC_INVALID_ARGUMENT, "time sampling argument is invalid");
        }
        if (transform_capacity < decoder->value->joint_count) fail(SKAC_BUFFER_TOO_SMALL, "transform buffer is smaller than joint count");
        if (mode != SKAC_TIME_CLAMP && mode != SKAC_TIME_LOOP) fail(SKAC_INVALID_ARGUMENT, "time mode is invalid");
        const double duration = (decoder->value->frame_count - 1) * decoder->value->frame_time;
        double sampled_time = time_seconds;
        if (mode == SKAC_TIME_LOOP && duration > 0.0) {
            sampled_time = SKAC_STANDARD::fmod(sampled_time, duration);
            if (sampled_time < 0.0) sampled_time += duration;
        } else {
            sampled_time = SKAC_STANDARD::max(0.0, SKAC_STANDARD::min(duration, sampled_time));
        }
        const double frame_position = sampled_time / decoder->value->frame_time;
        const uint32_t first = SKAC_STANDARD::min(
            static_cast<uint32_t>(SKAC_STANDARD::floor(frame_position)), decoder->value->frame_count - 1
        );
        const uint32_t second = SKAC_STANDARD::min(first + 1, decoder->value->frame_count - 1);
        const double amount = SKAC_STANDARD::max(0.0, SKAC_STANDARD::min(1.0, frame_position - first));
        for (uint32_t joint = 0; joint < decoder->value->joint_count; ++joint) {
            const size_t first_index = pose_index(*decoder->value, first, joint);
            const size_t second_index = pose_index(*decoder->value, second, joint);
            const Quaternion rotation = slerp(
                decoder->value->rotations[first_index], decoder->value->rotations[second_index], amount
            );
            SKAC_STANDARD::array<double, 3> translation{};
            for (size_t axis = 0; axis < 3; ++axis) {
                translation[axis] = decoder->value->translations[first_index][axis] + amount *
                    (decoder->value->translations[second_index][axis] - decoder->value->translations[first_index][axis]);
            }
            write_transform(rotation, translation, out_transforms[joint]);
        }
    });
}

const char* skac_runtime_last_error(void) {
    return g_last_error.c_str();
}

}  // extern "C"
