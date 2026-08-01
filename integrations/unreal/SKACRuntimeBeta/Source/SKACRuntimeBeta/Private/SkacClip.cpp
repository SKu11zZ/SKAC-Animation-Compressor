#include "SkacClip.h"

#include "Misc/Compression.h"

FSkacClip::~FSkacClip()
{
    Close();
}

bool FSkacClip::Open(const TArray<uint8>& Container, FString& OutError)
{
    Close();
    if (Container.IsEmpty())
    {
        OutError = TEXT("The .skac container is empty.");
        return false;
    }

    const skac_result Result = skac_decoder_open_container(
        Container.GetData(),
        static_cast<size_t>(Container.Num()),
        &FSkacClip::InflateZlib,
        nullptr,
        &Decoder
    );
    if (Result != SKAC_OK)
    {
        OutError = LastError();
        Decoder = nullptr;
        return false;
    }
    if (skac_decoder_get_info(Decoder, &Info) != SKAC_OK)
    {
        OutError = LastError();
        Close();
        return false;
    }
    NativePose.SetNumUninitialized(static_cast<int32>(Info.joint_count));
    return true;
}

bool FSkacClip::OpenRetargeter(
    const TArray<uint8>& ProfileJson,
    const TArray<uint8>& TargetSkeletonJson,
    FString& OutError
)
{
    CloseRetargeter();
    if (Decoder == nullptr)
    {
        OutError = TEXT("Open a .skac clip before compiling its Profile.");
        return false;
    }
    if (ProfileJson.IsEmpty() || TargetSkeletonJson.IsEmpty())
    {
        OutError = TEXT("The Profile or runtime target skeleton JSON is empty.");
        return false;
    }
    const skac_result Result = skac_retargeter_create(
        Decoder,
        reinterpret_cast<const char*>(ProfileJson.GetData()),
        static_cast<size_t>(ProfileJson.Num()),
        reinterpret_cast<const char*>(TargetSkeletonJson.GetData()),
        static_cast<size_t>(TargetSkeletonJson.Num()),
        &Retargeter
    );
    if (Result != SKAC_OK || skac_retargeter_get_info(Retargeter, &RetargetInfo) != SKAC_OK)
    {
        OutError = LastError();
        CloseRetargeter();
        return false;
    }
    TargetNativePose.SetNumUninitialized(static_cast<int32>(RetargetInfo.target_joint_count));
    return true;
}

void FSkacClip::CloseRetargeter()
{
    if (Retargeter != nullptr)
    {
        skac_retargeter_close(Retargeter);
        Retargeter = nullptr;
    }
    RetargetInfo = {};
    TargetNativePose.Reset();
}

void FSkacClip::Close()
{
    CloseRetargeter();
    if (Decoder != nullptr)
    {
        skac_decoder_close(Decoder);
        Decoder = nullptr;
    }
    Info = {};
    NativePose.Reset();
}

FString FSkacClip::GetJointName(uint32 JointIndex) const
{
    const char* Name = nullptr;
    if (Decoder == nullptr ||
        skac_decoder_get_joint_name(Decoder, JointIndex, &Name) != SKAC_OK)
    {
        return FString();
    }
    return UTF8_TO_TCHAR(Name);
}

int32 FSkacClip::GetJointParent(uint32 JointIndex) const
{
    int32 Parent = INDEX_NONE;
    if (Decoder == nullptr ||
        skac_decoder_get_joint_parent(Decoder, JointIndex, &Parent) != SKAC_OK)
    {
        return INDEX_NONE;
    }
    return Parent;
}

FString FSkacClip::GetTargetJointName(uint32 JointIndex) const
{
    const char* Name = nullptr;
    if (Retargeter == nullptr ||
        skac_retargeter_get_joint_name(Retargeter, JointIndex, &Name) != SKAC_OK)
    {
        return FString();
    }
    return UTF8_TO_TCHAR(Name);
}

int32 FSkacClip::GetTargetJointParent(uint32 JointIndex) const
{
    int32 Parent = INDEX_NONE;
    if (Retargeter == nullptr ||
        skac_retargeter_get_joint_parent(Retargeter, JointIndex, &Parent) != SKAC_OK)
    {
        return INDEX_NONE;
    }
    return Parent;
}

bool FSkacClip::SampleFrame(
    uint32 FrameIndex,
    TArray<FTransform>& OutLocalPose,
    FString& OutError
)
{
    if (Decoder == nullptr)
    {
        OutError = TEXT("No .skac clip is open.");
        return false;
    }
    const skac_result Result = skac_decoder_sample_frame(
        Decoder,
        FrameIndex,
        NativePose.GetData(),
        static_cast<size_t>(NativePose.Num())
    );
    if (Result != SKAC_OK)
    {
        OutError = LastError();
        return false;
    }
    return CopyPose(NativePose, OutLocalPose, OutError);
}

bool FSkacClip::SampleTime(
    double TimeSeconds,
    bool bLoop,
    TArray<FTransform>& OutLocalPose,
    FString& OutError
)
{
    if (Decoder == nullptr)
    {
        OutError = TEXT("No .skac clip is open.");
        return false;
    }
    const skac_result Result = skac_decoder_sample_time(
        Decoder,
        TimeSeconds,
        bLoop ? SKAC_TIME_LOOP : SKAC_TIME_CLAMP,
        NativePose.GetData(),
        static_cast<size_t>(NativePose.Num())
    );
    if (Result != SKAC_OK)
    {
        OutError = LastError();
        return false;
    }
    return CopyPose(NativePose, OutLocalPose, OutError);
}

bool FSkacClip::SampleRetargetedFrame(
    uint32 FrameIndex,
    TArray<FTransform>& OutLocalPose,
    FString& OutError
)
{
    if (Retargeter == nullptr)
    {
        OutError = TEXT("No frozen Profile is open.");
        return false;
    }
    const skac_result Result = skac_retargeter_sample_frame(
        Retargeter,
        FrameIndex,
        TargetNativePose.GetData(),
        static_cast<size_t>(TargetNativePose.Num())
    );
    if (Result != SKAC_OK)
    {
        OutError = LastError();
        return false;
    }
    return CopyPose(TargetNativePose, OutLocalPose, OutError);
}

bool FSkacClip::SampleRetargetedTime(
    double TimeSeconds,
    bool bLoop,
    TArray<FTransform>& OutLocalPose,
    FString& OutError
)
{
    if (Retargeter == nullptr)
    {
        OutError = TEXT("No frozen Profile is open.");
        return false;
    }
    const skac_result Result = skac_retargeter_sample_time(
        Retargeter,
        TimeSeconds,
        bLoop ? SKAC_TIME_LOOP : SKAC_TIME_CLAMP,
        TargetNativePose.GetData(),
        static_cast<size_t>(TargetNativePose.Num())
    );
    if (Result != SKAC_OK)
    {
        OutError = LastError();
        return false;
    }
    return CopyPose(TargetNativePose, OutLocalPose, OutError);
}

skac_result FSkacClip::InflateZlib(
    const uint8_t* Compressed,
    size_t CompressedSize,
    uint8_t* Destination,
    size_t DestinationSize,
    void*
)
{
    const bool bInflated = FCompression::UncompressMemory(
        NAME_Zlib,
        Destination,
        static_cast<int64>(DestinationSize),
        Compressed,
        static_cast<int64>(CompressedSize)
    );
    return bInflated ? SKAC_OK : SKAC_DECOMPRESSION_FAILED;
}

bool FSkacClip::CopyPose(
    const TArray<skac_transform>& Source,
    TArray<FTransform>& OutLocalPose,
    FString& OutError
)
{
    OutLocalPose.SetNumUninitialized(Source.Num());
    for (int32 Index = 0; Index < Source.Num(); ++Index)
    {
        const skac_transform& Value = Source[Index];
        OutLocalPose[Index] = FTransform(
            FQuat(Value.rotation_x, Value.rotation_y, Value.rotation_z, Value.rotation_w),
            FVector(Value.translation_x, Value.translation_y, Value.translation_z)
        );
    }
    OutError.Reset();
    return true;
}

FString FSkacClip::LastError()
{
    return UTF8_TO_TCHAR(skac_runtime_last_error());
}
