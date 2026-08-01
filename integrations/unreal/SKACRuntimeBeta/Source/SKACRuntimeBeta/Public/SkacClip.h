#pragma once

#include "CoreMinimal.h"
#include "skac_runtime.h"

class SKACRUNTIMEBETA_API FSkacClip
{
public:
    FSkacClip() = default;
    ~FSkacClip();

    FSkacClip(const FSkacClip&) = delete;
    FSkacClip& operator=(const FSkacClip&) = delete;

    bool Open(const TArray<uint8>& Container, FString& OutError);
    bool OpenRetargeter(
        const TArray<uint8>& ProfileJson,
        const TArray<uint8>& TargetSkeletonJson,
        FString& OutError
    );
    void CloseRetargeter();
    void Close();
    bool IsOpen() const { return Decoder != nullptr; }
    bool HasRetargeter() const { return Retargeter != nullptr; }

    uint32 GetFrameCount() const { return Info.frame_count; }
    uint32 GetJointCount() const { return Info.joint_count; }
    double GetFrameTimeSeconds() const { return Info.frame_time_seconds; }
    double GetDurationSeconds() const { return Info.duration_seconds; }

    FString GetJointName(uint32 JointIndex) const;
    int32 GetJointParent(uint32 JointIndex) const;
    uint32 GetTargetJointCount() const { return RetargetInfo.target_joint_count; }
    FString GetTargetJointName(uint32 JointIndex) const;
    int32 GetTargetJointParent(uint32 JointIndex) const;

    bool SampleFrame(uint32 FrameIndex, TArray<FTransform>& OutLocalPose, FString& OutError);
    bool SampleTime(
        double TimeSeconds,
        bool bLoop,
        TArray<FTransform>& OutLocalPose,
        FString& OutError
    );
    bool SampleRetargetedFrame(
        uint32 FrameIndex,
        TArray<FTransform>& OutLocalPose,
        FString& OutError
    );
    bool SampleRetargetedTime(
        double TimeSeconds,
        bool bLoop,
        TArray<FTransform>& OutLocalPose,
        FString& OutError
    );

private:
    static skac_result InflateZlib(
        const uint8_t* Compressed,
        size_t CompressedSize,
        uint8_t* Destination,
        size_t DestinationSize,
        void* UserData
    );

    bool CopyPose(
        const TArray<skac_transform>& Source,
        TArray<FTransform>& OutLocalPose,
        FString& OutError
    );
    static FString LastError();

    skac_decoder* Decoder = nullptr;
    skac_retargeter* Retargeter = nullptr;
    skac_clip_info Info{};
    skac_retarget_info RetargetInfo{};
    TArray<skac_transform> NativePose;
    TArray<skac_transform> TargetNativePose;
};
