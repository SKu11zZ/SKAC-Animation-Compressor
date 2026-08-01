using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace Skac.Runtime
{
    public enum SkacTimeMode : int
    {
        Clamp = 0,
        Loop = 1
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct SkacTransform
    {
        public float RotationX;
        public float RotationY;
        public float RotationZ;
        public float RotationW;
        public float TranslationX;
        public float TranslationY;
        public float TranslationZ;

        public Quaternion Rotation =>
            new Quaternion(RotationX, RotationY, RotationZ, RotationW);

        public Vector3 Translation =>
            new Vector3(TranslationX, TranslationY, TranslationZ);
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct SkacClipInfo
    {
        public uint AbiVersion;
        public uint FrameCount;
        public uint JointCount;
        public double FrameTimeSeconds;
        public double DurationSeconds;
    }

    public sealed class SkacClip : IDisposable
    {
        private const int Ok = 0;
        private IntPtr decoder;

        private SkacClip(IntPtr handle, SkacClipInfo info)
        {
            decoder = handle;
            Info = info;
        }

        public SkacClipInfo Info { get; }

        public static SkacClip Open(byte[] container)
        {
            if (container == null || container.Length == 0)
                throw new ArgumentException("The .skac container is empty.", nameof(container));

            IntPtr handle;
            ThrowIfFailed(Native.OpenMemory(container, (UIntPtr)container.Length, out handle));
            try
            {
                SkacClipInfo info;
                ThrowIfFailed(Native.GetInfo(handle, out info));
                return new SkacClip(handle, info);
            }
            catch
            {
                Native.Close(handle);
                throw;
            }
        }

        public string GetJointName(uint jointIndex)
        {
            EnsureOpen();
            IntPtr value;
            ThrowIfFailed(Native.GetJointName(decoder, jointIndex, out value));
            return Marshal.PtrToStringUTF8(value) ?? string.Empty;
        }

        public int GetJointParent(uint jointIndex)
        {
            EnsureOpen();
            int value;
            ThrowIfFailed(Native.GetJointParent(decoder, jointIndex, out value));
            return value;
        }

        public void SampleFrame(uint frameIndex, SkacTransform[] output)
        {
            EnsureOutput(output);
            ThrowIfFailed(Native.SampleFrame(
                decoder, frameIndex, output, (UIntPtr)output.Length));
        }

        public void SampleTime(double seconds, SkacTimeMode mode, SkacTransform[] output)
        {
            EnsureOutput(output);
            ThrowIfFailed(Native.SampleTime(
                decoder, seconds, mode, output, (UIntPtr)output.Length));
        }

        public void Dispose()
        {
            if (decoder == IntPtr.Zero)
                return;
            Native.Close(decoder);
            decoder = IntPtr.Zero;
            GC.SuppressFinalize(this);
        }

        ~SkacClip()
        {
            if (decoder != IntPtr.Zero)
                Native.Close(decoder);
        }

        private void EnsureOutput(SkacTransform[] output)
        {
            EnsureOpen();
            if (output == null || output.Length < Info.JointCount)
                throw new ArgumentException("Output must hold every joint transform.", nameof(output));
        }

        private void EnsureOpen()
        {
            if (decoder == IntPtr.Zero)
                throw new ObjectDisposedException(nameof(SkacClip));
        }

        private static void ThrowIfFailed(int result)
        {
            if (result == Ok)
                return;
            string message = Marshal.PtrToStringUTF8(Native.LastError()) ?? "Unknown native error.";
            throw new InvalidOperationException($"SKAC runtime error {result}: {message}");
        }

        private static class Native
        {
            private const string Library = "skac_runtime";

            [DllImport(Library, CallingConvention = CallingConvention.Cdecl,
                EntryPoint = "skac_decoder_open_memory")]
            internal static extern int OpenMemory(
                byte[] container, UIntPtr containerSize, out IntPtr decoder);

            [DllImport(Library, CallingConvention = CallingConvention.Cdecl,
                EntryPoint = "skac_decoder_close")]
            internal static extern void Close(IntPtr decoder);

            [DllImport(Library, CallingConvention = CallingConvention.Cdecl,
                EntryPoint = "skac_decoder_get_info")]
            internal static extern int GetInfo(IntPtr decoder, out SkacClipInfo info);

            [DllImport(Library, CallingConvention = CallingConvention.Cdecl,
                EntryPoint = "skac_decoder_get_joint_parent")]
            internal static extern int GetJointParent(
                IntPtr decoder, uint jointIndex, out int parentIndex);

            [DllImport(Library, CallingConvention = CallingConvention.Cdecl,
                EntryPoint = "skac_decoder_get_joint_name")]
            internal static extern int GetJointName(
                IntPtr decoder, uint jointIndex, out IntPtr utf8Name);

            [DllImport(Library, CallingConvention = CallingConvention.Cdecl,
                EntryPoint = "skac_decoder_sample_frame")]
            internal static extern int SampleFrame(
                IntPtr decoder,
                uint frameIndex,
                [Out] SkacTransform[] output,
                UIntPtr outputCapacity);

            [DllImport(Library, CallingConvention = CallingConvention.Cdecl,
                EntryPoint = "skac_decoder_sample_time")]
            internal static extern int SampleTime(
                IntPtr decoder,
                double seconds,
                SkacTimeMode mode,
                [Out] SkacTransform[] output,
                UIntPtr outputCapacity);

            [DllImport(Library, CallingConvention = CallingConvention.Cdecl,
                EntryPoint = "skac_runtime_last_error")]
            internal static extern IntPtr LastError();
        }
    }
}
