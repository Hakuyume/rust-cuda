use cuda_sys;

use super::Error;

impl Into<cuda_sys::cudaError> for Error {
    fn into(self) -> cuda_sys::cudaError {
        match self {
            Error::MemoryAllocation => cuda_sys::cudaError_cudaErrorMemoryAllocation,
            Error::InitializationError => cuda_sys::cudaError_cudaErrorInitializationError,
            Error::LaunchFailure => cuda_sys::cudaError_cudaErrorLaunchFailure,
            Error::PriorLaunchFailure => cuda_sys::cudaError_cudaErrorPriorLaunchFailure,
            Error::LaunchTimeout => cuda_sys::cudaError_cudaErrorLaunchTimeout,
            Error::LaunchOutOfResources => cuda_sys::cudaError_cudaErrorLaunchOutOfResources,
            Error::InvalidDeviceFunction => cuda_sys::cudaError_cudaErrorInvalidDeviceFunction,
            Error::InvalidConfiguration => cuda_sys::cudaError_cudaErrorInvalidConfiguration,
            Error::InvalidDevice => cuda_sys::cudaError_cudaErrorInvalidDevice,
            Error::InvalidValue => cuda_sys::cudaError_cudaErrorInvalidValue,
            Error::InvalidPitchValue => cuda_sys::cudaError_cudaErrorInvalidPitchValue,
            Error::InvalidSymbol => cuda_sys::cudaError_cudaErrorInvalidSymbol,
            Error::MapBufferObjectFailed => cuda_sys::cudaError_cudaErrorMapBufferObjectFailed,
            Error::UnmapBufferObjectFailed => cuda_sys::cudaError_cudaErrorUnmapBufferObjectFailed,
            Error::InvalidHostPointer => cuda_sys::cudaError_cudaErrorInvalidHostPointer,
            Error::InvalidDevicePointer => cuda_sys::cudaError_cudaErrorInvalidDevicePointer,
            Error::InvalidTexture => cuda_sys::cudaError_cudaErrorInvalidTexture,
            Error::InvalidTextureBinding => cuda_sys::cudaError_cudaErrorInvalidTextureBinding,
            Error::InvalidChannelDescriptor => {
                cuda_sys::cudaError_cudaErrorInvalidChannelDescriptor
            }
            Error::InvalidMemcpyDirection => cuda_sys::cudaError_cudaErrorInvalidMemcpyDirection,
            Error::AddressOfConstant => cuda_sys::cudaError_cudaErrorAddressOfConstant,
            Error::TextureFetchFailed => cuda_sys::cudaError_cudaErrorTextureFetchFailed,
            Error::TextureNotBound => cuda_sys::cudaError_cudaErrorTextureNotBound,
            Error::SynchronizationError => cuda_sys::cudaError_cudaErrorSynchronizationError,
            Error::InvalidFilterSetting => cuda_sys::cudaError_cudaErrorInvalidFilterSetting,
            Error::InvalidNormSetting => cuda_sys::cudaError_cudaErrorInvalidNormSetting,
            Error::MixedDeviceExecution => cuda_sys::cudaError_cudaErrorMixedDeviceExecution,
            Error::CudartUnloading => cuda_sys::cudaError_cudaErrorCudartUnloading,
            Error::Unknown => cuda_sys::cudaError_cudaErrorUnknown,
            Error::NotYetImplemented => cuda_sys::cudaError_cudaErrorNotYetImplemented,
            Error::MemoryValueTooLarge => cuda_sys::cudaError_cudaErrorMemoryValueTooLarge,
            Error::InvalidResourceHandle => cuda_sys::cudaError_cudaErrorInvalidResourceHandle,
            Error::NotReady => cuda_sys::cudaError_cudaErrorNotReady,
            Error::InsufficientDriver => cuda_sys::cudaError_cudaErrorInsufficientDriver,
            Error::SetOnActiveProcess => cuda_sys::cudaError_cudaErrorSetOnActiveProcess,
            Error::InvalidSurface => cuda_sys::cudaError_cudaErrorInvalidSurface,
            Error::NoDevice => cuda_sys::cudaError_cudaErrorNoDevice,
            Error::ECCUncorrectable => cuda_sys::cudaError_cudaErrorECCUncorrectable,
            Error::SharedObjectSymbolNotFound => {
                cuda_sys::cudaError_cudaErrorSharedObjectSymbolNotFound
            }
            Error::SharedObjectInitFailed => cuda_sys::cudaError_cudaErrorSharedObjectInitFailed,
            Error::UnsupportedLimit => cuda_sys::cudaError_cudaErrorUnsupportedLimit,
            Error::DuplicateVariableName => cuda_sys::cudaError_cudaErrorDuplicateVariableName,
            Error::DuplicateTextureName => cuda_sys::cudaError_cudaErrorDuplicateTextureName,
            Error::DuplicateSurfaceName => cuda_sys::cudaError_cudaErrorDuplicateSurfaceName,
            Error::DevicesUnavailable => cuda_sys::cudaError_cudaErrorDevicesUnavailable,
            Error::InvalidKernelImage => cuda_sys::cudaError_cudaErrorInvalidKernelImage,
            Error::NoKernelImageForDevice => cuda_sys::cudaError_cudaErrorNoKernelImageForDevice,
            Error::IncompatibleDriverContext => {
                cuda_sys::cudaError_cudaErrorIncompatibleDriverContext
            }
            Error::PeerAccessAlreadyEnabled => {
                cuda_sys::cudaError_cudaErrorPeerAccessAlreadyEnabled
            }
            Error::PeerAccessNotEnabled => cuda_sys::cudaError_cudaErrorPeerAccessNotEnabled,
            Error::DeviceAlreadyInUse => cuda_sys::cudaError_cudaErrorDeviceAlreadyInUse,
            Error::ProfilerDisabled => cuda_sys::cudaError_cudaErrorProfilerDisabled,
            Error::ProfilerNotInitialized => cuda_sys::cudaError_cudaErrorProfilerNotInitialized,
            Error::ProfilerAlreadyStarted => cuda_sys::cudaError_cudaErrorProfilerAlreadyStarted,
            Error::ProfilerAlreadyStopped => cuda_sys::cudaError_cudaErrorProfilerAlreadyStopped,
            Error::Assert => cuda_sys::cudaError_cudaErrorAssert,
            Error::TooManyPeers => cuda_sys::cudaError_cudaErrorTooManyPeers,
            Error::HostMemoryAlreadyRegistered => {
                cuda_sys::cudaError_cudaErrorHostMemoryAlreadyRegistered
            }
            Error::HostMemoryNotRegistered => cuda_sys::cudaError_cudaErrorHostMemoryNotRegistered,
            Error::OperatingSystem => cuda_sys::cudaError_cudaErrorOperatingSystem,
            Error::PeerAccessUnsupported => cuda_sys::cudaError_cudaErrorPeerAccessUnsupported,
            Error::LaunchMaxDepthExceeded => cuda_sys::cudaError_cudaErrorLaunchMaxDepthExceeded,
            Error::LaunchFileScopedTex => cuda_sys::cudaError_cudaErrorLaunchFileScopedTex,
            Error::LaunchFileScopedSurf => cuda_sys::cudaError_cudaErrorLaunchFileScopedSurf,
            Error::SyncDepthExceeded => cuda_sys::cudaError_cudaErrorSyncDepthExceeded,
            Error::LaunchPendingCountExceeded => {
                cuda_sys::cudaError_cudaErrorLaunchPendingCountExceeded
            }
            Error::NotPermitted => cuda_sys::cudaError_cudaErrorNotPermitted,
            Error::NotSupported => cuda_sys::cudaError_cudaErrorNotSupported,
            Error::HardwareStackError => cuda_sys::cudaError_cudaErrorHardwareStackError,
            Error::IllegalInstruction => cuda_sys::cudaError_cudaErrorIllegalInstruction,
            Error::MisalignedAddress => cuda_sys::cudaError_cudaErrorMisalignedAddress,
            Error::InvalidAddressSpace => cuda_sys::cudaError_cudaErrorInvalidAddressSpace,
            Error::InvalidPc => cuda_sys::cudaError_cudaErrorInvalidPc,
            Error::IllegalAddress => cuda_sys::cudaError_cudaErrorIllegalAddress,
            Error::InvalidPtx => cuda_sys::cudaError_cudaErrorInvalidPtx,
            Error::InvalidGraphicsContext => cuda_sys::cudaError_cudaErrorInvalidGraphicsContext,
            Error::NvlinkUncorrectable => cuda_sys::cudaError_cudaErrorNvlinkUncorrectable,
            Error::StartupFailure => cuda_sys::cudaError_cudaErrorStartupFailure,
        }
    }
}
