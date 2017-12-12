use std::result;

use cuda_sys;

use super::Error;

pub trait TryFrom<T>: Sized {
    type Error;
    fn try_from(T) -> result::Result<Self, Self::Error>;
}

impl TryFrom<cuda_sys::cudaError> for Error {
    type Error = ();
    fn try_from(value: cuda_sys::cudaError) -> result::Result<Error, ()> {
        match value {
            cuda_sys::cudaSuccess => Err(()),
            cuda_sys::cudaErrorMemoryAllocation => Ok(Error::MemoryAllocation),
            cuda_sys::cudaErrorInitializationError => Ok(Error::InitializationError),
            cuda_sys::cudaErrorLaunchFailure => Ok(Error::LaunchFailure),
            cuda_sys::cudaErrorPriorLaunchFailure => Ok(Error::PriorLaunchFailure),
            cuda_sys::cudaErrorLaunchTimeout => Ok(Error::LaunchTimeout),
            cuda_sys::cudaErrorLaunchOutOfResources => Ok(Error::LaunchOutOfResources),
            cuda_sys::cudaErrorInvalidDeviceFunction => Ok(Error::InvalidDeviceFunction),
            cuda_sys::cudaErrorInvalidConfiguration => Ok(Error::InvalidConfiguration),
            cuda_sys::cudaErrorInvalidDevice => Ok(Error::InvalidDevice),
            cuda_sys::cudaErrorInvalidValue => Ok(Error::InvalidValue),
            cuda_sys::cudaErrorInvalidPitchValue => Ok(Error::InvalidPitchValue),
            cuda_sys::cudaErrorInvalidSymbol => Ok(Error::InvalidSymbol),
            cuda_sys::cudaErrorMapBufferObjectFailed => Ok(Error::MapBufferObjectFailed),
            cuda_sys::cudaErrorUnmapBufferObjectFailed => Ok(Error::UnmapBufferObjectFailed),
            cuda_sys::cudaErrorInvalidHostPointer => Ok(Error::InvalidHostPointer),
            cuda_sys::cudaErrorInvalidDevicePointer => Ok(Error::InvalidDevicePointer),
            cuda_sys::cudaErrorInvalidTexture => Ok(Error::InvalidTexture),
            cuda_sys::cudaErrorInvalidTextureBinding => Ok(Error::InvalidTextureBinding),
            cuda_sys::cudaErrorInvalidChannelDescriptor => Ok(Error::InvalidChannelDescriptor),
            cuda_sys::cudaErrorInvalidMemcpyDirection => Ok(Error::InvalidMemcpyDirection),
            cuda_sys::cudaErrorAddressOfConstant => Ok(Error::AddressOfConstant),
            cuda_sys::cudaErrorTextureFetchFailed => Ok(Error::TextureFetchFailed),
            cuda_sys::cudaErrorTextureNotBound => Ok(Error::TextureNotBound),
            cuda_sys::cudaErrorSynchronizationError => Ok(Error::SynchronizationError),
            cuda_sys::cudaErrorInvalidFilterSetting => Ok(Error::InvalidFilterSetting),
            cuda_sys::cudaErrorInvalidNormSetting => Ok(Error::InvalidNormSetting),
            cuda_sys::cudaErrorMixedDeviceExecution => Ok(Error::MixedDeviceExecution),
            cuda_sys::cudaErrorCudartUnloading => Ok(Error::CudartUnloading),
            cuda_sys::cudaErrorUnknown => Ok(Error::Unknown),
            cuda_sys::cudaErrorNotYetImplemented => Ok(Error::NotYetImplemented),
            cuda_sys::cudaErrorMemoryValueTooLarge => Ok(Error::MemoryValueTooLarge),
            cuda_sys::cudaErrorInvalidResourceHandle => Ok(Error::InvalidResourceHandle),
            cuda_sys::cudaErrorNotReady => Ok(Error::NotReady),
            cuda_sys::cudaErrorInsufficientDriver => Ok(Error::InsufficientDriver),
            cuda_sys::cudaErrorSetOnActiveProcess => Ok(Error::SetOnActiveProcess),
            cuda_sys::cudaErrorInvalidSurface => Ok(Error::InvalidSurface),
            cuda_sys::cudaErrorNoDevice => Ok(Error::NoDevice),
            cuda_sys::cudaErrorECCUncorrectable => Ok(Error::ECCUncorrectable),
            cuda_sys::cudaErrorSharedObjectSymbolNotFound => Ok(Error::SharedObjectSymbolNotFound),
            cuda_sys::cudaErrorSharedObjectInitFailed => Ok(Error::SharedObjectInitFailed),
            cuda_sys::cudaErrorUnsupportedLimit => Ok(Error::UnsupportedLimit),
            cuda_sys::cudaErrorDuplicateVariableName => Ok(Error::DuplicateVariableName),
            cuda_sys::cudaErrorDuplicateTextureName => Ok(Error::DuplicateTextureName),
            cuda_sys::cudaErrorDuplicateSurfaceName => Ok(Error::DuplicateSurfaceName),
            cuda_sys::cudaErrorDevicesUnavailable => Ok(Error::DevicesUnavailable),
            cuda_sys::cudaErrorInvalidKernelImage => Ok(Error::InvalidKernelImage),
            cuda_sys::cudaErrorNoKernelImageForDevice => Ok(Error::NoKernelImageForDevice),
            cuda_sys::cudaErrorIncompatibleDriverContext => Ok(Error::IncompatibleDriverContext),
            cuda_sys::cudaErrorPeerAccessAlreadyEnabled => Ok(Error::PeerAccessAlreadyEnabled),
            cuda_sys::cudaErrorPeerAccessNotEnabled => Ok(Error::PeerAccessNotEnabled),
            cuda_sys::cudaErrorDeviceAlreadyInUse => Ok(Error::DeviceAlreadyInUse),
            cuda_sys::cudaErrorProfilerDisabled => Ok(Error::ProfilerDisabled),
            cuda_sys::cudaErrorProfilerNotInitialized => Ok(Error::ProfilerNotInitialized),
            cuda_sys::cudaErrorProfilerAlreadyStarted => Ok(Error::ProfilerAlreadyStarted),
            cuda_sys::cudaErrorProfilerAlreadyStopped => Ok(Error::ProfilerAlreadyStopped),
            cuda_sys::cudaErrorAssert => Ok(Error::Assert),
            cuda_sys::cudaErrorTooManyPeers => Ok(Error::TooManyPeers),
            cuda_sys::cudaErrorHostMemoryAlreadyRegistered => {
                Ok(Error::HostMemoryAlreadyRegistered)
            }
            cuda_sys::cudaErrorHostMemoryNotRegistered => Ok(Error::HostMemoryNotRegistered),
            cuda_sys::cudaErrorOperatingSystem => Ok(Error::OperatingSystem),
            cuda_sys::cudaErrorPeerAccessUnsupported => Ok(Error::PeerAccessUnsupported),
            cuda_sys::cudaErrorLaunchMaxDepthExceeded => Ok(Error::LaunchMaxDepthExceeded),
            cuda_sys::cudaErrorLaunchFileScopedTex => Ok(Error::LaunchFileScopedTex),
            cuda_sys::cudaErrorLaunchFileScopedSurf => Ok(Error::LaunchFileScopedSurf),
            cuda_sys::cudaErrorSyncDepthExceeded => Ok(Error::SyncDepthExceeded),
            cuda_sys::cudaErrorLaunchPendingCountExceeded => Ok(Error::LaunchPendingCountExceeded),
            cuda_sys::cudaErrorNotPermitted => Ok(Error::NotPermitted),
            cuda_sys::cudaErrorNotSupported => Ok(Error::NotSupported),
            cuda_sys::cudaErrorHardwareStackError => Ok(Error::HardwareStackError),
            cuda_sys::cudaErrorIllegalInstruction => Ok(Error::IllegalInstruction),
            cuda_sys::cudaErrorMisalignedAddress => Ok(Error::MisalignedAddress),
            cuda_sys::cudaErrorInvalidAddressSpace => Ok(Error::InvalidAddressSpace),
            cuda_sys::cudaErrorInvalidPc => Ok(Error::InvalidPc),
            cuda_sys::cudaErrorIllegalAddress => Ok(Error::IllegalAddress),
            cuda_sys::cudaErrorInvalidPtx => Ok(Error::InvalidPtx),
            cuda_sys::cudaErrorInvalidGraphicsContext => Ok(Error::InvalidGraphicsContext),
            cuda_sys::cudaErrorNvlinkUncorrectable => Ok(Error::NvlinkUncorrectable),
            cuda_sys::cudaErrorStartupFailure => Ok(Error::StartupFailure),
            _ => unreachable!(),
        }
    }
}
