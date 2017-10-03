use std::result;

use cuda_sys::cudaError;

use super::Error;

pub trait TryFrom<T>: Sized {
    type Error;
    fn try_from(T) -> result::Result<Self, Self::Error>;
}

impl TryFrom<cudaError> for Error {
    type Error = ();
    fn try_from(value: cudaError) -> result::Result<Error, ()> {
        match value {
            cudaError::cudaSuccess => Err(()),
            cudaError::cudaErrorMemoryAllocation => Ok(Error::MemoryAllocation),
            cudaError::cudaErrorInitializationError => Ok(Error::InitializationError),
            cudaError::cudaErrorLaunchFailure => Ok(Error::LaunchFailure),
            cudaError::cudaErrorPriorLaunchFailure => Ok(Error::PriorLaunchFailure),
            cudaError::cudaErrorLaunchTimeout => Ok(Error::LaunchTimeout),
            cudaError::cudaErrorLaunchOutOfResources => Ok(Error::LaunchOutOfResources),
            cudaError::cudaErrorInvalidDeviceFunction => Ok(Error::InvalidDeviceFunction),
            cudaError::cudaErrorInvalidConfiguration => Ok(Error::InvalidConfiguration),
            cudaError::cudaErrorInvalidDevice => Ok(Error::InvalidDevice),
            cudaError::cudaErrorInvalidValue => Ok(Error::InvalidValue),
            cudaError::cudaErrorInvalidPitchValue => Ok(Error::InvalidPitchValue),
            cudaError::cudaErrorInvalidSymbol => Ok(Error::InvalidSymbol),
            cudaError::cudaErrorMapBufferObjectFailed => Ok(Error::MapBufferObjectFailed),
            cudaError::cudaErrorUnmapBufferObjectFailed => Ok(Error::UnmapBufferObjectFailed),
            cudaError::cudaErrorInvalidHostPointer => Ok(Error::InvalidHostPointer),
            cudaError::cudaErrorInvalidDevicePointer => Ok(Error::InvalidDevicePointer),
            cudaError::cudaErrorInvalidTexture => Ok(Error::InvalidTexture),
            cudaError::cudaErrorInvalidTextureBinding => Ok(Error::InvalidTextureBinding),
            cudaError::cudaErrorInvalidChannelDescriptor => Ok(Error::InvalidChannelDescriptor),
            cudaError::cudaErrorInvalidMemcpyDirection => Ok(Error::InvalidMemcpyDirection),
            cudaError::cudaErrorAddressOfConstant => Ok(Error::AddressOfConstant),
            cudaError::cudaErrorTextureFetchFailed => Ok(Error::TextureFetchFailed),
            cudaError::cudaErrorTextureNotBound => Ok(Error::TextureNotBound),
            cudaError::cudaErrorSynchronizationError => Ok(Error::SynchronizationError),
            cudaError::cudaErrorInvalidFilterSetting => Ok(Error::InvalidFilterSetting),
            cudaError::cudaErrorInvalidNormSetting => Ok(Error::InvalidNormSetting),
            cudaError::cudaErrorMixedDeviceExecution => Ok(Error::MixedDeviceExecution),
            cudaError::cudaErrorCudartUnloading => Ok(Error::CudartUnloading),
            cudaError::cudaErrorUnknown => Ok(Error::Unknown),
            cudaError::cudaErrorNotYetImplemented => Ok(Error::NotYetImplemented),
            cudaError::cudaErrorMemoryValueTooLarge => Ok(Error::MemoryValueTooLarge),
            cudaError::cudaErrorInvalidResourceHandle => Ok(Error::InvalidResourceHandle),
            cudaError::cudaErrorNotReady => Ok(Error::NotReady),
            cudaError::cudaErrorInsufficientDriver => Ok(Error::InsufficientDriver),
            cudaError::cudaErrorSetOnActiveProcess => Ok(Error::SetOnActiveProcess),
            cudaError::cudaErrorInvalidSurface => Ok(Error::InvalidSurface),
            cudaError::cudaErrorNoDevice => Ok(Error::NoDevice),
            cudaError::cudaErrorECCUncorrectable => Ok(Error::ECCUncorrectable),
            cudaError::cudaErrorSharedObjectSymbolNotFound => Ok(Error::SharedObjectSymbolNotFound),
            cudaError::cudaErrorSharedObjectInitFailed => Ok(Error::SharedObjectInitFailed),
            cudaError::cudaErrorUnsupportedLimit => Ok(Error::UnsupportedLimit),
            cudaError::cudaErrorDuplicateVariableName => Ok(Error::DuplicateVariableName),
            cudaError::cudaErrorDuplicateTextureName => Ok(Error::DuplicateTextureName),
            cudaError::cudaErrorDuplicateSurfaceName => Ok(Error::DuplicateSurfaceName),
            cudaError::cudaErrorDevicesUnavailable => Ok(Error::DevicesUnavailable),
            cudaError::cudaErrorInvalidKernelImage => Ok(Error::InvalidKernelImage),
            cudaError::cudaErrorNoKernelImageForDevice => Ok(Error::NoKernelImageForDevice),
            cudaError::cudaErrorIncompatibleDriverContext => Ok(Error::IncompatibleDriverContext),
            cudaError::cudaErrorPeerAccessAlreadyEnabled => Ok(Error::PeerAccessAlreadyEnabled),
            cudaError::cudaErrorPeerAccessNotEnabled => Ok(Error::PeerAccessNotEnabled),
            cudaError::cudaErrorDeviceAlreadyInUse => Ok(Error::DeviceAlreadyInUse),
            cudaError::cudaErrorProfilerDisabled => Ok(Error::ProfilerDisabled),
            cudaError::cudaErrorProfilerNotInitialized => Ok(Error::ProfilerNotInitialized),
            cudaError::cudaErrorProfilerAlreadyStarted => Ok(Error::ProfilerAlreadyStarted),
            cudaError::cudaErrorProfilerAlreadyStopped => Ok(Error::ProfilerAlreadyStopped),
            cudaError::cudaErrorAssert => Ok(Error::Assert),
            cudaError::cudaErrorTooManyPeers => Ok(Error::TooManyPeers),
            cudaError::cudaErrorHostMemoryAlreadyRegistered => {
                Ok(Error::HostMemoryAlreadyRegistered)
            }
            cudaError::cudaErrorHostMemoryNotRegistered => Ok(Error::HostMemoryNotRegistered),
            cudaError::cudaErrorOperatingSystem => Ok(Error::OperatingSystem),
            cudaError::cudaErrorPeerAccessUnsupported => Ok(Error::PeerAccessUnsupported),
            cudaError::cudaErrorLaunchMaxDepthExceeded => Ok(Error::LaunchMaxDepthExceeded),
            cudaError::cudaErrorLaunchFileScopedTex => Ok(Error::LaunchFileScopedTex),
            cudaError::cudaErrorLaunchFileScopedSurf => Ok(Error::LaunchFileScopedSurf),
            cudaError::cudaErrorSyncDepthExceeded => Ok(Error::SyncDepthExceeded),
            cudaError::cudaErrorLaunchPendingCountExceeded => Ok(Error::LaunchPendingCountExceeded),
            cudaError::cudaErrorNotPermitted => Ok(Error::NotPermitted),
            cudaError::cudaErrorNotSupported => Ok(Error::NotSupported),
            cudaError::cudaErrorHardwareStackError => Ok(Error::HardwareStackError),
            cudaError::cudaErrorIllegalInstruction => Ok(Error::IllegalInstruction),
            cudaError::cudaErrorMisalignedAddress => Ok(Error::MisalignedAddress),
            cudaError::cudaErrorInvalidAddressSpace => Ok(Error::InvalidAddressSpace),
            cudaError::cudaErrorInvalidPc => Ok(Error::InvalidPc),
            cudaError::cudaErrorIllegalAddress => Ok(Error::IllegalAddress),
            cudaError::cudaErrorInvalidPtx => Ok(Error::InvalidPtx),
            cudaError::cudaErrorInvalidGraphicsContext => Ok(Error::InvalidGraphicsContext),
            cudaError::cudaErrorNvlinkUncorrectable => Ok(Error::NvlinkUncorrectable),
            cudaError::cudaErrorStartupFailure => Ok(Error::StartupFailure),
        }
    }
}
