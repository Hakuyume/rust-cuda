use std::result;

use cuda_sys;
use cuda_sys::cudaError::*;

use super::Error;

pub trait TryFrom<T>: Sized {
    type Error;
    fn try_from(T) -> result::Result<Self, Self::Error>;
}

impl TryFrom<cuda_sys::cudaError> for Error {
    type Error = ();
    fn try_from(value: cuda_sys::cudaError) -> result::Result<Error, ()> {
        match value {
            cudaSuccess => Err(()),
            cudaErrorMemoryAllocation => Ok(Error::MemoryAllocation),
            cudaErrorInitializationError => Ok(Error::InitializationError),
            cudaErrorLaunchFailure => Ok(Error::LaunchFailure),
            cudaErrorPriorLaunchFailure => Ok(Error::PriorLaunchFailure),
            cudaErrorLaunchTimeout => Ok(Error::LaunchTimeout),
            cudaErrorLaunchOutOfResources => Ok(Error::LaunchOutOfResources),
            cudaErrorInvalidDeviceFunction => Ok(Error::InvalidDeviceFunction),
            cudaErrorInvalidConfiguration => Ok(Error::InvalidConfiguration),
            cudaErrorInvalidDevice => Ok(Error::InvalidDevice),
            cudaErrorInvalidValue => Ok(Error::InvalidValue),
            cudaErrorInvalidPitchValue => Ok(Error::InvalidPitchValue),
            cudaErrorInvalidSymbol => Ok(Error::InvalidSymbol),
            cudaErrorMapBufferObjectFailed => Ok(Error::MapBufferObjectFailed),
            cudaErrorUnmapBufferObjectFailed => Ok(Error::UnmapBufferObjectFailed),
            cudaErrorInvalidHostPointer => Ok(Error::InvalidHostPointer),
            cudaErrorInvalidDevicePointer => Ok(Error::InvalidDevicePointer),
            cudaErrorInvalidTexture => Ok(Error::InvalidTexture),
            cudaErrorInvalidTextureBinding => Ok(Error::InvalidTextureBinding),
            cudaErrorInvalidChannelDescriptor => Ok(Error::InvalidChannelDescriptor),
            cudaErrorInvalidMemcpyDirection => Ok(Error::InvalidMemcpyDirection),
            cudaErrorAddressOfConstant => Ok(Error::AddressOfConstant),
            cudaErrorTextureFetchFailed => Ok(Error::TextureFetchFailed),
            cudaErrorTextureNotBound => Ok(Error::TextureNotBound),
            cudaErrorSynchronizationError => Ok(Error::SynchronizationError),
            cudaErrorInvalidFilterSetting => Ok(Error::InvalidFilterSetting),
            cudaErrorInvalidNormSetting => Ok(Error::InvalidNormSetting),
            cudaErrorMixedDeviceExecution => Ok(Error::MixedDeviceExecution),
            cudaErrorCudartUnloading => Ok(Error::CudartUnloading),
            cudaErrorUnknown => Ok(Error::Unknown),
            cudaErrorNotYetImplemented => Ok(Error::NotYetImplemented),
            cudaErrorMemoryValueTooLarge => Ok(Error::MemoryValueTooLarge),
            cudaErrorInvalidResourceHandle => Ok(Error::InvalidResourceHandle),
            cudaErrorNotReady => Ok(Error::NotReady),
            cudaErrorInsufficientDriver => Ok(Error::InsufficientDriver),
            cudaErrorSetOnActiveProcess => Ok(Error::SetOnActiveProcess),
            cudaErrorInvalidSurface => Ok(Error::InvalidSurface),
            cudaErrorNoDevice => Ok(Error::NoDevice),
            cudaErrorECCUncorrectable => Ok(Error::ECCUncorrectable),
            cudaErrorSharedObjectSymbolNotFound => Ok(Error::SharedObjectSymbolNotFound),
            cudaErrorSharedObjectInitFailed => Ok(Error::SharedObjectInitFailed),
            cudaErrorUnsupportedLimit => Ok(Error::UnsupportedLimit),
            cudaErrorDuplicateVariableName => Ok(Error::DuplicateVariableName),
            cudaErrorDuplicateTextureName => Ok(Error::DuplicateTextureName),
            cudaErrorDuplicateSurfaceName => Ok(Error::DuplicateSurfaceName),
            cudaErrorDevicesUnavailable => Ok(Error::DevicesUnavailable),
            cudaErrorInvalidKernelImage => Ok(Error::InvalidKernelImage),
            cudaErrorNoKernelImageForDevice => Ok(Error::NoKernelImageForDevice),
            cudaErrorIncompatibleDriverContext => Ok(Error::IncompatibleDriverContext),
            cudaErrorPeerAccessAlreadyEnabled => Ok(Error::PeerAccessAlreadyEnabled),
            cudaErrorPeerAccessNotEnabled => Ok(Error::PeerAccessNotEnabled),
            cudaErrorDeviceAlreadyInUse => Ok(Error::DeviceAlreadyInUse),
            cudaErrorProfilerDisabled => Ok(Error::ProfilerDisabled),
            cudaErrorProfilerNotInitialized => Ok(Error::ProfilerNotInitialized),
            cudaErrorProfilerAlreadyStarted => Ok(Error::ProfilerAlreadyStarted),
            cudaErrorProfilerAlreadyStopped => Ok(Error::ProfilerAlreadyStopped),
            cudaErrorAssert => Ok(Error::Assert),
            cudaErrorTooManyPeers => Ok(Error::TooManyPeers),
            cudaErrorHostMemoryAlreadyRegistered => Ok(Error::HostMemoryAlreadyRegistered),
            cudaErrorHostMemoryNotRegistered => Ok(Error::HostMemoryNotRegistered),
            cudaErrorOperatingSystem => Ok(Error::OperatingSystem),
            cudaErrorPeerAccessUnsupported => Ok(Error::PeerAccessUnsupported),
            cudaErrorLaunchMaxDepthExceeded => Ok(Error::LaunchMaxDepthExceeded),
            cudaErrorLaunchFileScopedTex => Ok(Error::LaunchFileScopedTex),
            cudaErrorLaunchFileScopedSurf => Ok(Error::LaunchFileScopedSurf),
            cudaErrorSyncDepthExceeded => Ok(Error::SyncDepthExceeded),
            cudaErrorLaunchPendingCountExceeded => Ok(Error::LaunchPendingCountExceeded),
            cudaErrorNotPermitted => Ok(Error::NotPermitted),
            cudaErrorNotSupported => Ok(Error::NotSupported),
            cudaErrorHardwareStackError => Ok(Error::HardwareStackError),
            cudaErrorIllegalInstruction => Ok(Error::IllegalInstruction),
            cudaErrorMisalignedAddress => Ok(Error::MisalignedAddress),
            cudaErrorInvalidAddressSpace => Ok(Error::InvalidAddressSpace),
            cudaErrorInvalidPc => Ok(Error::InvalidPc),
            cudaErrorIllegalAddress => Ok(Error::IllegalAddress),
            cudaErrorInvalidPtx => Ok(Error::InvalidPtx),
            cudaErrorInvalidGraphicsContext => Ok(Error::InvalidGraphicsContext),
            cudaErrorNvlinkUncorrectable => Ok(Error::NvlinkUncorrectable),
            cudaErrorStartupFailure => Ok(Error::StartupFailure),
        }
    }
}
