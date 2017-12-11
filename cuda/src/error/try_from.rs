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
            cuda_sys::cudaError_cudaSuccess => Err(()),
            cuda_sys::cudaError_cudaErrorMemoryAllocation => Ok(Error::MemoryAllocation),
            cuda_sys::cudaError_cudaErrorInitializationError => Ok(Error::InitializationError),
            cuda_sys::cudaError_cudaErrorLaunchFailure => Ok(Error::LaunchFailure),
            cuda_sys::cudaError_cudaErrorPriorLaunchFailure => Ok(Error::PriorLaunchFailure),
            cuda_sys::cudaError_cudaErrorLaunchTimeout => Ok(Error::LaunchTimeout),
            cuda_sys::cudaError_cudaErrorLaunchOutOfResources => Ok(Error::LaunchOutOfResources),
            cuda_sys::cudaError_cudaErrorInvalidDeviceFunction => Ok(Error::InvalidDeviceFunction),
            cuda_sys::cudaError_cudaErrorInvalidConfiguration => Ok(Error::InvalidConfiguration),
            cuda_sys::cudaError_cudaErrorInvalidDevice => Ok(Error::InvalidDevice),
            cuda_sys::cudaError_cudaErrorInvalidValue => Ok(Error::InvalidValue),
            cuda_sys::cudaError_cudaErrorInvalidPitchValue => Ok(Error::InvalidPitchValue),
            cuda_sys::cudaError_cudaErrorInvalidSymbol => Ok(Error::InvalidSymbol),
            cuda_sys::cudaError_cudaErrorMapBufferObjectFailed => Ok(Error::MapBufferObjectFailed),
            cuda_sys::cudaError_cudaErrorUnmapBufferObjectFailed => {
                Ok(Error::UnmapBufferObjectFailed)
            }
            cuda_sys::cudaError_cudaErrorInvalidHostPointer => Ok(Error::InvalidHostPointer),
            cuda_sys::cudaError_cudaErrorInvalidDevicePointer => Ok(Error::InvalidDevicePointer),
            cuda_sys::cudaError_cudaErrorInvalidTexture => Ok(Error::InvalidTexture),
            cuda_sys::cudaError_cudaErrorInvalidTextureBinding => Ok(Error::InvalidTextureBinding),
            cuda_sys::cudaError_cudaErrorInvalidChannelDescriptor => {
                Ok(Error::InvalidChannelDescriptor)
            }
            cuda_sys::cudaError_cudaErrorInvalidMemcpyDirection => {
                Ok(Error::InvalidMemcpyDirection)
            }
            cuda_sys::cudaError_cudaErrorAddressOfConstant => Ok(Error::AddressOfConstant),
            cuda_sys::cudaError_cudaErrorTextureFetchFailed => Ok(Error::TextureFetchFailed),
            cuda_sys::cudaError_cudaErrorTextureNotBound => Ok(Error::TextureNotBound),
            cuda_sys::cudaError_cudaErrorSynchronizationError => Ok(Error::SynchronizationError),
            cuda_sys::cudaError_cudaErrorInvalidFilterSetting => Ok(Error::InvalidFilterSetting),
            cuda_sys::cudaError_cudaErrorInvalidNormSetting => Ok(Error::InvalidNormSetting),
            cuda_sys::cudaError_cudaErrorMixedDeviceExecution => Ok(Error::MixedDeviceExecution),
            cuda_sys::cudaError_cudaErrorCudartUnloading => Ok(Error::CudartUnloading),
            cuda_sys::cudaError_cudaErrorUnknown => Ok(Error::Unknown),
            cuda_sys::cudaError_cudaErrorNotYetImplemented => Ok(Error::NotYetImplemented),
            cuda_sys::cudaError_cudaErrorMemoryValueTooLarge => Ok(Error::MemoryValueTooLarge),
            cuda_sys::cudaError_cudaErrorInvalidResourceHandle => Ok(Error::InvalidResourceHandle),
            cuda_sys::cudaError_cudaErrorNotReady => Ok(Error::NotReady),
            cuda_sys::cudaError_cudaErrorInsufficientDriver => Ok(Error::InsufficientDriver),
            cuda_sys::cudaError_cudaErrorSetOnActiveProcess => Ok(Error::SetOnActiveProcess),
            cuda_sys::cudaError_cudaErrorInvalidSurface => Ok(Error::InvalidSurface),
            cuda_sys::cudaError_cudaErrorNoDevice => Ok(Error::NoDevice),
            cuda_sys::cudaError_cudaErrorECCUncorrectable => Ok(Error::ECCUncorrectable),
            cuda_sys::cudaError_cudaErrorSharedObjectSymbolNotFound => {
                Ok(Error::SharedObjectSymbolNotFound)
            }
            cuda_sys::cudaError_cudaErrorSharedObjectInitFailed => {
                Ok(Error::SharedObjectInitFailed)
            }
            cuda_sys::cudaError_cudaErrorUnsupportedLimit => Ok(Error::UnsupportedLimit),
            cuda_sys::cudaError_cudaErrorDuplicateVariableName => Ok(Error::DuplicateVariableName),
            cuda_sys::cudaError_cudaErrorDuplicateTextureName => Ok(Error::DuplicateTextureName),
            cuda_sys::cudaError_cudaErrorDuplicateSurfaceName => Ok(Error::DuplicateSurfaceName),
            cuda_sys::cudaError_cudaErrorDevicesUnavailable => Ok(Error::DevicesUnavailable),
            cuda_sys::cudaError_cudaErrorInvalidKernelImage => Ok(Error::InvalidKernelImage),
            cuda_sys::cudaError_cudaErrorNoKernelImageForDevice => {
                Ok(Error::NoKernelImageForDevice)
            }
            cuda_sys::cudaError_cudaErrorIncompatibleDriverContext => {
                Ok(Error::IncompatibleDriverContext)
            }
            cuda_sys::cudaError_cudaErrorPeerAccessAlreadyEnabled => {
                Ok(Error::PeerAccessAlreadyEnabled)
            }
            cuda_sys::cudaError_cudaErrorPeerAccessNotEnabled => Ok(Error::PeerAccessNotEnabled),
            cuda_sys::cudaError_cudaErrorDeviceAlreadyInUse => Ok(Error::DeviceAlreadyInUse),
            cuda_sys::cudaError_cudaErrorProfilerDisabled => Ok(Error::ProfilerDisabled),
            cuda_sys::cudaError_cudaErrorProfilerNotInitialized => {
                Ok(Error::ProfilerNotInitialized)
            }
            cuda_sys::cudaError_cudaErrorProfilerAlreadyStarted => {
                Ok(Error::ProfilerAlreadyStarted)
            }
            cuda_sys::cudaError_cudaErrorProfilerAlreadyStopped => {
                Ok(Error::ProfilerAlreadyStopped)
            }
            cuda_sys::cudaError_cudaErrorAssert => Ok(Error::Assert),
            cuda_sys::cudaError_cudaErrorTooManyPeers => Ok(Error::TooManyPeers),
            cuda_sys::cudaError_cudaErrorHostMemoryAlreadyRegistered => {
                Ok(Error::HostMemoryAlreadyRegistered)
            }
            cuda_sys::cudaError_cudaErrorHostMemoryNotRegistered => {
                Ok(Error::HostMemoryNotRegistered)
            }
            cuda_sys::cudaError_cudaErrorOperatingSystem => Ok(Error::OperatingSystem),
            cuda_sys::cudaError_cudaErrorPeerAccessUnsupported => Ok(Error::PeerAccessUnsupported),
            cuda_sys::cudaError_cudaErrorLaunchMaxDepthExceeded => {
                Ok(Error::LaunchMaxDepthExceeded)
            }
            cuda_sys::cudaError_cudaErrorLaunchFileScopedTex => Ok(Error::LaunchFileScopedTex),
            cuda_sys::cudaError_cudaErrorLaunchFileScopedSurf => Ok(Error::LaunchFileScopedSurf),
            cuda_sys::cudaError_cudaErrorSyncDepthExceeded => Ok(Error::SyncDepthExceeded),
            cuda_sys::cudaError_cudaErrorLaunchPendingCountExceeded => {
                Ok(Error::LaunchPendingCountExceeded)
            }
            cuda_sys::cudaError_cudaErrorNotPermitted => Ok(Error::NotPermitted),
            cuda_sys::cudaError_cudaErrorNotSupported => Ok(Error::NotSupported),
            cuda_sys::cudaError_cudaErrorHardwareStackError => Ok(Error::HardwareStackError),
            cuda_sys::cudaError_cudaErrorIllegalInstruction => Ok(Error::IllegalInstruction),
            cuda_sys::cudaError_cudaErrorMisalignedAddress => Ok(Error::MisalignedAddress),
            cuda_sys::cudaError_cudaErrorInvalidAddressSpace => Ok(Error::InvalidAddressSpace),
            cuda_sys::cudaError_cudaErrorInvalidPc => Ok(Error::InvalidPc),
            cuda_sys::cudaError_cudaErrorIllegalAddress => Ok(Error::IllegalAddress),
            cuda_sys::cudaError_cudaErrorInvalidPtx => Ok(Error::InvalidPtx),
            cuda_sys::cudaError_cudaErrorInvalidGraphicsContext => {
                Ok(Error::InvalidGraphicsContext)
            }
            cuda_sys::cudaError_cudaErrorNvlinkUncorrectable => Ok(Error::NvlinkUncorrectable),
            cuda_sys::cudaError_cudaErrorStartupFailure => Ok(Error::StartupFailure),
            _ => unreachable!(),
        }
    }
}
