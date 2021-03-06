use cuda_sys;

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum Error {
    MemoryAllocation = cuda_sys::cudaErrorMemoryAllocation,
    InitializationError = cuda_sys::cudaErrorInitializationError,
    LaunchFailure = cuda_sys::cudaErrorLaunchFailure,
    PriorLaunchFailure = cuda_sys::cudaErrorPriorLaunchFailure,
    LaunchTimeout = cuda_sys::cudaErrorLaunchTimeout,
    LaunchOutOfResources = cuda_sys::cudaErrorLaunchOutOfResources,
    InvalidDeviceFunction = cuda_sys::cudaErrorInvalidDeviceFunction,
    InvalidConfiguration = cuda_sys::cudaErrorInvalidConfiguration,
    InvalidDevice = cuda_sys::cudaErrorInvalidDevice,
    InvalidValue = cuda_sys::cudaErrorInvalidValue,
    InvalidPitchValue = cuda_sys::cudaErrorInvalidPitchValue,
    InvalidSymbol = cuda_sys::cudaErrorInvalidSymbol,
    MapBufferObjectFailed = cuda_sys::cudaErrorMapBufferObjectFailed,
    UnmapBufferObjectFailed = cuda_sys::cudaErrorUnmapBufferObjectFailed,
    InvalidHostPointer = cuda_sys::cudaErrorInvalidHostPointer,
    InvalidDevicePointer = cuda_sys::cudaErrorInvalidDevicePointer,
    InvalidTexture = cuda_sys::cudaErrorInvalidTexture,
    InvalidTextureBinding = cuda_sys::cudaErrorInvalidTextureBinding,
    InvalidChannelDescriptor = cuda_sys::cudaErrorInvalidChannelDescriptor,
    InvalidMemcpyDirection = cuda_sys::cudaErrorInvalidMemcpyDirection,
    AddressOfConstant = cuda_sys::cudaErrorAddressOfConstant,
    TextureFetchFailed = cuda_sys::cudaErrorTextureFetchFailed,
    TextureNotBound = cuda_sys::cudaErrorTextureNotBound,
    SynchronizationError = cuda_sys::cudaErrorSynchronizationError,
    InvalidFilterSetting = cuda_sys::cudaErrorInvalidFilterSetting,
    InvalidNormSetting = cuda_sys::cudaErrorInvalidNormSetting,
    MixedDeviceExecution = cuda_sys::cudaErrorMixedDeviceExecution,
    CudartUnloading = cuda_sys::cudaErrorCudartUnloading,
    Unknown = cuda_sys::cudaErrorUnknown,
    NotYetImplemented = cuda_sys::cudaErrorNotYetImplemented,
    MemoryValueTooLarge = cuda_sys::cudaErrorMemoryValueTooLarge,
    InvalidResourceHandle = cuda_sys::cudaErrorInvalidResourceHandle,
    NotReady = cuda_sys::cudaErrorNotReady,
    InsufficientDriver = cuda_sys::cudaErrorInsufficientDriver,
    SetOnActiveProcess = cuda_sys::cudaErrorSetOnActiveProcess,
    InvalidSurface = cuda_sys::cudaErrorInvalidSurface,
    NoDevice = cuda_sys::cudaErrorNoDevice,
    ECCUncorrectable = cuda_sys::cudaErrorECCUncorrectable,
    SharedObjectSymbolNotFound = cuda_sys::cudaErrorSharedObjectSymbolNotFound,
    SharedObjectInitFailed = cuda_sys::cudaErrorSharedObjectInitFailed,
    UnsupportedLimit = cuda_sys::cudaErrorUnsupportedLimit,
    DuplicateVariableName = cuda_sys::cudaErrorDuplicateVariableName,
    DuplicateTextureName = cuda_sys::cudaErrorDuplicateTextureName,
    DuplicateSurfaceName = cuda_sys::cudaErrorDuplicateSurfaceName,
    DevicesUnavailable = cuda_sys::cudaErrorDevicesUnavailable,
    InvalidKernelImage = cuda_sys::cudaErrorInvalidKernelImage,
    NoKernelImageForDevice = cuda_sys::cudaErrorNoKernelImageForDevice,
    IncompatibleDriverContext = cuda_sys::cudaErrorIncompatibleDriverContext,
    PeerAccessAlreadyEnabled = cuda_sys::cudaErrorPeerAccessAlreadyEnabled,
    PeerAccessNotEnabled = cuda_sys::cudaErrorPeerAccessNotEnabled,
    DeviceAlreadyInUse = cuda_sys::cudaErrorDeviceAlreadyInUse,
    ProfilerDisabled = cuda_sys::cudaErrorProfilerDisabled,
    ProfilerNotInitialized = cuda_sys::cudaErrorProfilerNotInitialized,
    ProfilerAlreadyStarted = cuda_sys::cudaErrorProfilerAlreadyStarted,
    ProfilerAlreadyStopped = cuda_sys::cudaErrorProfilerAlreadyStopped,
    Assert = cuda_sys::cudaErrorAssert,
    TooManyPeers = cuda_sys::cudaErrorTooManyPeers,
    HostMemoryAlreadyRegistered = cuda_sys::cudaErrorHostMemoryAlreadyRegistered,
    HostMemoryNotRegistered = cuda_sys::cudaErrorHostMemoryNotRegistered,
    OperatingSystem = cuda_sys::cudaErrorOperatingSystem,
    PeerAccessUnsupported = cuda_sys::cudaErrorPeerAccessUnsupported,
    LaunchMaxDepthExceeded = cuda_sys::cudaErrorLaunchMaxDepthExceeded,
    LaunchFileScopedTex = cuda_sys::cudaErrorLaunchFileScopedTex,
    LaunchFileScopedSurf = cuda_sys::cudaErrorLaunchFileScopedSurf,
    SyncDepthExceeded = cuda_sys::cudaErrorSyncDepthExceeded,
    LaunchPendingCountExceeded = cuda_sys::cudaErrorLaunchPendingCountExceeded,
    NotPermitted = cuda_sys::cudaErrorNotPermitted,
    NotSupported = cuda_sys::cudaErrorNotSupported,
    HardwareStackError = cuda_sys::cudaErrorHardwareStackError,
    IllegalInstruction = cuda_sys::cudaErrorIllegalInstruction,
    MisalignedAddress = cuda_sys::cudaErrorMisalignedAddress,
    InvalidAddressSpace = cuda_sys::cudaErrorInvalidAddressSpace,
    InvalidPc = cuda_sys::cudaErrorInvalidPc,
    IllegalAddress = cuda_sys::cudaErrorIllegalAddress,
    InvalidPtx = cuda_sys::cudaErrorInvalidPtx,
    InvalidGraphicsContext = cuda_sys::cudaErrorInvalidGraphicsContext,
    NvlinkUncorrectable = cuda_sys::cudaErrorNvlinkUncorrectable,
    StartupFailure = cuda_sys::cudaErrorStartupFailure,
}
