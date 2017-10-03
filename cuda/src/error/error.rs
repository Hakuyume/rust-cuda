#[derive(Clone, Copy, Debug)]
pub enum Error {
    MemoryAllocation,
    InitializationError,
    LaunchFailure,
    PriorLaunchFailure,
    LaunchTimeout,
    LaunchOutOfResources,
    InvalidDeviceFunction,
    InvalidConfiguration,
    InvalidDevice,
    InvalidValue,
    InvalidPitchValue,
    InvalidSymbol,
    MapBufferObjectFailed,
    UnmapBufferObjectFailed,
    InvalidHostPointer,
    InvalidDevicePointer,
    InvalidTexture,
    InvalidTextureBinding,
    InvalidChannelDescriptor,
    InvalidMemcpyDirection,
    AddressOfConstant,
    TextureFetchFailed,
    TextureNotBound,
    SynchronizationError,
    InvalidFilterSetting,
    InvalidNormSetting,
    MixedDeviceExecution,
    CudartUnloading,
    Unknown,
    NotYetImplemented,
    MemoryValueTooLarge,
    InvalidResourceHandle,
    NotReady,
    InsufficientDriver,
    SetOnActiveProcess,
    InvalidSurface,
    NoDevice,
    ECCUncorrectable,
    SharedObjectSymbolNotFound,
    SharedObjectInitFailed,
    UnsupportedLimit,
    DuplicateVariableName,
    DuplicateTextureName,
    DuplicateSurfaceName,
    DevicesUnavailable,
    InvalidKernelImage,
    NoKernelImageForDevice,
    IncompatibleDriverContext,
    PeerAccessAlreadyEnabled,
    PeerAccessNotEnabled,
    DeviceAlreadyInUse,
    ProfilerDisabled,
    ProfilerNotInitialized,
    ProfilerAlreadyStarted,
    ProfilerAlreadyStopped,
    Assert,
    TooManyPeers,
    HostMemoryAlreadyRegistered,
    HostMemoryNotRegistered,
    OperatingSystem,
    PeerAccessUnsupported,
    LaunchMaxDepthExceeded,
    LaunchFileScopedTex,
    LaunchFileScopedSurf,
    SyncDepthExceeded,
    LaunchPendingCountExceeded,
    NotPermitted,
    NotSupported,
    HardwareStackError,
    IllegalInstruction,
    MisalignedAddress,
    InvalidAddressSpace,
    InvalidPc,
    IllegalAddress,
    InvalidPtx,
    InvalidGraphicsContext,
    NvlinkUncorrectable,
    StartupFailure,
}
