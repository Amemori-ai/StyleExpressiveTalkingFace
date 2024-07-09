from GlintCloud.error import (CloudError, G_ERROR_SYSTEM_OOM,
                              G_ERROR_SYSTEM_GPU_AVAILABLE,
                              G_ERROR_SYSTEM_DISK_RESOURCE,
                              G_ERROR_SYSTEM_FILE_OPEN,G_ERROR_FACE_NO_DETECTED,
                              G_ERROR_SYSTEM_FILE_FORMAT)

class SystemOOMError(CloudError):
    code = G_ERROR_SYSTEM_OOM

class SystemGPUAvailableError(CloudError):
    code = G_ERROR_SYSTEM_GPU_AVAILABLE

class SystemDiskResouceError(CloudError):
    code = G_ERROR_SYSTEM_DISK_RESOURCE   

class SystemFileOpenError(CloudError):
    code = G_ERROR_SYSTEM_FILE_OPEN

class FaceNoDetectedError(CloudError):
    code = G_ERROR_FACE_NO_DETECTED

class SystemFileFormatError(CloudError):
    code = G_ERROR_SYSTEM_FILE_FORMAT

