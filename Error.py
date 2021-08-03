class Error(Exception):
    """Base class for other exceptions"""
    pass

class ValueTooSmallError(Error):
    """Raised when the value is smaller than the valid"""
    pass

class ValueInvalidError(Error):
    """Raised when the value is invalid"""
    pass

class ValueConfigurationError(Error):
    """Raised when the class does not support this configuration"""
    pass