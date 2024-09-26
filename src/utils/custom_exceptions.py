class MalFunction(Exception):
    def __init__(self, message):
        super().__init__(message)
        return

    @classmethod
    def why(self):
        return "This exception is meant for situations where the code is not working as expected, but the reason is not clear."


class InvalidParameterDefinition(Exception):
    def __init__(self, message):
        super().__init__(message)
        return

    @classmethod
    def why(self):
        return "This exception is meant for situations where a parameter was given an invalid value. This check has to do with the parameter per se. No other parameter comes into play for this check."


class InconsistentParametersDefinition(Exception):
    def __init__(self, message):
        super().__init__(message)
        return

    @classmethod
    def why(self):
        return "This exception is meant for situations where a set of parameters which are not compatible among each other was given."


class InsufficientParameters(Exception):
    def __init__(self, message):
        super().__init__(message)
        return

    @classmethod
    def why(self):
        return "This exception is meant for situations where not even a minimal set of parameters has been defined."


class NoAvailableData(Exception):
    def __init__(self, message):
        super().__init__(message)
        return

    @classmethod
    def why(self):
        return "This exception is meant for situations where the requested information is not available."


class TypeException(Exception):
    def __init__(self, message):
        super().__init__(message)
        return

    @classmethod
    def why(self):
        return "This exception is meant for situations where the provided object type is not allowed."


class ShapeException(Exception):
    def __init__(self, message):
        super().__init__(message)
        return

    @classmethod
    def why(self):
        return "This exception is meant for situations where the provided multidimensional object, say an array, does not have a valid shape."


class WfmReadException(Exception):
    def __init__(self, message):
        super().__init__(message)
        return

    @classmethod
    def why(self):
        return "This exception is meant for situations where something went wrong reading a .WFM file (binary waveform file for Tektronix performance oscilloscope)."


class InvalidPhysicalConfiguration(Exception):
    def __init__(self, message):
        super().__init__(message)
        return

    @classmethod
    def why(self):
        return "This exception is meant for situations where the set of parameters do not define a physically possible configuration."
