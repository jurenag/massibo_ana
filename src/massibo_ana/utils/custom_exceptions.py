class MalFunction(Exception):
    def __init__(self, message):
        super().__init__(message)
        return

    @classmethod
    def why(self):
        return "This exception is meant for situations where the " \
        "code is not working as expected, but the reason is not " \
        "clear."


class InvalidParameterDefinition(Exception):
    def __init__(self, message):
        super().__init__(message)
        return

    @classmethod
    def why(self):
        return "This exception is meant for situations where a " \
        "parameter was given an invalid value. This check has to " \
        "do with the parameter per se. No other parameter comes " \
        "into play for this check."


class InconsistentParametersDefinition(Exception):
    def __init__(self, message):
        super().__init__(message)
        return

    @classmethod
    def why(self):
        return "This exception is meant for situations where a set " \
        "of parameters which are not compatible among each other was given."


class InsufficientParameters(Exception):
    def __init__(self, message):
        super().__init__(message)
        return

    @classmethod
    def why(self):
        return "This exception is meant for situations where not " \
        "even a minimal set of parameters has been defined."


class NoAvailableData(Exception):
    def __init__(self, message):
        super().__init__(message)
        return

    @classmethod
    def why(self):
        return "This exception is meant for situations where the " \
        "requested information is not available."


class TypeException(Exception):
    def __init__(self, message):
        super().__init__(message)
        return

    @classmethod
    def why(self):
        return "This exception is meant for situations where the " \
        "provided object type is not allowed."


class ShapeException(Exception):
    def __init__(self, message):
        super().__init__(message)
        return

    @classmethod
    def why(self):
        return "This exception is meant for situations where the " \
        "provided multidimensional object, say an array, does not " \
        "have a valid shape."


class WfmReadException(Exception):
    def __init__(self, message):
        super().__init__(message)
        return

    @classmethod
    def why(self):
        return "This exception is meant for situations where something " \
        "went wrong reading a .WFM file (binary waveform file for " \
        "Tektronix performance oscilloscope)."
    
class BinReadException(Exception):
    def __init__(self, message):
        super().__init__(message)
        return

    @classmethod
    def why(self):
        return "This exception is meant for situations where something " \
        "went wrong reading a .NPY or .BIN file (numpy or homemade binary " \
        "waveform file)."


class InvalidPhysicalConfiguration(Exception):
    def __init__(self, message):
        super().__init__(message)
        return

    @classmethod
    def why(self):
        return "This exception is meant for situations where the set " \
        "of parameters do not define a physically possible configuration."
    
class RestrictiveTimedelay(Exception):
    def __init__(self, message):
        super().__init__(message)
        return

    @classmethod
    def why(self):
        return "This exception is not meant for a general situation. " \
        "This exception is meant for the case where, in the " \
        "DarkNoiseMeas.compute_amplitude_levels() method, the" \
        "value given to the timedelay_cut variable is so big that " \
        "no samples were left."
    
class NotEnoughFitSamples(Exception):
    def __init__(self, message):
        super().__init__(message)
        return

    @classmethod
    def why(self):
        return "This exception is meant for a situations where " \
        "the number of samples (points) for fit is smaller than " \
        "the number of fitting parameters."
    
class RequestedPeaksNotFound(Exception):
    def __init__(self, message):
        super().__init__(message)
        return

    @classmethod
    def why(self):
        return "This exception is meant for situations where the " \
        "number of requested peaks were not found."