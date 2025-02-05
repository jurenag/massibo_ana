import abc as abc

import massibo_ana.utils.custom_exceptions as cuex


def generate_exception_message(issuer, code, extra_info=None):
    """This function gets two positional arguments:

    - issuer (string): Name of the function whose execution led to the exception.

    - code (integer scalar): Integer which identifies a certain exception. This
    integer is meant to help tracking down the exact point of the code where the
    exception ocurred, and so, make debugging easier.

    This function gets the following optional keyword argument:

    - extra_info (string): This string should contain extra information on the
    exception cause.

    This function returns an string which is meant to be the message that should
    be passed to an exception and which identifies the issuer of the exception
    and an additional integer code which could help tracking down the exception.
    If an string is passed to extra_info, it is appended to the exception message."""

    if type(issuer) != str:
        # generate_exception_message() cannot use its own functionality
        raise cuex.TypeException(
            "In function generate_exception_message(): ERR101: Invalid type."
        )

    if type(code) != int:
        raise cuex.TypeException(
            "In function generate_exception_message(): ERR102: Invalid type."
        )

    if extra_info is None:
        return f"In function {issuer}(): ERR{code}"
    else:
        if type(extra_info) != str:
            raise cuex.TypeException(
                "In function generate_exception_message(): ERR103: Invalid type."
            )
        return f"In function {issuer}(): ERR{code}: {extra_info}"


def check_type(input, *goal_type, exception_message="", verbosity=False):
    """This function gets the following positional arguments:

    - input (unspecified type).
    This function also gets the following optional positional arguments:

    - goal_type (list of types)
    This function also gets the following optional keyword argument:

    - exception_message (string): Message passed to the raised exception. Normally,
    for debugging purposes, this message should include information about the issuer
    of the exception, i.e. the function or method from which the check_type is being
    called.

    - verbosity (boolean scalar): Whether to print extra information.
    This function checks that the type of input belongs to goal_type. If so,
    this function returns True. Else, this function raises a TypeException.
    If no goal types are provided at all, it is understood that type(input)
    do not belong to goal_type and so, the function raises TypeException."""

    for i in range(len(goal_type)):

        # Cover the case of classes which derive from abc.ABC
        # whose type is not type(type(0)), but abc.ABCMETA
        if type(goal_type[i]) != type(type(0)) and type(goal_type[i]) != abc.ABCMeta:
            raise cuex.TypeException(generate_exception_message("check_type", 201))

    if type(exception_message) != type(""):
        raise cuex.TypeException(generate_exception_message("check_type", 202))

    if type(verbosity) != type(False):
        raise cuex.TypeException(generate_exception_message("check_type", 203))

    for i in range(len(goal_type)):

        if verbosity:
            print(
                "In function check_type(): ¿Is type("
                + str(input)
                + ") == "
                + str(goal_type[i])
                + "?: ",
                end="",
            )

        if type(input) == goal_type[i]:
            if verbosity:
                print("Yes")
            return True

        if verbosity:
            print("No")

    raise cuex.TypeException(exception_message)
