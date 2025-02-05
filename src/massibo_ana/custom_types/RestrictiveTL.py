from massibo_ana.custom_types.TypedList import TypedList


class RestrictiveTL(TypedList):

    def __init__(self, allowed_types, list_name=None, init_members=None):
        """The aim of this method is to implement a restrictive TypedList class which only allows
        member of N different types (or its derived types/classes) and its restrictiveness is locked from
        the very beginning. In agreement with TypedList class documentation, what this means is
        that a RestrictiveTL object is basically a TypedList object where its member_types attribute
        is set to allowed_types (i.e. a N-length list which contains N objects of type type) and its
        restrictivenessIsLocked attribute is set to True right after the initialization. Therefore, an
        object of this class is a list which can only allocate objects which are instance of at least
        one element within allowed_types (either their type match some type in allowed_types, or their
        type (class) is derived from some type within allowed_types). This method takes the following
        positional argument:

        - allowed_types (list of type objects): It is passed to the base class initializer as member_types.

        And the following keyword arguments:

        - list_name (string): It is passed to the base class initializer as list_name. It is
        the name of this list.
        - init_members (list): It is passed to the base class initializer as init_members.
        """

        super().__init__(
            list_name=list_name,
            member_types=allowed_types,
            init_members=init_members,
            restrictive=True,
        )
        
        self.lock_restrictiveness()
        return
