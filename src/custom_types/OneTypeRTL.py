from src.custom_types.RestrictiveTL import RestrictiveTL

class OneTypeRTL(RestrictiveTL):

    def __init__(self, allowed_type, list_name=None, init_members=None):

        """The aim of this method is the same as that of RestrictiveTL (see such class documentation
        for further information) but allowing for just one base type within the underlying typed list.  
        
        This method takes the following positional arguments:

        - allowed_type (type): [allowed_type] is passed to the base class initializer as allowed_types.

        And the following keyword arguments:

        - list_name (string): It is passed to the base class initializer as list_name. It is 
        the name of the typed list.
        - init_members (list): It is passed to the base class initializer as init_members.
        """

        self.__the_type = allowed_type
        super().__init__([allowed_type], list_name=list_name, init_members=init_members)
        return

    @property        
    def TheType(self):
        return self.__the_type
