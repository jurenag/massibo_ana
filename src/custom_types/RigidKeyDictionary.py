import numpy as np

import src.utils.htype as htype
import src.utils.custom_exceptions as cuex


class RigidKeyDictionary(dict):

    def __init__(self, potential_keys, is_typed=False, values_types=None):
        """This class aims to implement a dictionary whose potential keys are restricted.
        I.e. only a certain set of objects can belong to the dictionary keys. Optionally,
        keys in this dictionary could be typed, meaning that its associated values must
        comply with some type.

        This initializer gets the following positional argument:

        - potential_keys (list):self.__potential_keys is initialized with this parameter.
        Any key of a RigidKeyDictionary must be a copy of some object in its
        self.__potential_keys.

        This initializer gets the following optional keyword arguments:

        - is_typed (boolean, or a list of len(potential_keys) booleans): If the given
        parameter is equal to False (resp. True), then no (resp. every) potential key
        is typed. If the given parameter is a list of booleans, then is_typed[i] gives
        whether the i-th potential key (potential_key[i]) is typed or not.

        - values_types (list of types): This list must contain as many types as typed
        potential keys. If is_typed==False, or is_typed is a list full of "False", then
        values_types is ignored. If is_typed==True, then values_types must contain
        N types (where N is the number of potential keys). Otherwise, if is_typed is a
        list of booleans with M entries equal to True, with M>=1, then values_types must
        contain M types. In the last case, values_types[i] gives whether the type for
        the i-th typed key, up to the order set by potential_keys.

        For now, the potential keys and its typing property are fixed as of the object
        initialization. In the future, appropriate setters methods for these attributes
        could be implemented."""

        htype.check_type(
            potential_keys,
            list,
            exception_message=htype.generate_exception_message(
                "RigidKeyDictionary.__init__", 10001
            ),
        )

        fIsList = False
        if type(is_typed) == bool:
            pass
        elif type(is_typed) == list:
            if len(is_typed) != len(potential_keys):
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "RigidKeyDictionary.__init__", 10002
                    )
                )
            for element in is_typed:
                htype.check_type(
                    element,
                    bool,
                    exception_message=htype.generate_exception_message(
                        "RigidKeyDictionary.__init__", 10003
                    ),
                )
            fIsList = True
        else:
            raise cuex.TypeException(
                htype.generate_exception_message("RigidKeyDictionary.__init__", 10004)
            )

        fIgnoreValuesTypes = False
        if not fIsList:
            if is_typed is False:
                fIgnoreValuesTypes = True
        else:
            if bool(np.prod(np.logical_not(is_typed))):
                fIgnoreValuesTypes = True

        # In this case, we have to check
        # the proper format of values_types
        if not fIgnoreValuesTypes:
            htype.check_type(
                values_types,
                list,
                exception_message=htype.generate_exception_message(
                    "RigidKeyDictionary.__init__", 10005
                ),
            )
            for element in values_types:
                htype.check_type(
                    element,
                    type,
                    exception_message=htype.generate_exception_message(
                        "RigidKeyDictionary.__init__", 10006
                    ),
                )

            if not fIsList:  # is_typed must be True
                aux = len(potential_keys)
            else:
                aux = np.array(is_typed).sum()

            if len(values_types) != aux:  # aux is the number of typed potential keys
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "RigidKeyDictionary.__init__", 10007
                    )
                )

        self.__potential_keys = potential_keys
        self.__values_types = None

        # Either is_typed==False or
        # is_typed==[False, False, ..., False]
        if fIgnoreValuesTypes:
            self.__values_types = [object for x in self.__potential_keys]
        else:
            if not fIsList:  # is_typed must be True
                self.__values_types = (
                    values_types  # len(values_types)==len(potential_keys)
                )
            else:  # In this case, aux is equal to
                # the number of True's in is_typed
                gen = (n for n in range(0, aux))
                self.__values_types = [
                    values_types[next(gen)] if is_typed[i] is True else object
                    for i in range(len(potential_keys))
                ]
        return

    @property
    def PotentialKeys(self):
        return self.__potential_keys

    @property
    def ValuesTypes(self):
        return self.__values_types

    def __setitem__(self, __key, __value):

        try:
            idx = self.__potential_keys.index(__key)
        # __key was not found
        # in self.__potential_keys
        except ValueError:
            return
        if isinstance(__value, self.__values_types[idx]):
            return super().__setitem__(__key, __value)
        return

    def update(self, a_dict):
        """This method overrides the update method of the base class. This method
        updates the RigidKeyDictionary with the key/value pairs that are present
        within the given dictionary, a_dict, overwriting existing keys. This method
        returns None."""

        htype.check_type(
            a_dict,
            dict,
            exception_message=htype.generate_exception_message(
                "RigidKeyDictionary.update()", 20001
            ),
        )

        for key in a_dict.keys():
            self.__setitem__(key, a_dict[key])
        return
