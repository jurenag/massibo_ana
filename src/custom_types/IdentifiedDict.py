import os
import json
import inspect

import src.utils.htype as htype
import src.utils.custom_exceptions as cuex


class IdentifiedDict(dict):

    def __init__(self, input_data, ignored_fields=[]):
        """This class, which inherits from python's built-in dictionary, aims
        to model a dictionary with an identifier (self.__id) which might be
        inferred from the dictionary entries. This initializer takes the
        following positional argument:

        - input_data (dictionary): The initial dictionary.

        This initializer takes the following keyword argument:

        - ignored_fields (list): The keys in input_data which are also in
        ignored_fields are not loaded into self. By default, ignored_fields
        is an empty list."""

        htype.check_type(
            input_data,
            dict,
            exception_message=htype.generate_exception_message(
                "IdentifiedDict.__init__", 91127
            ),
        )
        htype.check_type(
            ignored_fields,
            list,
            exception_message=htype.generate_exception_message(
                "IdentifiedDict.__init__", 67019
            ),
        )

        input_data_ = input_data.copy()

        # Yes, iterate over input_data, not
        # input_data_, because the size of
        # input_data_ may change through the loop

        for key in input_data.keys():
            if key in ignored_fields:
                del input_data_[key]

        self.__id = None
        super().__init__(input_data_)
        return

    @property
    def ID(self):
        return self.__id

    def assign_identifier(self, id_callable, *args, **kwargs):
        """This method gets the following positional argument:

        - id_callable (callable): Its return type must be string.
        - args, kwargs: They are passed to id_callable.

        This method sets the self.__id attribute to
        id_callable(self, *args, **kwargs). In addition to the
        case when id_callable is an external function or an static
        method, this method is also able to handle the case when
        id_callable is a method of self."""

        if not callable(id_callable):
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "IdentifiedDict.assign_identifier", 18253
                )
            )
        signature = inspect.signature(id_callable)
        if signature.return_annotation != str:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "IdentifiedDict.assign_identifier",
                    50038,
                    extra_info="The return type of the given callable must be hinted as str.",
                )
            )

        id_callable_ = None
        if hasattr(self, id_callable.__name__):
            id_callable_ = getattr(self, id_callable.__name__)
            self.__id = id_callable_(*args, **kwargs)

            # When the passed callable is a method
            # of self, the first argument of the
            # method is self implicitly. Passing
            # self explicitly would be equivalent
            # to calling self.a_method(self, *args, **kwargs)

        else:
            id_callable_ = id_callable
            self.__id = id_callable_(self, *args, **kwargs)

        return

    @staticmethod
    def lowercase_and_remove_redundancy(input_seq):
        """This method gets the following positional argument:

        - input (tuple/list of strings): If it is a tuple (resp. list),
        then this function returns a tuple (resp. list).

        This method converts every entry of input to its lowercase
        version and removes every duplicated entry. After performing
        these two actions, this method returns the resulting sequence.
        The output keeps the ordering of the input.
        """

        htype.check_type(
            input_seq,
            tuple,
            list,
            exception_message=htype.generate_exception_message(
                "IdentifiedDict.lowercase_and_remove_redundancy", 25103
            ),
        )

        for x in input_seq:
            htype.check_type(
                x,
                str,
                exception_message=htype.generate_exception_message(
                    "IdentifiedDict.lowercase_and_remove_redundancy", 89772
                ),
            )

        aux = [x.lower() for x in input_seq]

        # We need to iterate backwards,
        # since, at each iteration, we
        # may delete an entry from aux
        for i in reversed(range(len(aux))):

            # Check whether this element is
            # duplicated in the sub-sequence
            # before the current element
            if aux[i] in aux[:i]:

                # If so, this element is duplicated,
                # so delete it
                del aux[i]

        # Up to this point, we have erased the duplicated
        # entries but keeping the input_seq ordering

        if type(input_seq) == tuple:
            return tuple(aux)
        else:
            return list(aux)

    @classmethod
    def from_json_file(cls, json_filepath, ignored_fields=[]):
        """This class method is meant to be an alternative initializer.
        This method gets the following positional argument:

        - json_filepath (string): The path to the json file which contains
        the dictionary to be loaded.

        This method gets the following keyword argument:

        - ignored_fields (list): It is given to the ignored_fields parameter
        of the class initializer. The keys in the input json file which are
        also in ignored_fields are not loaded into the IdentifiedDict object.
        By default, ignored_fields is an empty list.

        This class method lets you create an IdentifiedDict object from a
        json file."""

        htype.check_type(
            json_filepath,
            str,
            exception_message=htype.generate_exception_message(
                "IdentifiedDict.from_json_file", 33808
            ),
        )

        htype.check_type(
            ignored_fields,
            list,
            exception_message=htype.generate_exception_message(
                "IdentifiedDict.from_json_file", 22667
            ),
        )

        if not os.path.exists(json_filepath) or not os.path.isfile(json_filepath):
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "IdentifiedDict.from_json_file",
                    67892,
                    extra_info=f"The given path does not exist or is not a file: {json_filepath}",
                )
            )

        extension = os.path.splitext(json_filepath)[1]
        if extension != ".json":
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "IdentifiedDict.from_json_file",
                    23636,
                    extra_info=f"The extension of the given filepath ({extension}) must match '.json'.",
                )
            )

        with open(json_filepath, "r") as file:
            data = json.load(file)

        return cls(data, ignored_fields=ignored_fields)
