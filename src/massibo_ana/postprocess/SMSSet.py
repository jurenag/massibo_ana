import os
import inspect
import numpy as np
import pandas as pd
import typing

import massibo_ana.utils.htype as htype
import massibo_ana.utils.custom_exceptions as cuex
from massibo_ana.custom_types.OneTypeRTL import OneTypeRTL
from massibo_ana.custom_types.IdentifiedDict import IdentifiedDict
from massibo_ana.postprocess.SiPMMeasSummary import SiPMMeasSummary


class SMSSet(OneTypeRTL):

    def __init__(
        self,
        sipmmeas_summaries,
        *args,
        list_name="default",
        identifier_to_assign="standard",
        **kwargs,
    ):
        """This class aims to model a set of SiPMMeasSummary objects.
        This initializer gets the following positional arguments:

        - sipmmeas_summaries (tuple/list of SiPMMeasSummary objects): The
        objects in this sequence are loaded into the underlying typed
        list of this object.

        - args: Passed to member.assign_identifier, for every member
        in this set.

        This initializer also gets the following keyword argument:

        - list_name (string): This string is passed to the 'list_name'
        parameter of the parent initializer. It is meant to be the
        name of this set. If this argument is not given, then the
        list name will be set to 'default'.

        - identifier_to_assign (string): Using a hardcoded mapping, this
        parameter controls the type of identifier that is assigned for
        every SiPMMeasSummary in this SMSSet, i.e. the callable that is
        passed to x.assign_identifier(callable, *args, **kwargs), for every
        x in this SMSSet. For the time being, there's only one type of
        identifier that can be assigned by SiPMMeasSummary, which is the
        standard one (assigned by SiPMMeasSummary.standard_identifier), so
        the current mapping is:

            - 'standard' (except for upper- or lower- case conversion) -> SiPMMeasSummary.standard_identifier

        - kwargs: Passed to member.assign_identifier, for every member
        of this set."""

        htype.check_type(
            sipmmeas_summaries,
            tuple,
            list,
            exception_message=htype.generate_exception_message(
                "SMSSet.__init__", 72114
            ),
        )
        for x in sipmmeas_summaries:
            htype.check_type(
                x,
                SiPMMeasSummary,
                exception_message=htype.generate_exception_message(
                    "SMSSet.__init__", 98872
                ),
            )

        # list_name is managed by the parent class initializer
        htype.check_type(
            identifier_to_assign,
            str,
            exception_message=htype.generate_exception_message(
                "SMSSet.__init__", 96733
            ),
        )
        super().__init__(
            SiPMMeasSummary, list_name=list_name, init_members=list(sipmmeas_summaries)
        )

        for member in self:

            if identifier_to_assign.lower() == "standard":
                identifier_assigner = member.standard_identifier
            else:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "SMSSet.__init__",
                        77490,
                        extra_info=f"The given identifier_to_assign ({identifier_to_assign}) is not recognized.",
                    )
                )

            # The identifier assigner that is passed to SiPMMeasSummary.assign_identifier
            # (actually IdentifedDict.assign_identifier) might be a method of each member
            # in self, therefore accesing such method needs to be done at each iteration

            member.assign_identifier(identifier_assigner, *args, **kwargs)
        return

    def do_for_every_member(self, the_callable, *args, **kwargs):
        """This method gets the following arguments:

        - the_callable (callable): Its first argument must be a positional
        argument which must be hinted either as IdentifiedDict, or as a
        typing.Union object containing IdentifiedDict. The reason for
        not looking for SiPMMeasSummary instead of IdentifiedDict is that
        the type-hinting of the allowed callables (if they are defined
        as methods of SiPMMeasSummary) should include (quoted) forward
        references to SiPMMeasSummary. For more information, check
        https://peps.python.org/pep-0484/#forward-references . Thus, I could
        use a forward reference to SiPMMeasSummary in the allowed callables
        (if they are defined as SiPMMeasSummary methods), but, in that case,
        inspecting the signature for the type-hints is not straightforward,
        because what you get at the inspection phase is actually a forward-
        reference, not its associated type. A workaround for this is to
        define the allowed callables outside SiPMMeasSummary, however, most
        of them are conceptually better suited to be SiPMMeasSummary methods,
        since they perform tasks which should apply to SiPMMeasSummary.

        - args, kwargs: Passed to the_callable

        This method iterates through this SMSSet and, for every member,
        it runs the_callable(member, *args, **kwargs)."""

        if not callable(the_callable):
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("SMSSet.do_for_every_member", 91372)
            )
        signature = inspect.signature(the_callable)
        if len(signature.parameters) < 1:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SMSSet.do_for_every_member",
                    49216,
                    extra_info="The signature of the given callable must have one argument at least.",
                )
            )
        first_parameter = list(signature.parameters.keys())[0]
        fp_annotation = signature.parameters[first_parameter].annotation

        # Happens if the first parameter is hinted using a typing.Union, meaning
        # that its type could be any of the types within an specified set of types
        if isinstance(fp_annotation, typing._UnionGenericAlias):

            if not IdentifiedDict in typing.get_args(fp_annotation):
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "SMSSet.do_for_every_member", 12280
                    )
                )

        # If the first parameter is hinted using just one type, then it must be IdentifiedDict
        elif fp_annotation != IdentifiedDict:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SMSSet.do_for_every_member",
                    42804,
                    extra_info="The first parameter of the given callable is not hinted, or is hinted as a non-recognized type.",
                )
            )
        for member in self:
            the_callable(member, *args, **kwargs)
        return

    def find_by_identifer(self, identifier, verbose=False):
        """This method gets the following positional argument:

        - identifier (any object except for None): The identifier of the
        member to be found.

        This method gets the following keyword argument:

        - verbose (bool): Whether to print functioning-related messages.

        This method sequentially iterates through this SMSSet. For each
        member, it compares identifier with member.ID. If they match, then
        this method returns the member. If no member is found, then this
        method returns None. The members whose identifier match None are
        considered as unidentified and are ignored for the search."""

        if identifier is None:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("SMSSet.find_by_identifer", 20970)
            )
        htype.check_type(
            verbose,
            bool,
            exception_message=htype.generate_exception_message(
                "SMSSet.find_by_identifer", 80385
            ),
        )
        for member in self:
            if member.ID is None:
                if verbose:
                    print(
                        "In method SMSSet.find_by_identifier(): An unidentified member was spotted (its ID matches None). This member is skipped."
                    )
                continue
            elif member.ID == identifier:
                return member

        if verbose:
            print(
                f"In method SMSSet.find_by_identifier(): No member with the given identifier ({identifier}) was found."
            )

        # This point is reached if no member with the given identifier was found
        return None

    def compare(
        self,
        other,
        fields_to_compare=[],
        comparing_callables=[],
        verbose=False,
        *args,
        **kwargs,
    ):
        """This method gets the following argument:

        - other (SMSSet): This SMSSet is compared against self,
        member-wise and field-wise, up to fields_to_compare.

        This method gets the following keyword arguments:

        - fields_to_compare (list): The keys within self's members
        and other's members which are compared by the given callables.

        - comparing_callables (list of len(fields_to_compare) callables):
        For each member in self whose self.__id matches the identifier of
        one member of other, their entries for the key which matches
        fields_to_compare[i] (if such key exists in both members to be
        compared) are compared among each other by means of
        comparing_callables[i]. Every entry in comparing_callables must
        have at least two positional arguments.

        - verbose (bool): Whether to print functioning-related messages.

        - args, kwargs: Passed to comparing_callables[i] for every i.

        This method returns a dictionary, say output, whose keys match the
        identifiers of the members in self which were also spotted in other.
        The members whose identifier match None are considered as unidentified
        and are ignored. The identifiers which were not simultaneously spotted
        in both, self and other, do not belong to output.keys(). For each key,
        say id, its value (output[id]) is a dictionary, whose entries include
        those entries within fields_to_compare which were spotted in both: the
        member in self whose identifier matches id, and the member in other
        whose identifier matches id. If this field-to-compare was not spotted
        in both simultaneously, then such field will not be part of
        output[id].keys(). For the fields-to-compare within fields_to_compare
        which were indeed spotted in both members, say fields_to_compare[i], the
        value of its matching sub-dictionary (i.e. output[id][fields_to_compare[i]])
        is the result of comparing the field in the member in self against the field
        in the member in other, by means of comparing_callable[i]. More precisely,
        such value is computed as:

        output[id][fields_to_compare[i]] = comparing_callable[i](   self.find_by_identifier(id)[fields_to_compare[i]],
                                                                    other.find_by_identifier(id)[fields_to_compare[i]],
                                                                    *args, **kwargs).

        If the fields_to_compare list is empty, then all of the sub-dictionaries
        of output are emtpy dictionaries.
        """

        htype.check_type(
            other,
            SMSSet,
            exception_message=htype.generate_exception_message("SMSSet.compare", 43804),
        )
        htype.check_type(
            fields_to_compare,
            list,
            exception_message=htype.generate_exception_message("SMSSet.compare", 46975),
        )
        htype.check_type(
            comparing_callables,
            list,
            exception_message=htype.generate_exception_message("SMSSet.compare", 55749),
        )
        if len(comparing_callables) != len(fields_to_compare):
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("SMSSet.compare", 75382)
            )
        for elem in comparing_callables:
            if not callable(elem):
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message("SMSSet.compare", 10490)
                )
            signature = inspect.signature(elem)
            if len(signature.parameters) < 2:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "SMSSet.compare",
                        76981,
                        extra_info="The signature of the comparing callables must have at least two arguments.",
                    )
                )
            for parameter in list(signature.parameters.keys())[:2]:
                if signature.parameters[parameter].default != inspect.Parameter.empty:
                    raise cuex.InvalidParameterDefinition(
                        htype.generate_exception_message(
                            "SMSSet.compare",
                            75584,
                            extra_info="The two first arguments of the comparing callables must be positional arguments.",
                        )
                    )
        htype.check_type(
            verbose,
            bool,
            exception_message=htype.generate_exception_message("SMSSet.compare", 66322),
        )
        output = {}

        for member in self:
            if member.ID is None:
                if verbose:
                    print(
                        "In method SMSSet.compare(): An unidentified member was spotted (its ID matches None). This member is skipped."
                    )
            else:
                current_ID = member.ID
                matching_member = other.find_by_identifer(current_ID, verbose=verbose)

                # If no member in other was found to have
                # an identifier matching current_ID, then
                # SMSSet.find_by_identifier() returns None
                if matching_member is None:
                    continue
                else:
                    output[current_ID] = {}
                    for i in range(len(fields_to_compare)):
                        if (
                            fields_to_compare[i] in member.keys()
                            and fields_to_compare[i] in matching_member.keys()
                        ):
                            output[current_ID][
                                fields_to_compare[i]
                            ] = comparing_callables[i](
                                member[fields_to_compare[i]],
                                matching_member[fields_to_compare[i]],
                                *args,
                                **kwargs,
                            )
                        elif verbose:
                            print(
                                f"In method SMSSet.compare(): The field {fields_to_compare[i]} was not simultaneously found in both members to be compared. This field is skipped."
                            )
        return output

    @classmethod
    def from_excel_file(
        cls,
        excel_filepath,
        *args,
        sheet_name=0,
        header=0,
        usecols=None,
        list_name="default",
        identifier_to_assign="standard",
        ignored_fields=[],
        **kwargs,
    ):
        """This class method is meant to be an alternative initializer,
        which lets you create a SMSSet object from an excel file. This
        class method adds one SummaryMeasSummary object to this SMSSet
        per row in the excel file. This class method gets the following
        positional arguments:

        - excel_filepath (string): The path to the excel file which contains
        the data to be loaded. It is given to the 'io' parameter of the
        pandas.read_excel function.

        - args: Passed to the SMSSet initializer, which in turn passes them
        to member.assign_identifier, for every member in this set.

        - sheet_name (integer or string): The zero-indexed position (integer case)
        or the name (string case) of the sheet to be loaded. It is given to the
        'sheet_name' parameter of the pandas.read_excel function.

        - header (integer): Row (0-indexed) to use for the column labels of the
        parsed DataFrame. It is given to the 'header' parameter of the
        pandas.read_excel.

        - usecols (None, or str, or list of int, or list of str, or callable):
        Passed to the 'usecols' argument of pandas.read_excel. The documentation
        for that argument in the pandas 2.2 API reference is:

            - If None, then parse all columns.
            - If str, then indicates comma separated list of Excel column
            letters and column ranges (e.g. “A:E” or “A,C,E:F”). Ranges are
            inclusive of both sides.
            - If list of int, then indicates list of column numbers to be
            parsed (0-indexed).
            - If list of string, then indicates list of column names to be
            parsed.
            - If callable, then evaluate each column name against it and
            parse the column if the callable returns True.

        - list_name (string): This string is passed to the 'list_name' parameter
        of the SMSSet initializer. It is meant to be the name of this set. For
        more information, check the SMSSet.__init__ documentation.

        - identifier_to_assign (string): This string is passed to the
        'identifier_to_assign' parameter of the SMSSet initializer. This
        parameter controls the type of identifier that is assigned for
        every SiPMMeasSummary in this SMSSet. For more information, check
        the SMSSet.__init__ documentation.

        - ignored_fields (list of str): This argument is passed to the
        'ignored_fields' argument of SiPMMeasSummary, for every read row/member
        from the excel file. The column names which are set to be read, up
        to the value given to the 'usecols' parameter, will be actually ignored
        if they match any string within ignored_fields. By default, ignored_fields
        is an empty list, meaning that every field is read. Although
        SiPMMeasSummary.__init___ only constrains ignored_fields to be a list,
        this class method constrains it to be a list of strings, since the column
        names in the excel file are read to strings by pandas.read_excel.

        - kwargs: Passed to the SMSSet initializer, which in turn passes them
        to member.assign_identifier, for every member in this set."""

        htype.check_type(
            excel_filepath,
            str,
            exception_message=htype.generate_exception_message(
                "SMSSet.from_excel_file", 54432
            ),
        )
        htype.check_type(
            sheet_name,
            int,
            np.int64,
            str,
            exception_message=htype.generate_exception_message(
                "SMSSet.from_excel_file", 43201
            ),
        )
        htype.check_type(
            header,
            int,
            np.int64,
            exception_message=htype.generate_exception_message(
                "SMSSet.from_excel_file", 77598
            ),
        )
        if usecols is not None or not callable(usecols):

            htype.check_type(
                usecols,
                str,
                list,
                exception_message=htype.generate_exception_message(
                    "SMSSet.from_excel_file", 95177
                ),
            )
            if type(usecols) == list:
                for x in usecols:
                    htype.check_type(
                        x,
                        int,
                        str,
                        exception_message=htype.generate_exception_message(
                            "SMSSet.from_excel_file", 13233
                        ),
                    )
        htype.check_type(
            list_name,
            str,
            exception_message=htype.generate_exception_message(
                "SMSSet.from_excel_file", 50729
            ),
        )
        htype.check_type(
            identifier_to_assign,
            str,
            exception_message=htype.generate_exception_message(
                "SMSSet.from_excel_file", 58836
            ),
        )
        htype.check_type(
            ignored_fields,
            list,
            exception_message=htype.generate_exception_message(
                "SMSSet.from_excel_file", 28328
            ),
        )
        for x in ignored_fields:
            htype.check_type(
                x,
                str,
                exception_message=htype.generate_exception_message(
                    "SMSSet.from_excel_file", 63706
                ),
            )
        input_df = pd.read_excel(
            excel_filepath, sheet_name=sheet_name, header=header, usecols=usecols
        )

        set = [
            SiPMMeasSummary(input_df.iloc[i].to_dict(), ignored_fields=ignored_fields)
            for i in range(len(input_df))
        ]
        return cls(
            set,
            *args,
            list_name=list_name,
            identifier_to_assign=identifier_to_assign,
            **kwargs,
        )

    @classmethod
    def from_jsons_folder(
        cls,
        jsons_folderpath,
        *args,
        filename_filter=lambda x: True,
        filter_kwargs={},
        list_name="default",
        identifier_to_assign="standard",
        ignored_fields=[],
        **kwargs,
    ):
        """This class method is meant to be an alternative initializer,
        which lets you create a SMSSet object from a folder-path. This
        class method adds one SiPMMeasSummary object to this SMSSet
        per json-file spotted in the given folder. This class method
        gets the following positional arguments:

        - jsons_folderpath (string): A path which must point to an
        already existing directory. This location is where the json files
        will be looked for.

        - args: Passed to the SMSSet initializer, which in turn passes
        them to member.assign_identifier, for every member in this set.

        This class method also gets the following keyword arguments:

        - filename_filter (callable): Its first argument must be a
        positional argument called 'filename'. Such positional
        argument must be hinted as str. The rest of the arguments,
        if any, must be keyword arguments. The output of this callable
        must be hinted as bool. This callable is used to filter the
        filenames in the given folder. For every filename in the folder,
        filename_filter(filename) is called. If it returns True, then
        the file is read. If it returns False, then the file is skipped.

        - filter_kwargs (dictionary): This dictionary is interpreted as
        a set of keyword arguments which are directly passed to
        filename_filter everytime it is called. No checks are done on
        whether filename_filter actually expects the parameters given
        to filter_kwargs. It is the user responsibility to ensure that
        the specified keyword arguments are expected by filename_filter.

        - list_name (string): This string is passed to the 'list_name'
        parameter of the SMSSet initializer. It is meant to be the name
        of this set. For more information, check the SMSSet.__init__
        documentation.

        - identifier_to_assign (string): This string is passed to the
        'identifier_to_assign' parameter of the SMSSet initializer. This
        parameter controls the type of identifier that is assigned for
        every SiPMMeasSummary in this SMSSet. For more information, check
        the SMSSet.__init__ documentation.

        - ignored_fields (list of str): This argument is passed to the
        'ignored_fields' argument of SiPMMeasSummary.from_json_file(),
        for every read json file. The entries within each spotted json
        file whose key matches any string within ignored_fields, will
        not be read into its SiPMMeasSummary object. By default,
        ignored_fields is an empty list, meaning that every field is read.
        Although SiPMMeasSummary.__init___ only constrains ignored_fields
        to be a list, this class method constrains it to be a list of
        strings (it is assumed that json keys are strings).

        - kwargs: Passed to the SMSSet initializer, which in turn passes them
        to member.assign_identifier, for every member in this set."""

        htype.check_type(
            jsons_folderpath,
            str,
            exception_message=htype.generate_exception_message(
                "SMSSet.from_jsons_folder", 71939
            ),
        )
        if not os.path.isdir(jsons_folderpath):
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("SMSSet.from_jsons_folder", 47368)
            )
        if not callable(filename_filter):
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message("SMSSet.from_jsons_folder", 69212)
            )
        signature = inspect.signature(filename_filter)
        if (
            len(signature.parameters) < 1
        ):  # The file-name filters gets at least one argument
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SMSSet.from_jsons_folder",
                    30583,
                    extra_info="The signature of the given file-name filter must have one and only one argument.",
                )
            )

        if (
            list(signature.parameters.keys())[0] != "filename"
        ):  # The first argument gotten by the filter is called 'filename'
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SMSSet.from_jsons_folder",
                    87610,
                    extra_info="The name of the only parameter of the signature of the given file-name filter must be 'filename'.",
                )
            )

        # The file-name filter gets no default arguments, i.e. it is a positional argument
        if signature.parameters["filename"].default != inspect.Parameter.empty:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SMSSet.from_jsons_folder",
                    93994,
                    extra_info="The first parameter of the signature of the given file-name filter must be a positional argument.",
                )
            )

        for parameter in list(signature.parameters.keys())[
            1:
        ]:  # The rest of the parameters must be keyword arguments
            if signature.parameters[parameter].default == inspect.Parameter.empty:
                raise cuex.InvalidParameterDefinition(
                    htype.generate_exception_message(
                        "SMSSet.from_jsons_folder",
                        67631,
                        extra_info="The rest of the parameters of the given file-name filter must be keyword arguments.",
                    )
                )
        if signature.parameters["filename"].annotation != str:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SMSSet.from_jsons_folder",
                    79554,
                    extra_info="The type of the only parameter of the signature of the given file-name filter must be hinted as str.",
                )
            )
        if signature.return_annotation != bool:
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "SMSSet.from_jsons_folder",
                    64857,
                    extra_info="The return type of the given file-name filter must be hinted as bool.",
                )
            )
        htype.check_type(
            filter_kwargs,
            dict,
            exception_message=htype.generate_exception_message(
                "SMSSet.from_jsons_folder", 35587
            ),
        )
        # list_name is managed by the class initializer
        # identifier_to_assign is managed by the class initializer

        htype.check_type(
            ignored_fields,
            list,
            exception_message=htype.generate_exception_message(
                "SMSSet.from_jsons_folder", 45065
            ),
        )
        for x in ignored_fields:
            htype.check_type(
                x,
                str,
                exception_message=htype.generate_exception_message(
                    "SMSSet.from_jsons_folder", 87109
                ),
            )
        aux = []
        for filename in os.listdir(jsons_folderpath):
            if filename.endswith(".json"):
                if filename_filter(filename, **filter_kwargs):
                    filepath = os.path.join(jsons_folderpath, filename)
                    if os.path.isfile(filepath):
                        aux.append(
                            SiPMMeasSummary.from_json_file(
                                filepath, ignored_fields=ignored_fields
                            )
                        )

        return cls(
            aux,
            *args,
            list_name=list_name,
            identifier_to_assign=identifier_to_assign,
            **kwargs,
        )

    @staticmethod
    def absolute_difference(x, y):
        """This method gets the following positional arguments:

        - x, y (int or float): The values to be compared.

        This method returns the absolute difference between x and y."""

        htype.check_type(
            x,
            int,
            np.int64,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "SMSSet.absolute_difference", 44355
            ),
        )
        htype.check_type(
            y,
            int,
            np.int64,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "SMSSet.absolute_difference", 33788
            ),
        )
        return abs(x - y)

    @staticmethod
    def difference(x, y):
        """This method gets the following positional arguments:

        - x, y (int or float): The values to be compared.

        This method returns the difference between x and y."""

        htype.check_type(
            x,
            int,
            np.int64,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "SMSSet.difference", 81206
            ),
        )
        htype.check_type(
            y,
            int,
            np.int64,
            float,
            np.float64,
            exception_message=htype.generate_exception_message(
                "SMSSet.difference", 99411
            ),
        )
        return x - y

    @staticmethod
    def return_first_of_pair(x, y):
        return x

    @staticmethod
    def return_second_of_pair(x, y):
        return y

    @staticmethod
    def filename_startswith(filename: str, prefix="") -> bool:
        """This method gets the following positional arguments:

        - filename (str): The filename to be compared.

        - prefix (str): The prefix to be compared against filename.

        This method returns True if filename starts with prefix, and False
        otherwise."""

        htype.check_type(
            filename,
            str,
            exception_message=htype.generate_exception_message(
                "SMSSet.filename_startswith", 71031
            ),
        )
        htype.check_type(
            prefix,
            str,
            exception_message=htype.generate_exception_message(
                "SMSSet.filename_startswith", 60163
            ),
        )
        return filename.startswith(prefix)
