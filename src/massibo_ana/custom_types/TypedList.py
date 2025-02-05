import numpy as np

import massibo_ana.utils.htype as htype


class TypedList(list):

    def __init__(
        self, list_name=None, member_types=None, init_members=None, restrictive=False
    ):
        """This class inherits from python built-in list. It aims to implement a list that
        can be configured to be "hard typed". By "hard typed" list we mean a list whose
        members types must belong to a certain list of types. This class offers some features
        regarding the hard-typing configuration, i.e. the hard-typing can be activated or
        deactivated. The typed list is also given an unique ID (an integer number). As of
        now, we may refer to a TypedList instance as a "collection".

        This initializer takes the following keyword arguments:

        - list_name (string): Name of the typed-list. If it is not given, then this name
        is set to "default".
        - member_types (list of types): List with which self.__member_types is initialized.
        Its meaning depends on the value of self.__is_restrictive.
        - init_members (list of objects): The objects within init_members are candidates to
        be added to the new typed-list. They will end up belonging to the typed-list depending
        on self.__is_restrictive and self.__member_types.
        - restrictive (scalar boolean): Initial value to give to self.__is_restrictive. While
        self.__is_restrictive==False, all objects regardless its type can be added to the
        typed-list. However, if self.__is_restrictive==True, only objects which are instance
        of some element in self.__member_types, can be added. This means that only objects
        whose type belongs to self.__member_types, or whose type (class) is derived from a
        type (class) which belongs to self.__member_types, can be added to the typed-list.
        """

        htype.check_type(
            restrictive,
            bool,
            exception_message=htype.generate_exception_message(
                "TypedList.__init__", 10001
            ),
        )

        if list_name is not None:
            htype.check_type(
                list_name,
                str,
                exception_message=htype.generate_exception_message(
                    "TypedList.__init__", 10002
                ),
            )

        if member_types is not None:
            htype.check_type(
                member_types,
                list,
                exception_message=htype.generate_exception_message(
                    "TypedList.__init__", 10003
                ),
            )

            for i in range(len(member_types)):
                htype.check_type(
                    member_types[i],
                    type,
                    exception_message=htype.generate_exception_message(
                        "TypedList.__init__", 10004
                    ),
                )
            self.__member_types = member_types
        else:
            self.__member_types = []

        self.__is_restrictive = restrictive
        self.__restrictiveness_is_locked = False

        if list_name is None:
            self.__name = "default"
        else:
            self.__name = list_name

        self.ID_ = self.get_new_ID()

        if init_members is not None:
            htype.check_type(
                init_members,
                list,
                exception_message=htype.generate_exception_message(
                    "TypedList.__init__", 10005
                ),
            )
            for i in range(len(init_members)):
                self.append(init_members[i])
        return

    # Getters
    @property
    def Members(self):
        return self

    @property
    def MemberTypes(self):
        return self.__member_types

    @property
    def IsRestrictive(self):
        return self.__is_restrictive

    @property
    def Name(self):
        return self.__name

    # Setters
    @Name.setter
    def Name(self, input):
        self.__name = input
        return

    # Representation
    def __str__(self):

        print("------- Element List -------")
        print("--- index, type, member ----")
        for i in range(self.__len__()):
            print(i, ", ", type(self.Members[i]), ", ", self.Members[i])
        return ""

    # Tools
    def lock_restrictiveness(self):
        """This method sets the flag self.__restrictiveness_is_locked to True. Once that flag
        is set to True, the value of the flag self.__is_restrictive cannot be switched.
        """
        self.__restrictiveness_is_locked = True
        return

    def switch_restrictiveness(self):
        """If the flag self.__restrictiveness_is_locked is False, then this method switches
        the flag __is_restrictive. Switching it from True to False causes no effect in any
        instance variable, apart from self.__is_restrictive. Switching it from False to True
        automatically adds every type from self.__members elements to self.__member_types.
        If the flag self.__restrictiveness_is_locked is True, then this method does nothing.
        """

        if not self.__restrictiveness_is_locked:
            if self.IsRestrictive == False:
                for i in range(self.__len__()):
                    if type(self.Members[i]) not in self.MemberTypes:
                        self.MemberTypes.append(type(self.Members[i]))

            self.__is_restrictive = not self.IsRestrictive
        else:
            raise Exception(
                htype.generate_exception_message(
                    "TypedList.switch_restrictiveness",
                    20001,
                    extra_info="Restrictiveness is locked. It cannot be switched.",
                )
            )

    def append(self, __candidate_object):
        """This method overrides builtin list.append(). Its aim is to add
        __candidate_object to the typed list. If __is_restrictive==False, the addition
        is performed unconditionally. If else, the addition is performed only if
        type(candidate_element) belongs to self.__member_types. Also, unlike list.append(),
        this method returns True if __candidate_object was finally added to the typed-list,
        and False if else.
        """

        fAdd = False
        if self.IsRestrictive == False:
            fAdd = True
        else:
            for type in self.MemberTypes:
                if isinstance(__candidate_object, type):
                    fAdd = True
                    break
        if fAdd:
            super().append(__candidate_object)
        else:
            print(
                "In function TypedList.try_adding_an_element():ERR0: Could not add an element."
            )
        return fAdd

    def extend(self, a_list):
        """This method overrides builtin list.extend() for TypedList objects."""

        htype.check_type(
            a_list,
            list,
            TypedList,
            exception_message=htype.generate_exception_message(
                "TypedList.extend", 30001
            ),
        )
        for i in range(len(a_list)):
            self.append(a_list[i])  # Type checks is managed by TypedList.append().
        return

    def remove_member_by_index(self, i):
        """This method lets you remove one member of the typed list by providing its matching
        iterator value within the members list. This method returns True if the removal was
        successful, and False if else."""

        htype.check_type(
            i,
            int,
            np.int64,
            exception_message=htype.generate_exception_message(
                "TypedList.remove_member_by_index()", 40001
            ),
        )

        if i < 0 or i > (self.__len__() - 1):
            raise Exception(
                htype.generate_exception_message(
                    "TypedList.remove_member_by_index()",
                    40002,
                    extra_info="The provided value for the iterator is not valid.",
                )
            )
        del self.Members[i]
        return True

    next_id_ = 0

    @classmethod
    def get_new_ID(cls):
        cls.next_id_ += 1
        return cls.next_id_ - 1
