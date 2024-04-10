import numpy as np

import src.utils.htype as htype
import src.utils.custom_exceptions as cuex
from src.custom_types.RigidKeyDictionary import RigidKeyDictionary
from src.custom_types.TypedList import TypedList

class ListsRKD(RigidKeyDictionary):

    def __init__(self, potential_keys, is_subtyped=False, values_subtypes=None):

        """This class aims to implement a RigidKeyDictionary whose keys are all typed.
        In particular, every value must be of type list or TypedList (which inherits 
        from list). See RigidKeyDictionary and TypedList classes documentation for 
        more information. Since every value is a list, as of now we will refer to 
        the elements in such lists as subvalues. The types of these subvalues are 
        the subtypes. This initializer gets the following mandatory positional argument:

        -potential_keys (list): It is passed to the initializer of RigidKeyDictionary as 
        potential_keys. See RigidKeyInitializer.__init__ docstring for more information.

        This initializer also gets the following keyword arguments:

        - is_subtyped (boolean, or a list of len(potential_keys) booleans): If the given 
        parameter is equal to False (resp. True), then no (resp. every) potential key 
        is subtyped (a potential key is said to be subtyped if the elements in its value 
        must comply with a certain type). If the given parameter is a list of booleans, 
        then is_subtyped[i] gives whether the i-th potential key (potential_key[i]) is 
        subtyped or not.

        - values_subtypes (list of types): This list must contain as many types as typed 
        potential keys. If is_subtyped==False, or is_subtyped is a list full of "False", 
        then values_subtypes is ignored. If is_subtyped==True, then values_subtypes must 
        contain N types (where N is the number of potential keys). Otherwise, if is_subtyped 
        is a list of booleans with M entries equal to True, with M>=1, then values_subtypes 
        must contain M types. In the last case, values_subtypes[i] gives the subtype for 
        the i-th typed key, up to the order set by potential_keys. 
        
        For now, the potential keys and its typing property are fixed as of the object
        initialization. In the future, appropriate setters methods for these attributes 
        could be implemented."""

        htype.check_type(   potential_keys, list,
                            exception_message=htype.generate_exception_message("ListsRKD.__init__", 10001))
        fIsList = False
        if type(is_subtyped)==bool:
            pass
        elif type(is_subtyped)==list:
            if len(is_subtyped)!=len(potential_keys):
                raise cuex.InvalidParameterDefinition(htype.generate_exception_message("ListsRKD.__init__", 10002))
            for element in is_subtyped:
                htype.check_type(   element, bool,
                                    exception_message=htype.generate_exception_message("ListsRKD.__init__", 10003))
            fIsList = True
        else:
            raise cuex.TypeException(htype.generate_exception_message("ListsRKD.__init__", 10004))
        
        fIgnoreValuesSubTypes = False
        if not fIsList:
            if is_subtyped is False:
                fIgnoreValuesSubTypes=True
        else:
            if bool(np.prod(np.logical_not(is_subtyped))):
                fIgnoreValuesSubTypes = True

        if not fIgnoreValuesSubTypes:   # In this case, we have to check 
                                        # the proper format of values_types
            htype.check_type(   values_subtypes, list,
                                exception_message=htype.generate_exception_message("ListsRKD.__init__", 10005))
            for element in values_subtypes:
                htype.check_type(   element, type, 
                                    exception_message=htype.generate_exception_message("ListsRKD.__init__", 10006))

            if not fIsList: # is_typed must be True
                aux = len(potential_keys)
            else:
                aux = np.array(is_subtyped).sum()

            if len(values_subtypes)!=aux:  # aux is the number of typed potential keys
                raise cuex.InvalidParameterDefinition(htype.generate_exception_message("ListsRKD.__init__", 10007))

        values_types = []
        self.__values_subtypes = None
        if fIgnoreValuesSubTypes:   # Either is_subtyped==False or
                                    # is_subtyped==[False, False, ..., False]
            values_types = [list for x in potential_keys]
            self.__values_subtypes = [object for x in potential_keys]
        else:
            if not fIsList: # is_subtyped must be True
                values_types = [TypedList for x in potential_keys] 
                self.__values_subtypes = values_subtypes
            else:   # In this case, aux is equal to 
                    # the number of True's in is_subtyped
                values_types = [
                    TypedList if is_subtyped[i] is True else list 
                    for i in range(len(potential_keys))
                ]

                gen = (n for n in range(0, aux))
                self.__values_subtypes = [
                    values_subtypes[next(gen)] if is_subtyped[i] is True else object 
                    for i in range(len(potential_keys))
                ]

        super().__init__(   potential_keys, 
                            is_typed=True, 
                            values_types=values_types)
        return
    
    @property
    def ValuesSubtypes(self):
        return self.__values_subtypes
    
    def __setitem__(self, __key, __value, append=False):

        """ This method overrides RigidKeyDictionary.__setitem__() method. This method
        gets the following positional arguments:
        
        - __key (object): Proposed key for a new entry of the ListsRKD whose value
        would be __value.
        - __value (list or TypedList): Proposed value for a new entry of the ListsRKD
        whose key would be __key. 

        For the entry to be actually added to the dictionary, some Addition Conditions 
        (AC) must be met:
        - AC1: __key belongs to self.PotentialKeys.
        - AC2: isinstance(__value, list) (resp. isinstance(__value, TypedList)) if the 
        given key is not subtyped (resp. is subtyped).
        - AC3: Elements of __value comply with its subtype.

        This method gets the following keyword argument:

        - append (boolean): Since the values of a ListsRKD are instances of list, the 
        functionality of the __setitem__ method can be naturally extended in the 
        following way. This parameter only makes a difference if AC1, AC2 and AC3 are met
        and __key is already in self.keys(), i.e. if there's already one entry with such 
        key. Then, if append==False, the entry (__key, __value) is added to the ListsRKD, 
        maybe overwritting a previous entry with the same key. Else, assume append==True. 
        In this case, if __key is not already present in the ListsRKD, the entry (__key, 
        __value) is added to the ListsRKD. If __key was already present in the ListsRKD, 
        say in an entry (__key, previous_value), then this entry is updated by appending 
        all of the entries of __value to previous_value, respecting the order set by 
        __value.   """

        try:    #AC1
            idx = self.PotentialKeys.index(__key)
        except ValueError:  # __key was not found
                            # in self.PotentialKeys
            return
        
        if not isinstance(__value, self.ValuesTypes[idx]):  # AC2
            raise cuex.InvalidParameterDefinition(htype.generate_exception_message( "ListsRKD.__setitem__",
                                                                                    20001))
        fPass = True                    
        for i in range(len(__value)):   
            if not isinstance(__value[i], self.ValuesSubtypes[idx]): # AC3
                fPass = False

        if fPass:
            if __key in self.keys() and append:
                value = self.__getitem__(__key).extend(__value) # self.__getitem__(__key) must be
                                                                # a list or a TypedList. extend 
                                                                # method is suitably overriden for
                                                                # TypedList's
            else:
                value = __value

            return super().__setitem__(__key, value)
        else:
            return