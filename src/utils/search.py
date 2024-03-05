import numpy as np

import src.utils.htype as htype
import src.utils.custom_exceptions as cuex

def array_equality(a, b):

    """This function gets two numpy arrays which must have the same shape and returns 
    True (resp. False) if they are equal (resp. not equal) entry-wise."""

    htype.check_type(   a, np.ndarray,
                        exception_message=htype.generate_exception_message("array_equality", 101))
    
    htype.check_type(   b, np.ndarray,
                        exception_message=htype.generate_exception_message("array_equality", 102))
    
    if np.shape(a)!=np.shape(b):
        raise cuex.ShapeException(htype.generate_exception_message("array_equality", 103))
        
    return bool(np.prod(a==b))

def find_neighbours(x_base, new_x):

    """This function gets:

    - x_base (unidimensional float numpy array): Must be sorted in ascending order
    and must contain at least two entries.
    - new_x (scalar float)

    This function searches for the nearest neighbours of new_x within x_base. This 
    function returns four scalar arguments, say i_low, warning_low, i_up and warning_up. 
    In this order, i_low (resp. i_up) is the iterator value for the nearest smaller 
    (bigger) neighbour of new_x within x_base. warning_low (warning_up) is False if 
    there is at least one entry within x_base that is smaller (bigger) than new_x. It 
    is True if else. In the latter case, the returned iterator value for the matching 
    neighbour is equal to -1, signalling that the search failed.
    """

    htype.check_type(   x_base, np.ndarray,
                        exception_message=htype.generate_exception_message( "find_neighbours", 
                                                                            201))
    if np.ndim(x_base)!=1:
        raise cuex.ShapeException(htype.generate_exception_message( "find_neighbours", 
                                                                    202))   
    if np.shape(x_base)[0]<2:
        raise cuex.ShapeException(htype.generate_exception_message( "find_neighbours", 
                                                                    203,
                                                                    extra_info="The input array must have at least two entries."))
    htype.check_type(   new_x, float, np.float64,
                        exception_message=htype.generate_exception_message( "find_neighbours", 
                                                                            204))
    
    if not array_equality(np.argsort(x_base), np.array(range(np.shape(x_base)[0]))):
        raise cuex.InvalidParameterDefinition(  htype.generate_exception_message(   "find_neighbours", 
                                                                                    205,
                                                                                    extra_info="The input array must be increasingly ordered."))
    fDone = False

    if new_x<x_base[0]:

        warning_low = True  
        warning_up = False

        i_low = -1
        i_up = 0

        fDone = True

    i = 1

    while not fDone and i<len(x_base):

        if new_x<x_base[i]: # At this point, the values within 
                            # base_x become bigger than new_x
            i_low = i-1
            i_up = i

            warning_low = False
            warning_up = False

            fDone = True

        i += 1
    
    if not fDone:

        warning_low = False
        warning_up = True

        i_low = x_base.shape[0]-1
        i_up = -1

        fDone = True

    if not fDone:
        raise cuex.MalFunction(htype.generate_exception_message("find_neighbours",
                                                                206,
                                                                extra_info="Something is not working as expected. fDone must be True at this point."))
    return i_low, warning_low, i_up, warning_up

def find_nearest_neighbour(x_base, new_x):

    """This function gets:

    - x_base (unidimensional float numpy array): Must be sorted in ascending order
    and must contain at least two entries.
    - new_x (scalar float)

    This function looks for the nearest neighbour of new_x among the entries of x_base. 
    This function returns the iterator value for such nearest neighbour, as well as the 
    value of the desired entry itself."""

    # All of the requirements that must be met by x_base 
    # and new_x are checked in find_neighbours()

    i_low, warning_low, i_up, warning_up = find_neighbours(x_base, new_x) 

    if warning_low==True and warning_up==False:

        return i_up, x_base[i_up]
    
    elif warning_low==False and warning_up==True:

        return i_low, x_base[i_low]
    
    else:                                           # If x_base contains at least one element, then warning_low 
                                                    # and warning_up cannot be True at the same time, so this 
        up_distance = np.abs(new_x-x_base[i_up])    # case is warning_low=warning_up=True. In this case, decide 
        low_distance = np.abs(new_x-x_base[i_low])  # which neighbour is actually closer to new_x
            
        if up_distance<=low_distance:
            return i_up, x_base[i_up]
        else:
            return i_low, x_base[i_low]