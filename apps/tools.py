# The contents of this file, which are basically utilities used by the applications, 
# should be integrated elsehow into the massibo_ana package. This is a temporal solution.

from massibo_ana.core.Waveform import Waveform
from massibo_ana.core.SiPMMeas import SiPMMeas
from massibo_ana.core.GainMeas import GainMeas
from massibo_ana.core.DarkNoiseMeas import DarkNoiseMeas
import massibo_ana.utils.htype as htype
import massibo_ana.utils.custom_exceptions as cuex

import os
import copy
import math
import numpy as np
import pandas as pd
from scipy import interpolate as spinter
from scipy import optimize as spopt
from typing import List, Optional

strip_ids_of_set = {
    1: [
        '1425', '1426', '1429', '1431', '1432', 
        '1435', '1436', '1437', '1438', '1441', 
        '1442', '1443', '1444', '1445', '1446', 
        '1447', '1449', '1451', '1452', '1456'
    ],
    2: [
        # The following boards were measured with OVs 2.6, 3. and 3.4 V
        '2210', '2213', '2214', '2216', '2218', '2219',
        # The following boards were measured with OVs 2.6, 3. and 4. V
        '2220', '2221', '2223', '2224', '2229', '2230',
        '2234', '2239', '2240', '2243', '2251',
        # The following boards were measured with OVs 2., 3. and 4. V
        '2252', '2253', '2254'
    ],
    3: [
        # Measured with OVs 2., 3. and 4. V
        '5030', '5033', '5036', '5049', '5050',
        '5058', '5063', '6019', '6023', '6024',
        '6025', '5976', '6029', '6034', '4918',
        '6042', '6043', '6049', '6055', '6058'
    ],
    4: [
        # Measured with OVs 2., 3. and 4. V
        '416', '318', '287', '450', '304',
        '412', '225', '298', '311', '449',
        '257', '439', '250', '297', '309',
        '332', '540', '385', '477', '451'
    ],
    5: [
        # Measured with OVs 2., 3. and 4. V
        '1081', '1097', '1076', '1087', '1088',
        '1094', '1467', '1077', '1085', '1461',
        '1079', '1089', '1091', '1080', '1302',
        '1276', '1273', '1083', '1086', '1281'
    ],
    6: [
        # Measured with OVs 2., 3. and 4. V
        '7281', '7286', '7291', '7296', '8549',
        '8550', '8563', '8566', '8570', '8572',
        '8576', '8579', '8581', '8589', '8590',
        '8596', '8606', '8612', '8614'
    ],
    7: [
        # Measured with OVs 2., 3. and 4. V
        '5787', '5788', '5790', '7299', '7300',
        '7304', '7306', '7308', '7309', '7310',
        '7315', '7317', '7337', '7343', '7350',
        '7320', '7330', '7334', '7352', '7353'
    ],
    8: [
        # Measured with OVs 2., 3. and 4. V
        '1534', '1538', '1540', '1541', '1542',
        '1543', '1546', '1547', '1550', '1553',
        '1556', '1569', '1562', '1563', '1564',
        '1566', '1567', '1568', '1572', '1573'
    ]

}

thresholds = {
    'DCR_mHz_per_mm2': {
        'pre_threshold': 160,
        'threshold': 200
    },
    'XTP': {
        'pre_threshold': 0.28,
        'threshold': 0.35
    },
    'APP': {
        'pre_threshold': 0.04,
        'threshold': 0.05
    }
}

def plot_histogram(
        axes,
        *input_data,
        bins=None,
        hist_range=None,
        density=False,
        xlabel=None,
        ylabel=None,
        legend_labels=None,
        linewidth=1.,
        figtitle=None,
        fontsize=12,
        transparency_parameter=1.,
        yticks_step=None,
        colourful=False
    ):
    """This function gets the following positional arguments:
    
    - axes (matplotlib.axes.Axes): The axes where the histogram
    is to be plotted.
    - input_data (list of numpy arrays): The data to be plotted.
    The i-th numpy array within this list gives the samples for
    the i-th histogram.
    
    This function gets the following keyword arguments:
    
    - bins (int): The number of bins to be used in every histogram.
    It is given to the 'bins' parameter of the matplotlib.axes.Axes.hist
    method.
    - hist_range (tuple of floats): The range to be used in every
    histogram. It is given to the 'range' parameter of the
    matplotlib.axes.Axes.hist method.
    - density (boolean scalar): Whether to plot a probability density
    function or a counts-histogram. It is given to the 'density'
    parameter of the matplotlib.axes.Axes.hist method.
    - xlabel (string): The label to be used in the x-axis of the plot.
    - ylabel (string): The label to be used in the y-axis of the plot.
    - legend_labels (list of strings): The labels to be used in the
    legend of the plot. The i-th string within this list is the label
    for the i-th histogram.
    - linewidth (float): The linewidth to be used in the histograms.
    It is given to the 'linewidth' parameter of the matplotlib.axes.Axes.hist
    method.
    - figtitle (string): The title to be used in the plot.
    - fontsize (int): The fontsize to be used for the axes labels and
    the axes title.
    - transparency_parameter (float): The transparency to be used in
    the histograms. It is given to the 'alpha' parameter of the
    matplotlib.axes.Axes.hist method.
    - yticks_step (int): The step to be used in the y-axis ticks.
    - colourful (boolean scalar): If True, different colours are used
    for the histograms. If False, grayscale is used.

    This function plots the histograms of the given data in the given
    axes. The histograms are plotted using the matplotlib.axes.Axes.hist
    method.
    """

    legend_labels_ = [None for aux in input_data]
    fShowLegend = False
    if legend_labels is not None:
        legend_labels_ = legend_labels
        fShowLegend = True
    
    def infinite_color_generator():
        aux = [
            'blue', 'green', 'red', 'cyan', 'magenta', 'yellow',
            'black', 'orange', 'purple', 'brown', 'pink', 'gray',
            'olive', 'gold', 'lime', 'navy', 'teal', 'indigo',
            'violet', 'turquoise', 'darkred', 'darkblue',
            'darkgreen', 'darkorange', 'silver'
        ] if colourful else [
            'black', 'darkgray', 'gray', 'lightgray'
        ] # Grayscale

        i = 0
        while True:
            yield aux[i]
            i += 1

            if i==len(aux):
                i = 0

    gen = infinite_color_generator()

    n = []
    for i in range(len(input_data)):
        aux, bins, _ = axes.hist(
            input_data[i], 
            bins=bins,
            range=hist_range,
            density=density,
            alpha=transparency_parameter,
            label=legend_labels_[i],
            histtype='step',
            edgecolor=next(gen),
            linewidth=linewidth
        )

        n.append(aux)

    max_ytick = math.ceil(np.max(np.array([np.max(entry) for entry in n])))+1

    if yticks_step is None:
        yticks_step = math.ceil(max_ytick/5.)

    yticks, ytick = [0], 0
    while ytick<max_ytick:
        ytick += yticks_step
        yticks.append(ytick)

    axes.set_yticks(yticks)

    axes.set_xlabel(xlabel, fontsize=fontsize)
    axes.set_ylabel(ylabel, fontsize=fontsize)
    axes.set_title(figtitle, fontsize=fontsize)
    axes.grid()

    if fShowLegend:
        axes.legend()

def find_first_peak_width_at_level(
        x, 
        y, 
        level, 
        verbose=False
    ):
    """This function gets:
    
    - x, y (unidimensional float numpy array): x and y
    must have the same length. Jointly, they describe
    a set of 2D-points which make up a curve.
    - level (scalar float): It must belong to 
    (y[0], np.max(y)), i.e. it must be bigger
    than y[0] and smaller than np.max(y).
    - verbose (boolean scalar): Whether to print
    functioning-related messages.
    
    Assume that f interpolates the set of points 
    {x[i],y[i]}_i using cubic splines. Then, this 
    function returns b-a, where a (resp. b) is the
    first (resp. second) solution for x to 
    f(x)=level. By first (resp. second) solution we
    mean the smallest (resp. second smallest) value
    for x which matches f(x)=level.
    
    Since level\in(y[0], np.max(y)), at least one
    solution for f(x)=level exists. However, the
    existence of a second solution is not guaranteed.
    If a second solution does not exist, then this
    function raises an Exception.
    """

    # Skipping type-checks

    if len(x)!=len(y):
        raise cuex.InvalidParameterDefinition(
            htype.generate_exception_message(
                "find_first_peak_width_at_level", 
                1
            )
        )
    
    # Filter out infs and nans
    y_, mask = Waveform.filter_infs_and_nans(y, get_mask=True)
    x_ = x[mask]

    if level<=y_[0] or level>=np.max(y_):
        raise cuex.InvalidParameterDefinition(
            htype.generate_exception_message(
                "find_first_peak_width_at_level", 
                2
            )
        )

    first_solution_interval = (None, None)
    second_solution_interval = (None, None)

    for i in range(len(y_)):
        if y_[i]>level:
            first_solution_interval = (i-1, i)
            if verbose:
                print(
                    "Found first sign change in "
                    f"(y_[i],y_[i+1])={(y_[i],y_[i+1])}"
                )
            last_i = i
            break
    
    fFoundSecondSolution = False
    for i in range(last_i+1, len(y_)):
        if y_[i]<level:
            second_solution_interval = (i-1, i)
            if verbose:
                print(
                    f"Found second sign change in "
                    "(y_[i],y_[i+1])={(y_[i],y_[i+1])}"
                )
            fFoundSecondSolution = True
            break

    if not fFoundSecondSolution:
        raise Exception(
            "In function find_first_peak_width_at_level: "
            "A second solution was not found."
        )

    # Build interpolator 
    interpolator = spinter.CubicSpline(
        x_, 
        y_-level
    )    
    # Finding roots of g(x)=f(x)-level
    # is equivalent to finding solutions
    # of f(x)=level for x

    first_solution = spopt.brentq(
        interpolator, 
        x_[first_solution_interval[0]], 
        x_[first_solution_interval[1]],
        full_output=False
    )
    
    second_solution = spopt.brentq(
        interpolator, 
        x_[second_solution_interval[0]], 
        x_[second_solution_interval[1]],
        full_output=False
    )
    
    return second_solution-first_solution

def cluster_integers_by_contiguity(list_of_integers):
    """This function gets a list of integers, list_of_integers,
    then clusters them by contiguity. This is better understood 
    using an example:
    
        If list_of_integers is [1,2,3,5,6,8,10,11,12,13,16], 
        then this function will return the following list: 
        '[[1,3],[5,6],[8,8],[10,13],[16,16]]'.

    A more extreme is the one where we give the following input;
    [-30,-2,200,-29,1,2,3,5,6,8,10,11,12,13,16,-28],
    and get the following output:
    [[-30, -28], [-2, -2], [1, 3], [5, 6], [8, 8], [10, 13], [16, 16], [200, 200]]
    """

    htype.check_type(
        list_of_integers,
        list,
        exception_message=htype.generate_exception_message(
            "cluster_integers_and_produce_grouped_string",
            1
        )
    )
    
    # Isolate this ill-formed case
    if len(list_of_integers)==1:
        return [[list_of_integers[0], list_of_integers[0]]] 
    elif len(list_of_integers)<1:
        raise cuex.InvalidParameterDefinition(
            htype.generate_exception_message(
                "cluster_integers_and_produce_grouped_string",
                2
            )
        )

    for aux in list_of_integers:
        htype.check_type(
            aux,
            int,
            exception_message=htype.generate_exception_message(
                "cluster_integers_and_produce_grouped_string",
                3
            )
        )

    sorted_list_of_integers = sorted(list_of_integers)
    
    extremals = []
    extremals.append([sorted_list_of_integers[0]])
    
    # The last integer has an exclusive treatment
    for i in range(1, len(sorted_list_of_integers)-1):
        if sorted_list_of_integers[i]==sorted_list_of_integers[i-1]+1:
            pass
        else: # We have stepped into a new group
            extremals[-1].append(sorted_list_of_integers[i-1])
            extremals.append([sorted_list_of_integers[i]])

    # Taking care of the last element of the given list
    if sorted_list_of_integers[-1]==sorted_list_of_integers[-2]+1:
        extremals[-1].append(sorted_list_of_integers[-1])
    else:
        extremals[-1].append(sorted_list_of_integers[-2])
        extremals.append([sorted_list_of_integers[-1], sorted_list_of_integers[-1]])

    return extremals

def get_string_of_contiguously_clustered_integers(cluster):
    """This function gets a list of lists, cluster, where
    each list within cluster contains two integers, and
    outputs an string out of such list. For cluster equal
    to 
    [[i0,i1],[i2,i2],[i3,i4],...,[iM,iM],[iN,iR]], 
    the output is
    'i0-i1, i2, i3-i4,...,iM, iN-iR."""

    htype.check_type(
        cluster,
        list,
        exception_message=htype.generate_exception_message(
            "get_string_of_contiguously_clustered_integers",
            1
        )
    )

    for aux in cluster:
        htype.check_type(
            aux,
            list,
            exception_message=htype.generate_exception_message(
                "get_string_of_contiguously_clustered_integers",
                2
            )
        )
        
        if len(aux)!=2: 
            raise cuex.InvalidParameterDefinition(
                htype.generate_exception_message(
                    "get_string_of_contiguously_clustered_integers",
                    3
                )
            )
        
        for elem in aux:
            htype.check_type(
                elem,
                int,
                exception_message=htype.generate_exception_message(
                    "get_string_of_contiguously_clustered_integers",
                    4
                )
            )

    result = ''
    for aux in cluster:
        if aux[0]==aux[1]:
            result += f"{aux[0]}, "
        else:
            result += f"{aux[0]}-{aux[1]}, "
    return result[:-2]

def swap(x, y):
    return y, x
        
def decide_colour(
        sample,
        mean,
        std,
        discern_sign=True,
        red_above=True
    ):
    """red_above only makes a difference if discern_sign is True.
    If discern_sign is False, then the returned colours are either
    white or red(s)."""
    
    blue = (0.204,0.596,0.859)
    light_blue = (0.522,0.757,0.914)
    light_light_blue = (0.839,0.918,0.972)
    white = (1.0, 1.0, 1.0)
    light_light_red = (1, 0.8, 0.8)
    light_red = (1.0, 0.5, 0.5)
    red = (1.0, 0.0, 0.0)

    color_map = {   3:red, 2:light_red, 1:light_light_red, 
                    0:white,
                    -1:light_light_blue, -2:light_blue, -3:blue}
    if not red_above:        
        color_map[3], color_map[-3] = swap(color_map[3], color_map[-3])
        color_map[2], color_map[-2] = swap(color_map[2], color_map[-2])
        color_map[1], color_map[-1] = swap(color_map[1], color_map[-1])

    if math.isnan(sample):
        return color_map[0] # White
        
    fCode = None
    if discern_sign:
        aux = sample-mean
        if aux<=-3.*std:
            fCode = -3
        elif aux<=-2.*std:
            fCode = -2
        elif aux<=-1.*std:
            fCode = -1
        elif aux<=std:
            fCode = 0
        elif aux<=2.*std:
            fCode = 1
        elif aux<=3.*std:
            fCode = 2
        else:
            fCode = 3
    else:
        aux = np.abs(sample-mean)
        if aux<=std:
            fCode = 0
        elif aux<=2.*std:
            fCode = 1
        elif aux<=3.*std:
            fCode = 2
        else:
            fCode = 3

        multiplier = 1 if red_above else -1
        fCode *= multiplier

    return color_map[fCode]
    
def analyze_it(
        filename,
        analyzable_prefixes=[],
        conjuntive_analyzable_marks=[],
        disyuntive_analyzable_marks=[]
    ):
    """This function gets the following positional argument:
    
    - filename (string): Name of a file
    
    This function gets the following keyword arguments:
    
    - analyzable_prefixes (list of strings)
    - conjuntive_analyzable_marks (list of strings)
    - disyuntive_analyzable_marks (list of strings)
    
    This function returns True if filename starts with any of the
    strings contained within analyzable_prefixes and contains
    every substring contained within conjuntive_analyzable_marks,
    and at least one substring contained within analyzable_marks.
    Otherwise, it returns False."""

    htype.check_type(
        filename,
        str,
        exception_message=htype.generate_exception_message(
            "analyze_it", 
            1
        )
    )
    
    htype.check_type(
        analyzable_prefixes,
        list,
        exception_message=htype.generate_exception_message(
            "analyze_it", 
            2
        )
    )
    
    for elem in analyzable_prefixes:
        htype.check_type(
            elem,
            str,
            exception_message=htype.generate_exception_message(
                "analyze_it", 
                3
            )
        )

    htype.check_type(
        conjuntive_analyzable_marks,
        list,
        exception_message=htype.generate_exception_message(
            "analyze_it", 
            4
        )
    )
    
    for elem in conjuntive_analyzable_marks:
        htype.check_type(
            elem,
            str,
            exception_message=htype.generate_exception_message(
                "analyze_it", 
                5
            )
        )
        
    htype.check_type(
        disyuntive_analyzable_marks,
        list,
        exception_message=htype.generate_exception_message(
            "analyze_it", 
            6
        )
    )
    
    for elem in disyuntive_analyzable_marks:
        htype.check_type(
            elem,
            str,
            exception_message=htype.generate_exception_message(
                "analyze_it", 
                7
            )
        )
        
    fPrefixConditionIsMet = False
    for prefix in analyzable_prefixes:
        if filename.startswith(prefix):
            fPrefixConditionIsMet = True
            break
    
    if not fPrefixConditionIsMet:
        return False
    else:
        for mark in conjuntive_analyzable_marks:
            if mark not in filename:
                return False
        for mark in disyuntive_analyzable_marks:
            if mark in filename:
                return True
            
        # Reaching this point means that the prefix condition is met
        # and that the conjuntive_analyzable_marks condition is met,
        # but the disyuntive_analyzable_marks condition is not met, so
        # return False.    
        return False
    
def filter_out_repeated_smss(
        smsset,
        return_filter_history=False,
        verbose=False
    ):
    """This function gets the following positional argument:
    
    - smsset (SMSSet): The objects within this SMSSet object must
    have an entry for the 'thermal_cycle' key.

    This function gets the following keyword arguments:

    - return_filter_history (bool): If True, this function
    additionally returns a dictionary which gives information
    about the measurements that were filtered out from the
    given SMSSet object. The keys of this dictionary are the
    IDs of such objects, and the values are lists which contain
    the thermal cycles of the measurements that were filtered out
    for each ID.

    - verbose (bool): Whether to print functioning-related information.

    This function returns a SMSSet whose SiPMMeasSummary are taken
    from the input SMSSet object, but their IDs are unique among the
    objects in the returned SMSSet. The objects which are not kept are
    those whose ID is repeated in the input SMSSet and their
    'thermal_cycle' key is not the maximum among the repeated IDs.
    This is the way to ensure that the returned SMSSet contains the
    most recent measurements."""

    aux_smsset = copy.deepcopy(smsset)
    filter_history = {}

    i=0
    while i < len(aux_smsset):
        max_thermal_cycle = aux_smsset[i]['thermal_cycle']
        j = i+1
        while j < len(aux_smsset):
            if aux_smsset[j].ID == aux_smsset[i].ID:
                if aux_smsset[j]['thermal_cycle'] > max_thermal_cycle:
                    max_thermal_cycle = aux_smsset[j]['thermal_cycle']

                    # Since we are fixing the i-th element of the
                    # list, so that we can compare its ID to the
                    # rest of the elements of the list by using a
                    # second iterator (j, with j>i), we cannot modify
                    # the strip ID of the i-th element of the list
                    # during the search. So, in case that's the 
                    # element we need to delete, because its thermal 
                    # cycle is actually smaller than that of the 
                    # j-th element of the list, we shall first put 
                    # the object with the smaller thermal cycle at 
                    # the j-th position (remember j>i), and then 
                    # delete it.
                    aux_smsset[i], aux_smsset[j] = aux_smsset[j], aux_smsset[i]

                elif aux_smsset[j]['thermal_cycle'] == max_thermal_cycle:                                       
                    raise Exception(
                        f"In function filter_repeated_old_smss(): Spotted two "
                        "SiPMMeasSummary objects with the same ID "
                        f"({aux_smsset[i].ID}) and the same entry for "
                        f"'thermal_cycle' ({aux_smsset[i]['thermal_cycle']}).")
                
                if verbose:
                    print(
                        f"Deleting SiPMMeasSummary object with repeated ID "
                        f"{aux_smsset[i].ID} and thermal cycle "
                        f"{aux_smsset[j]['thermal_cycle']} (keeping one "
                        f"with thermal_cycle={max_thermal_cycle}")
                try:
                    filter_history[aux_smsset[j].ID].append(aux_smsset[j]['thermal_cycle'])
                except KeyError:
                    filter_history[aux_smsset[j].ID] = [aux_smsset[j]['thermal_cycle']]

                # No need to increment j, since we are
                # deleting the j-th element of the list
                del aux_smsset[j]

            else:
                j += 1
        i += 1
    
    if not return_filter_history:
        return aux_smsset
    else:   
        return aux_smsset, filter_history
    
def strip_ID_vs_sipm_location_dataframe(
        data_df, 
        field_to_show, 
        significant_figures=2
    ):
    """Returns a pandas DataFrame with 7 rows:
    
    - the row i=0 is the header of the table and contains
    the strip ID values found in the given data_df DataFrame.
    - the i-th column, with i>=1, contains the value of the
    field_to_show entry for the i-th SiPM location of the
    strip with strip ID equal to the i-th column header.
    
    Note that the values of the returned DataFrame are
    strings in scientific notation with significant_figures
    significant figures."""

    groups = data_df.groupby('strip_ID').groups
    table = pd.DataFrame(
        columns=groups.keys(),
        index=range(1,7)
    )

    for strip_ID in groups.keys():
        for row_idx in groups[strip_ID]:
            sipm_location = data_df['sipm_location'][row_idx]

            # This works because I know for sure
            # that sipm_location\in[1,2,3,4,5,6]
            table.loc[sipm_location, strip_ID] = \
                scientific_notation_str(
                    data_df[field_to_show][row_idx],
                    ndigits=significant_figures
                )

    return table

def build_list_of_SiPMMeas_objects(
        jsons_folder_path: str,
        strips_to_analyze: List[str],
        analyzable_marks: List[str],
        set_to_analyze: Optional[int] = None,
        is_gain_meas: bool = False,
        verbose: bool = False
) -> List[SiPMMeas]:
    """This function gets the following positional argument:
    
    - jsons_folder_path (string): The folder path where to look
    for the JSON files from which SiPMMeas objects (either GainMeas
    or DarkNoiseMeas objects, up to the is_gain_meas parameter)
    will be loaded.
    - strips_to_analyze (list of strings): This parameter only makes
    a difference if set_to_analyze is None. In such case, it gives
    the strip IDs of the strips whose measurements are to be analyzed.
    - analyzable_marks (list of strings): For a JSON file to be
    loaded, its name must contain at least one of the strings
    contained within the analyzable_marks list.

    This function gets the following keyword arguments:

    - set_to_analyze (int or None): If this parameter is not null,
    then it must be set to an integer which identifies the measurements
    set to be analyzed. I.e. the strip IDs to be analyzed will be
    taken from a global variable called strip_ids_of_set, which is a
    dictionary whose keys are integers (set numbers) and whose values
    are lists of strip IDs. If this parameter is null, then the
    strips_to_analyze parameter above is used to determine which strips
    are to be analyzed.
    - is_gain_meas (bool): Whether the SiPMMeas objects to be
    loaded are GainMeas or DarkNoiseMeas objects. If True (resp.
    False), only JSON files whose name contains the 'gain' (resp.
    darknoise) substring will be loaded.
    - verbose (bool): Whether to print functioning-related messages.

    This function returns a list of SiPMMeas objects, each of
    which has been loaded from a single JSON file in the specified
    folder."""

    if set_to_analyze is None:
        strips_to_analyze_ = strips_to_analyze

    else:
        try:
            strips_to_analyze_ = strip_ids_of_set[set_to_analyze]

        except KeyError:
            raise cuex.NoAvailableData(
                "In function build_list_of_SiPMMeas_objects(): "
                f"Set {set_to_analyze} not found in available "
                f"sets: {tuple(strip_ids_of_set.keys())}"
            )
        
    if verbose:
        print(
            "In function build_list_of_SiPMMeas_objects(): "
            f"Targetting the following strip IDs: {tuple(strips_to_analyze_)}"
        )

    if is_gain_meas:
        cls = GainMeas
    else:
        cls = DarkNoiseMeas

    output = []
    for json_file in sorted(os.listdir(jsons_folder_path)):

        if analyze_it(
            json_file, 
            analyzable_prefixes=[
                strip_id+'-' for strip_id in strips_to_analyze_
            ],
            conjuntive_analyzable_marks=['gain'] if is_gain_meas else ['darknoise'],
            disyuntive_analyzable_marks=analyzable_marks
        ):

            if verbose:
                print(
                    "In function build_list_of_SiPMMeas_objects():"
                    f" Reading {json_file}"
                )

            filepath = os.path.join(
                jsons_folder_path,
                json_file
            )

            if os.path.isfile(filepath):
                output.append(cls.from_json_file(filepath))
    
    return output

def order_list_of_SiPMMeas_objects(
    input: List[SiPMMeas]
) -> List[SiPMMeas]:
    """This function gets the following positional argument:
    
    - input (list of SiPMMeas objects): The list of SiPMMeas
    objects to be ordered.

    This function returns a list of SiPMMeas objects, which is 
    the input list but ordered according to the StripID and the
    SiPMLocation attributes of each SiPMMeas object."""

    # Assuming that, if the first element of input is a DarkNoiseMeas,
    # object then all the elements of input are DarkNoiseMeas objects
    if isinstance(input[0], DarkNoiseMeas):
        # Implement the ordering-criterion here
        # This ordering criterion assumes that the SiPMLocation
        # is not bigger or equal to 10. Otherwise the ordering
        # will not work.

        ordering_idcs = np.argsort(
            [(10 * int(aux.StripID)) + int(aux.SiPMLocation) for aux in input]
        )
    
    # Assuming that, if the first element of input is not a DarkNoiseMeas
    # object, then every object is a GainMeas object
    else:
        # Implement the ordering-criterion here
        # This ordering criterion assumes that the SiPMLocation
        # and the Overvoltage_V attributes are not bigger or equal
        # to 10. Otherwise the ordering will not work.
        ordering_idcs = np.argsort(
            [(100 * int(aux.StripID)) + (10 * int(aux.SiPMLocation)) + aux.Overvoltage_V for aux in input]
        )

    n_objects = len(input)

    # Craft the ordered list
    output = [None] * n_objects
    for i in range(n_objects):
        output[ordering_idcs[i]] = input[i]

    return output

def filter_out_zeros_and_infs(
        input: list
) -> list:
    """This function gets the following positional argument:

    - input (list of floats): The list to be filtered.

    This function returns a list which is the input list but
    without the entries which are either 0.0 or inf."""

    for i in reversed(range(len(input))):
        if input[i]==0. or math.isinf(input[i]):
            del input[i]

    return input

def scientific_notation_str(
        number,
        ndigits
    ):
    """This function gets the following positional arguments:

    - number (scalar float): The number to be converted to
    scientific notation.
    - ndigits (scalar int): The number of digits to be shown
    in the scientific notation.

    This function returns a string which is the scientific
    notation of the input number with the specified number
    of digits."""

    return f"{number:.{ndigits - 1}e}"

def natural_numbers_generator():

    i = 1
    while True:
        yield i
        i += 1