import os
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import subprocess
import papermill
import yaml
import shutil

TESTS_DATA_FOLDER = 'tests/regression/data'
SETS_PATHS_YAML = 'tests/regression/sets_filepaths.yml'
TEMPORAL_WORKSPACE_FOLDER = 'tests/tmp'
OUTPUT_FOLDER = 'tests/regression/output'
SETS = list(range(1, 9))

def run_analysis(
        set_number: int,
        config_path: str,
        temporal_workspace_folder: str,
    ):
    """This function runs the analysis for a given set number, using
    the data that should have been copied to the temporal workspace
    folder by the bring_data_to_temporal_workspace() function. The
    function first runs the preprocessor, and then it runs either
    the darknoise analysis or the gain analysis, depending on
    whether the name of the config file contains 'darknoise' or
    'gain'. If the name of the config file does not contain any
    of those two keywords, the function raises an exception with
    an informative message.
    """

    _run_preprocessor(
        set_number,
        temporal_workspace_folder
    )

    if 'darknoise' in os.path.basename(config_path):
        _run_darknoise_analysis(
            config_path,
            temporal_workspace_folder
        )

    elif 'gain' in os.path.basename(config_path):
        _run_gain_analysis(
            config_path,
            temporal_workspace_folder
        )

    else:
        raise Exception(
            "In function run_analysis(): The name of the config file "
            f"{config_path} does not contain 'darknoise' nor 'gain', "
            "so the type of analysis to run cannot be determined."
        )

def _run_preprocessor(
        set_number: int,
        temporal_workspace_folder: str
):
    """This function runs the preprocessor for a given set number,
    using the data that should have been copied to the temporal
    workspace folder by the bring_data_to_temporal_workspace() function.
    The function modifies the preprocessor parameters YAML file to
    set the correct input data folder, workspace folder, and input
    data format code for the given set number, and then it runs the
    preprocessor by executing the corresponding shell script. After
    running the preprocessor, the function restores the original
    preprocessor parameters YAML file, by moving the backup copy to
    the original location.
    """

    # Create a backup copy of the original preprocessor
    # parameters YAML file, to restore it after running
    # the preprocessor
    preprocessor_parameters_path = Path(
        "apps/parameters_examples/preprocessor_parameters.yml"
    )
    preprocessor_parameters_backup_path = Path(
        "apps/parameters_examples/preprocessor_parameters_backup.yml"
    )
    if not preprocessor_parameters_backup_path.exists():
        subprocess.run(
            [
                "cp",
                str(preprocessor_parameters_path),
                str(preprocessor_parameters_backup_path)
            ],
            check=True
        )
    else:
        raise Exception(
            "In function _run_preprocessor(): The backup copy of the "
            "preprocessor parameters YAML file already exists. This "
            "should not happen, because the backup copy should be "
            "removed after running the preprocessor for each set."
        )

    # Load the preprocessor parameters YAML file, change the
    # values of the parameters, and save the modified YAML file
    with open(preprocessor_parameters_path, 'r') as file:
        preprocessor_parameters = yaml.safe_load(file)

    preprocessor_parameters['input_data_folderpath'] = \
        str(Path(temporal_workspace_folder) / 'input_data')
    preprocessor_parameters['workspace_dir'] = \
        str(Path(temporal_workspace_folder))
    preprocessor_parameters['input_data_format_code'] = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 2,
        7: 2,
        8: 2
    }[set_number]

    with open(preprocessor_parameters_path, 'w') as file:
        yaml.safe_dump(preprocessor_parameters, file)

    subprocess.run(
        ["./ci/aux/b1_run_preprocessor.sh"],
        check=True
    )

    # Restore the original preprocessor parameters YAML file
    subprocess.run(
        [
            "mv",
            str(preprocessor_parameters_backup_path),
            str(preprocessor_parameters_path)
        ],
        check=True
    )

    return

def _run_darknoise_analysis(
        path_to_darknoise_YAML: str,
        temporal_workspace_folder: str
):
    """This function runs the darknoise analysis for a given set number,
    using the data that should have been copied to the temporal
    workspace folder by the bring_data_to_temporal_workspace() function,
    and that should have been preprocessed by the _run_preprocessor()
    function. The function modifies the darknoise analysis parameters
    YAML file to set the correct workspace folder and to disable the
    generation of the report, and then it runs the darknoise analysis
    by executing the corresponding shell script. After running the
    darknoise analysis, the function restores the original darknoise
    analysis parameters YAML file, by moving the backup copy to the
    original location.
    """

    # Create a backup copy of the original darknoise analysis
    # parameters YAML file, to restore it after running
    # the darknoise analysis
    darknoise_parameters_path = Path(path_to_darknoise_YAML)
    darknoise_parameters_backup_path = Path(
        path_to_darknoise_YAML.replace('.yml', '_backup.yml')
    )
    if not darknoise_parameters_backup_path.exists():
        subprocess.run(
            [
                "cp",
                darknoise_parameters_path,
                darknoise_parameters_backup_path
            ],
            check=True
        )
    else:
        raise Exception(
            "In function _run_darknoise_analysis(): The backup copy of the "
            "darknoise analysis parameters YAML file already exists. This "
            "should not happen, because the backup copy should be "
            "removed after running the darknoise analysis for each set."
        )
    
    # Load the darknoise analysis parameters YAML file, change the
    # values of the parameters, and save the modified YAML file
    with open(darknoise_parameters_path, 'r') as file:
        darknoise_parameters = yaml.safe_load(file)

    darknoise_parameters['workspace_dir'] = str(
        Path(temporal_workspace_folder)
    )
    darknoise_parameters['generate_report'] = False
    
    with open(darknoise_parameters_path, 'w') as file:
        yaml.safe_dump(darknoise_parameters, file)

    subprocess.run(
        [
            "./ci/aux/b3_run_darknoise_batch_analyzer.sh",
            path_to_darknoise_YAML
        ],
        check=True
    )

    # Restore the original darknoise analysis parameters YAML file
    subprocess.run(
        [
            "mv",
            darknoise_parameters_backup_path,
            darknoise_parameters_path
        ],
        check=True
    )

    return

def _run_gain_analysis(
        path_to_gain_YAML: str,
        temporal_workspace_folder: str
):
    """Should do the same as _run_darknoise_analysis(),
    but for the gain analysis."""
    pass

def find_config_and_desired_output(
        set_number: int,
        tests_data_folder: str
    ) -> tuple[str, str]:
    """Given a set number and the path to the folder
    which contains the regression test data, this function
    searches for the corresponding set folder, and within
    that folder, it looks for the YAML config file and the
    expected output CSV file. It returns the paths to both
    files, in the mentioned order. If any of the required
    files or folders are not found, it raises an exception
    with an informative message.
    """

    set_folder_path = None
    pattern = f"set_{set_number}"
    for folder_path in Path(tests_data_folder).iterdir():
        if folder_path.is_dir() and pattern in folder_path.name:
            set_folder_path = folder_path
            break
    
    if set_folder_path is None:
        raise FileNotFoundError(
            "In function find_config_and_desired_output(): No folder "
            f"found for set number {set_number}"
        )
    
    input_yaml_config_path = None
    for file in Path(set_folder_path).iterdir():
        if file.is_file() and file.suffix == '.yml':
            input_yaml_config_path = file
            break

    if input_yaml_config_path is None:
        raise FileNotFoundError(
            "In function find_config_and_desired_output(): No YAML "
            f"config file found in folder {set_folder_path}"
        )
    
    expected_csv_path = None
    for file in Path(set_folder_path).iterdir():
        if file.is_file() and file.suffix == '.csv':
            expected_csv_path = file
            break

    if expected_csv_path is None:
        raise FileNotFoundError(
            "In function find_config_and_desired_output(): No expected "
            f"output CSV file found in folder {set_folder_path}"
        )
    
    return str(input_yaml_config_path), str(expected_csv_path)

def bring_data_to_temporal_workspace(
        set_number: int,
        temporal_workspace_folder: str,
        sets_paths_yaml: str
    ) -> None:
    """Given a set number, the path to the YAML file which
    contains the paths to the folders with the data for each
    set, and the path to the temporal workspace folder, this
    function looks for the path to the folder with the data
    for the given set number in the YAML file, and then
    copies that folder into the temporal workspace folder,
    under the input_data subfolder. If any of the required
    files or folders are not found, it raises an exception
    with an informative message.
    """
    
    with open(sets_paths_yaml, 'r') as f:
        sets_paths = yaml.safe_load(f)

    try:
        set_path = sets_paths[set_number]

    except KeyError:
        raise Exception(
            "In function bring_data_to_temporal_workspace(): No "
            f"path found for set number {set_number} in YAML file "
            f"{sets_paths_yaml}"
        )
    
    # Copy the set_path folder into temporal_workspace / input_data
    destination_folder = Path(temporal_workspace_folder) / 'input_data'
    subprocess.run(
        ["cp", "-r", set_path, destination_folder],
        check=True
    )

    return

def find_generated_output_csv(
        temporal_workspace_folder: str
    ) -> Path:
    """Find the path to the generated output CSV file in
    the temporal workspace folder.
    
    This function looks into the summary subfolder of the
    temporal workspace folder, and it looks for the (unique)
    generated output folder that should have been created
    after running the analysis for a set. Then, it looks for
    the CSV file that should have been generated in that
    folder. Before returning the path to that CSV file, it
    renames it by adding the 'computed_' prefix to the name
    so that the labels in the generated comparison plots let
    the user tell apart the generated output from the expected
    output. If any of the required folders or files are not
    found, it raises an exception with an informative message.
    """

    output_folder = Path(temporal_workspace_folder) / 'summary'
    for dir in output_folder.iterdir():
        # The workspace should have been reset before running
        # the analysis for each set, so look for the (unique)
        # results folder that should have been generated in
        # summary after running the analysis
        if dir.is_dir() and not dir.name.startswith('.'):
            for file in dir.iterdir():
                if file.is_file() and file.suffix == '.csv':

                    # Add the 'computed_' prefix to the name of the 
                    # generated output CSV file and return it so that
                    # the labels in the generated comparison plots let
                    # the user tell apart the generated output from
                    # the expected output

                    new_file_path = file.parent / f"computed_{file.name}"
                    file.rename(new_file_path)

                    return new_file_path
                
    raise Exception(
        "In function find_generated_output_csv(): Could not find "
        f"a single generated output folder in {output_folder}"
    )

def assert_dataframes_compatibility(
        needed_columns: list,
        expected_dataframe: pd.DataFrame,
        computed_dataframe: pd.DataFrame,
) -> None:
    """This function takes as input two dataframes and a
    list of needed columns, and it makes sure that

        1)  The expected dataframe contains all the 
            columns that are needed for the comparison
        2)  The computed dataframe contains, at least,
            all the columns that are present in the
            expected dataframe
        3)  Both dataframes have the same number of
            rows (i.e. same number of SiPMs)
    """

    assert set(needed_columns).issubset(expected_dataframe.columns), (
        "In function assert_dataframes_compatibility(): The expected "
        "output CSV file does not contain all the columns that are "
        "needed for the comparison between the generated output and "
        "the expected output"
    )

    assert set(expected_dataframe.columns).issubset(computed_dataframe.columns), (
        "In function assert_dataframes_compatibility(): The generated "
        "output CSV file does not contain all the columns that are present "
        "in the expected output CSV file"
    )

    assert len(expected_dataframe) == len(computed_dataframe), (
        "In function assert_dataframes_compatibility(): The generated "
        "output CSV file does not have the same number of rows as the "
        "expected output CSV file"
    )

    return

def generate_comparison_plots(
        expected_csv_path: str,
        computed_csv_path: str,
        set_number: int,
        is_darknoise: bool,
        output_folder_path: str,
    ):
    """This function takes as input the paths to the expected
    output CSV file and to the generated output CSV file, and it
    generates comparison plots between both CSV files. The generated
    plots are saved in a PDF file in the given output folder, with a
    a name that includes the set number. The function uses the
    papermill library to execute the notebook
    apps/csv_vs_csv_comparison.ipynb, which is responsible for
    generating the comparison plots. The function passes the paths
    to the expected and generated CSV files as parameters to the
    notebook, as well as the path where the output PDF file should
    be saved.
    """

    test_type = 'darknoise' if is_darknoise else 'gain'

    papermill.execute_notebook(
        "apps/csv_vs_csv_comparison.ipynb",
        "/dev/null", # Throw away the executed notebook
        parameters={
            'csv_filepaths': [
                str(expected_csv_path),
                str(computed_csv_path)
            ],
            'REPORT_OUTPUT_FILEPATH': \
                output_folder_path + \
                    f'/set_{set_number}_{test_type}_comparison_plots.pdf'
        }
    )

    return

@pytest.mark.parametrize(
    "is_darknoise",
    [True, False],
    ids=["darknoise", "gain"]
)
@pytest.mark.parametrize(
    "set_number",
    SETS,
    ids=[f"Set {set_number}" for set_number in SETS]
)
def test_regression(
        is_darknoise: bool,
        set_number: int,
        pytestconfig,
        reset_workspace, # custom fixture with 'function' scope
        tests_data_folder: str = TESTS_DATA_FOLDER,
        sets_paths_yaml: str = SETS_PATHS_YAML,
        temporal_workspace_folder: str = TEMPORAL_WORKSPACE_FOLDER,
    ):
    """This function implements one iteration of the regression
    tests. One iteration is defined by the type of run (either
    gain or darknoise) and the set number. For each iteration,
    it runs the analysis, and then it compares the generated
    output CSV file with the expected output CSV file, by checking
    that the values in the generated output CSV file are close enough
    to the values in the expected output CSV file, for a given list
    of columns and their corresponding relative and absolute
    tolerances. The function also generates comparison plots between
    the generated output and the expected output, by executing a
    notebook with the papermill library, and it saves the generated
    plots in a PDF file in the output folder. The function returns
    early with a warning message (without raising an error) in case
    the required data for one iteration (YAML config file and expected
    output CSV file) is not found.
    """

    test_type = 'darknoise' if is_darknoise else 'gain'

    try:
        input_yaml_config_path, expected_csv_path = \
            find_config_and_desired_output(
                set_number,
                tests_data_folder+'/'+test_type
            )
    except FileNotFoundError:
        print(
            f"In function test_regression(): The {test_type} test "
            f"for set number {set_number} will be skipped, because the "
            f"the required data (YAML config file and expected output "
            "CSV file) were not found"
        )
        return

    input_yaml_config_path = os.path.abspath(input_yaml_config_path)
    expected_csv_path = os.path.abspath(expected_csv_path)

    bring_data_to_temporal_workspace(
        set_number,
        temporal_workspace_folder,
        sets_paths_yaml,
    )

    run_analysis(
        set_number,
        input_yaml_config_path,
        temporal_workspace_folder
    )

    computed_csv_path = str(
        find_generated_output_csv(
            temporal_workspace_folder
        )
    )

    # Save a copy of the computed CSV file in the output folder
    shutil.copy(
        computed_csv_path,
        pytestconfig.rootpath / OUTPUT_FOLDER / \
            f'set_{set_number}_{test_type}_computed.csv'
    )

    generate_comparison_plots(
        expected_csv_path,
        computed_csv_path,
        set_number,
        is_darknoise,
        str(pytestconfig.rootpath / OUTPUT_FOLDER)
    )

    computed_dataframe = pd.read_csv(
        computed_csv_path
    )

    expected_dataframe = pd.read_csv(expected_csv_path)
    
    # Dict of [relative tolerance, absolute tolerance] lists
    columns_to_compare = {
        'darknoise': {
            'N_events': [0.0, 0.0],
            'acquisition_time_min': [0.0, 0.0],
            'half_a_pe': [0.01, 0.0],
            'one_and_a_half_pe': [0.01, 0.0],
            'DC#': [0.01, 0.0],
            'DCR_mHz_per_mm2': [0.01, 0.0],
            'DCR_mHz_per_mm2_error': [0.01, 0.0],
            'XTP': [0.01, 0.0],
            'APP': [0.01, 0.0],
            'bursts_number': [0.0, 0.0],
            'time_elapsed_during_bursts_in_min': [0.0, 0.0],
            'bursts_contribution_to_DCR_mHz_per_mm2': [0.0, 0.0],
            'burstless_DC#': [0.01, 0.0],
            'burstless_DCR_mHz_per_mm2': [0.01, 0.0],
            'burstless_DCR_mHz_per_mm2_error': [0.01, 0.0],
            'burstless_XTP': [0.01, 0.0],
            'burstless_APP': [0.01, 0.0],
            'analysis_reliability': [0.0, 0.0]
        },
        'gain': {
            'N_events': [0.0, 0.0],
            'acquisition_time_min': [0.0, 0.0],
            'gain_in_#e-': [0.01, 0.0],
            'analysis_reliability': [0.0, 0.0]
        }
    }[test_type]

    assert_dataframes_compatibility(
        list(columns_to_compare.keys()),
        expected_dataframe,
        computed_dataframe
    )

    for column in columns_to_compare:
        np.testing.assert_allclose(
            computed_dataframe[column].values, # actual
            expected_dataframe[column].values, # desired
            rtol=columns_to_compare[column][0],
            atol=columns_to_compare[column][1],
            err_msg=f"Mismatch in column {column} for {set_number}"
        )

    return