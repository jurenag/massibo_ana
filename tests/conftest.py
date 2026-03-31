import pytest
from pathlib import Path
import subprocess

@pytest.fixture(scope="session", autouse=True)
def validate_current_working_directory(pytestconfig):
    """This fixture validates that the tests are being
    run from the repository root folder. If not, it raises
    an exception with an informative message. It is run
    automatically at the beginning of any test session.
    """

    assert Path.cwd() == pytestconfig.rootpath, (
        "The current working directory is not the repository "
        f"root folder, but {Path.cwd()}. Please run the tests "
        f"from the repository root folder ({pytestconfig.rootpath})."
    )

    return

@pytest.fixture(scope='function')
def reset_workspace(pytestconfig) -> None:
    """This fixture resets the temporal workspace by
    deleting all the files and folders in it, and then
    recreating the necessary folder structure. If used,
    it is run before each test function.
    """

    workspace_path = \
        Path(pytestconfig.rootpath) / 'tests/tmp'
    
    for folder_name in [
        'input_data',
        'aux',
        'load',
        'data',
        'summary'
    ]:
        folder_path = workspace_path / folder_name
        
        if folder_path.exists():
            subprocess.run(
                ["rm", "-rf", folder_path],
                check=True
            )

        folder_path.mkdir(exist_ok=False)

    return