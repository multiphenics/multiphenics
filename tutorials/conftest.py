# Copyright (C) 2016-2022 by the multiphenics authors
#
# This file is part of multiphenics.
#
# multiphenics is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# multiphenics is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with multiphenics. If not, see <http://www.gnu.org/licenses/>.
#

import os
import sys
import importlib
import pytest
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

def pytest_collect_file(file_path, path, parent):
    """
    Hook into py.test to collect tutorial files.
    """
    if (
        file_path.suffix == ".py" and file_path.name not in "conftest.py"
            and
        all(exclude not in str(file_path.parent) for exclude in ("data", "formulations", "models", "utils"))
    ):
        return TutorialFile.from_parent(parent=parent, path=file_path)

def pytest_pycollect_makemodule(module_path, path, parent):
    """
    Hook into py.test to avoid collecting twice tutorial files explicitly provided on the command lines
    """
    if (
        module_path.suffix == ".py" and module_path.name not in "conftest.py"
            and
        all(exclude not in str(module_path.parent) for exclude in ("data", "formulations", "models", "utils"))
    ):
        return DoNothingFile.from_parent(parent=parent, path=module_path)

class TutorialFile(pytest.File):
    """
    Custom file handler for tutorial files
    """

    def collect(self):
        yield TutorialItem.from_parent(parent=self, name="run_tutorial -> " + os.path.relpath(str(self.path), str(self.parent.path)))

class TutorialItem(pytest.Item):
    """
    Handle the execution of the tutorial
    """

    def __init__(self, name, parent):
        super(TutorialItem, self).__init__(name, parent)

    def runtest(self):
        os.chdir(self.parent.path.parent)
        sys.path.append(str(self.parent.path.parent))
        spec = importlib.util.spec_from_file_location(self.name, str(self.parent.path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        plt.close('all') # do not trigger matplotlib max_open_warning

    def reportinfo(self):
        return self.path, 0, self.name

class DoNothingFile(pytest.File):
    """
    Custom file handler to avoid running twice tutorial files explicitly provided on the command lines
    """

    def collect(self):
        return []
