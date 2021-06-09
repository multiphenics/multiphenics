# Copyright (C) 2016-2021 by the multiphenics authors
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

"""
test_bndModel = test different boundary conditions for multiscale models in
comparison to the single scale simulation. For the mesh chosen it should agree
with the given errors. Also 'lag' and 'lin' error should match.

test_bndModel = test different the parallel implementation with different
number of cores against the standalone implementation. All errors should be
technically very close to zero.
"""


import os
import numpy as np
from timeit import default_timer as timer
import dolfin as df


def test_bndModel():

    Lx, Ly, Ny, Nx, NxyMicro = 0.5, 2.0, 12, 3, 50

    start = timer()
    print("simulating single scale")
    os.system("python bar_single_scale.py %d %d" % (Nx, Ny))
    end = timer()
    print('finished in ', end - start)

    for bndModel in ['per', 'lin', 'MR', 'lag']:
        suffix = "{0} {1} {2} {3} > log_{3}.txt".format(Nx, Ny,
                                                        NxyMicro, bndModel)

        start = timer()
        print("simulating using " + bndModel)
        os.system("python bar_multiscale_standalone.py " + suffix)
        end = timer()
        print('finished in ', end - start)

    mesh = df.RectangleMesh(df.Point(0.0, 0.0), df.Point(Lx, Ly),
                            Nx, Ny, "right/left")

    Uh = df.VectorFunctionSpace(mesh, "Lagrange", 1)

    uh0 = df.Function(Uh)

    with df.XDMFFile("bar_single_scale.xdmf") as f:
        f.read_checkpoint(uh0, 'u')

    error = {}
    ehtemp = df.Function(Uh)

    for bndModel in ['per', 'lin', 'MR', 'lag']:

        with df.XDMFFile("bar_multiscale_standalone_%s.xdmf" % bndModel) as f:
            f.read_checkpoint(ehtemp, 'u')

        ehtemp.vector().set_local(ehtemp.vector().get_local()[:] -
                                  uh0.vector().get_local()[:])

        error[bndModel] = df.norm(ehtemp)

    assert np.abs(error['lin'] - error['lag']) < 1e-12
    assert np.abs(error['per'] - 0.00458937125497013) < 1e-12
    assert np.abs(error['lin'] - 0.0061056841749225176) < 1e-12
    assert np.abs(error['MR'] - 0.0009530615548645162) < 1e-12


def test_parallel():

    Lx, Ly, Ny, Nx, NxyMicro = 0.5, 2.0, 12, 3, 50
    bndModel = 'per'
    suffix = "%d %d %d %s > log_%s.txt" % (
        Nx, Ny, NxyMicro, bndModel, bndModel)

    start = timer()
    print("simulating single core")
    os.system("python bar_multiscale_standalone.py " + suffix)
    end = timer()
    print('finished in ', end - start)

    for nProc in [1, 2, 3, 4]:
        start = timer()
        print("simulating using nProc",  nProc)
        os.system("mpirun -n %d python bar_multiscale_parallel.py " %
                  nProc + suffix)
        end = timer()
        print('finished in ', end - start)

    mesh = df.RectangleMesh(df.Point(0.0, 0.0), df.Point(Lx, Ly),
                            Nx, Ny, "right/left")

    Uh = df.VectorFunctionSpace(mesh, "Lagrange", 1)

    uh0 = df.Function(Uh)

    with df.XDMFFile("bar_multiscale_standalone_%s.xdmf" % bndModel) as f:
        f.read_checkpoint(uh0, 'u')

    error = {}
    ehtemp = df.Function(Uh)

    for nProc in [1, 2, 3, 4]:

        with df.XDMFFile("bar_multiscale_parallel_%s_np%d.xdmf" %
                         (bndModel, nProc)) as f:
            f.read_checkpoint(ehtemp, 'u')

        ehtemp.vector().set_local(ehtemp.vector().get_local()[:] -
                                  uh0.vector().get_local()[:])

        error[nProc] = df.norm(ehtemp)

    assert np.max(np.array(list(error.values()))) < 1e-13
