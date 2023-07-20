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

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl
import dolfin as df
from .multiscale_formulation import MultiscaleFormulation


class FormulationMinimallyConstrained(MultiscaleFormulation):

    def other_spaces(self):
        return [df.TensorFunctionSpace(self.mesh, "Real", 0)]

    def other_restrictions(self):
        return [None]

    def blocks(self):
        aa, ff = super(FormulationMinimallyConstrained, self).blocks()

        n = ufl.FacetNormal(self.mesh)
        ds = ufl.Measure("ds", self.mesh)

        u, P = self.uu_[0], self.uu_[2]
        v, Q = self.vv_[0], self.vv_[2]

        aa[0].append(- ufl.inner(P, ufl.outer(v, n)) * ds)
        aa[1].append(0)
        aa.append([- ufl.inner(Q, ufl.outer(u, n)) * ds, 0, 0])

        ff.append(0)

        return [aa, ff]
