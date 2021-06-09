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

import dolfin as df
from multiscale_formulation import MultiscaleFormulation


class FormulationMinimallyConstrained(MultiscaleFormulation):

    def otherSpaces(self):
        return [df.TensorFunctionSpace(self.mesh, "Real", 0)]

    def otherRestrictions(self):
        return [None]

    def blocks(self):
        aa, ff = super(FormulationMinimallyConstrained, self).blocks()

        n = df.FacetNormal(self.mesh)
        ds = df.Measure('ds', self.mesh)

        u, P = self.uu_[0], self.uu_[2]
        v, Q = self.vv_[0], self.vv_[2]

        aa[0].append(- df.inner(P, df.outer(v, n))*ds)
        aa[1].append(0)
        aa.append([- df.inner(Q, df.outer(u, n))*ds, 0, 0])

        ff.append(0)

        return [aa, ff]
