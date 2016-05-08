# Copyright (C) 2016 by the block_ext authors
#
# This file is part of block_ext.
#
# block_ext is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# block_ext is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with block_ext. If not, see <http://www.gnu.org/licenses/>.
#

from dolfin import NonlinearProblem, as_backend_type
from block_assemble import block_assemble
from monolithic_matrix import MonolithicMatrix
from monolithic_vector import MonolithicVector

class BlockNonlinearProblem(NonlinearProblem):
    def __init__(self, residual_form, block_solution, bcs, jacobian_form):
        NonlinearProblem.__init__(self)
        # Store the input arguments
        self.residual_form = residual_form
        self.jacobian_form = jacobian_form
        self.block_solution = block_solution
        self.bcs = bcs
        # Assemble residual and jacobian in order to
        # have block storage with appropriate sizes
        # and sparsity patterns
        self.block_residual = block_assemble(residual_form)
        self.block_jacobian = block_assemble(jacobian_form)
        # Add monolithic wrappers to PETSc objects, initialized
        # the first time F or J are called
        self.monolithic_residual = None
        self.monolithic_jacobian = None
        # Declare a monolithic_solution vector. The easiest way is to define a temporary monolithic matrix
        monolithic_jacobian = MonolithicMatrix(self.block_jacobian, preallocate=False)
        monolithic_solution, monolithic_residual = monolithic_jacobian.create_monolithic_vectors(self.block_solution.block_vector(), self.block_residual)
        self.monolithic_solution = monolithic_solution
        # Add the current block_solution as initial guess
        self.monolithic_solution.zero(); self.monolithic_solution.block_add(self.block_solution.block_vector())
        
    def F(self, fenics_residual, fenics_solution):
        # Initialize monolithic wrappers (only once)
        if self.monolithic_residual is None:
            self.monolithic_residual = MonolithicVector(self.block_residual, as_backend_type(fenics_residual).vec())
        # Shorthands
        residual_form = self.residual_form
        block_residual = self.block_residual
        block_solution = self.block_solution
        monolithic_residual = self.monolithic_residual # wrapper of the second input argument
        monolithic_solution = self.monolithic_solution # wrapper of the third input argument
        bcs = self.bcs
        # Convert monolithic_solution into block_solution
        monolithic_solution.copy_values_to(block_solution.block_vector())
        # Assemble
        block_assemble(residual_form, block_tensor=block_residual)
        bcs.apply(block_residual, block_solution.block_vector())
        # Copy values from block_residual/solution into monolithic_residual/solution
        monolithic_residual.zero(); monolithic_residual.block_add(block_residual)
        monolithic_solution.zero(); monolithic_solution.block_add(block_solution.block_vector())
        
    def J(self, fenics_jacobian, _):
        # Initialize monolithic wrappers (only once)
        if self.monolithic_jacobian is None:
            self.monolithic_jacobian = MonolithicMatrix(self.block_jacobian, as_backend_type(fenics_jacobian).mat())
        # Shorthands
        jacobian_form = self.jacobian_form
        block_jacobian = self.block_jacobian
        bcs = self.bcs
        monolithic_jacobian = self.monolithic_jacobian # wrapper of the second input argument
        # Assemble
        block_assemble(jacobian_form, block_tensor=block_jacobian)
        bcs.apply(block_jacobian)
        # Copy values from block_jacobian into monolithic_jacobian
        monolithic_jacobian.zero(); monolithic_jacobian.block_add(block_jacobian)
        
        