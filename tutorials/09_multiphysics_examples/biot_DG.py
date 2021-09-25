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

from ufl import avg, div, FiniteElement, grad, inner, jump, Measure, sym, VectorElement
from dolfin import (CellVolume, Constant, dot, FacetArea, FacetNormal, Function, FunctionSpace,
                    Identity, Mesh, MeshFunction, MPI, parameters, TensorFunctionSpace, XDMFFile,
                    VectorFunctionSpace, TestFunction, assemble)
from multiphenics import (block_assemble, block_assign, BlockDirichletBC, BlockElement, BlockFunction,
                          BlockFunctionSpace, block_solve, block_split, BlockTestFunction,
                          BlockTrialFunction, DirichletBC)

import csv
import math
import numpy as np

parameters["ghost_mode"] = "shared_facet" # required by dS

"""
Biot"s equations
Two-field discontinuous Galerkin,
Kadeethum T, Nick HM, Lee S, Ballarin F.
Enriched Galerkin discretization for modeling poroelasticity and permeability alteration in
heterogeneous porous media. Journal of Computational Physics. 2021 Feb;427:110030.

permeability is heterogeneous but not deformable
"""

def E_nu_to_mu_lmbda(E, nu):
    mu = E/(2*(1.0+nu))
    lmbda = (nu*E)/((1-2*nu)*(1+nu))
    return (mu, lmbda)

def K_nu_to_E(K, nu):
    return 3*K*(1-2*nu)

def Ks_cal(alpha, K):
    if alpha == 1.0:
        Ks = 1e35
    else:
        Ks = K/(1.0-alpha)
    return Ks

def strain(u):
    return sym(grad(u))

def avg_w(x, w):
    return (w*x("+")+(1-w)*x("-"))

def k_normal(k, n):
    return dot(dot(np.transpose(n), k), n)

def k_plus(k, n):
    return dot(dot(n("+"), k("+")), n("+"))

def k_minus(k, n):
    return dot(dot(n("-"), k("-")), n("-"))

def weight_e(k, n):
    return (k_minus(k, n))/(k_plus(k, n)+k_minus(k, n))

def k_e(k, n):
    return (2*k_plus(k, n)*k_minus(k, n)/(k_plus(k, n)+k_minus(k, n)))

def k_har(k):
    return (2*k*k/(k+k))

def init_scalar_parameter(p, p_value, index, sub):
    local_p = p.vector().get_local()
    cell_indices = np.where(sub.array() == index)[0]
    cell_indices = cell_indices[cell_indices < local_p.size]
    local_p[cell_indices] = p_value
    p.vector().set_local(local_p)
    p.vector().apply("insert")

def init_from_file_parameter_scalar_to_tensor(p, filename):
    local_p = p.vector().get_local()
    local_range = p.vector().local_range()
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
        for (i, row) in enumerate(readCSV):
            if 4*i >= local_range[0] and 4*i + 3 < local_range[1]:
                local_p[4*i - local_range[0]] = math.pow(10, row[0])
                local_p[4*i + 1 - local_range[0]] = 0.
                local_p[4*i + 2 - local_range[0]] = 0.
                local_p[4*i + 3 - local_range[0]] = math.pow(10, row[0])
    p.vector().set_local(local_p)
    p.vector().apply("insert")

mesh = Mesh("data/biot_2mat.xml")
subdomains = MeshFunction("size_t", mesh, "data/biot_2mat_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/biot_2mat_facet_region.xml")

# Block function space
V_element = VectorElement("CG", mesh.ufl_cell(), 2)
Q_element = FiniteElement("DG", mesh.ufl_cell(), 1)
W_element = BlockElement(V_element, Q_element)
W = BlockFunctionSpace(mesh, W_element)

PM = FunctionSpace(mesh, "DG", 0)
TM = TensorFunctionSpace(mesh, "DG", 0)

I = Identity(mesh.topology().dim())

dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
dS = Measure("dS", domain=mesh, subdomain_data=boundaries)

# Test and trial functions
vq = BlockTestFunction(W)
(v, q) = block_split(vq)
up = BlockTrialFunction(W)
(u, p) = block_split(up)

w = BlockFunction(W)
w0 = BlockFunction(W)
(u0, p0) = block_split(w0)

n = FacetNormal(mesh)
vc = CellVolume(mesh)
fc = FacetArea(mesh)

h = vc/fc
h_avg = (vc("+") + vc("-"))/(2*avg(fc))

penalty1 = 1.0
penalty2 = 10.0
theta = 1.0

# Constitutive parameters
K = 1000.e3
nu = 0.25
E = K_nu_to_E(K, nu) # Pa 14

(mu_l, lmbda_l) = E_nu_to_mu_lmbda(E, nu)

f_stress_y = Constant(-1.e3)

f = Constant((0.0, 0.0)) # sink/source for displacement
g = Constant(0.0) # sink/source for velocity

p_D1 = 0.0

alpha = 1.0

rho1 = 1000.0
vis1 = 1.e-3
cf1 = 1.e-10
phi1 = 0.2
cf1 = 1e-10
ct1 = phi1*cf1

rho2 = 1000.0
vis2 = 1.e-3
cf2 = 1.e-10
phi2 = 0.2
cf2 = 1e-10
ct2 = phi2*cf2

rho_values = [rho1, rho2]
vis_values = [vis1, vis2]
cf_values = [cf1, cf2]
phi_values = [phi1, phi2]
ct_values = [ct1, ct2]

rho = Function(PM)
vis = Function(PM)
phi = Function(PM)
ct = Function(PM)
k = Function(TM)

init_scalar_parameter(rho, rho_values[0], 500, subdomains)
init_scalar_parameter(vis, vis_values[0], 500, subdomains)
init_scalar_parameter(phi, phi_values[0], 500, subdomains)
init_scalar_parameter(ct, ct_values[0], 500, subdomains)

init_scalar_parameter(rho, rho_values[1], 501, subdomains)
init_scalar_parameter(vis, vis_values[1], 501, subdomains)
init_scalar_parameter(phi, phi_values[1], 501, subdomains)
init_scalar_parameter(ct, ct_values[1], 501, subdomains)

init_from_file_parameter_scalar_to_tensor(k, "data/het_0.csv")

T = 500.0
t = 0.0
dt = 20.0

bcd1 = DirichletBC(W.sub(0).sub(0), Constant(0.0), boundaries, 1) # No normal displacement for solid on left side
bcd3 = DirichletBC(W.sub(0).sub(0), Constant(0.0), boundaries, 3) # No normal displacement for solid on right side
bcd4 = DirichletBC(W.sub(0).sub(1), Constant(0.0), boundaries, 4) # No normal displacement for solid on bottom side
bcs = BlockDirichletBC([[bcd1, bcd3, bcd4], []])

a = inner(2*mu_l*strain(u)+lmbda_l*div(u)*I, sym(grad(v)))*dx

b = inner(-alpha*p*I, sym(grad(v)))*dx

c = rho*alpha*div(u)*q*dx

d = (
    rho*ct*p*q*dx + dt*dot(rho*k/vis*grad(p), grad(q))*dx
    - dt*dot(avg_w(rho*k/vis*grad(p), weight_e(k, n)), jump(q, n))*dS
    - theta*dt*dot(avg_w(rho*k/vis*grad(q), weight_e(k, n)), jump(p, n))*dS
    + dt*penalty1/h_avg*avg(rho)*k_e(k, n)/avg(vis)*dot(jump(p, n), jump(q, n))*dS
    - dt*dot(rho*k/vis*grad(p), q*n)*ds(2)
    - dt*dot(rho*k/vis*grad(q), p*n)*ds(2)
    + dt*(penalty2/h*rho/vis*dot(dot(n, k), n)*dot(p*n, q*n))*ds(2)
)

lhs = [[a, b],
       [c, d]]

f_u = (
    inner(f, v)*dx
    + dot(f_stress_y*n, v)*ds(2)
)

f_p = (
    rho*alpha*div(u0)*q*dx
    + rho*ct*p0*q*dx + dt*g*q*dx
    - dt*dot(p_D1*n, rho*k/vis*grad(q))*ds(2)
    + dt*(penalty2/h*rho/vis*dot(dot(n, k), n)*dot(p_D1*n, q*n))*ds(2)
)

rhs = [f_u, f_p]

# Mass conservation
V_CON = VectorFunctionSpace(mesh, "CG", 2)
P_CON = FunctionSpace(mesh, "DG", 1)
u_con = Function(V_CON)
p_con = Function(P_CON)
con = TestFunction(PM)
mass_con = (rho*alpha*div(u_con-u0)/dt*con*dx
    + rho*ct*(p_con-p0)/dt*con*dx
    - dot(avg_w(rho*k/vis*grad(p_con), weight_e(k, n)), jump(con, n))*dS
    + penalty1/h_avg*avg(rho)*k_e(k, n)/avg(vis)*dot(jump(p_con, n), jump(con, n))*dS
    - dot(rho*k/vis*grad(p_con), n)*con*ds(2)
    + penalty2/h*rho/vis*dot(dot(n, k), n)*(p_con-p_D1)*con*ds(2))

if MPI.size(MPI.comm_world) == 1:
    xdmu = XDMFFile("biot_u.xdmf")
    xdmp = XDMFFile("biot_p.xdmf")

    while t < T:
        t += dt
        print("solving at time", t, flush=True)
        AA = block_assemble(lhs)
        FF = block_assemble(rhs)
        bcs.apply(AA)
        bcs.apply(FF)
        block_solve(AA, w.block_vector(), FF, "mumps")
        block_assign(w0, w)
        xdmu.write(w[0], t)
        xdmp.write(w[1], t)

        u_con.assign(w[0])
        p_con.assign(w[1])
        mass = assemble(mass_con)
        print("average mass residual: ", np.average(np.abs(mass[:])))

else:
    pass  # TODO this tutorial is currently skipped in parallel due to issue #1
