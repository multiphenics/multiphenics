## multiphenics -- easy prototyping of multiphysics problems in FEniCS ##
![multiphenics -- easy prototyping of multiphysics problems in FEniCS](https://raw.githubusercontent.com/multiphenics/multiphenics/master/docs/multiphenics-logo-small.png "multiphenics -- easy prototyping of multiphysics problems in FEniCS")

### 0. Introduction
**multiphenics** is a python library that aims at providing tools in FEniCS for an easy prototyping of multiphysics problems on conforming meshes. In particular, it facilitates the definition of subdomain/boundary restricted variables and enables the definition of the problem by means of a block structure.

### 1. Prerequisites
**multiphenics** requires **FEniCS** (>= 2018.1.0, python 3) with PETSc and SLEPc.

Support for **FEniCSx** is available in the [new multiphenicsx library](https://github.com/multiphenics/multiphenicsx). New and existing users of **FEniCS** and **multiphenics** are encouraged to gradually move to **FEniCSx** and **multiphenicsx**.

### 2. Installation and usage
Simply clone the **multiphenics** public repository
```
git clone https://github.com/multiphenics/multiphenics.git
```
and install the package by typing
```
python3 setup.py install
```

#### 2.1. multiphenics docker image
If you want to try **multiphenics** out but do not have **FEniCS** already installed, you can [pull our docker image from Docker Hub](https://hub.docker.com/r/multiphenics/multiphenics/). All required dependencies are already installed. **multiphenics** tutorials and tests are located at
```
$FENICS_HOME/multiphenics
```

### 3. Tutorials
Several tutorials are provided in the [**tutorials** subfolder](https://github.com/multiphenics/multiphenics/tree/master/tutorials).
* **Tutorial 01**: block Poisson test case, to introduce the block notation used in the library.
* **Tutorial 02**: Navier-Stokes problem using block matrices.
* **Tutorial 03**: weak imposition of Dirichlet boundary conditions by Lagrange multipliers using block matrices and discarding interior degrees of freedom.
* **Tutorial 04**: computation of the inf-sup constant for a Stokes problem assembled using block matrices.
* **Tutorial 05**: computation of the inf-sup constant for the problem presented in tutorial 3.
* **Tutorial 06**: several examples on optimal control problems, with different state equations (elliptic, Stokes, Navier-Stokes), control (distributed or boundary) and observation (distributed or boundary).
* **Tutorial 07**: generation of restrictions for meshes obtained from gmsh.
* **Tutorial 08**: how to get the list of degrees of freedom associated to a specific restriction, and use it e.g. to perform local modifications to assembled tensors.
* **Tutorial 09**: applications of **multiphenics** to multiphysics problems. We are looking forward to receiving further multiphysics examples from our users!
* **Tutorial 10**: applications of **multiphenics** to multiscale problems. We are looking forward to receiving further multiscale examples from our users!

### 4. Authors and contributors
**multiphenics** is currently developed and maintained at [Università Cattolica del Sacro Cuore](https://www.unicatt.it/) by [Dr. Francesco Ballarin](https://www.francescoballarin.it) in collaboration with [Prof. Gianluigi Rozza](https://people.sissa.it/~grozza/)'s group at [SISSA mathLab](http://mathlab.sissa.it/). The financial support of the [AROMA-CFD ERC CoG project](https://people.sissa.it/~grozza/aroma-cfd/) is gratefully acknowledged. Please see the [AUTHORS file](https://github.com/multiphenics/multiphenics/blob/master/AUTHORS) for a list of contributors.

Contact us by [email](mailto:francesco.ballarin@unicatt.it) for further information or questions about **multiphenics**, or open an issue on [our issue tracker](https://github.com/multiphenics/multiphenics/issues). **multiphenics** is at an early development stage, so contributions improving either the code or the documentation are welcome, both as patches or [pull requests](https://github.com/multiphenics/multiphenics/pulls).

### 5. Related resources
* Block matrix support in [DOLFIN-X](https://github.com/FEniCS/dolfinx), either as MatNest or monolithic matrices. In **multiphenics** we always assemble block matrices into a monolithic matrix, and also support possible restriction of the unknowns to subdomains and/or boundaries.
* Mixed dimensional branch in [DOLFIN](https://bitbucket.org/fenics-project/dolfin/branch/cecile/mixed-dimensional) pursues a similar goal to **multiphenics**. It requires the user the install the corresponding branches of [UFL](https://bitbucket.org/fenics-project/ufl/branch/cecile/mixed-dimensional) and [FFC](https://bitbucket.org/fenics-project/ffc/branch/cecile/mixed-dimensional). In contrast, **multiphenics** does not require any change to the underlying FEniCS installation.
* multimesh support in FEniCS, which aims at providing support for problems on non conforming meshes. In **multiphenics** we are rather interested in conforming meshes, with possible restriction of the unknowns to subdomains and/or boundaries.
* [CutFEM](http://www.cutfem.org/), an unfitted finite element framework for multi-physics problems that relies on the FEniCS project.
* [CBC.Block](https://bitbucket.org/fenics-apps/cbc.block/) for the definition of block matrices and vectors in FEniCS.
* Weak imposition of Dirichlet Dirichlet boundary conditions by Lagrange multipliers is a frequently asked question on FEniCS support forums [[1](https://fenicsproject.org/qa/), [2](https://fenicsproject.discourse.group/)]. Some answers provide possible solutions to the problem (e.g. constraining the useless degrees of freedom by DirichletBC), which however may result in an unnecessarily large system to be solved. **multiphenics** handles subdomain/boundary restricted variables in an efficient and automatic way.
* Please contact us by [email](mailto:francesco.ballarin@unicatt.it) if you have other related resources.

### 6. How to cite
If you use **multiphenics** in your work, please cite the [multiphenics website](http://mathlab.sissa.it/multiphenics).

### 7. License
Like all core **FEniCS** components, **multiphenics** is freely available under the GNU LGPL, version 3.
