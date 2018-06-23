# PdeFiniteDifferenceKernels

This API is a collection of CUDA kernels used for solving Partial Differential Equations (PDE) by means of Finite Difference. 

The extent of this API is to provide a library for applying the Method of Lines (MoL) to (up to) 3D hyperbolic and parabolic PDEs. Note that I wrote no support for elliptic PDEs.

The goal was to write the smallest amount of lines of code, and re-use of all my existing CUDA code, see [CudaLight](https://github.com/pmontalb/CudaLight) and [CudaLightKernels](https://github.com/pmontalb/CudaLightKernels). For this reason I decided to focus my attention to just linear PDE with no source term, so that everything can be solved by the application of a linear operator.

## Types
- <i>SolverType</i>: defines the time discretization solver type: the most popular ODE methods are there
- <i>SpaceDiscretizerType</i>: defines the space discretization type: One-sided, Centered and Lax-Wendroff-Style
- <i>BoundaryConditionType</i>: defines the boundary conditions: Dirichlet, Neumann and Periodic

## Convenience Structures
- <i>BoundaryCondition1D, BoundaryCondition2D</i>: wrapper for left/right/up/down boundary conditions
- <i>FiniteDifferenceInput1D, FiniteDifferenceInput2D</i>: wrapper for all the common inputs used by CUDA kernels

## Kernels structure
I kept the same conventions used in [CudaLightKernels](https://github.com/pmontalb/CudaLightKernels)
