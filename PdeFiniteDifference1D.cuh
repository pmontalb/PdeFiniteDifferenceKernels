#include <Common.cuh>
#include <Flags.cuh>
#include <Types.h>

#include "FiniteDifferenceTypes.h"

EXTERN_C
{
	/**
	*	Calculates the space discretization for the advection-diffusion equation
			u_t = velocity * u_x + diffusion * u_xx
	*/
	EXPORT int _MakeSpaceDiscretizer1D(MemoryTile spaceDiscretizer, const FiniteDifferenceInput1D input);

	/**
	*	Calculates the time discretization for the ODE (that comes from the space discretization of the advection-diffusion PDE)
			u' = L * u
	*/
    EXPORT int _MakeTimeDiscretizerAdvectionDiffusion(MemoryCube timeDiscretizer, const MemoryTile spaceDiscretizer, const SolverType solverType, const double dt);

	/**
	*	Calculates the time discretization for the ODE (that comes from the space discretization of the advection-diffusion PDE)
			u'' = L * u
	*/
	EXPORT int _MakeTimeDiscretizerWaveEquation(MemoryCube timeDiscretizer, const MemoryTile spaceDiscretizer, const SolverType solverType, const double dt);

	/**
	*	Evolve the solution for nSteps steps using the time discretizer, and sets the boundary conditions
	*	N.B.: 
	*		* solution is a memory tile, as multi-step solvers require the solution history
	*		* if provided, workBuffer is a previously allocated buffer used for matrix-vector multiplication
	*		* the more the steps, the more efficient it will be, as there's less overhead in creating and destroying volatile buffers
	*/
	EXPORT int _Iterate1D(MemoryTile solution, const MemoryCube timeDiscretizer, const FiniteDifferenceInput1D input, const unsigned nSteps);
}

template <typename T>
GLOBAL void __MakeSpaceDiscretizer1D__(T* RESTRICT spaceDiscretizer, const T* RESTRICT grid, const T* RESTRICT velocity, const T* RESTRICT diffusion, const SpaceDiscretizerType discretizerType, const T dt, const unsigned sz);

template <typename T>
GLOBAL void __SetBoundaryConditions1D__(T* RESTRICT solution, T* RESTRICT timeDiscretizer, const T leftValue, const T rightValue, const BoundaryConditionType leftBoundaryConditionType, const BoundaryConditionType rightBoundaryConditionType, const T* RESTRICT grid, const unsigned sz);