#pragma once
#include <Common.cuh>
#include <Flags.cuh>
#include <Types.h>

#include "FiniteDifferenceTypes.h"

EXTERN_C
{
	/**
	*	Calculates the space discretization for the advection-diffusion equation
			u_t = (xVelocity, yVelocity) * grad(u) + diffusion * laplacian(u)
	*/
	EXPORT int _MakeSpaceDiscretizer2D(MemoryTile& spaceDiscretizer, const FiniteDifferenceInput2D& input);

	EXPORT int _SetBoundaryConditions2D(MemoryTile& solution, const FiniteDifferenceInput2D& input);

	/**
	*	Evolve the solution for nSteps steps using the time discretizer, and sets the boundary conditions
	*	N.B.:
	*		* solution is a memory tile, as multi-step solvers require the solution history
	*		* if provided, workBuffer is a previously allocated buffer used for matrix-vector multiplication
	*		* the more the steps, the more efficient it will be, as there's less overhead in creating and destroying volatile buffers
	*/
	EXPORT int _Iterate2D(MemoryTile& solution, const MemoryCube& timeDiscretizer, const FiniteDifferenceInput2D& input, const unsigned nSteps);
}

template <typename T>
GLOBAL void __MakeSpaceDiscretizer2D__(T* RESTRICT spaceDiscretizer, 
									   const T* RESTRICT xGrid, const T* RESTRICT yGrid, 
									   const T* RESTRICT xVelocity, const T* RESTRICT yVelocity,
									   const T* RESTRICT diffusion, 
									   const SpaceDiscretizerType discretizerType, const T dt, const unsigned nRows, const unsigned nCols);

template <typename T>
GLOBAL void __SetBoundaryConditions2D__(T* RESTRICT solution,
										const T leftValue, const T rightValue, const T downValue, const T upValue,
										const BoundaryConditionType leftBoundaryConditionType,
										const BoundaryConditionType rightBoundaryConditionType,
										const BoundaryConditionType downBoundaryConditionType,
										const BoundaryConditionType upBoundaryConditionType,
										const T* RESTRICT xGrid, const T* RESTRICT yGrid, const unsigned nRows, const unsigned nCols);