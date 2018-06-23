#include <Common.cuh>
#include <Flags.cuh>
#include <Types.h>

#include "FiniteDifferenceTypes.h"

#include <CuBlasWrappers.cuh>
#include <BufferInitializer.cuh>
#include <MemoryManager.cuh>

#include <array>

namespace detail
{
	/**
	*	Evolve the solution using the time discretizer.
	*	N.B.: solution is a memory tile, as some solver might require the solution history
	*	N.B.2: if provided, workBuffer is a previously allocated buffer used for matrix-vector multiplication
	*/
	int _Advance(MemoryTile solution, const MemoryCube timeDiscretizer, MemoryTile workBuffer, const bool overwriteBuffer);

	// N is the size of the Butcher tableau table
	// aMatrix is the lower triangular tableau matrix. If the diagonal is populated the method is an implicit RK
	// bvector is the vector used for composing the "k"'s 
	// WARNING: This doesn't support dense aMatrix, but only lower triangular
	template<unsigned N>
	int _MakeRungeKuttaDiscretizer(const std::array<double, N * (N + 1) / 2>& aMatrix,
								   const std::array<double, N>& bVector,
								   const double dt,
								   const MemoryTile& spaceDiscretizer,
								   const MemoryTile& timeDiscretizer)
	{
		auto getLowerTriangularIndex = [](const unsigned i, const unsigned j)
		{
			return j + i * (i + 1) / 2;
		};

		MemoryCube kVector(0, timeDiscretizer.nRows, timeDiscretizer.nCols, N, timeDiscretizer.memorySpace, timeDiscretizer.mathDomain);
		_Alloc(kVector);

		MemoryTile kRhs(timeDiscretizer); // kRhs is a working buffer that stores k_i r.h.s.
		_Alloc(kRhs);

		// loop for calculating k_i
		for (unsigned i = 0; i < N; ++i)
		{
			_Eye(kRhs);

			// aMatrix * k multiplication
			for (unsigned j = 0; j < i; ++j)
			{
				MemoryTile k_j;
				extractMatrixBufferFromCube(k_j, kVector, j);
				if (aMatrix[getLowerTriangularIndex(i, j)] != 0.0)
					_AddEqualMatrix(kRhs, k_j, MatrixOperation::None, MatrixOperation::None, aMatrix[getLowerTriangularIndex(i, j)] * dt);
			}

			MemoryTile k_i;
			extractMatrixBufferFromCube(k_i, kVector, i);
			_Multiply(k_i, spaceDiscretizer, kRhs, spaceDiscretizer.nRows, kRhs.nRows);

			if (aMatrix[getLowerTriangularIndex(i, i)] != 0.0)
			{
				// re-set kRhs instead of allocating kLhs
				_Eye(kRhs);
				_AddEqual(kRhs, spaceDiscretizer, -aMatrix[getLowerTriangularIndex(i, i)] * dt);
				_Solve(kRhs, k_i);
			}
		}

		//now that all kVector items are set, fo the b * k multiplication
		_Eye(timeDiscretizer);  // initialise time discretizer with the identity
		for (unsigned j = 0; j < N; ++j)
		{
			MemoryTile k_j;
			extractMatrixBufferFromCube(k_j, kVector, j);
			_AddEqualMatrix(timeDiscretizer, k_j, MatrixOperation::None, MatrixOperation::None, bVector[j] * dt);
		}

		_Free(kVector);
		_Free(kRhs);

		return cudaGetLastError();
	}


	int _MakeRungeKuttaGaussLegendre(const double dt, const MemoryTile& spaceDiscretizer, const MemoryTile& timeDiscretizer);
}

EXTERN_C
{
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
}

