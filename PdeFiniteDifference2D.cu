#include "PdeFiniteDifference2D.cuh"
#include "PdeFiniteDifference.cuh"

EXTERN_C
{
	/**
	*	Calculates the space discretization for the advection-diffusion equation
			u_t = (xVelocity, yVelocity) * grad(u) + diffusion * laplacian(u)
	*/
	EXPORT int _MakeSpaceDiscretizer2D(MemoryTile& spaceDiscretizer, const FiniteDifferenceInput2D& input)
    {
		if (input.spaceDiscretizerType == SpaceDiscretizerType::Null)
			return CudaKernelException::_NotImplementedException;

		const dim3 grid(32, 32);
		const dim3 blocks(8, 8);

		switch (spaceDiscretizer.mathDomain)
		{
			case MathDomain::Float:
				CUDA_CALL_XY(__MakeSpaceDiscretizer2D__<float>, grid, blocks,
					             (float*)spaceDiscretizer.pointer, (float*)input.xGrid.pointer, (float*)input.yGrid.pointer,
								 (float*)input.xVelocity.pointer, (float*)input.yVelocity.pointer,
								 (float*)input.diffusion.pointer, input.spaceDiscretizerType, (float)input.dt, input.xGrid.size, input.yGrid.size);
				break;
			case MathDomain::Double:
				CUDA_CALL_XY(__MakeSpaceDiscretizer2D__<double>, grid, blocks,
					             (double*)spaceDiscretizer.pointer, (double*)input.xGrid.pointer, (double*)input.yGrid.pointer,
								 (double*)input.xVelocity.pointer, (double*)input.yVelocity.pointer,
								 (double*)input.diffusion.pointer, input.spaceDiscretizerType, input.dt, input.xGrid.size, input.yGrid.size);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}
		return cudaGetLastError();
    }

	EXPORT int _SetBoundaryConditions2D(MemoryTile& solution, const FiniteDifferenceInput2D& input)
	{
		const dim3 grid(32, 32);
		const dim3 blocks(8, 8);

		switch (solution.mathDomain)
		{
			case MathDomain::Float:
				CUDA_CALL_XY(__SetBoundaryConditions2D__<float>, grid, blocks,
					             (float*)solution.pointer, 
								 (float)input.boundaryConditions.left.value, (float)input.boundaryConditions.right.value, 
								 (float)input.boundaryConditions.down.value, (float)input.boundaryConditions.up.value,
								 input.boundaryConditions.left.type, input.boundaryConditions.right.type, 
								 input.boundaryConditions.down.type, input.boundaryConditions.up.type,
								 (float*)input.xGrid.pointer, (float*)input.yGrid.pointer,
								 input.xGrid.size, input.yGrid.size);
				break;
			case MathDomain::Double:
				CUDA_CALL_XY(__SetBoundaryConditions2D__<double>, grid, blocks,
					             (double*)solution.pointer,
								 input.boundaryConditions.left.value, input.boundaryConditions.right.value,
								 input.boundaryConditions.down.value, input.boundaryConditions.up.value,
								 input.boundaryConditions.left.type, input.boundaryConditions.right.type,
								 input.boundaryConditions.down.type, input.boundaryConditions.up.type,
								 (double*)input.xGrid.pointer, (double*)input.yGrid.pointer,
								 input.xGrid.size, input.yGrid.size);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}
		return cudaGetLastError();
    }

	/**
	*	Evolve the solution for nSteps steps using the time discretizer, and sets the boundary conditions
	*	N.B.:
	*		* solution is a memory tile, as multi-step solvers require the solution history
	*		* if provided, workBuffer is a previously allocated buffer used for matrix-vector multiplication
	*		* the more the steps, the more efficient it will be, as there's less overhead in creating and destroying volatile buffers
	*/
	EXPORT int _Iterate2D(MemoryTile& solution, const MemoryCube& timeDiscretizer, const FiniteDifferenceInput2D& input, const unsigned nSteps)
	{
		// allocate a volatile buffer, used for the matrix-vector dot-product
		MemoryTile workBuffer = solution;
		_Alloc(workBuffer);

		bool overwriteBuffer = true;
		int err = 0;
		for (unsigned n = 0; n < nSteps; ++n)
		{
			err = detail::_Advance(solution, timeDiscretizer, workBuffer, overwriteBuffer);
			if (err)
				return err;

			// set boundary conditions
			err = _SetBoundaryConditions2D(overwriteBuffer ? workBuffer : solution, input);
			if (err)
				return err;

			overwriteBuffer = !overwriteBuffer;
		}

		if (!overwriteBuffer)  // need the negation here, as it's set at the end of the loop!
							   // copy the result back from working buffer and free it
			_DeviceToDeviceCopy(solution, workBuffer);

		_Free(workBuffer);

		return cudaGetLastError();
    }
}

// the dimension of the spaceDiscretizer is (nRows * nCols) x (nRows * nCols)
// the spaceDiscretizer is a block tridiagonal matrix
// j iterates over the nCols blocks of dimension (nRows x nRows)
// i iterates over the pentadiagonal sub-blocks
#define COORD(X, Y) (((X + j * nRows) + sz * (Y + j * nRows)))
#define COORD_PLUS(X) (((X + j * nRows) + sz * (X + (j + 1) * nRows)))
#define COORD_MINUS(X) (((X + j * nRows) + sz * (X + (j - 1) * nRows)))

template <typename T>
GLOBAL void __MakeSpaceDiscretizer2D__(T* RESTRICT spaceDiscretizer,
									   const T* RESTRICT xGrid, const T* RESTRICT yGrid,
									   const T* RESTRICT xVelocity, const T* RESTRICT yVelocity,
									   const T* RESTRICT diffusion,
									   const SpaceDiscretizerType discretizerType, const T dt, const unsigned nRows, const unsigned nCols)
{
	const unsigned sz = nRows * nCols;

	unsigned int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stepX = gridDim.x * blockDim.x;

	unsigned int tidY = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int stepY = gridDim.y * blockDim.y;

	// loop over blocks
	for (unsigned j = tidY + 1; j < nCols - 1; j += stepY)
	{
		// space discretization over y-axis
		const T dyPlus = yGrid[j + 1] - yGrid[j];
		const T dyMinus = yGrid[j] - yGrid[j - 1];
		const T dy = dyPlus + dyMinus;

		const T yMultiplierMinus = 1.0 / (dyMinus * dy);
		const T yMultiplierPlus = 1.0 / (dyPlus  * dy);

		// loop inside blocks
		for (unsigned i = tidX + 1; i < nRows - 1; i += stepX)
		{
			// since diffusion might depend on x and y, this needs to stay inside the inner loop
			__MakeSpaceDiscretizerWorker__<T>(spaceDiscretizer[COORD_PLUS(i)], spaceDiscretizer[COORD(i, i)], spaceDiscretizer[COORD_MINUS(i)], discretizerType,
											  yVelocity[j], diffusion[i + nRows * j],
											  dyPlus, dyMinus, dy, yMultiplierMinus, yMultiplierPlus, dt);

			// space discretization over x-axis
			const T dxPlus = xGrid[i + 1] - xGrid[i];
			const T dxMinus = xGrid[i] - xGrid[i - 1];
			const T dx = dxPlus + dxMinus;

			const T xMultiplierMinus = 1.0 / (dxMinus * dx);
			const T xMultiplierPlus = 1.0 / (dxPlus  * dx);

			__MakeSpaceDiscretizerWorker__<T>(spaceDiscretizer[COORD(i, i + 1)], spaceDiscretizer[COORD(i, i)], spaceDiscretizer[COORD(i, i - 1)], discretizerType,
											  xVelocity[i], diffusion[i + nRows * j],
											  dxPlus, dxMinus, dx, xMultiplierMinus, xMultiplierPlus, dt);
		}
	}
}

#undef COORD
#undef COORD_PLUS
#undef COORD_MINUS
#define COORD(X, Y) (X) + (nRows * (Y))

template <typename T>
GLOBAL void __SetBoundaryConditions2D__(T* RESTRICT solution,
										const T leftValue, const T rightValue, const T downValue, const T upValue,
										const BoundaryConditionType leftBoundaryConditionType,
										const BoundaryConditionType rightBoundaryConditionType,
										const BoundaryConditionType downBoundaryConditionType,
										const BoundaryConditionType upBoundaryConditionType,
										const T* RESTRICT xGrid, const T* RESTRICT yGrid, 
										const unsigned nRows, const unsigned nCols)
{
	unsigned int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stepX = gridDim.x * blockDim.x;

	unsigned int tidY = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int stepY = gridDim.y * blockDim.y;

	for (unsigned i = tidX + 1; i < nRows - 1; i += stepX)
		solution[COORD(i, 0)] = solution[COORD(i, 1)] - leftValue * (xGrid[1] - xGrid[0]);

	// working out boundary conditions except from the 4 corners
    #pragma region Left BC

	switch (leftBoundaryConditionType)
	{
		case BoundaryConditionType::Dirichlet:
			for (unsigned i = tidX + 1; i < nRows - 1; i += stepX)
				solution[COORD(i, 0)] = leftValue;
			break;
		case BoundaryConditionType::Neumann:
			for (unsigned i = tidX + 1; i < nRows - 1; i += stepX)
				solution[COORD(i, 0)] = solution[COORD(i, 1)] - leftValue * (yGrid[1] - yGrid[0]);
			break;
		case BoundaryConditionType::Periodic:
			for (unsigned i = tidX + 1; i < nRows - 1; i += stepX)
				solution[COORD(i, 0)] = solution[COORD(i, nCols - 2)];
			break;
		default:
			break;
	}

    #pragma endregion

    #pragma region Right BC

	switch (rightBoundaryConditionType)
	{
		case BoundaryConditionType::Dirichlet:
			for (unsigned i = tidX + 1; i < nRows - 1; i += stepX)
				solution[COORD(i, nCols - 1)] = rightValue;
			break;
		case BoundaryConditionType::Neumann:
			for (unsigned i = tidX + 1; i < nRows - 1; i += stepX)
				solution[COORD(i, nCols - 1)] = solution[COORD(i, nCols - 2)] - rightValue * (yGrid[nCols - 1] - yGrid[nCols - 2]);
			break;
		case BoundaryConditionType::Periodic:
			for (unsigned i = tidX + 1; i < nRows - 1; i += stepX)
				solution[COORD(i, nCols - 1)] = solution[COORD(i, 1)];
			break;
		default:
			break;
	}

    #pragma endregion

    #pragma region Up BC

	switch (upBoundaryConditionType)
	{
		case BoundaryConditionType::Dirichlet:
			for (unsigned j = tidY + 1; j < nCols - 1; j += stepY)
				solution[COORD(0, j)] = upValue;
			break;
		case BoundaryConditionType::Neumann:
			for (unsigned j = tidY + 1; j < nCols - 1; j += stepY)
				solution[COORD(0, j)] = solution[COORD(1, j)] - upValue * (xGrid[1] - xGrid[0]);
			break;
		case BoundaryConditionType::Periodic:
			for (unsigned j = tidY + 1; j < nCols - 1; j += stepY)
				solution[COORD(0, j)] = solution[COORD(nRows - 2, j)];
			break;
		default:
			break;
	}

    #pragma endregion

    #pragma region Down BC

	switch (downBoundaryConditionType)
	{
		case BoundaryConditionType::Dirichlet:
			for (unsigned j = tidY + 1; j < nCols - 1; j += stepY)
				solution[COORD(nRows - 1, j)] = downValue;
			break;
		case BoundaryConditionType::Neumann:
			for (unsigned j = tidY + 1; j < nCols - 1; j += stepY)
				solution[COORD(nRows - 1, j)] = solution[COORD(nRows - 2, j)] - downValue * (xGrid[nCols - 1] - xGrid[nCols - 2]);
			break;
		case BoundaryConditionType::Periodic:
			for (unsigned j = tidY + 1; j < nCols - 1; j += stepY)
				solution[COORD(nRows - 1, j)] = solution[COORD(1, j)];
			break;
		default:
			break;
	}

    #pragma endregion

	// first thread x and y deals with the 4 corners: sets them to an average of neighborhood points
	if (tidX + tidY == 0)
	{
		// top left
		solution[COORD(0, 0)] = .5 * (solution[COORD(0, 1)] + solution[COORD(1, 0)]);

		// top right
		solution[COORD(0, nCols - 1)] = .5 * (solution[COORD(0, nCols - 2)] + solution[COORD(1, nCols - 1)]);

		// bottom left
		solution[COORD(nRows - 1, 0)] = .5 * (solution[COORD(nRows - 2, 0)] + solution[COORD(nRows - 1, 1)]);

		// bottom right
		solution[COORD(nRows - 1, nCols - 1)] = .5 * (solution[COORD(nRows - 1, nCols - 2)] + solution[COORD(nRows - 2, nCols - 1)]);
	}
}
