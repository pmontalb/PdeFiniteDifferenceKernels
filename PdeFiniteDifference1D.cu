#include "PdeFiniteDifference1D.cuh"
#include "PdeFiniteDifference.cuh"

EXTERN_C
{
	EXPORT int _MakeSpaceDiscretizer1D(MemoryTile spaceDiscretizer, const FiniteDifferenceInput1D input)
	{
	    if (input.spaceDiscretizerType == SpaceDiscretizerType::Null)
			return CudaKernelException::_NotImplementedException;

		switch (spaceDiscretizer.mathDomain)
		{
			case MathDomain::Float:
				CUDA_CALL_SINGLE(__MakeSpaceDiscretizer1D__<float>, (float*)spaceDiscretizer.pointer, (float*)input.grid.pointer, (float*)input.velocity.pointer, (float*)input.diffusion.pointer, input.spaceDiscretizerType, (float)input.dt, input.grid.size);
				break;
			case MathDomain::Double:
				CUDA_CALL_DOUBLE(__MakeSpaceDiscretizer1D__<double>, (double*)spaceDiscretizer.pointer, (double*)input.grid.pointer, (double*)input.velocity.pointer, (double*)input.diffusion.pointer, input.spaceDiscretizerType, input.dt, input.grid.size);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}
		return cudaGetLastError();
	}

	/**
	*	Sets the boundary conditions in the solution. It's a bit of a waste calling a kernel<<<1, 1>>>, but I found no other good way!
	*/
    EXPORT int _SetBoundaryConditions1D(MemoryTile solution, const FiniteDifferenceInput1D input)
	{
		switch (solution.mathDomain)
		{
			case MathDomain::Float:
				CUDA_CALL_XY(__SetBoundaryConditions1D__<float>, 1, 1, (float*)solution.pointer, (float)input.boundaryConditions.left.value, (float)input.boundaryConditions.right.value, input.boundaryConditions.left.type, input.boundaryConditions.right.type, (float*)input.grid.pointer, solution.nRows);
				break;
			case MathDomain::Double:
				CUDA_CALL_XY(__SetBoundaryConditions1D__<double>, 1, 1, (double*)solution.pointer, input.boundaryConditions.left.value, input.boundaryConditions.right.value, input.boundaryConditions.left.type, input.boundaryConditions.right.type, (double*)input.grid.pointer, solution.nRows);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}
		return cudaGetLastError();
	}

	EXPORT int _Iterate1D(MemoryTile solution, const MemoryCube timeDiscretizer, const FiniteDifferenceInput1D input, const unsigned nSteps)
	{
		// allocate a volatile buffer, used for the matrix-vector dot-product
		MemoryTile workBuffer = MemoryTile(solution);
		_Alloc(workBuffer);

		bool overwriteBuffer = true;
		int err = 0;
		for (unsigned n = 0; n < nSteps; ++n)
		{
			err = detail::_Advance(solution, timeDiscretizer, workBuffer, overwriteBuffer);
			if (err)
				return err;

			// set boundary conditions
			err = _SetBoundaryConditions1D(overwriteBuffer ? workBuffer : solution, input);
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

template <typename T>
GLOBAL void __MakeSpaceDiscretizer1D__(T* RESTRICT spaceDiscretizer, const T* RESTRICT grid, const T* RESTRICT velocity, const T* RESTRICT diffusion, const SpaceDiscretizerType discretizerType, const T dt, const unsigned sz)
{
	CUDA_FUNCTION_PROLOGUE;

	for (unsigned i = tid + 1; i < sz - 1; i += step)
	{
		const T dxPlus = grid[i + 1] - grid[i];
		const T dxMinus = grid[i] - grid[i - 1];
		const T dx = dxPlus + dxMinus;
		
		const T multiplierMinus = 1.0 / (dxMinus * dx);
		const T multiplierPlus = 1.0 / (dxPlus  * dx);

		__MakeSpaceDiscretizerWorker__<T>(spaceDiscretizer[i + sz * (i + 1)], spaceDiscretizer[i + sz * i], spaceDiscretizer[i + sz * (i - 1)], discretizerType,
									   velocity[i], diffusion[i],
									   dxPlus, dxMinus, dx, multiplierMinus, multiplierPlus, dt);
	}
}

template <typename T>
GLOBAL void __SetBoundaryConditions1D__(T* RESTRICT solution, const T leftValue, const T rightValue, const BoundaryConditionType leftBoundaryConditionType, const BoundaryConditionType rightBoundaryConditionType, const T* RESTRICT grid, const unsigned sz)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// update boundary condition only for the most recent solution, which is the first column of the solution matrix
	if (tid == 0)
	{
		switch (leftBoundaryConditionType)
		{
			case BoundaryConditionType::Dirichlet:
				solution[0] = leftValue;
				break;
			case BoundaryConditionType::Neumann:
				solution[0] = solution[1] - leftValue * (grid[1] - grid[0]);
				break;
			case BoundaryConditionType::Periodic:
				solution[0] = solution[sz - 2];
				break;
			default:
				break;
		}

		switch (rightBoundaryConditionType)
		{
			case BoundaryConditionType::Dirichlet:
				solution[sz - 1] = rightValue;
				break;
			case BoundaryConditionType::Neumann:
				solution[sz - 1] = solution[sz - 2] - rightValue * (grid[sz - 1] - grid[sz - 2]);
				break;
			case BoundaryConditionType::Periodic:
				solution[sz - 1] = solution[1];
			default:
				break;
		}
	}
}