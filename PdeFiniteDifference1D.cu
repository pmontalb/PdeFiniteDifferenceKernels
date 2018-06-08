#include "PdeFiniteDifference1D.cuh"

#include <CuBlasWrappers.cuh>
#include <BufferInitializer.cuh>
#include <MemoryManager.cuh>

namespace detail
{
	/**
	*	Evolve the solution using the time discretizer.
	*	N.B.: solution is a memory tile, as some solver might require the solution history
	*	N.B.2: if provided, workBuffer is a previously allocated buffer used for matrix-vector multiplication
	*/
	int _Advance(MemoryTile solution, const MemoryCube timeDiscretizer, MemoryTile workBuffer, const bool overwriteBuffer)
	{
		// this is to support multi-step algorithms: each solution is multiplied by a different time discretizer
		MemoryBuffer _solution(solution.pointer, solution.nRows, solution.memorySpace, solution.mathDomain);
		MemoryBuffer _buffer(workBuffer.pointer, workBuffer.nRows, workBuffer.memorySpace, workBuffer.mathDomain);
		MemoryTile _timeDiscretizer(timeDiscretizer.pointer, timeDiscretizer.nRows, timeDiscretizer.nCols, timeDiscretizer.memorySpace, timeDiscretizer.mathDomain);

		// work out where to write the matrix-vector dot-product
		MemoryBuffer *_out, *_in;
		if (overwriteBuffer)
		{
			_out = &_buffer;
			_in = &_solution;
		}
		else
		{
			_in = &_buffer;
			_out = &_solution;
		}

		// multiplicate each solution with the respective time discretizer
		for (unsigned i = 0; i < solution.nCols; ++i)
		{
			_buffer.pointer = workBuffer.pointer + i * _buffer.TotalSize();
			_solution.pointer = solution.pointer + i * _buffer.TotalSize();
			_timeDiscretizer.pointer = timeDiscretizer.pointer + i * _timeDiscretizer.TotalSize();
			_Dot(*_out, _timeDiscretizer, *_in);
		}

		return cudaGetLastError();
	}

	/**
	*	Sets the boundary conditions in the solution. It's a bit of a waste calling a kernel<<<1, 1>>>, but I found no other good way!
	*/
	int _SetBoundaryConditions1D(MemoryTile solution, const FiniteDifferenceInput1D input)
	{
		if (input.boundaryConditions.left.type == BoundaryConditionType::Periodic && input.boundaryConditions.right.type == BoundaryConditionType::Periodic)
			return 0;  // no need to call the kernel!

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
}

EXTERN_C
{
	EXPORT int _MakeSpaceDiscretizer1D(MemoryTile spaceDiscretizer, const FiniteDifferenceInput1D input)
	{
		switch (spaceDiscretizer.mathDomain)
		{
			case MathDomain::Float:
				CUDA_CALL_SINGLE(__MakeSpaceDiscretizer1D__<float>, (float*)spaceDiscretizer.pointer, (float*)input.grid.pointer, (float*)input.velocity.pointer, (float*)input.diffusion.pointer, input.boundaryConditions.left.type, input.boundaryConditions.right.type, input.grid.size);
				break;
			case MathDomain::Double:
				CUDA_CALL_DOUBLE(__MakeSpaceDiscretizer1D__<double>, (double*)spaceDiscretizer.pointer, (double*)input.grid.pointer, (double*)input.velocity.pointer, (double*)input.diffusion.pointer, input.boundaryConditions.left.type, input.boundaryConditions.right.type, input.grid.size);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}
		return cudaGetLastError();
	}

	EXPORT int _MakeTimeDiscretizer1D(MemoryCube timeDiscretizer, const MemoryTile spaceDiscretizer, const FiniteDifferenceInput1D input)
	{
		switch (input.solverType)
		{
			case SolverType::ExplicitEuler:
			{
				// A = I + L * dt

				MemoryTile _timeDiscretizer = static_cast<MemoryTile>(timeDiscretizer);
				_Eye(_timeDiscretizer);
				_AddEqual(_timeDiscretizer, spaceDiscretizer, input.dt);
				break;
			}
			case SolverType::ImplicitEuler:
			{
				// A = (I - L * dt)^(-1)

				MemoryTile _timeDiscretizer = static_cast<MemoryTile>(timeDiscretizer);
				_Eye(_timeDiscretizer);
				_AddEqual(_timeDiscretizer, spaceDiscretizer, -input.dt);
				_Invert(_timeDiscretizer);
				break;
			}
			case SolverType::CrankNicolson:
			{
				// A = (I - L * .5 * dt)^(-1) * (I + L * .5 * dt)

				MemoryTile _timeDiscretizer = static_cast<MemoryTile>(timeDiscretizer);
				_Eye(_timeDiscretizer);

				// copy timeDiscretizer into leftOperator volatile buffer
				MemoryTile leftOperator(_timeDiscretizer);
				_Alloc(leftOperator);
				_DeviceToDeviceCopy(leftOperator, _timeDiscretizer);

				// left and right operator
				_AddEqual(leftOperator, spaceDiscretizer, -.5 * input.dt);  // A = I - .5 * dt
				_AddEqual(timeDiscretizer, spaceDiscretizer, .5 * input.dt);  // B = I + .5 * dt
				_Solve(leftOperator, _timeDiscretizer);

				_Free(leftOperator);
			}
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
			err = detail::_SetBoundaryConditions1D(overwriteBuffer ? workBuffer : solution, input);
			if (err)
				return err;

			overwriteBuffer = !overwriteBuffer;
		}

		if (overwriteBuffer)
		{
			// copy the result back from working buffer and free it
			_DeviceToDeviceCopy(solution, workBuffer);
			_Free(workBuffer);
		}

		return cudaGetLastError();
	}
}

template <typename T>
GLOBAL void __MakeSpaceDiscretizer1D__(T* RESTRICT spaceDiscretizer, const T* RESTRICT grid, const T* RESTRICT velocity, const T* RESTRICT diffusion, const BoundaryConditionType leftBoundaryConditionType, const BoundaryConditionType rightBoundaryConditionType, const unsigned sz)
{
	CUDA_FUNCTION_PROLOGUE;

	// first thread deals with boundary condition
	if (tid == 0)
	{
		// (consider that identity is added later on!)

		// Dirichlet: nothing to do

		// Neumann: -2 1 ...  0 0
		//			 0 0 ...  0 0
		//			 0 0 ... -1 0

		// Periodic: -1 0 ... 1  0
		//			  0 0 ... 0  0
		//			  0 1 ... 0 -1

		if (leftBoundaryConditionType == BoundaryConditionType::Neumann)
		{
			spaceDiscretizer[0] = static_cast<T>(-2.0);
			spaceDiscretizer[sz] = static_cast<T>(1.0);
		}
		else if (leftBoundaryConditionType == BoundaryConditionType::Periodic)
		{
			spaceDiscretizer[0] = static_cast<T>(-1.0);
			spaceDiscretizer[0 + sz * (sz - 2)] = static_cast<T>(1.0);
		}

		if (rightBoundaryConditionType == BoundaryConditionType::Neumann)
		{
			//spaceDiscretizer[sz - 1 + sz * (sz - 2)] = static_cast<T>(0.0);
			spaceDiscretizer[sz - 1 + sz * (sz - 1)] = static_cast<T>(-1.0);
		}
		else if (rightBoundaryConditionType == BoundaryConditionType::Periodic)
		{
			spaceDiscretizer[sz - 1 + sz * (sz - 2)] = static_cast<T>(0.0);
			spaceDiscretizer[sz - 1 + sz * 2] = static_cast<T>(1.0);
		}
	}

	for (unsigned i = tid + 1; i < sz - 1; i += step)
	{
		const T dxPlus = grid[i + 1] - grid[i];
		const T dxMinus = grid[i] - grid[i - 1];
		const T dx = dxPlus + dxMinus;

		// 3-point centered spatial finite difference that accounts for uneven space mesh
		spaceDiscretizer[i + sz * (i - 1)] = (-dxPlus  * velocity[i] + static_cast<T>(2.0) * diffusion[i]) / (dxMinus * dx);
		spaceDiscretizer[i + sz * (i + 1)] = (dxMinus * velocity[i] + static_cast<T>(2.0) * diffusion[i]) / (dxPlus  * dx);
		spaceDiscretizer[i + sz * i] = -spaceDiscretizer[i + sz * (i - 1)] - spaceDiscretizer[i + sz * (i + 1)];
	}
}

template <typename T>
GLOBAL void __SetBoundaryConditions1D__(T* RESTRICT solution, const T leftValue, const T rightValue, const BoundaryConditionType leftBoundaryConditionType, const BoundaryConditionType rightBoundaryConditionType, const T* RESTRICT grid, const unsigned sz)
{
	CUDA_FUNCTION_PROLOGUE;

	// update boundary condition only for the most recent solution, which is the first column of the solution matrix
	if (tid == 0)
	{
		switch (leftBoundaryConditionType)
		{
			case BoundaryConditionType::Dirichlet:
				solution[0] = leftValue;
			case BoundaryConditionType::Neumann:
				solution[0] = leftValue * (grid[1] - grid[0]);
			case BoundaryConditionType::Periodic:
				// this is already done with the matrix multiplication!
				//solution[0] = solution[0];
			default:
				break;
		}

		switch (leftBoundaryConditionType)
		{
			case BoundaryConditionType::Dirichlet:
				solution[sz - 1] = rightValue;
			case BoundaryConditionType::Neumann:
				solution[sz - 1] = rightValue * (grid[sz - 1] - grid[sz - 2]);
			case BoundaryConditionType::Periodic:
				// this is already done with the matrix multiplication!
				//solution[sz - 1] = rightValue * (grid[sz - 1] - grid[sz - 2]);
			default:
				break;
		}
	}
}