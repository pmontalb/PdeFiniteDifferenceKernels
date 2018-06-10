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
	int _Advance1D(MemoryTile solution, const MemoryCube timeDiscretizer, MemoryTile workBuffer, const bool overwriteBuffer)
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
			_solution.pointer = solution.pointer + i * _solution.TotalSize();
			_timeDiscretizer.pointer = timeDiscretizer.pointer + i * _timeDiscretizer.TotalSize();
			_Dot(*_out, _timeDiscretizer, *_in);
		}

		// add the partial results into the latest solution
		const ptr_t& outPtr = overwriteBuffer ? workBuffer.pointer : solution.pointer;
		for (unsigned i = 1; i < solution.nCols; ++i)
		{
			// copy the input solution into the older solution buffers
			_out->pointer = outPtr + i * _buffer.TotalSize();
			_DeviceToDeviceCopy(*_out, *_in);

			// re-use _in for convenience
			_out->pointer = outPtr;
			_in->pointer = _out->pointer + i * _solution.TotalSize();
			_AddEqual(*_out, *_in);
		}
		

		return cudaGetLastError();
	}

	/**
	*	Sets the boundary conditions in the solution. It's a bit of a waste calling a kernel<<<1, 1>>>, but I found no other good way!
	*/
	int _SetBoundaryConditions1D(MemoryTile solution, const MemoryCube timeDiscretizer, const FiniteDifferenceInput1D input)
	{
		if (input.boundaryConditions.left.type == BoundaryConditionType::Periodic && input.boundaryConditions.right.type == BoundaryConditionType::Periodic)
			return 0;  // no need to call the kernel!

		switch (solution.mathDomain)
		{
			case MathDomain::Float:
				CUDA_CALL_XY(__SetBoundaryConditions1D__<float>, 1, 1, (float*)solution.pointer, (float*)timeDiscretizer.pointer, (float)input.boundaryConditions.left.value, (float)input.boundaryConditions.right.value, input.boundaryConditions.left.type, input.boundaryConditions.right.type, (float*)input.grid.pointer, solution.nRows);
				break;
			case MathDomain::Double:
				CUDA_CALL_XY(__SetBoundaryConditions1D__<double>, 1, 1, (double*)solution.pointer, (double*)timeDiscretizer.pointer, input.boundaryConditions.left.value, input.boundaryConditions.right.value, input.boundaryConditions.left.type, input.boundaryConditions.right.type, (double*)input.grid.pointer, solution.nRows);
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
				CUDA_CALL_SINGLE(__MakeSpaceDiscretizer1D__<float>, (float*)spaceDiscretizer.pointer, (float*)input.grid.pointer, (float*)input.velocity.pointer, (float*)input.diffusion.pointer, input.boundaryConditions.left.type, input.boundaryConditions.right.type, (float)input.dt, input.grid.size);
				break;
			case MathDomain::Double:
				CUDA_CALL_DOUBLE(__MakeSpaceDiscretizer1D__<double>, (double*)spaceDiscretizer.pointer, (double*)input.grid.pointer, (double*)input.velocity.pointer, (double*)input.diffusion.pointer, input.boundaryConditions.left.type, input.boundaryConditions.right.type, input.dt, input.grid.size);
				break;
			default:
				return CudaKernelException::_NotImplementedException;
		}
		return cudaGetLastError();
	}

	EXPORT int _MakeTimeDiscretizer1D(MemoryCube timeDiscretizer, const MemoryTile spaceDiscretizer, const FiniteDifferenceInput1D input)
	{
		MemoryTile _timeDiscretizer;
		extractMatrixBufferFromCube(_timeDiscretizer, timeDiscretizer, 0);

		switch (input.solverType)
		{
			case SolverType::ExplicitEuler:
				// A = I + L * dt
				assert(timeDiscretizer.nCubes == 1);

				_Eye(_timeDiscretizer);
				_AddEqual(_timeDiscretizer, spaceDiscretizer, input.dt);
				break;

			case SolverType::ImplicitEuler:
				// A = (I - L * dt)^(-1)
				assert(timeDiscretizer.nCubes == 1);

				_Eye(_timeDiscretizer);
				_AddEqual(_timeDiscretizer, spaceDiscretizer, -input.dt);
				_Invert(_timeDiscretizer);
				break;

			case SolverType::CrankNicolson:
			{
				// A = (I - L * .5 * dt)^(-1) * (I + L * .5 * dt)
				assert(timeDiscretizer.nCubes == 1);

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

			case SolverType::AdamsBashforth2:
				// A_{n + 1} = (I + L * 1.5 * dt)
				assert(timeDiscretizer.nCubes == 2);
				
				_Eye(_timeDiscretizer);
				_AddEqual(_timeDiscretizer, spaceDiscretizer, 1.5 * input.dt);  // A = I - .5 * dt

				// A_{n} = - L * .5 * dt
				_timeDiscretizer.pointer += _timeDiscretizer.nRows * _timeDiscretizer.nCols * _timeDiscretizer.ElementarySize();
				_DeviceToDeviceCopy(_timeDiscretizer, spaceDiscretizer);
				_Scale(_timeDiscretizer, -.5 * input.dt);
				break;

			case SolverType::AdamsMouldon2:
			{
				// A_{n + 1} = (I - L * 5 / 12 * dt)^(-1) * (I + L * 2.0 / 3.0 * dt)
				assert(timeDiscretizer.nCubes == 2);

				// copy timeDiscretizer into leftOperator volatile buffer
				MemoryTile leftOperator(_timeDiscretizer);
				_Alloc(leftOperator);
				_Eye(leftOperator);
				_AddEqual(leftOperator, spaceDiscretizer, -5.0 / 12.0 * input.dt);  // A = I - .5 * dt

				_Eye(_timeDiscretizer);
				_AddEqual(_timeDiscretizer, spaceDiscretizer, 2.0 / 3.0 * input.dt);  // A = I - .5 * dt
				_Solve(leftOperator, _timeDiscretizer);

				// A_{n} = (I - L * 5 / 12 * dt)^(-1) * (- L *  1.0 / 12.0 * dt)
				_timeDiscretizer.pointer += _timeDiscretizer.nRows * _timeDiscretizer.nCols * _timeDiscretizer.ElementarySize();
				_DeviceToDeviceCopy(_timeDiscretizer, spaceDiscretizer);
				_Scale(_timeDiscretizer, -1.0 / 12.0 * input.dt);
				_Solve(leftOperator, _timeDiscretizer);
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
			err = detail::_Advance1D(solution, timeDiscretizer, workBuffer, overwriteBuffer);
			if (err)
				return err;

			// set boundary conditions
			err = detail::_SetBoundaryConditions1D(overwriteBuffer ? workBuffer : solution, timeDiscretizer, input);
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
GLOBAL void __MakeSpaceDiscretizer1D__(T* RESTRICT spaceDiscretizer, const T* RESTRICT grid, const T* RESTRICT velocity, const T* RESTRICT diffusion, const BoundaryConditionType leftBoundaryConditionType, const BoundaryConditionType rightBoundaryConditionType, const T dt, const unsigned sz)
{
	CUDA_FUNCTION_PROLOGUE;

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
GLOBAL void __SetBoundaryConditions1D__(T* RESTRICT solution, T* RESTRICT timeDiscretizer, const T leftValue, const T rightValue, const BoundaryConditionType leftBoundaryConditionType, const BoundaryConditionType rightBoundaryConditionType, const T* RESTRICT grid, const unsigned sz)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// update boundary condition only for the most recent solution, which is the first column of the solution matrix
	if (tid == 0)
	{
		// Dirichlet:  1 0 ...  0 0
		//			   0 0 ...  0 0
		//			   0 0 ...  0 1

		// Neumann: -2 1 ...  0 0
		//			 0 0 ...  0 0
		//			 0 0 ... -1 0

		// Periodic: -1 0 ... 1  0
		//			  0 0 ... 0  0
		//			  0 1 ... 0 -1

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