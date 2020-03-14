#include "PdeFiniteDifference.cuh"

namespace detail
{
	/**
	*	Evolve the solution using the time discretizer.
	*	N.B.: solution is a memory tile, as some solver might require the solution history
	*	N.B.2: if provided, workBuffer is a previously allocated buffer used for matrix-vector multiplication
	*/
	int _Advance(MemoryTile& solution, const MemoryCube& timeDiscretizer, MemoryTile& workBuffer, const bool overwriteBuffer)
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
		const ptr_t inPtr = _in->pointer;
		const ptr_t outPtr = _out->pointer;

		// multiplicate each solution with the respective time discretizer
		for (unsigned i = 0; i < solution.nCols; ++i)
		{
			_buffer.pointer = workBuffer.pointer + i * _buffer.TotalSize();
			_solution.pointer = solution.pointer + i * _solution.TotalSize();
			_timeDiscretizer.pointer = timeDiscretizer.pointer + i * _timeDiscretizer.TotalSize();
			_Dot(*_out, _timeDiscretizer, *_in);
		}

		// add the partial results into the latest solution

		for (unsigned i = 1; i < solution.nCols; ++i)
		{
			// cumulative sum of each step contribution into the first column
			_out->pointer = outPtr;
			_in->pointer = outPtr + i * _in->TotalSize();  // re-use _in for convenience!
			_AddEqual(*_out, *_in);

			// copy the input solution into the older solution buffers
			_out->pointer = _in->pointer;
			_in->pointer = inPtr + i * _in->TotalSize();
			_DeviceToDeviceCopy(*_out, *_in);
		}

		return cudaGetLastError();
	}

	int _MakeRungeKuttaGaussLegendre(const double dt,
									 const MemoryTile& spaceDiscretizer,
									 MemoryTile& timeDiscretizer)
	{
		constexpr double a00 = { .25 };
		constexpr double sqrt3 = { 1.73205080756888 };
		constexpr double a01 = { .25 - sqrt3 / 6.0 };
		constexpr double a10 = { .25 + sqrt3 / 6.0 };
		constexpr double a11 = { .25 };

		MemoryTile eye(timeDiscretizer);
		_Alloc(eye);
		_Eye(eye);

		MemoryTile A(timeDiscretizer);
		_Alloc(A);
		_Add(A, eye, spaceDiscretizer, -a00 * dt);

		MemoryTile B(timeDiscretizer);
		_Alloc(B);
		_DeviceToDeviceCopy(B, spaceDiscretizer);
		_Solve(A, B);
		_Scale(B, a10 * dt);

		MemoryTile C(timeDiscretizer);
		_Alloc(C);
		_DeviceToDeviceCopy(C, B);
		_Scale(C, a01 * dt);
		_AddEqualMatrix(C, eye, MatrixOperation::None, MatrixOperation::None, 1.0, a11 * dt);

		MemoryTile C2(timeDiscretizer);
		_Alloc(C2);
		_DeviceToDeviceCopy(C2, C);
		_Multiply(C, spaceDiscretizer, C2, MatrixOperation::None, MatrixOperation::None);
		_Free(C2);

		MemoryTile D(timeDiscretizer);
		_Alloc(D);
		_Add(D, C, eye, -1);

		MemoryTile E(timeDiscretizer);
		_Alloc(E);
		_Add(E, eye, B);

		MemoryTile k_2(timeDiscretizer);
		_Alloc(k_2);
		_Multiply(k_2, spaceDiscretizer, E, MatrixOperation::None, MatrixOperation::None);
		_Solve(D, k_2);

		MemoryTile F(timeDiscretizer);
		_Alloc(F);
		_Add(F, eye, k_2, a01 * dt);

		MemoryTile k_1(timeDiscretizer);
		_Alloc(k_1);
		_Multiply(k_1, spaceDiscretizer, F, MatrixOperation::None, MatrixOperation::None);
		_Solve(A, k_1);

		_Eye(timeDiscretizer);
		_AddEqualMatrix(k_1, k_2);
		_AddEqualMatrix(timeDiscretizer, k_1, MatrixOperation::None, MatrixOperation::None, 1.0, .5 * dt);

		_Free(eye);
		_Free(A);
		_Free(B);
		_Free(C);
		_Free(D);
		_Free(E);
		_Free(F);

		_Free(k_1);
		_Free(k_2);

		return cudaGetLastError();
	}
}

EXTERN_C
{

	EXPORT int _MakeTimeDiscretizerAdvectionDiffusion(MemoryCube& timeDiscretizer, const MemoryTile& spaceDiscretizer, const SolverType solverType, const double dt)
	{
		MemoryTile _timeDiscretizer;
		ExtractMatrixBufferFromCube(_timeDiscretizer, timeDiscretizer, 0);

		switch (solverType)
		{
			case SolverType::ExplicitEuler:
				// A = I + L * dt
				assert(timeDiscretizer.nCubes == 1);

				_Eye(_timeDiscretizer);
				_AddEqualMatrix(_timeDiscretizer, spaceDiscretizer, MatrixOperation::None, MatrixOperation::None, 1.0, dt);
				break;

			case SolverType::ImplicitEuler:
				// A = (I - L * dt)^(-1)
				assert(timeDiscretizer.nCubes == 1);

				_Eye(_timeDiscretizer);
				_AddEqualMatrix(_timeDiscretizer, spaceDiscretizer, MatrixOperation::None, MatrixOperation::None, 1.0, -dt);
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
				_AddEqualMatrix(leftOperator, spaceDiscretizer, MatrixOperation::None, MatrixOperation::None, 1.0, -.5 * dt);  // A = I - .5 * dt
				_AddEqualMatrix(timeDiscretizer, spaceDiscretizer, MatrixOperation::None, MatrixOperation::None, 1.0, .5 * dt);  // B = I + .5 * dt
				_Solve(leftOperator, _timeDiscretizer);

				_Free(leftOperator);
			}
			break;

			case SolverType::RungeKuttaRalston:
				assert(timeDiscretizer.nCubes == 1);
				detail::_MakeRungeKuttaDiscretizer<2>({ 0,
													  2.0 / 3.0, 0 },
													  { .25, .75 }, dt, spaceDiscretizer, _timeDiscretizer);
				break;
			case SolverType::RungeKutta3:
				assert(timeDiscretizer.nCubes == 1);
				detail::_MakeRungeKuttaDiscretizer<3>({ 0,
													  .5, .0,
													  -1,  2, 0 },
													  { 1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0 }, dt, spaceDiscretizer, _timeDiscretizer);
				break;
			case SolverType::RungeKutta4:
				assert(timeDiscretizer.nCubes == 1);
				detail::_MakeRungeKuttaDiscretizer<4>({ 0,
													  .5, .0,
													  0, .5, 0,
													  0,  0, 1, 0 },
													  { 1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0 }, dt, spaceDiscretizer, _timeDiscretizer);
				break;
			case SolverType::RungeKuttaThreeEight:
				assert(timeDiscretizer.nCubes == 1);
				detail::_MakeRungeKuttaDiscretizer<4>({ 0,
													  1.0 / 3.0, .0,
													  -1.0 / 3.0,  1, 0,
													  1, -1, 1, 0 },
													  { 1.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0, 1.0 / 8.0 }, dt, spaceDiscretizer, _timeDiscretizer);
				break;
			case SolverType::RungeKuttaGaussLegendre4:
				assert(timeDiscretizer.nCubes == 1);
				detail::_MakeRungeKuttaGaussLegendre(dt, spaceDiscretizer, _timeDiscretizer);
				break;

			case SolverType::RichardsonExtrapolation2:
			{
				assert(timeDiscretizer.nCubes == 1);
				_Eye(_timeDiscretizer);
				_AddEqualMatrix(_timeDiscretizer, spaceDiscretizer, MatrixOperation::None, MatrixOperation::None, 1.0, -dt);
				_Invert(_timeDiscretizer);
				_Scale(timeDiscretizer, -1.0);

				MemoryTile halfIteration(_timeDiscretizer);
				_Alloc(halfIteration);
				_Eye(halfIteration);
				_AddEqualMatrix(halfIteration, spaceDiscretizer, MatrixOperation::None, MatrixOperation::None, 1.0, -.5 * dt);

				MemoryTile halfIterationSquared(_timeDiscretizer);
				_Alloc(halfIterationSquared);
				_Multiply(halfIterationSquared, halfIteration, halfIteration, MatrixOperation::None, MatrixOperation::None);
				_Invert(halfIterationSquared);

				_AddEqualMatrix(_timeDiscretizer, halfIterationSquared, MatrixOperation::None, MatrixOperation::None, 1.0, 2.0);

				_Free(halfIteration);
				_Free(halfIterationSquared);
			}
			break;

			case SolverType::RichardsonExtrapolation3:
			{
				assert(timeDiscretizer.nCubes == 1);
				_Eye(_timeDiscretizer);
				_AddEqualMatrix(_timeDiscretizer, spaceDiscretizer, MatrixOperation::None, MatrixOperation::None, 1.0, -dt);
				_Invert(_timeDiscretizer);

				// - F
				_Scale(_timeDiscretizer, -1.0);

				MemoryTile halfIteration(_timeDiscretizer);
				_Alloc(halfIteration);
				_Eye(halfIteration);
				_AddEqualMatrix(halfIteration, spaceDiscretizer, MatrixOperation::None, MatrixOperation::None, 1.0, -.5 * dt);

				MemoryTile halfIterationSquared(_timeDiscretizer);
				_Alloc(halfIterationSquared);
				_Multiply(halfIterationSquared, halfIteration, halfIteration, MatrixOperation::None, MatrixOperation::None);
				_Invert(halfIterationSquared);  // H

				MemoryTile quarterIteration(_timeDiscretizer);
				_Alloc(quarterIteration);
				_Eye(quarterIteration);
				_AddEqualMatrix(quarterIteration, spaceDiscretizer, MatrixOperation::None, MatrixOperation::None, 1.0, -.25 * dt);

				MemoryTile quarterIterationFour(_timeDiscretizer);
				_Alloc(quarterIterationFour);
				_Multiply(halfIteration, quarterIteration, quarterIteration, MatrixOperation::None, MatrixOperation::None);  // re-use halfIteration for convenience
				_Multiply(quarterIterationFour, halfIteration, halfIteration, MatrixOperation::None, MatrixOperation::None);
				_Invert(quarterIterationFour);  // Q

												// 2 * H - F
				_AddEqualMatrix(_timeDiscretizer, halfIterationSquared, MatrixOperation::None, MatrixOperation::None, 1.0, 2.0);

				// -(2 * H - F) / 3
				_Scale(_timeDiscretizer, -1.0 / 3.0);

				// 2 * Q - H
				_Scale(halfIterationSquared, -1);
				_AddEqualMatrix(halfIterationSquared, quarterIterationFour, MatrixOperation::None, MatrixOperation::None, 1.0, 2.0);

				// (2 * Q - H) * 4/3 - (2 * H - F) / 3
				_AddEqualMatrix(_timeDiscretizer, halfIterationSquared, MatrixOperation::None, MatrixOperation::None, 1.0, 4.0 / 3.0);

				_Free(halfIteration);
				_Free(halfIterationSquared);
				_Free(quarterIteration);
				_Free(quarterIterationFour);
			}
			break;

			case SolverType::AdamsBashforth2:
				// A_{n + 1} = (I + L * 1.5 * dt)
				assert(timeDiscretizer.nCubes == 2);

				_Eye(_timeDiscretizer);
				_AddEqual(_timeDiscretizer, spaceDiscretizer, 1.5 * dt);  // A = I + 1.5 * dt

				// A_{n} = - L * .5 * dt
				_timeDiscretizer.pointer += _timeDiscretizer.nRows * _timeDiscretizer.nCols * _timeDiscretizer.ElementarySize();
				_DeviceToDeviceCopy(_timeDiscretizer, spaceDiscretizer);
				_Scale(_timeDiscretizer, -.5 * dt);
				break;

			case SolverType::AdamsMouldon2:
			{
				// A_{n + 1} = (I - L * 5 / 12 * dt)^(-1) * (I + L * 2.0 / 3.0 * dt)
				assert(timeDiscretizer.nCubes == 2);

				// copy timeDiscretizer into leftOperator volatile buffer
				MemoryTile leftOperator(_timeDiscretizer);
				_Alloc(leftOperator);
				_Eye(leftOperator);
				_AddEqual(leftOperator, spaceDiscretizer, -5.0 / 12.0 * dt);  // A = I - .5 * dt

				_Eye(_timeDiscretizer);
				_AddEqual(_timeDiscretizer, spaceDiscretizer, 2.0 / 3.0 * dt);  // A = I - .5 * dt
				_Solve(leftOperator, _timeDiscretizer);

				// A_{n} = (I - L * 5 / 12 * dt)^(-1) * (- L *  1.0 / 12.0 * dt)
				_timeDiscretizer.pointer += _timeDiscretizer.nRows * _timeDiscretizer.nCols * _timeDiscretizer.ElementarySize();
				_DeviceToDeviceCopy(_timeDiscretizer, spaceDiscretizer);
				_Scale(_timeDiscretizer, -1.0 / 12.0 * dt);
				_Solve(leftOperator, _timeDiscretizer);
			}
			break;

			default:
				return CudaKernelException::_NotImplementedException;
		}

		return cudaGetLastError();
	}

    EXPORT int _MakeTimeDiscretizerWaveEquation(MemoryCube& timeDiscretizer, const MemoryTile& spaceDiscretizer, const SolverType solverType, const double dt)
	{
		MemoryTile _timeDiscretizer;
		ExtractMatrixBufferFromCube(_timeDiscretizer, timeDiscretizer, 0);

		switch (solverType)
		{
			case SolverType::ExplicitEuler:
			{
				// A = I
				assert(timeDiscretizer.nCubes == 1);

				_Eye(_timeDiscretizer);
				break;
			}

			case SolverType::ImplicitEuler:
			{
				// A = (I - L * dt^2)^(-1)
				assert(timeDiscretizer.nCubes == 1);

				_Eye(_timeDiscretizer);
				_AddEqualMatrix(_timeDiscretizer, spaceDiscretizer, MatrixOperation::None, MatrixOperation::None, 1.0, -dt * dt);
				_Invert(_timeDiscretizer);
				break;
			}

			default:
				return CudaKernelException::_NotImplementedException;
		}

		return cudaGetLastError();
	}
}
