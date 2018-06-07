
#include <Flags.cuh>

EXTERN_C
{
	enum class SolverType
	{
		Null = 0,
		ExplicitEuler = 1,
		ImplicitEuler = 2,
		CrankNicolson = 3,
	};

	enum class BoundaryConditionType
	{
		Null = 0,
		Dirichlet = 1,
		Neumann = 2,
		Periodic = 3,
	};

	class BoundaryCondition1D
	{
	public:
		BoundaryConditionType type = BoundaryConditionType::Neumann;
		double left = 0.0;
		double right = 0.0;

		BoundaryCondition1D(const BoundaryConditionType type, const double left, const double right)
			: type(type), left(left), right(right)
		{
		}

		virtual ~BoundaryCondition1D() noexcept = default;
		BoundaryCondition1D(const BoundaryCondition1D& rhs) noexcept = default;
		BoundaryCondition1D(BoundaryCondition1D&& rhs) noexcept = default;
		BoundaryCondition1D& operator=(const BoundaryCondition1D& rhs) noexcept = default;
		BoundaryCondition1D& operator=(BoundaryCondition1D&& rhs) noexcept = default;
	};

	struct FiniteDifferenceInput1D
	{
		/**
		* Time discretization mesh size
		*/
		double dt = 0.0;

		/**
		* Space discretization mesh
		*/
		MemoryBuffer grid;

		/**
		* Advection coefficient
		*/
		MemoryBuffer velocity; 

		/**
		* Diffusion coefficient
		*/
		MemoryBuffer diffusion;

		/**
		* Solver Type
		*/
		SolverType solverType;

		/**
		* Left/Right boundary conditions
		*/
		BoundaryCondition1D boundaryConditions;

		FiniteDifferenceInput1D(double dt,
								MemoryBuffer grid,
								MemoryBuffer velocity,
								MemoryBuffer diffusion,
								SolverType solverType,
								BoundaryCondition1D boundaryConditions)
			:
			dt(dt),
			grid(grid),
			velocity(velocity),
			diffusion(diffusion),
			solverType(solverType),
			boundaryConditions(boundaryConditions)
		{
		}

		virtual ~FiniteDifferenceInput1D() noexcept = default;
		FiniteDifferenceInput1D(const FiniteDifferenceInput1D& rhs) noexcept = default;
		FiniteDifferenceInput1D(FiniteDifferenceInput1D&& rhs) noexcept = default;
		FiniteDifferenceInput1D& operator=(const FiniteDifferenceInput1D& rhs) noexcept = default;
		FiniteDifferenceInput1D& operator=(FiniteDifferenceInput1D&& rhs) noexcept = default;
	};
}