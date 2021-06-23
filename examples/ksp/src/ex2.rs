//! Concepts: KSP^basic parallel example;
//! Concepts: KSP^Laplacian, 2d
//! Concepts: Laplacian, 2d
//! Processors: n
//!
//! To run:
//! ```text
//! $ cargo build --bin ex2
//! $ mpiexec -n 1 target/debug/ex2
//! Norm of error 1.56044e-4, Iters 6
//! $ mpiexec -n 2 target/debug/ex2
//! Norm of error 4.11674e-4, Iters 7
//! $ mpiexec -n 5 target/debug/ex2 -m 80 -n 70
//! Norm of error 2.28509e-3, Iters 83
//! ```

static HELP_MSG: &str = "Solves a linear system in parallel with KSP.
Input parameters include:\n\n";

use petsc_rs::prelude::*;
use structopt::StructOpt;

mod opt;
use opt::*;

#[derive(Debug, StructOpt)]
#[structopt(name = "ex2", about = HELP_MSG)]
struct Opt {
    /// number of mesh points in x-direction
    #[structopt(short, default_value = "8")]
    m: PetscInt,
    
    /// number of mesh points in y-direction
    #[structopt(short, default_value = "7")]
    n: PetscInt,

    /// write exact solution vector to stdout
    #[structopt(short, long)]
    view_exact_sol: bool,

    /// use `-- -help` for petsc help
    #[structopt(subcommand)]
    sub: Option<PetscOpt>,
}

fn main() -> petsc_rs::Result<()> {
    let Opt {m, n, view_exact_sol, sub: ext_args} = Opt::from_args();
    let petsc_args = PetscOpt::petsc_args(ext_args); // Is there an easier way to do this

    // optionally initialize mpi
    // let _univ = mpi::initialize().unwrap();
    // init with no options
    let petsc = Petsc::builder()
        .args(petsc_args)
        .help_msg(HELP_MSG)
        .init()?;

    // or init with no options
    // let petsc = Petsc::init_no_args()?;

    petsc_println!(petsc.world(), "Hello parallel world of {} processes!", petsc.world().size() );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*
        Create parallel matrix, specifying only its global dimensions.
        When using MatCreate(), the matrix format can be specified at
        runtime. Also, the parallel partitioning of the matrix is
        determined by PETSc at runtime.

        Performance tuning note:  For problems of substantial size,
        preallocation of matrix memory is crucial for attaining good
        performance. See the matrix chapter of the users manual for details.
    */

    #[allow(non_snake_case)]
    let mut A = petsc.mat_create()?;
    A.set_sizes(None, None, Some(m*n), Some(m*n))?;
    A.set_from_options()?;
    A.mpi_aij_set_preallocation(5, None, 5, None)?;
    A.seq_aij_set_preallocation(5, None)?;
    A.seq_sb_aij_set_preallocation(1, 5, None)?;
    A.mpi_sb_aij_set_preallocation(1, 5, None, 5, None)?;
    A.mpi_sell_set_preallocation(5, None, 5, None)?;
    A.seq_sell_set_preallocation(5, None)?;

    /*
        Currently, all PETSc parallel matrix formats are partitioned by
        contiguous chunks of rows across the processors.  Determine which
        rows of the matrix are locally owned.
    */
  let mat_ownership_range = A.get_ownership_range()?;

    /*
        Set matrix elements for the 2-D, five-point stencil in parallel.
        - Each processor needs to insert only elements that it owns
            locally (but any non-local elements will be sent to the
            appropriate processor during matrix assembly).
        - Always specify global rows and columns of matrix entries.

        Note: this uses the less common natural ordering that orders first
        all the unknowns for x = h then for x = 2h etc; Hence you see J = Ii +- n
        instead of J = I +- m as you might expect. The more standard ordering
        would first do all variables for y = h, then y = 2h etc.

        Note MatAssemblyBegin(), MatAssemblyEnd() are automatically call by `assemble_with()`.
    */
    // Note, `PetscScalar` could be a complex number, so best practice is to instead of giving
    // float literals (i.e. `1.5`) when a function takes a `PetscScalar` wrap in in a `from`
    // call. E.x. `PetscScalar::from(1.5)`. This will do nothing if `PetscScalar` in a real number,
    // but if `PetscScalar` is complex it will construct a complex value which the imaginary part being
    // set to `0`.
    A.assemble_with(mat_ownership_range.map(|ii| {
            let mut data_vec = vec![];
            let i = ii/n;
            let j = ii - i*n;
            if i > 0   { let jj = ii - n; data_vec.push((ii, jj, PetscScalar::from(-1.0))); }
            if i < m-1 { let jj = ii + n; data_vec.push((ii, jj, PetscScalar::from(-1.0))); }
            if j > 0   { let jj = ii - 1; data_vec.push((ii, jj, PetscScalar::from(-1.0))); }
            if j < n-1 { let jj = ii + 1; data_vec.push((ii, jj, PetscScalar::from(-1.0))); }
            data_vec.push((ii, ii, PetscScalar::from(4.0)));
            data_vec
        }).flatten(), 
        InsertMode::ADD_VALUES, MatAssemblyType::MAT_FINAL_ASSEMBLY)?;

    /* A is symmetric. Set symmetric flag to enable ICC/Cholesky preconditioner */
    A.set_option(MatOption::MAT_SYMMETRIC, true)?;

    /*
        Create parallel vectors.
        - We form 1 vector from scratch and then duplicate as needed.
        - When using VecCreate(), VecSetSizes and VecSetFromOptions()
            in this example, we specify only the
            vector's global dimension; the parallel partitioning is determined
            at runtime.
        - When solving a linear system, the vectors and matrices MUST
            be partitioned accordingly.  PETSc automatically generates
            appropriately partitioned matrices and vectors when MatCreate()
            and VecCreate() are used with the same communicator.
        - The user can alternatively specify the local vector and matrix
            dimensions when more sophisticated partitioning is needed
            (replacing the PETSC_DECIDE argument in the VecSetSizes() statement
            below).
    */
    let mut u = petsc.vec_create()?;
    u.set_name("Exact Solution")?;
    u.set_sizes(None, Some(n*m))?;
    u.set_from_options()?;
    let mut b = u.duplicate()?;
    let mut x = u.duplicate()?;

    /*
        Set exact solution; then compute right-hand-side vector.
        By default we use an exact solution of a vector with all
        elements of 1.0;
    */
    u.set_all(PetscScalar::from(1.0))?;
    Mat::mult(&A, &u, &mut b)?;

    // View the exact solution vector if desired
    if view_exact_sol
    {
        let viewer = Viewer::create_ascii_stdout(petsc.world())?;
        u.view_with(Some(&viewer))?;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    let mut ksp = petsc.ksp_create()?;

    /*
        Set operators. Here the matrix that defines the linear system
        also serves as the matrix that defines the preconditioner.
    */
    #[allow(non_snake_case)]
    let rc_A = std::rc::Rc::new(A);
    ksp.set_operators(Some(rc_A.clone()), Some(rc_A.clone()))?;

    /*
        Set linear solver defaults for this problem (optional).
        - By extracting the KSP and PC contexts from the KSP context,
          we can then directly call any KSP and PC routines to set
          various options.
        - The following two statements are optional; all of these
          parameters could alternatively be specified at runtime via
          KSPSetFromOptions().  All of these defaults can be
          overridden at runtime, as indicated below.
    */
    ksp.set_tolerances(Some(1.0e-2/(((m+1)*(n+1)) as PetscReal)), Some(1.0e-50), None, None)?;

    /*
        Set runtime options, e.g.,
            `-- -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
        These options will override those specified above as long as
        KSPSetFromOptions() is called _after_ any other customization
        routines.
    */
    ksp.set_from_options()?;

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Solve the linear system
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ksp.solve(Some(&b), &mut x)?;


    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check the solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    x.axpy(PetscScalar::from(-1.0), &u)?;
    let x_norm = x.norm(NormType::NORM_2)?;
    let iters = ksp.get_iteration_number()?;
    petsc_println!(petsc.world(), "Norm of error {:.5e}, Iters {}", x_norm, iters);

    /*
        All PETSc objects are automatically destroyed when they are no longer needed.
        PetscFinalize() is also automatically called.
    */

    // return
    Ok(())
}
