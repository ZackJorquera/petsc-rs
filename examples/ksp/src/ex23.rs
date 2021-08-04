//! This file will show how to do the ksp ex23 example in rust using the petsc-rs bindings.
//!
//! Concepts: KSP^basic parallel example for solving a system of linear equations
//! Processors: n
//!
//! To run:
//! ```test
//! $ cargo build --bin ksp-ex23
//! $ mpiexec -n 1 target/debug/ksp-ex23
//! Norm of error 2.41202e-15, Iters 5
//! $ mpiexec -n 2 target/debug/ksp-ex23
//! Norm of error 2.41202e-15, Iters 5
//! ```
//!
//! Note: The corresponding uniprocessor example is ex1.rs

static HELP_MSG: &str = "Solves a tridiagonal linear system with KSP.\n\n";

use petsc_rs::prelude::*;
use mpi::traits::*;

fn main() -> petsc_rs::Result<()> {
    // optionally initialize mpi
    // let _univ = mpi::initialize().unwrap();
    // init with no options
    let petsc = Petsc::builder()
        .args(std::env::args())
        .help_msg(HELP_MSG)
        .init()?;

    // or init with no options
    // let petsc = Petsc::init_no_args()?;

    let n = petsc.options_try_get_int("-n")?.unwrap_or(10);

    petsc_println!(petsc.world(), "(petsc_println!) Hello parallel world of {} processes!", petsc.world().size() )?;

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*
        Create vectors.  Note that we form 1 vector from scratch and
        then duplicate as needed.
    */
    let mut x = petsc.vec_create()?;
    x.set_name("Solution")?;
    x.set_sizes(None, Some(n))?;
    x.set_from_options()?;
    let mut b = x.duplicate()?;
    let mut u = x.duplicate()?;

    /* 
        Identify the starting and ending mesh points on each
        processor for the interior part of the mesh. We let PETSc decide
        above.
    */

    let vec_ownership_range = x.get_ownership_range()?;
    let local_size = x.get_local_size()?;


    /*
        Create matrix.  When using MatCreate(), the matrix format can
        be specified at runtime.

        Performance tuning note:  For problems of substantial size,
        preallocation of matrix memory is crucial for attaining good
        performance. See the matrix chapter of the users manual for details.

        We pass in nlocal as the "local" size of the matrix to force it
        to have the same parallel layout as the vector created above.
    */
    #[allow(non_snake_case)]
    let mut A = petsc.mat_create()?;
    A.set_sizes(Some(local_size), Some(local_size), Some(n), Some(n))?;
    A.set_from_options()?;
    A.set_up()?;

    /*
        Assemble matrix
    */
    // Note, `PetscScalar` could be a complex number, so best practice is to instead of giving
    // float literals (i.e. `1.5`) when a function takes a `PetscScalar` wrap in in a `from`
    // call. E.x. `PetscScalar::from(1.5)`. This will do nothing if `PetscScalar` in a real number,
    // but if `PetscScalar` is complex it will construct a complex value with the imaginary part being
    // set to `0`.
    A.assemble_with(vec_ownership_range.map(|i| (-1..=1).map(move |j| (i,i+j))).flatten()
            .filter(|&(i,j)| i < n && j < n) // we could also filter out negatives, but assemble_with does that for us
            .map(|(i,j)| { if i == j { (i, j, PetscScalar::from(2.0)) }
                           else { (i, j, PetscScalar::from(-1.0)) } }),
        InsertMode::INSERT_VALUES, MatAssemblyType::MAT_FINAL_ASSEMBLY)?;

    /*
        Set exact solution; then compute right-hand-side vector.
    */
    u.set_all(PetscScalar::from(1.0))?;
    Mat::mult(&A, &u, &mut b)?;
    // petsc_println!(petsc, "b: {:?}", b.get_values(0..n)?)?;

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    let mut ksp = petsc.ksp_create()?;

    /*
        Set operators. Here the matrix that defines the linear system
        also serves as the matrix that defines the preconditioner.
    */
    ksp.set_operators(&A, &A)?;
    
    /*
        Set linear solver defaults for this problem (optional).
        - By extracting the KSP and PC contexts from the KSP context,
          we can then directly call any KSP and PC routines to set
          various options.
        - The following four statements are optional; all of these
          parameters could alternatively be specified at runtime via
          KSPSetFromOptions();
    */
    let pc = ksp.get_pc_mut()?;
    pc.set_type(PCType::PCJACOBI)?;
    ksp.set_tolerances(Some(1.0e-5), None, None, None)?;

    /*
        Set runtime options, e.g.,
            `-- -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>`
        These options will override those specified above as long as
        KSPSetFromOptions() is called _after_ any other customization
        routines.
    */
    ksp.set_from_options()?;

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Solve the linear system
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ksp.solve(Some(&b), &mut x)?;

    /*
        View solver info; we could instead use the option -ksp_view to
        print this info to the screen at the conclusion of KSPSolve().
    */
    let viewer = Viewer::create_ascii_stdout(petsc.world())?;
    ksp.view_with(Some(&viewer))?;

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check the solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    x.axpy(PetscScalar::from(-1.0), &u)?;
    let x_norm = x.norm(NormType::NORM_2)?;
    let iters = ksp.get_iteration_number()?;
    petsc_println!(petsc.world(), "Norm of error {:.5e}, Iters {}", x_norm, iters)?;

    /*
        All PETSc objects are automatically destroyed when they are no longer needed.
        PetscFinalize() is also automatically called.
    */

    // return
    Ok(())
}
