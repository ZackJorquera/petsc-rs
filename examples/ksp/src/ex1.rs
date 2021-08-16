//! This file will show how to do the kps ex1 example in rust using the petsc-rs bindings.
//!
//! Concepts: KSP^solving a system of linear equations
//! Processors: 1
//!
//! Use "petsc_rs::prelude::*" to get direct access to all important petsc-rs bindings.
//! Use "mpi::traits::*" to get access to all mpi traits which allow you to call things like `world.size()`.
//!
//! To run:
//! ```text
//! $ cargo build --bin ksp-ex1
//! $ target/debug/ksp-ex1
//! Norm of error 2.41202e-15, Iters 5
//! $ mpiexec -n 1 target/debug/ksp-ex1
//! Norm of error 2.41202e-15, Iters 5
//! $ target/debug/ksp-ex1 -n 100
//! Norm of error 1.14852e-2, Iters 318
//! ```
//!
//! Note:  The corresponding parallel example is ex23.rs

static HELP_MSG: &str = "Solves a tridiagonal linear system with KSP.\n\n";

use petsc_rs::prelude::*;
use mpi::traits::*;

fn main() -> petsc_rs::Result<()> {
    let petsc = Petsc::builder()
        .args(std::env::args())
        .help_msg(HELP_MSG)
        .init()?;

    let n = petsc.options_try_get_int("-n")?.unwrap_or(10);
    let show_solution = petsc.options_try_get_bool("-show_solution")?.unwrap_or(false);

    if petsc.world().size() != 1 {
        Petsc::set_error(petsc.world(), PetscErrorKind::PETSC_ERR_WRONG_MPI_SIZE,
            "This is a uniprocessor example only!")?;
    }

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //   Compute the matrix and right-hand-side vector that define
    //   the linear system, Ax = b.
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    // Create vectors.  Note that we form 1 vector from scratch and
    // then duplicate as needed.
    let mut x = petsc.vec_create()?;
    x.set_name("Solution")?;
    x.set_sizes(None, n)?;
    x.set_from_options()?;
    let mut b = x.clone();
    let mut u = x.clone();

    #[allow(non_snake_case)]
    let mut A = petsc.mat_create()?;
    A.set_sizes(None, None, n, n)?;
    A.set_from_options()?;
    A.set_up()?;

    // Assemble matrix:
    // Note, `PetscScalar` could be a complex number, so best practice is to instead of giving
    // float literals (i.e. `1.5`) when a function takes a `PetscScalar` wrap in in a `from`
    // call. E.x. `PetscScalar::from(1.5)`. This will do nothing if `PetscScalar` in a real number,
    // but if `PetscScalar` is complex it will construct a complex value with the imaginary part being
    // set to `0`.
    A.assemble_with((0..n).map(|i| (-1..=1).map(move |j| (i,i+j))).flatten()
            // we could also filter out negatives, but `assemble_with` does that for us
            .filter(|&(i, j)| i < n && j < n)
            .map(|(i,j)| if i == j { (i, j, PetscScalar::from(2.0)) }
                         else { (i, j, PetscScalar::from(-1.0)) }),
        InsertMode::INSERT_VALUES, MatAssemblyType::MAT_FINAL_ASSEMBLY)?;

    // Set exact solution; then compute right-hand-side vector.
    u.set_all(PetscScalar::from(1.0))?;
    Mat::mult(&A, &u, &mut b)?;

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //          Create the linear solver and set various options
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    let mut ksp = petsc.ksp_create()?;

    // Set operators. Here the matrix that defines the linear system
    // also serves as the matrix that defines the preconditioner.
    ksp.set_operators(&A, &A)?;

    // Set linear solver defaults for this problem (optional).
    // - By extracting the KSP and PC contexts from the KSP context,
    //     we can then directly call any KSP and PC routines to set
    //     various options.
    // - The following statements are optional; all of these
    //     parameters could alternatively be specified at runtime via
    //     `KSP::set_from_options()`.
    let pc = ksp.get_pc_or_create()?;
    pc.set_type(PCType::PCJACOBI)?;
    ksp.set_tolerances(1.0e-5, None, None, None)?;

    // Set runtime options, e.g.,
    //     `-ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>`
    // These options will override those specified above as long as
    // `KSP::set_from_options()` is called _after_ any other customization
    // routines.
    ksp.set_from_options()?;

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //                  Solve the linear system
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ksp.solve(&b, &mut x)?;

    // View solver info; we could instead use the option -ksp_view to
    // print this info to the screen at the conclusion of `KSP::solve()`.
    let viewer = Viewer::create_ascii_stdout(petsc.world())?;
    viewer.view(&ksp)?;

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //                Check the solution and clean up
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if show_solution {
        viewer.view(&x)?;
        // Or we can do the following. Note, in a multi-process comm
        // world we should instead use `petsc_println_sync!`.
        println!("{}: {:.2}", x.get_name()?, *x.view()?);
    }
    x.axpy(PetscScalar::from(-1.0), &u)?;
    let x_norm = x.norm(NormType::NORM_2)?;
    let iters = ksp.get_iteration_number()?;
    petsc_println!(petsc.world(), "Norm of error {:.5e}, Iters {}", x_norm, iters)?;

    // All PETSc objects are automatically destroyed when they are no longer needed.
    // PetscFinalize() is also automatically called.

    // return
    Ok(())
}
