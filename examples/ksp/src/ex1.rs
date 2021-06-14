//! This file will show how to do the kps ex1 example in rust using the petsc-rs bindings.
//!
//! Concepts: KSP^solving a system of linear equations
//! Processors: 1
//!
//! Use "petsc_rs::prelude::*" to get direct access to all important petsc-rs bindings
//!     and mpi traits which allow you to call things like `world.size()`.
//!
//! To run:
//! ```text
//! $ cargo build --bin ex1
//! $ target/debug/ex1
//! Norm of error 2.41202e-15, Iters 5
//! $ mpiexec -n 1 target/debug/ex1
//! Norm of error 2.41202e-15, Iters 5
//! $ target/debug/ex1 -n 100
//! Norm of error 1.14852e-2, Iters 318
//! ```
//!
//! Note:  The corresponding parallel example is ex23.rs


static HELP_MSG: &str = "Solves a tridiagonal linear system with KSP.\n\n";

use petsc_rs::prelude::*;
use structopt::StructOpt;

mod opt;
use opt::*;

// TODO: make this more stream-lined, maybe add to the petsc-rs lib
#[derive(Debug, StructOpt)]
#[structopt(name = "ex1", about = HELP_MSG)]
struct Opt {
    /// Size of the vector and matrix
    #[structopt(short, long, default_value = "10")]
    num_elems: i32,

    /// use `-- -help` for petsc help
    #[structopt(subcommand)]
    sub: Option<PetscOpt>,
}

fn main() -> petsc_rs::Result<()> {
    // Note, this does not work. It runs of all precesses which isn't what we want
    let Opt {num_elems: n, sub: ext_args} = Opt::from_args();
    let petsc_args = PetscOpt::petsc_args(ext_args); // Is there an easier way to do this

    // optionally initialize mpi
    // let _univ = mpi::initialize().unwrap();
    // init with options
    let petsc = Petsc::builder()
        .args(petsc_args)
        .help_msg(HELP_MSG)
        .init()?;

    // or init with no options
    // let petsc = Petsc::init_no_args()?;

    if petsc.world().size() != 1
    {
        Petsc::set_error(petsc.world(), PetscErrorKind::PETSC_ERROR_WRONG_MPI_SIZE, "This is a uniprocessor example only!")?;
    }

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

    #[allow(non_snake_case)]
    let mut A = petsc.mat_create()?;
    A.set_sizes(None, None, Some(n), Some(n))?;
    A.set_from_options()?;
    A.set_up()?;

    /*
        Assemble matrix
    */
    A.assemble_with((0..n).map(|i| (-1..=1).map(move |j| (i,i+j))).flatten()
            .filter(|&(i, j)| i < n && j < n) // we could also filter out negatives, but assemble_with does that for us
            .map(|(i,j)| if i == j { (i, j, 2.0) }
                         else { (i, j, -1.0) }),
        InsertMode::INSERT_VALUES, MatAssemblyType::MAT_FINAL_ASSEMBLY)?;

    /*
        Set exact solution; then compute right-hand-side vector.
    */
    u.set_all(1.0)?;
    Mat::mult(&A, &u, &mut b)?;

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
    ksp.solve(&b, &mut x)?;

    /*
        View solver info; we could instead use the option -ksp_view to
        print this info to the screen at the conclusion of KSPSolve().
    */
    let viewer = Viewer::ascii_get_stdout(petsc.world())?;
    ksp.view(&viewer)?;

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check the solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    x.axpy(-1.0, &u)?;
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
