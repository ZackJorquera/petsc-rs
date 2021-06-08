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
//! $ cargo build ex1
//! $ target/debug/ex1
//! Norm of error 2.41202e-15, Iters 5
//! $ target/debug/ex1 -n 100
//! Norm of error 1.14852e-2, Iters 318
//! ```
//!
//! Note:  The corresponding parallel example is ex23.rs


static HELP_MSG: &'static str = "Solves a tridiagonal linear system with KSP.\n\n";

use petsc_rs::prelude::*;
use structopt::StructOpt;

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

#[derive(Debug, PartialEq, StructOpt)]
enum PetscOpt {
    /// use `-- -help` for petsc help
    #[structopt(name = "Petsc Args", external_subcommand)]
    PetscArgs(Vec<String>),
}

impl PetscOpt
{
    fn petsc_args(self_op: Option<Self>) -> Vec<String>
    {
        match self_op
        {
            Some(PetscOpt::PetscArgs(mut vec)) => {
                vec.push(std::env::args().next().unwrap());
                vec.rotate_right(1);
                vec
            },
            _ => vec![std::env::args().next().unwrap()]
        }
    }
}

fn main() -> petsc_rs::Result<()> {
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
        petsc.set_error(PetscErrorKind::PETSC_ERROR_WRONG_MPI_SIZE, "This is a uniprocessor example only!")?;
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
    let value = vec![-1.0, 2.0, -1.0];
    for i in 1..n-1
    {
        let col = vec![i-1, i, i+1];
        A.set_values(1, &vec![i], 3, &col, &value, InsertMode::INSERT_VALUES)?;
    }
    let col = vec![n-2, n-1];
    A.set_values(1, &vec![n-1], 2, &col, &value, InsertMode::INSERT_VALUES)?;
    let col = vec![1, 0];
    A.set_values(1, &vec![0], 2, &col, &value, InsertMode::INSERT_VALUES)?;
    A.assembly_begin(MatAssemblyType::MAT_FINAL_ASSEMBLY)?;
    A.assembly_end(MatAssemblyType::MAT_FINAL_ASSEMBLY)?;

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
    // TODO

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check the solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    x.axpy(-1.0, &u)?;
    let x_norm = x.norm(NormType::NORM_2)?;
    let iters = ksp.get_iteration_number()?;
    petsc_println!(petsc, "Norm of error {:.5e}, Iters {}", x_norm, iters);

    /*
        All PETSc objects are automatically destroyed when they are no longer needed.
        PetscFinalize() is also automatically called.
    */

    // return
    Ok(())
}
