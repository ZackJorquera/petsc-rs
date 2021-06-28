//! Concepts: MPI^split world;
//! Concepts: KSP^solving a system of linear equations
//! Concepts: KSP^Laplacian, 2d
//! Processors: 2+
//!
//! To run:
//! ```text
//! $ cargo build --bin ex2
//! $ mpiexec -n 2 target/debug/ex
//! (ex2) Norm of error 1.56044e-4, Iters 6
//! (ex23) Norm of error 2.41202e-15, Iters 5
//! $ mpiexec -n 3 target/debug/ex2
//! (ex2) Norm of error 4.11674e-4, Iters 7
//! (ex23) Norm of error 2.41202e-15, Iters 5
//! $ mpiexec -n 5 target/debug/ex -m 80 -n 70 -k 100
//! (ex2) Norm of error 1.48926e-3, Iters 70
//! (ex23) Norm of error 1.14852e-2, Iters 318
//! ```

static HELP_MSG: &str = "Solves a linear system in parallel with KSP.
Input parameters include:\n\n";

use mpi;
use mpi::topology::Color;
use petsc_rs::prelude::*;
use structopt::StructOpt;

#[derive(Debug, PartialEq, StructOpt)]
pub enum PetscOpt {
    /// use `-- -help` for petsc help
    #[structopt(name = "Petsc Args", external_subcommand)]
    PetscArgs(Vec<String>),
}

impl PetscOpt
{
    pub fn petsc_args(self_op: Option<Self>) -> Vec<String>
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

#[derive(Debug, StructOpt)]
#[structopt(name = "ex2", about = HELP_MSG)]
struct Opt {
    /// number of mesh points in x-direction (for ex2)
    #[structopt(short, default_value = "8")]
    m: PetscInt,
    
    /// number of mesh points in y-direction (for ex2)
    #[structopt(short, default_value = "7")]
    n: PetscInt,

    /// write exact solution vector to stdout (for ex2)
    #[structopt(short, long)]
    view_stuff: bool,

    /// Size of the vector and matrix (for ex23)
    #[structopt(short = "k", long, default_value = "10")]
    num_elems: PetscInt,

    /// use `-- -help` for petsc help
    #[structopt(subcommand)]
    sub: Option<PetscOpt>,
}

fn main() -> petsc_rs::Result<()> {
    let Opt {m, n, view_stuff, num_elems, sub: ext_args} = Opt::from_args();
    let petsc_args = PetscOpt::petsc_args(ext_args); // Is there an easier way to do this

    // optionally initialize mpi
    let univ = mpi::initialize().unwrap();
    let world = univ.world();

    if world.size() == 1
    {
        // We cant use Petsc::set_error because we haven't initialized PETSc yet
        panic!("This is strictly not a uniprocessor example!");
    }

    // split world into two (off processes with even rank and processes odd rank).
    let my_color = world.rank() % 2;
    let comm = world.split_by_color(Color::with_value(my_color)).unwrap();

    // init with no options
    let petsc = Petsc::builder()
        .args(petsc_args)
        .world(Box::new(comm))
        .help_msg(HELP_MSG)
        .init()?;

    // or init with no options
    // let petsc = Petsc::init_no_args()?;

    // This will print twice, by the odd comm and the even comm
    petsc_println!(petsc.world(), "Hello parallel world (color: {}) of {} processes!", my_color, petsc.world().size() )?;

    // Have the first process group do one thing and the second process group do another
    if my_color == 0 {
        do_ksp_ex2(&petsc, m, n, view_stuff)
    } else {
        do_ksp_ex23(&petsc, num_elems, view_stuff)
    }
}

// TODO: make these be constructed better, right now we have two copies of this code
fn do_ksp_ex2(petsc: &Petsc, m: PetscInt, n: PetscInt, view_exact_sol: bool) -> petsc_rs::Result<()> {
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
    petsc_println!(petsc.world(), "(ex2) Norm of error {:.5e}, Iters {}", x_norm, iters)?;

    /*
        All PETSc objects are automatically destroyed when they are no longer needed.
        PetscFinalize() is also automatically called.
    */

    // return
    Ok(())
}

fn do_ksp_ex23(petsc: &Petsc, n: PetscInt, view_ksp: bool) -> petsc_rs::Result<()> {
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
    A.assemble_with(vec_ownership_range.map(|i| (-1..=1).map(move |j| (i,i+j))).flatten()
            .filter(|&(i,j)| i < n && j < n) // we could also filter out negatives, but assemble_with does that for us
            .map(|(i,j)| if i == j { (i, j, PetscScalar::from(2.0)) }
                         else { (i, j, PetscScalar::from(-1.0)) }),
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
    ksp.solve(Some(&b), &mut x)?;

    /*
        View solver info; we could instead use the option -ksp_view to
        print this info to the screen at the conclusion of KSPSolve().
    */
    if view_ksp {
        let viewer = Viewer::create_ascii_stdout(petsc.world())?;
        ksp.view_with(Some(&viewer))?;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check the solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    x.axpy(PetscScalar::from(-1.0), &u)?;
    let x_norm = x.norm(NormType::NORM_2)?;
    let iters = ksp.get_iteration_number()?;
    petsc_println!(petsc.world(), "(ex23) Norm of error {:.5e}, Iters {}", x_norm, iters)?;

    /*
        All PETSc objects are automatically destroyed when they are no longer needed.
        PetscFinalize() is also automatically called.
    */

    // return
    Ok(())
}
