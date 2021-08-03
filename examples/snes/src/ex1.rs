//! Concepts: SNES^basic example
//! Processors: 1
//!
//! To run:
//! ```text
//! $ cargo build --bin snes-ex1
//! $ mpiexec -n 1 target/debug/snes-ex1
//! ```

static HELP_MSG: &str = "Newton's method for a two-variable system, sequential.\n\n";

use petsc_rs::prelude::*;
use mpi::traits::*;

fn main() -> petsc_rs::Result<()> {
    let n = 2;

    // optionally initialize mpi
    // let _univ = mpi::initialize().unwrap();
    // init with no options
    let petsc = Petsc::builder()
        .args(std::env::args())
        .help_msg(HELP_MSG)
        .init()?;

    // or init with no options
    // let petsc = Petsc::init_no_args()?;

    let hard_flg = petsc.options_try_get_bool("-hard_flg")?.unwrap_or(false);

    if petsc.world().size() != 1
    {
        Petsc::set_error(petsc.world(), PetscErrorKind::PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!")?;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix and vector data structures; set corresponding routines
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*
        Create vectors for solution and nonlinear function
    */
    let mut x = petsc.vec_create()?;
    x.set_sizes(None, Some(n))?;
    x.set_from_options()?;
    let mut r = x.duplicate()?;
    x.set_name("soln")?;

    /*
        Create Jacobian matrix data structure
    */
    #[allow(non_snake_case)]
    let mut J = petsc.mat_create()?;
    J.set_sizes(None, None, Some(n), Some(n))?;
    J.set_from_options()?;
    J.set_up()?;

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    let mut snes = petsc.snes_create()?;

    if !hard_flg {
        /*
            Set function evaluation routine and vector with closure.
        */
        snes.set_function(&mut r, |_snes: &SNES, x: &Vector, f: &mut Vector| {
            let x_view = x.view()?;
            let mut f_view = f.view_mut()?;

            f_view[0] = x_view[0]*x_view[0] + x_view[0]*x_view[1] - 3.0;
            f_view[1] = x_view[0]*x_view[1] + x_view[1]*x_view[1] - 6.0;

            Ok(())
        })?;
        /*
            Set Jacobian matrix data structure and Jacobian evaluation routine with closure
        */
        snes.set_jacobian_single_mat(&mut J, |_snes: &SNES, x: &Vector, jac: &mut Mat| {
            let x_view = x.view()?;

            jac.assemble_with([(0,0,2.0*x_view[0]), (0,1,x_view[0]), (1,0,x_view[1]), (1,1,x_view[0]+2.0*x_view[1])],
                InsertMode::INSERT_VALUES, MatAssemblyType::MAT_FINAL_ASSEMBLY)?;

            Ok(())
        })?;
    } else {
        // We can also use functions, not closures for input
        snes.set_function(&mut r, from_function2)?;
        snes.set_jacobian_single_mat(&mut J, from_jacobian2)?;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Customize nonlinear solver; set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*
        Set linear solver defaults for this problem. By extracting the
        KSP and PC contexts from the SNES context, we can then
        directly call any KSP and PC routines to set various options.
    */
    let ksp = snes.get_ksp_mut()?;
    let pc = ksp.get_pc_mut()?;
    pc.set_type(PCType::PCNONE)?;
    ksp.set_tolerances(Some(1.0e-4), None, None, Some(20))?;

    /*
        Set SNES/KSP/KSP/PC runtime options, e.g.,
            -snes_view -snes_monitor -ksp_type <ksp> -pc_type <pc>
        These options will override those specified above as long as
        SNESSetFromOptions() is called _after_ any other customization
        routines.
    */
    snes.set_from_options()?;

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Set initial guess
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*
        Note: The user should initialize the vector, x, with the initial guess
        for the nonlinear solver prior to calling SNESSolve().  In particular,
        to employ an initial guess of zero, the user should explicitly set
        this vector to zero by calling VecSet().
    */
    // Note, `PetscScalar` could be a complex number, so best practice is to instead of giving
    // float literals (i.e. `1.5`) when a function takes a `PetscScalar` wrap in in a `from`
    // call. E.x. `PetscScalar::from(1.5)`. This will do nothing if `PetscScalar` in a real number,
    // but if `PetscScalar` is complex it will construct a complex value which the imaginary part being
    // set to `0`.
    if !hard_flg {
        x.set_all(PetscScalar::from(0.5))?;
    } else {
        let mut x_view = x.view_mut()?;
        x_view[0] = PetscScalar::from(2.0);
        x_view[1] = PetscScalar::from(3.0);
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Solve Nonlinear System
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    snes.solve(None, &mut x)?;
    if hard_flg {
        //todo!();
        let viewer = Viewer::create_ascii_stdout(petsc.world())?;
        x.view_with(Some(&viewer))?;
        //r.view_with(Some(&viewer))?;
        // TODO: view x, r, and f (f is from SNESGetFunction)
    }
    
    // return
    Ok(())
}

fn from_function2(_snes: &SNES, x: &Vector, f: &mut Vector) -> Result<(), petsc_rs::snes::DomainOrPetscError> {
    let x_view = x.view()?;
    let mut f_view = f.view_mut()?;

    f_view[0] = PetscScalar::sin(3.0 * x_view[0]) + x_view[0];
    f_view[1] = x_view[1];

    Ok(())
}

fn from_jacobian2(_snes: &SNES, x: &Vector, jac: &mut Mat) -> Result<(), petsc_rs::snes::DomainOrPetscError> {
    let x_view = x.view()?;

    jac.assemble_with([(0,0,3.0*PetscScalar::cos(3.0*x_view[0])+1.0), (1,1,PetscScalar::from(1.0))],
        InsertMode::INSERT_VALUES, MatAssemblyType::MAT_FINAL_ASSEMBLY)?;

    Ok(())
}
