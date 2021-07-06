//! Concepts: SNES^basic uniprocessor example
//! Concepts: SNES^setting a user-defined monitoring routine
//! Processors: 1
//!
//! To run:
//! ```text
//! $ cargo build --bin snes-ex2
//! $ mpiexec -n 1 target/debug/snes-ex2
//! Norm of error 1.49751e-10, Iters 3
//! ```

static HELP_MSG: &str = "Newton method to solve u'' + u^{2} = f, sequentially.\n\
This example employs a user-defined monitoring routine.\n\n";

use petsc_rs::prelude::*;

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

    let n = petsc.options_try_get_int("-n")?.unwrap_or(5);

    if petsc.world().size() != 1 {
        Petsc::set_error(petsc.world(), PetscErrorKind::PETSC_ERROR_WRONG_MPI_SIZE, "This is a uniprocessor example only!")?;
    }

    // Note, `PetscScalar` could be a complex number, so best practice is to instead of giving
    // float literals (i.e. `1.5`) when a function takes a `PetscScalar` wrap in in a `from`
    // call. E.x. `PetscScalar::from(1.5)`. This will do nothing if `PetscScalar` in a real number,
    // but if `PetscScalar` is complex it will construct a complex value which the imaginary part being
    // set to `0`.
    let h = 1.0/(PetscScalar::from(n as PetscReal) - 1.0);

    /*
        Note that we form 1 vector from scratch and then duplicate as needed.
        Set names for some vectors to facilitate monitoring (optional)
    */
    let mut x = petsc.vec_create()?;
    x.set_sizes(None, Some(n))?;
    x.set_from_options()?;
    let mut r = x.duplicate()?;
    let mut g = x.duplicate()?;
    let mut u = x.duplicate()?;
    x.set_name("Approximate Solution")?;
    u.set_name("Exact Solution")?;
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Initialize application:
        Store right-hand-side of PDE and exact solution
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    u.assemble_with((0..n).map(|i| {
        let xp = PetscScalar::from(i as PetscReal) * h;
        (i, xp.powi(3))
    }), InsertMode::INSERT_VALUES)?;
    g.assemble_with((0..n).map(|i| {
        let xp = PetscScalar::from(i as PetscReal) * h;
        (i, 6.0*xp + (xp+1.0e-12).powi(6)) // +1.e-12 is to prevent 0^6
    }), InsertMode::INSERT_VALUES)?;

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Set initial guess
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*
        Note: The user should initialize the vector, x, with the initial guess
        for the nonlinear solver prior to calling SNESSolve().  In particular,
        to employ an initial guess of zero, the user should explicitly set
        this vector to zero by calling VecSet().
    */
    x.set_all(PetscScalar::from(0.5))?;

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    let mut snes = petsc.snes_create()?;

    /*
        Set function evaluation routine and vector
    */
    snes.set_function(Some(&mut r), |_snes, x: &Vector, f: &mut Vector| {
        /*
            Evaluates nonlinear function, F(x).

            Input Parameters:
            .  snes - the SNES context
            .  x - input vector

            Output Parameter:
            .  f - function vector
        */

        let x_view = x.view()?;
        let mut f_view = f.view_mut()?;
        let g_view = g.view()?;

        let d = PetscScalar::from(n as PetscReal - 1.0).powi(2);

        f_view[0] = x_view[0];
        for i in 1..(n as usize - 1) {
            f_view[i] = d*(x_view[i-1] - 2.0*x_view[i] + x_view[i+1]) + x_view[i]*x_view[i] - g_view[i];
        }
        f_view[n as usize - 1] = x_view[n as usize - 1] - 1.0;

        Ok(())
    })?;

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Create matrix data structure; set Jacobian evaluation routine
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    #[allow(non_snake_case)]
    let mut J = petsc.mat_create()?;
    J.set_sizes(None, None, Some(n), Some(n))?;
    J.set_from_options()?;
    J.seq_aij_set_preallocation(3, None)?;

    /*
        Set Jacobian matrix data structure and default Jacobian evaluation
        routine. User can override with:
        -snes_fd : default finite differencing approximation of Jacobian
        -snes_mf : matrix-free Newton-Krylov method with no preconditioning
                    (unless user explicitly sets preconditioner)
        -snes_mf_operator : form preconditioning matrix as set by the user,
                            but use matrix-free approx for Jacobian-vector
                            products within Newton-Krylov method
    */
    snes.set_jacobian_single_mat(J,|_snes, x: &Vector, ap_mat: &mut Mat| {
        /*
            Evaluates Jacobian matrix.

            Input Parameters:
            .  snes - the SNES context
            .  x - input vector

            Output Parameters:
            .  ap_mat - Jacobian matrix (also used as preconditioning matrix)
        */

        let x_view = x.view()?;

        let d = (PetscScalar::from(n as PetscReal) - 1.0).powi(2);

        ap_mat.assemble_with((0..n).map(|i| if i == 0 || i == n-1{ vec![(i,i,PetscScalar::from(1.0))] }
                                            else { vec![(i,i-1,d), (i,i,-2.0*d+2.0*x_view[i as usize]), (i,i+1,d)] })
                .flatten(), 
            InsertMode::INSERT_VALUES, MatAssemblyType::MAT_FINAL_ASSEMBLY)?;

        Ok(())
    })?;
    
    // Set an optional user-defined monitoring routine
    snes.monitor_set(|snes, its, fnorm| {
        petsc_println!(petsc.world(), "iter: {}, SNES function norm: {:.5e}", its, fnorm)?;
        let x = snes.get_solution()?;
        x.view_with(None)?;
        Ok(())
    })?;

    /*
        Set SNES/KSP/KSP/PC runtime options, e.g.,
            -snes_view -snes_monitor -ksp_type <ksp> -pc_type <pc>
    */
    snes.set_from_options()?;

    /*
        Print parameters used for convergence testing (optional) ... just
        to demonstrate this routine; this information is also printed with
        the option -snes_view
    */
    let tols = snes.get_tolerances()?;
    petsc_println!(petsc.world(), "atol={:.5e}, rtol={:.5e}, stol={:.5e}, maxit={}, maxf={}",
        tols.0, tols.1, tols.2, tols.3, tols.4)?;


    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Solve Nonlinear System
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    snes.solve(None, &mut x)?;

    /*
        Check the error
    */
    x.axpy(PetscScalar::from(-1.0), &u)?;
    let x_norm = x.norm(NormType::NORM_2)?;
    let iters = snes.get_iteration_number()?;
    petsc_println!(petsc.world(), "Norm of error {:.5e}, Iters {}", x_norm, iters)?;
    
    // return
    Ok(())
}
