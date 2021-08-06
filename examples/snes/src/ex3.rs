//! Concepts: SNES^basic parallel example
//! Concepts: SNES^setting a user-defined monitoring routine
//! Processors: n
//!
//! To run:
//! ```text
//! $ cargo build --bin snes-ex3
//! $ mpiexec -n 1 target/debug/snes-ex3
//! $ mpiexec -n 2 target/debug/snes-ex3
//! ```

static HELP_MSG: &str = "Newton methods to solve u'' + u^{2} = f in parallel.\n\n";

use petsc_rs::prelude::*;

struct Opt {
    n: PetscInt,
    test_jacobian_domain_error: bool,
    view_initial: bool,
    user_precond: bool,
    post_check_iterates: bool,
    pre_check_iterates: bool,
    check_tol: PetscReal,
}

impl PetscOpt for Opt {
    fn from_petsc_opt_builder(pob: &mut PetscOptBuilder) -> petsc_rs::Result<Self> {
        let n = pob.options_int("-n", "", "snes-ex3", 5)?;
        let test_jacobian_domain_error = pob.options_bool("-test_jacobian_domain_error", "", "snes-ex3", false)?;
        let view_initial = pob.options_bool("-view_initial", "", "snes-ex3", false)?;
        let user_precond = pob.options_bool("-user_precond", "", "snes-ex3", false)?;
        let post_check_iterates = pob.options_bool("-post_check_iterates", "", "snes-ex3", false)?;
        let pre_check_iterates = pob.options_bool("-pre_check_iterates", "", "snes-ex3", false)?;
        let check_tol = pob.options_real("-check_tol", "", "snes-ex3", 1.0)?;
        Ok(Opt { n, test_jacobian_domain_error, view_initial, user_precond, post_check_iterates,
            pre_check_iterates, check_tol })
    }
}

fn main() -> petsc_rs::Result<()> {
    let petsc = Petsc::builder()
        .args(std::env::args())
        .help_msg(HELP_MSG)
        .init()?;
    
    let Opt { n, test_jacobian_domain_error, view_initial, user_precond, post_check_iterates,
            pre_check_iterates, check_tol } = petsc.options_get()?;

    let h = 1.0/(PetscScalar::from(n as PetscReal) - 1.0);

    let mut da = DM::da_create_1d(petsc.world(), DMBoundaryType::DM_BOUNDARY_NONE, n, 1, 1, None)?;
    da.set_from_options()?;
    da.set_up()?;

    let mut x = da.create_global_vector()?;
    let mut f = x.duplicate()?;
    let mut u = x.duplicate()?;
    let mut last_step = x.duplicate()?;
    let mut r = x.duplicate()?;
    x.set_name("Solution")?;
    u.set_name("Exact Solution")?;
    f.set_name("Forcing Function")?;

    let (xs, _, _, xm, _, _) = da.da_get_corners()?;
    {
        let mut ff = da.da_vec_view_mut(&mut f)?;
        let mut uu = da.da_vec_view_mut(&mut u)?;

        let _ = (0..xm).fold(h*xs as PetscReal, |xp,i| {
            ff[i as usize] = 6.0*xp + (xp+1.0e-12).powi(6);
            uu[i as usize] = xp*xp*xp;
            xp+h
        });
    }

    if view_initial {
        u.view_with(None)?;
        f.view_with(None)?;
    }

    #[allow(non_snake_case)]
    let mut J = petsc.mat_create()?;
    J.set_sizes(None, None, Some(n), Some(n))?;
    J.set_from_options()?;
    J.seq_aij_set_preallocation(3, None)?;
    J.mpi_aij_set_preallocation(3, None, 3, None)?;

    let mut snes = petsc.snes_create()?;

    snes.set_function(&mut r, |_snes, x, y| {
        let mut x_local = da.get_local_vector()?;

        da.global_to_local(x, InsertMode::INSERT_VALUES, &mut x_local)?;

        let (_, m, _, _, _, _, _, _, _, _, _, _, _) = da.da_get_info()?;
        let (xs, _, _, _xm, _, _) = da.da_get_corners()?;
        let (gxs, _, _, _gxm, _, _) = da.da_get_ghost_corners()?;
        let ghost_point_offset = xs-gxs;

        let d = 1.0/(h*h);

        let x_view = da.da_vec_view(&x_local)?;
        let mut y_view = da.da_vec_view_mut(y)?;
        let f_view = da.da_vec_view(&f)?;

        y_view.indexed_iter_mut().map(|(pat, v)| (pat[0], v) )
            .for_each(|(i, v)| {
                let ii = i+ghost_point_offset as usize;
                if i + xs as usize == 0 {
                    *v = x_view[ii];
                } else if i + xs as usize == (m-1) as usize {
                    *v = x_view[ii] - 1.0;
                } else {
                    *v = d * (x_view[ii-1] - 2.0 * x_view[ii] + x_view[ii+1]) + x_view[ii] * x_view[ii] - f_view[i];
                }
            });

        Ok(())
    })?;

    snes.set_jacobian_single_mat(&mut J, |_snes, x, jac| {
        let xx = da.da_vec_view(&x)?;

        let (_, m, _, _, _, _, _, _, _, _, _, _, _) = da.da_get_info()?;
        let (xs, _, _, xm, _, _) = da.da_get_corners()?;

        let d = 1.0/(h*h);

        jac.assemble_with((xs..xs+xm).map(|i| {
                if i == 0 { 
                    vec![ (i, i, PetscScalar::from(1.0)) ]
                } else if i == m-1 {
                    vec![ (i, i, PetscScalar::from(1.0)) ]
                } else {
                    vec![(i,i-1,d),
                         (i,i,-2.0*d+2.0*xx[(i-xs) as usize]),
                         (i,i+1,d)]
                }
            }).flatten(),
        InsertMode::INSERT_VALUES, MatAssemblyType::MAT_FINAL_ASSEMBLY)?;

        
        if test_jacobian_domain_error {
            Err(DomainErr)
        } else {
            Ok(())
        }
    })?;

    if user_precond {
        let pc = snes.get_ksp_or_create()?.get_pc_or_create()?;
        pc.set_type(PCType::PCSHELL)?;
        // Identity preconditioner:
        pc.shell_set_apply(|_pc, xin, xout| xout.copy_data_from(xin) )?;
    }

    snes.monitor_set(|snes, its, fnorm| {
        petsc_println!(petsc.world(), "iter: {}, SNES function norm: {:.5e}", its, fnorm)?;
        let x = snes.get_solution()?;
        x.view_with(None)?;
        Ok(())
    })?;
    
    snes.set_from_options()?;

    
    if post_check_iterates {
        petsc_println!(petsc.world(), "Activating post step checking routine")?;
        snes.linesearch_set_post_check(|_ls, snes, _x_cur, _y, x, _y_mod, x_mod| { 
            let it = snes.get_iteration_number()?;
            if it > 0 {
                petsc_println!(petsc.world(), "Checking candidate step at iteration {} with tolerance {}", it, check_tol)?;
                let xa_last = da.da_vec_view(&last_step)?;
                let mut xa = da.da_vec_view_mut(x)?;
                let (_xs, _, _, _xm, _, _) = da.da_get_corners()?;

                xa_last.indexed_iter().map(|(pat, _)|  pat[0])
                    .for_each(|i| { 
                        let rdiff = if xa[i].abs() == 0.0 { 2.0*check_tol }
                            else { ((xa[i] - xa_last[i])/xa[i]).abs() };
                        if rdiff > check_tol {
                            xa[i] = 0.5*(xa[i] + xa_last[i]);
                            *x_mod = true;
                        }
                    });
            }
            last_step.copy_data_from(&x)
        })?;
    }
    if pre_check_iterates {
        petsc_println!(petsc.world(), "Activating pre step checking routine")?;
        snes.linesearch_set_pre_check(|_ls, _snes, _x, _y, _y_mod| {
            Ok(())
        })?;
    }

    let tols = snes.get_tolerances()?;
    petsc_println!(petsc.world(), "atol={:.5e}, rtol={:.5e}, stol={:.5e}, maxit={}, maxf={}",
        tols.0, tols.1, tols.2, tols.3, tols.4)?;

    x.set_all(PetscScalar::from(0.5))?;

    snes.solve(None, &mut x)?;
    let it_num = snes.get_iteration_number()?;
    petsc_println!(petsc.world(), "Number of SNES iterations = {}", it_num)?;

    x.axpy(PetscScalar::from(-1.0), &u)?;
    let norm = x.norm(NormType::NORM_2)?;
    
    petsc_println!(petsc.world(), "Norm of error {:.5e} Iterations {}", norm, it_num)?;
    if test_jacobian_domain_error {
        let snes_type = snes.get_type_str()?;
        petsc_println!(petsc.world(), "SNES type: {}", snes_type)?;
    }

    // return
    Ok(())
}
