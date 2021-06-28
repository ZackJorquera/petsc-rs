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

fn main() -> petsc_rs::Result<()> {
    // TODO: make these be command line inputs
    let n = 5;
    let test_jacobian_domain_error = false;
    let view_initial = true;
    let user_precond = false;

    let petsc = Petsc::builder()
        .args(std::env::args())
        .help_msg(HELP_MSG)
        .init()?;

    let h = 1.0/(PetscScalar::from(n as PetscReal) - 1.0);

    let mut da = DM::da_create_1d(petsc.world(), DMBoundaryType::DM_BOUNDARY_NONE, n, 1, 1, None)?;
    da.set_from_options()?;
    da.set_up()?;

    let mut x = da.create_global_vector()?;
    let r = x.duplicate()?;
    let mut f = x.duplicate()?;
    let mut u = x.duplicate()?;
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

    let mut snes = petsc.snes_create()?;

    snes.set_function(Some(r), |_snes, x, y| {
        // Note, is the ex3.c file, this is a `DMGetLocalVector` not a `DMCreateLocalVector`.
        // TODO: make it use `DMGetLocalVector` to be consistent with c examples
        let mut x_local = da.create_local_vector()?;

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
    
    #[allow(non_snake_case)]
    let mut J = petsc.mat_create()?;
    J.set_sizes(None, None, Some(n), Some(n))?;
    J.set_from_options()?;
    J.seq_aij_set_preallocation(3, None)?;
    J.mpi_aij_set_preallocation(3, None, 3, None)?;

    snes.set_jacobian_single_mat(J, |_snes, x, jac| {
        if test_jacobian_domain_error {
            todo!();
            // return Ok(());
        }

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

        Ok(())
    })?;

    if user_precond {
        let pc = snes.get_ksp_mut()?.get_pc_mut()?;
        pc.set_type(PCType::PCSHELL)?;
        todo!()
    }

    // TODO: set monitor:
    // SNESMonitorSet(snes,Monitor,&monP,0);
    
    snes.set_from_options()?;

    // TODO: linesearch studd
    // SNESGetLineSearch(snes, &linesearch);
    //
    // PetscOptionsHasName(NULL,NULL,"-post_check_iterates",&post_check);
    // if (post_check) {
    //   PetscPrintf(PETSC_COMM_WORLD,"Activating post step checking routine\n");
    //   SNESLineSearchSetPostCheck(linesearch,PostCheck,&checkP);
    //   VecDuplicate(x,&(checkP.last_step));
    //   
    //   checkP.tolerance = 1.0;
    //   checkP.user      = &ctx;
    //   
    //   PetscOptionsGetReal(NULL,NULL,"-check_tol",&checkP.tolerance,NULL);
    // }
    // 
    // PetscOptionsHasName(NULL,NULL,"-post_setsubksp",&post_setsubksp);
    // if (post_setsubksp) {
    //   PetscPrintf(PETSC_COMM_WORLD,"Activating post setsubksp\n");
    //   SNESLineSearchSetPostCheck(linesearch,PostSetSubKSP,&checkP1);
    // }
    //  
    // PetscOptionsHasName(NULL,NULL,"-pre_check_iterates",&pre_check);
    // if (pre_check) {
    //   PetscPrintf(PETSC_COMM_WORLD,"Activating pre step checking routine\n");
    //   SNESLineSearchSetPreCheck(linesearch,PreCheck,&checkP);
    // }

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
        todo!();
        // let snes_type = snes.get_type()?;
        // petsc_println!(petsc.world(), "SNES type {:?}", snes_type)?;
    }

    // return
    Ok(())
}
