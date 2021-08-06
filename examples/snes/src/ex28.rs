//! Solve a PDE coupled to an algebraic system in 1D
//! 
//! PDE (U):
//!     -(k u_x)_x = 1 on (0,1), subject to u(0) = 0, u(1) = 1
//! Algebraic (K):
//!     exp(k-1) + k = 1/(1/(1+u) + 1/(1+u_x^2))
//! 
//! The discretization places k at staggered points, and a separate DMDA is used for each "physics".
//! 
//! This example is a prototype for coupling in multi-physics problems, therefore residual evaluation and assembly for
//! each problem (referred to as U and K) are written separately.  This permits the same "physics" code to be used for
//! solving each uncoupled problem as well as the coupled system.  In particular, run with -problem_type 0 to solve only
//! problem U (with K fixed), -problem_type 1 to solve only K (with U fixed), and -problem_type 2 to solve both at once.
//! 
//! In all cases, a fully-assembled analytic Jacobian is available, so the systems can be solved with a direct solve or
//! any other standard method.  Additionally, by running with
//! 
//!   -pack_dm_mat_type nest
//! 
//! The same code assembles a coupled matrix where each block is stored separately, which allows the use of PCFieldSplit
//! without copying values to extract submatrices.
//! 
//!
//! To run:
//! ```text
//! $ cargo build --bin snes-ex28
//! $ mpiexec -n 1 target/debug/snes-ex28
//! $ mpiexec -n 2 target/debug/snes-ex28
//! ```

static HELP_MSG: &str = "1D multiphysics prototype with analytic Jacobians to solve individual problems and a coupled problem.\n\n";

use std::{ops::DerefMut, rc::Rc};

use petsc_rs::prelude::*;

struct Opt {
    problem_type: PetscInt,
    pass_dm: bool,
}

impl PetscOpt for Opt {
    fn from_petsc_opt_builder(pob: &mut PetscOptBuilder) -> petsc_rs::Result<Self> {
        let problem_type = pob.options_int("-problem_type", "", "snes-ex28", 0)?;
        let pass_dm = pob.options_bool("-pass_dm", "", "snes-ex28", false)?;
        Ok(Opt { problem_type, pass_dm })
    }
}

fn main() -> petsc_rs::Result<()> {
    let petsc = Petsc::builder()
        .args(std::env::args())
        .help_msg(HELP_MSG)
        .init()?;
    
    let Opt { problem_type, pass_dm } = petsc.options_get()?;

    let mut dau = DM::da_create_1d(petsc.world(), DMBoundaryType::DM_BOUNDARY_NONE, 10, 1, 1, None)?;
    // DMSetOptionsPrefix(dau,"u_");
    dau.set_from_options()?;
    dau.set_up()?;
    dau.da_set_feild_name(0, "u")?;
    let (lxu, _, _) = dau.da_get_ownership_ranges()?;
    let (_, m, _, _, _sizes, _, _, _, _, _, _, _, _) = dau.da_get_info()?;
    let mut lxk = lxu;
    lxk[0] -= 1;

    let mut dak = DM::da_create_1d(petsc.world(), DMBoundaryType::DM_BOUNDARY_NONE, m-1, 1, 1, lxk.as_ref())?;
    // DMSetOptionsPrefix(dak,"k_");
    dak.set_from_options()?;
    dak.set_up()?;
    dak.da_set_feild_name(0, "k")?;

    let mut pack = DM::composite_create(petsc.world(), [dau, dak])?;
    pack.set_from_options()?;

    let mut x = pack.create_global_vector()?;
    let mut f = x.clone();

    let mut isg = pack.composite_get_global_indexsets()?;
    let mut local_vecs = pack.composite_create_local_vectors()?;
    pack.composite_scatter(&x, &mut local_vecs)?;

    {
        let dms = pack.composite_dms().unwrap();
        let mut x_parts = pack.composite_get_access_mut(&mut x)?;
        let (mut k_vec, mut u_vec) = (x_parts.pop().unwrap(), x_parts.pop().unwrap());
        let mut u_view = dms[0].da_vec_view_mut(&mut u_vec)?;
        let mut k_view = dms[1].da_vec_view_mut(&mut k_vec)?;
        let (xsu, _, _, _xmu, _, _) = dms[0].da_get_corners()?;
        let (xsk, _, _, _xmk, _, _) = dms[1].da_get_corners()?;
        let hx = PetscScalar::from(1.0)/(m-1) as PetscReal;
        u_view.indexed_iter_mut().map(|(s, u)| ((s[0]+xsu as usize) as PetscReal, u))
            .for_each(|(i, u)| {
                *u = i * hx * (1.0 - i * hx);
            });
        k_view.indexed_iter_mut().map(|(s, k)| ((s[0]+xsk as usize) as PetscReal, k))
            .for_each(|(i, k)| {
                *k = 1.0 + 0.5*PetscScalar::sin(2.0*std::f64::consts::PI as PetscReal * i * hx);
            });
    }
    
    pack.composite_scatter(&x, &mut local_vecs)?;
    let (k_loc, u_loc) = (local_vecs.pop().unwrap(), local_vecs.pop().unwrap());

    let dms = pack.composite_dms()?;

    match problem_type {
        0 => {
            let mut x_parts = pack.composite_get_access_mut(&mut x)?;
            let mut f_parts = pack.composite_get_access_mut(&mut f)?;

            let mut b_mat = dms[0].create_matrix()?;

            let mut snes = SNES::create(petsc.world())?;
            snes.set_dm(dms[0].clone())?; // TODO: does this work
            snes.set_function(f_parts[0].deref_mut(), |snes, x, y| {
                // TODO: idk if we need to do this, we might be able to just use the dms
                // from outside of the closure
                let dms = [snes.try_get_dm().unwrap(), &dms[1]];
                let (_, mx, _, _, _, _, _, _, _, _, _, _, _) = dms[0].da_get_info()?;
                let (xs, _, _, _xm, _, _) = dms[0].da_get_corners()?;
                let (xs_k, _, _, _xm_k, _, _) = dms[1].da_get_corners()?;
                let (gxs, _, _, _gxm, _, _) = dms[0].da_get_ghost_corners()?;
                
                // TODO: this `pack` is not using the dm from `snes.try_get_dm().unwrap()`
                // It is using the one that we called clone on (line 114). Is that a problem.
                // Do we even need to use `snes.try_get_dm().unwrap()` at all if this works?
                let mut local_vecs = pack.composite_get_local_vectors()?;
                // we have to do the map because local_vecs are not vectors
                pack.composite_scatter(x, local_vecs.iter_mut().map(|bv| bv.deref_mut()))?;
                let mut u_loc = local_vecs.remove(0);

                dms[0].global_to_local(x, InsertMode::INSERT_VALUES, &mut u_loc)?;

                form_func_u(&petsc, (m, mx, xs, xs_k, gxs), (dms[0], dms[1]), &u_loc, &k_loc, y)?;

                Ok(())
            })?;

            snes.set_jacobian_single_mat(&mut b_mat, |snes, x, jac| {
                // TODO: idk if we need to do this, we might be able to just use the dms
                // from outside of the closure
                let dms = [snes.try_get_dm().unwrap(), &dms[1]];
                let (_, mx, _, _, _, _, _, _, _, _, _, _, _) = dms[0].da_get_info()?;
                let (xs, _, _, xm, _, _) = dms[0].da_get_corners()?;
                let (xs_k, _, _, _xm_k, _, _) = dms[1].da_get_corners()?;
                let (gxs, _, _, _gxm, _, _) = dms[0].da_get_ghost_corners()?;

                let mut local_vecs = pack.composite_get_local_vectors()?;
                pack.composite_scatter(x, local_vecs.iter_mut().map(|bv| bv.deref_mut()))?;
                let mut u_loc = local_vecs.remove(0);

                dms[0].global_to_local(x, InsertMode::INSERT_VALUES, &mut u_loc)?;

                form_jac_u(&petsc, (mx, xm, xs, xs_k, gxs), (dms[0], dms[1]), &u_loc, &k_loc, jac)?;
                jac.assemble(MatAssemblyType::MAT_FINAL_ASSEMBLY)?;

                Ok(())
            })?;
            snes.set_from_options()?;

            snes.solve(None, &mut x_parts[0])?;
        },
        1 => {
            let mut x_parts = pack.composite_get_access_mut(&mut x)?;
            let mut f_parts = pack.composite_get_access_mut(&mut f)?;

            let mut b_mat = dms[1].create_matrix()?;
    
            let mut snes = SNES::create(petsc.world())?;
            
            snes.set_dm(dms[1].clone())?; // TODO: does this work
            
            snes.set_function(f_parts[1].deref_mut(), |snes, x, y| {
                // TODO: idk if we need to do this, we might be able to just use the dms
                // from outside of the closure
                let dms = [&dms[0], snes.try_get_dm().unwrap()];
                let (_, mx, _, _, _, _, _, _, _, _, _, _, _) = dms[1].da_get_info()?;
                let (xs, _, _, _xm, _, _) = dms[1].da_get_corners()?;
                let (xs_u, _, _, _xm_u, _, _) = dms[0].da_get_corners()?;
                let (gxs, _, _, _gxm, _, _) = dms[1].da_get_ghost_corners()?;
                
                let mut local_vecs = pack.composite_get_local_vectors()?;
                pack.composite_scatter(x, local_vecs.iter_mut().map(|bv| bv.deref_mut()))?;
                let mut k_loc = local_vecs.remove(1);

                dms[1].global_to_local(x, InsertMode::INSERT_VALUES, &mut k_loc)?;

                form_func_k(&petsc, (m, mx, xs, xs_u, gxs), (dms[0], dms[1]), &u_loc, &k_loc, y)?;

                Ok(())
            })?;
            snes.set_jacobian_single_mat(&mut b_mat, |snes, x, jac| {
                // TODO: idk if we need to do this, we might be able to just use the dms
                // from outside of the closure
                let dms = [&dms[0], snes.try_get_dm().unwrap()];
                let (_, mx, _, _, _, _, _, _, _, _, _, _, _) = dms[1].da_get_info()?;
                let (xs, _, _, xm, _, _) = dms[1].da_get_corners()?;
                let (xs_u, _, _, _xm_u, _, _) = dms[0].da_get_corners()?;
                let (gxs, _, _, _gxm, _, _) = dms[1].da_get_ghost_corners()?;
                
                let mut local_vecs = pack.composite_get_local_vectors()?;
                pack.composite_scatter(x, local_vecs.iter_mut().map(|bv| bv.deref_mut()))?;
                let mut k_loc = local_vecs.remove(1);

                dms[1].global_to_local(x, InsertMode::INSERT_VALUES, &mut k_loc)?;

                form_jac_k(&petsc, (mx, xm, xs, xs_u, gxs), (dms[0], dms[1]), &u_loc, &k_loc, jac)?;
                jac.assemble(MatAssemblyType::MAT_FINAL_ASSEMBLY)?;

                Ok(())
            })?;
            snes.set_from_options()?;

            snes.solve(None, &mut x_parts[1])?;
        },
        2 => {
            let mut b_mat = pack.create_matrix()?;

            let mut snes = SNES::create(petsc.world())?;

            if !pass_dm {
                let pc = snes.get_ksp_or_create()?.get_pc_or_create()?;
                pc.field_split_set_is(Some("u"), isg.remove(0))?;
                pc.field_split_set_is(Some("k"), isg.remove(0))?;
            } else {
                snes.set_dm(pack.clone())?; // TODO: will this work? what does cloning do in this case
            }

            // This example does not correctly allocate off-diagonal blocks.
            // These options allows new nonzeros (slow).
            b_mat.set_option(MatOption::MAT_NEW_NONZERO_LOCATION_ERR, false)?;
            b_mat.set_option(MatOption::MAT_NEW_NONZERO_ALLOCATION_ERR, false)?;
            snes.set_function(Some(&mut f), |snes, x, y| {
                // TODO: idk if we need to do this, we might be able to just use the dms
                // from outside of the closure
                let pack = snes.try_get_dm().unwrap_or(&pack);
                let dms = snes.try_get_dm().map(|pack| pack.composite_dms().ok()).flatten().unwrap_or(dms);
                let (_, mx_u, _, _, _, _, _, _, _, _, _, _, _) = dms[0].da_get_info()?;
                let (_, mx_k, _, _, _, _, _, _, _, _, _, _, _) = dms[1].da_get_info()?;
                let (xs_u, _, _, _xm_u, _, _) = dms[0].da_get_corners()?;
                let (xs_k, _, _, _xm_k, _, _) = dms[1].da_get_corners()?;
                let (gxs_u, _, _, _gxm_u, _, _) = dms[0].da_get_ghost_corners()?;
                let (gxs_k, _, _, _gxm_k, _, _) = dms[1].da_get_ghost_corners()?;

                let mut local_vecs = pack.composite_get_local_vectors()?;
                pack.composite_scatter(x, local_vecs.iter_mut().map(|bv| bv.deref_mut()))?;
                let (k_loc, u_loc) = (local_vecs.pop().unwrap(), local_vecs.pop().unwrap());
                let mut y_parts = pack.composite_get_access_mut(y)?;
                let (mut y_k, mut y_u) = (y_parts.pop().unwrap(), y_parts.pop().unwrap());

                form_func_u(&petsc, (m, mx_u, xs_u, xs_k, gxs_u), (&dms[0], &dms[1]), &u_loc, &k_loc, &mut y_u)?;
                form_func_k(&petsc, (m, mx_k, xs_k, xs_u, gxs_k), (&dms[0], &dms[1]), &u_loc, &k_loc, &mut y_k)?;

                Ok(())
            })?;
            snes.set_jacobian_single_mat(&mut b_mat, |snes, x, jac| {
                // TODO: idk if we need to do this, we might be able to just use the dms/pack
                // from outside of the closure
                let pack = snes.try_get_dm().unwrap_or(&pack);
                let dms = snes.try_get_dm().map(|pack| pack.composite_dms().ok()).flatten().unwrap_or(dms);
                let (_, mx_u, _, _, _, _, _, _, _, _, _, _, _) = dms[0].da_get_info()?;
                let (_, mx_k, _, _, _, _, _, _, _, _, _, _, _) = dms[1].da_get_info()?;
                let (xs_u, _, _, xm_u, _, _) = dms[0].da_get_corners()?;
                let (xs_k, _, _, xm_k, _, _) = dms[1].da_get_corners()?;
                let (gxs_u, _, _, _gxm_u, _, _) = dms[0].da_get_ghost_corners()?;
                let (gxs_k, _, _, _gxm_k, _, _) = dms[1].da_get_ghost_corners()?;

                let mut local_vecs = pack.composite_create_local_vectors()?;
                pack.composite_scatter(x, &mut local_vecs)?;
                let (k_loc, u_loc) = (local_vecs.pop().unwrap(), local_vecs.pop().unwrap());

                let is = pack.composite_get_local_indexsets()?.into_iter()
                    .map(|is| Rc::new(is)).collect::<Vec<_>>();

                form_jac_u(&petsc, (mx_u, xm_u, xs_u, xs_k, gxs_u), (&dms[0], &dms[1]), &u_loc, &k_loc,
                    &mut *jac.get_local_sub_matrix_mut(is[0].clone(), is[0].clone())?)?;

                if !jac.type_compare(MatType::MATNEST)? {
                    form_jac_uk(&petsc, (mx_u, xm_u, xs_u, xs_k, gxs_u, gxs_k), (&dms[0], &dms[1]), &u_loc, &k_loc,
                        &mut *jac.get_local_sub_matrix_mut(is[0].clone(), is[1].clone())?)?;

                    form_jac_ku(&petsc, (mx_u, xm_k, xs_k, xs_u, gxs_k, gxs_u), (&dms[0], &dms[1]), &u_loc, &k_loc,
                        &mut *jac.get_local_sub_matrix_mut(is[1].clone(), is[0].clone())?)?;
                }

                form_jac_k(&petsc, (mx_k, xm_k, xs_k, xs_u, gxs_k), (&dms[0], &dms[1]), &u_loc, &k_loc,
                    &mut *jac.get_local_sub_matrix_mut(is[1].clone(), is[1].clone())?)?;

                jac.assemble(MatAssemblyType::MAT_FINAL_ASSEMBLY)?;

                Ok(())
            })?;
            snes.set_from_options()?;

            snes.solve(None, &mut x)?;
        },
        _ => {
            Petsc::set_error(petsc.world(), PetscErrorKind::PETSC_ERR_USER_INPUT,
                format!("{} is not valid for `-problem_type`. Please use 0, 1, or 2.", problem_type))?;
        }
    }

    x.view_with(None)?;

    // return
    Ok(())
}

fn form_func_u<'a>(_petsc: &Petsc, info: (PetscInt, PetscInt, PetscInt, PetscInt, PetscInt),
    dms: (&DM<'a, '_>, &DM<'a, '_>), u_loc: &Vector<'a>, k_loc: &Vector<'a>, y: &mut Vector<'a>) -> petsc_rs::Result<()>
{
    let (m, mx, xs, xs_k, gxs) = info;

    let u_view = dms.0.da_vec_view(&u_loc)?;
    let mut y_view = dms.0.da_vec_view_mut(y)?;
    let k_view = dms.1.da_vec_view(&k_loc)?;
    
    let hx = PetscScalar::from(1.0)/mx as PetscReal;

    y_view.indexed_iter_mut().map(|(s, y)| (s[0], s[0]+(xs-gxs) as usize, y))
        .for_each(|(i, ii, y)| {
            let ii_k = ii + (xs - xs_k) as usize;
            if i + xs as usize == 0 {
                *y = 1.0/hx * u_view[ii];
            } else if i + xs as usize == (m-1) as usize {
                *y = 1.0/hx*(u_view[ii] - 1.0)
            } else {
                *y = hx*((k_view[ii_k-1]*(u_view[ii]-u_view[ii-1]) - k_view[ii_k]*(u_view[ii+1]-u_view[ii]))/(hx*hx) - 1.0)
            }
        });

    Ok(())
}

fn form_func_k<'a>(_petsc: &Petsc, info: (PetscInt, PetscInt, PetscInt, PetscInt, PetscInt),
    dms: (&DM<'a, '_>, &DM<'a, '_>), u_loc: &Vector<'a>, k_loc: &Vector<'a>, y: &mut Vector<'a>) -> petsc_rs::Result<()>
{
    let (_m, mx, xs, xs_u, gxs) = info;

    let u_view = dms.0.da_vec_view(&u_loc)?;
    let mut y_view = dms.1.da_vec_view_mut(y)?;
    let k_view = dms.1.da_vec_view(&k_loc)?;

    let hx = PetscScalar::from(1.0)/mx as PetscReal;

    y_view.indexed_iter_mut().map(|(s, y)| (s[0], s[0]+(xs-gxs) as usize, y))
        .for_each(|(_i, ii, y)| {
            let ii_u = ii - (xs_u - xs) as usize;
            let ubar = 0.5 * (u_view[ii_u+1] + u_view[ii_u]);
            let gradu = (u_view[ii_u+1]-u_view[ii_u])/hx;
            let g = 1.0 + gradu*gradu;
            let w = 1.0/(1.0+ubar) + 1.0/g;
            *y = hx * (PetscScalar::exp(k_view[ii] - 1.0) + k_view[ii] - 1.0/w);
        });

    Ok(())
}

fn form_jac_u<'a>(_petsc: &Petsc, info: (PetscInt, PetscInt, PetscInt, PetscInt, PetscInt),
    dms: (&DM<'a, '_>, &DM<'a, '_>), u_loc: &Vector<'a>, k_loc: &Vector<'a>, jac: &mut Mat<'a, '_>) -> petsc_rs::Result<()>
{
    let (mx, xm, xs, xs_k, gxs) = info;
    
    let _u_view = dms.0.da_vec_view(&u_loc)?;
    let k_view = dms.1.da_vec_view(&k_loc)?;

    let hx = PetscScalar::from(1.0)/mx as PetscReal;

    jac.set_local_values_with_batched((xs..xs+xm).map(|i| {
        let ii_k = (i - xs_k + (xs - gxs)) as usize;
        let row = vec![i-gxs];
        
        if i == 0 || i == mx-1 {
            (row.clone(), row, vec![1.0/hx])
        } else {
            let cols = vec![row[0]-1, row[0], row[0]+1];
            (row, cols, vec![-k_view[ii_k-1]/hx,(k_view[ii_k-1]+k_view[ii_k])/hx,-k_view[ii_k]/hx])
        }
    }), InsertMode::INSERT_VALUES)
}

fn form_jac_k<'a>(_petsc: &Petsc, info: (PetscInt, PetscInt, PetscInt, PetscInt, PetscInt),
    dms: (&DM<'a, '_>, &DM<'a, '_>), u_loc: &Vector<'a>, k_loc: &Vector<'a>, jac: &mut Mat<'a, '_>) -> petsc_rs::Result<()>
{
    let (mx, xm, xs, _xs_u, gxs) = info;
    
    let _u_view = dms.0.da_vec_view(&u_loc)?;
    let k_view = dms.1.da_vec_view(&k_loc)?;

    let hx = PetscScalar::from(1.0)/mx as PetscReal;

    jac.set_local_values_with((xs..xs+xm).map(|i| {
        let ii = (i-gxs) as usize;
        let row = i-gxs;
        (row, row, hx*(PetscScalar::exp(k_view[ii]-1.0)+ 1.0))
    }), InsertMode::INSERT_VALUES)
}

fn form_jac_uk<'a>(_petsc: &Petsc, info: (PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt),
    dms: (&DM<'a, '_>, &DM<'a, '_>), u_loc: &Vector<'a>, k_loc: &Vector<'a>, jac: &mut Mat<'a, '_>) -> petsc_rs::Result<()>
{
    let (mx, xm, xs, _xs_k, gxs, gxs_k) = info;
    
    let u_view = dms.0.da_vec_view(&u_loc)?;
    let _k_view = dms.1.da_vec_view(&k_loc)?;

    let hx = PetscScalar::from(1.0)/(mx as PetscReal);

    jac.set_local_values_with_batched((xs..xs+xm).filter_map(|i| {
        let ii_u = (i - gxs) as usize;
        let row = vec![i - gxs];
        let cols = vec![i - gxs_k - 1, i - gxs_k];
        
        if i == 0 || i == mx-1 {
            None
        } else {
            Some((row, cols, vec![(u_view[ii_u] - u_view[ii_u-1])/hx, (u_view[ii_u] - u_view[ii_u+1])/hx]))
        }
    }), InsertMode::INSERT_VALUES)
}

fn form_jac_ku<'a>(_petsc: &Petsc, info: (PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt),
    dms: (&DM<'a, '_>, &DM<'a, '_>), u_loc: &Vector<'a>, k_loc: &Vector<'a>, jac: &mut Mat<'a, '_>) -> petsc_rs::Result<()>
{
    let (mx_u, xm, xs, xs_u, gxs, gxs_u) = info;
    
    let u_view = dms.0.da_vec_view(&u_loc)?;
    let _k_view = dms.1.da_vec_view(&k_loc)?;

    let hx = PetscScalar::from(1.0)/(mx_u as PetscReal - 1.0);

    jac.set_local_values_with_batched((xs..xs+xm).map(|i| {
        let ii_u = ((i - gxs) - (xs_u - xs)) as usize;
        let row = vec![i - gxs];
        let cols = vec![i - gxs_u, i + 1 - gxs_u ];

        let ubar     = 0.5*(u_view[ii_u]+u_view[ii_u+1]);
        let ubar_l   = 0.5;
        let ubar_r   = 0.5;
        let gradu    = (u_view[ii_u+1]-u_view[ii_u])/hx;
        let gradu_l  = -1.0/hx;
        let gradu_r  = 1.0/hx;
        let g        = 1.0 + gradu.powi(2);
        let g_gradu  = 2.0*gradu;
        let w        = 1.0/(1.0+ubar) + 1.0/g;
        let w_ubar   = -1.0/(1.0+ubar).powi(2);
        let w_gradu  = -g_gradu/g.powi(2);
        let iw       = 1.0/w;
        let iw_ubar  = -w_ubar * iw.powi(2);
        let iw_gradu = -w_gradu * iw.powi(2);
        
        (row, cols, vec![-hx*(iw_ubar*ubar_l + iw_gradu*gradu_l), -hx*(iw_ubar*ubar_r + iw_gradu*gradu_r)])
    }), InsertMode::INSERT_VALUES)
}
