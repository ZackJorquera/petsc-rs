//! This file will show how to do the ksp ex29 example in rust using the petsc-rs bindings.
//!
//! Concepts: KSP^solving a system of linear equations
//! Concepts: KSP^Laplacian, 1d
//! Processors: n
//!
//! Partial differential equation
//!    d  (1 + e*sine(2*pi*k*x)) du = 1, 0 < x < 1,
//!    --                        --
//!    dx                        dx
//! with boundary conditions
//!    u = 0 for x = 0, x = 1
//! This uses multigrid to solve the linear system
//!
//! To run:
//! ```test
//! $ cargo build --bin ksp-ex29
//! $ mpiexec -n 1 target/debug/ksp-ex29
//! $ mpiexec -n 2 target/debug/ksp-ex29
//! ```

static HELP_MSG: &str = "Solves 1D variable coefficient Laplacian using multigrid.\n\n";

use mpi::traits::*;
use petsc_rs::prelude::*;

struct Opt {
    /// The conductivity
    k: PetscInt,

    /// The width of the Gaussian source
    e: PetscReal,
}

impl PetscOpt for Opt {
    fn from_petsc_opt_builder(pob: &mut PetscOptBuilder) -> petsc_rs::Result<Self> {
        let k = pob.options_int("-k", "", "ksp-ex25", 1)?;
        let e = pob.options_real("-e", "", "ksp-ex25", 0.99)?;
        Ok(Opt { k, e })
    }
}

fn main() -> petsc_rs::Result<()> {
    let petsc = Petsc::builder()
        .args(std::env::args())
        .help_msg(HELP_MSG)
        .init()?;

    let Opt { k, e } = petsc.options_get()?;

    petsc_println!(
        petsc.world(),
        "(petsc_println!) Hello parallel world of {} processes!",
        petsc.world().size()
    )?;

    let mut ksp = petsc.ksp_create()?;
    let mut da = DM::da_create_1d(
        petsc.world(),
        DMBoundaryType::DM_BOUNDARY_NONE,
        128,
        1,
        1,
        None,
    )?;
    da.set_from_options()?;
    da.set_up()?;

    let mut x = da.create_global_vector()?;

    ksp.set_dm(da)?;
    ksp.set_compute_rhs(|ksp, b| {
        // We will define the forcing function $f = e^{-x^2/\nu} e^{-y^2/\nu}$
        let dm = ksp.try_get_dm().unwrap();
        let (_, mx, _, _, _, _, _, _, _, _, _, _, _) = dm.da_get_info()?;
        let h = 1.0 / (PetscScalar::from(mx as PetscReal) - 1.0);
        let (xs, _, _, xm, _, _) = dm.da_get_corners()?;

        b.set_all(h)?;

        {
            let mut b_view = dm.da_vec_view_mut(b)?;
            if xs == 0 {
                b_view[0] = PetscScalar::from(0.0);
            }
            if xs + xm == mx {
                b_view[xm as usize - 1] = PetscScalar::from(0.0);
            }
        }

        b.assemble()?;

        Ok(())
    })?;

    ksp.set_compute_operators(|ksp, _, jac| {
        let dm = ksp.try_get_dm().unwrap();
        let (_, mx, _, _, _, _, _, _, _, _, _, _, _) = dm.da_get_info()?;
        let h = 1.0 / (PetscScalar::from(mx as PetscReal) - 1.0);
        let (xs, _, _, xm, _, _) = dm.da_get_corners()?;
        let pi = PetscScalar::from(std::f64::consts::PI as PetscReal);

        jac.assemble_with_stencil(
            (xs..xs + xm)
                .map(|i| {
                    let row = MatStencil {
                        i,
                        j: 0,
                        c: 0,
                        k: 0,
                    };
                    if i == 0 || i == mx - 1 {
                        vec![(row, row, 2.0 / h)]
                    } else {
                        let xlow = h * i as PetscReal - 0.5 * h;
                        let xhigh = xlow + h;
                        vec![
                            (
                                row,
                                MatStencil {
                                    i: i - 1,
                                    j: 0,
                                    c: 0,
                                    k: 0,
                                },
                                (-1.0 - e * PetscScalar::sin(2.0 * pi * k as PetscReal * xlow)) / h,
                            ),
                            (
                                row,
                                row,
                                (2.0 + e * PetscScalar::sin(2.0 * pi * k as PetscReal * xlow)
                                    + e * PetscScalar::sin(2.0 * pi * k as PetscReal * xhigh))
                                    / h,
                            ),
                            (
                                row,
                                MatStencil {
                                    i: i + 1,
                                    j: 0,
                                    c: 0,
                                    k: 0,
                                },
                                (-1.0 - e * PetscScalar::sin(2.0 * pi * k as PetscReal * xhigh))
                                    / h,
                            ),
                        ]
                    }
                })
                .flatten(),
            InsertMode::INSERT_VALUES,
            MatAssemblyType::MAT_FINAL_ASSEMBLY,
        )?;

        Ok(())
    })?;

    ksp.set_from_options()?;
    ksp.set_up()?;

    ksp.solve(None, &mut x)?;

    let b = ksp.get_rhs()?;
    let (a_mat, _) = ksp.get_operators_or_create()?;
    let mut b2 = b.duplicate()?;
    Mat::mult(&a_mat, &x, &mut b2)?;
    b2.axpy(PetscScalar::from(-1.0), &b)?;
    let r_norm = b2.norm(NormType::NORM_INFINITY)?;
    petsc_println!(petsc.world(), "Residual norm: {:.5e}", r_norm)?;

    let iters = ksp.get_iteration_number()?;
    petsc_println!(petsc.world(), "Iters {}", iters)?;

    // ksp.view_with(Some(&petsc.viewer_create_ascii_stdout()?))?;
    x.view_with(Some(&petsc.viewer_create_ascii_stdout()?))?;
    petsc_println_sync!(
        petsc.world(),
        "Process [{}]\n{:.5e}",
        petsc.world().rank(),
        *ksp.try_get_dm().unwrap().da_vec_view(&x)?
    )?;

    // return
    Ok(())
}
