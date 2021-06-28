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

use petsc_rs::prelude::*;
use structopt::StructOpt;

mod opt;
use opt::*;

#[derive(Debug, StructOpt)]
#[structopt(name = "ex25", about = HELP_MSG)]
struct Opt {
    /// The conductivity
    #[structopt(short, default_value = "1")]
    k: i32, // real

    /// The width of the Gaussian source
    #[structopt(short, default_value = "0.99")]
    e: f64, // real

    /// use `-- -help` for petsc help
    #[structopt(subcommand)]
    sub: Option<PetscOpt>,
}

fn main() -> petsc_rs::Result<()> {
    let Opt {k, e, sub: ext_args} = Opt::from_args();
    let petsc_args = PetscOpt::petsc_args(ext_args); // Is there an easier way to do this

    let petsc = Petsc::builder()
        .args(petsc_args)
        .help_msg(HELP_MSG)
        .init()?;

    petsc_println!(petsc.world(), "(petsc_println!) Hello parallel world of {} processes!", petsc.world().size() )?;

    let mut ksp = petsc.ksp_create()?;
    let mut da = DM::da_create_1d(petsc.world(), DMBoundaryType::DM_BOUNDARY_NONE, 128, 1, 1, None)?;
    da.set_from_options()?;
    da.set_up()?;

    let mut x = da.create_global_vector()?;

    ksp.set_dm(da)?;
    ksp.set_compute_rhs(|_ksp, dm, b| {
        // We will define the forcing function $f = e^{-x^2/\nu} e^{-y^2/\nu}$
        let (_, mx, _, _, _, _, _, _, _, _, _, _, _) = dm.da_get_info()?;
        let h = 1.0 / (mx as f64 - 1.0);
        let (xs, _, _, xm, _, _) = dm.da_get_corners()?;

        b.set_all(h)?;

        {
            let mut b_view = dm.da_vec_view_mut(b)?;
            if xs == 0 { b_view[0] = 0.0; }
            if xs + xm == mx { b_view[xm as usize -1] = 0.0; }
        }

        b.assemble()?;

        Ok(())
    })?;

    ksp.set_compute_operators(|_ksp, dm, _, jac| {
        let (_, mx, _, _, _, _, _, _, _, _, _, _, _) = dm.da_get_info()?;
        let h = 1.0 / (mx as f64 - 1.0);
        let (xs, _, _, xm, _, _) = dm.da_get_corners()?;

        jac.assemble_with_stencil((xs..xs+xm)
                .map(|i| {
                    let row = MatStencil { i, j: 0, c: 0, k: 0 };
                    if i == 0 || i == mx-1 {
                        vec![(row, row, 2.0/h)]
                    } else {
                        let xlow  = h* i as f64 - 0.5*h;
                        let xhigh  = xlow + h;
                        vec![
                            (row, MatStencil { i: i-1, j: 0, c: 0, k: 0 },
                                (-1.0 - e * f64::sin(2.0 * std::f64::consts::PI * k as f64 * xlow))/h),
                            (row, row,
                                (2.0 + e * f64::sin(2.0 * std::f64::consts::PI * k as f64 * xlow) 
                                + e * f64::sin(2.0 * std::f64::consts::PI * k as f64 * xhigh))/h),
                            (row, MatStencil { i: i+1, j: 0, c: 0, k: 0 },
                                (-1.0 - e * f64::sin(2.0 * std::f64::consts::PI * k as f64 * xhigh))/h),
                        ]
                    }
                }).flatten(), 
            InsertMode::INSERT_VALUES, MatAssemblyType::MAT_FINAL_ASSEMBLY)?;

        Ok(())
    })?;

    ksp.set_from_options()?;
    ksp.set_up()?;

    ksp.solve(None, &mut x)?;

    let (a_mat,_) = ksp.get_operators()?;
    let b = ksp.get_rhs()?;
    let mut b2 = b.duplicate()?;
    Mat::mult(&a_mat,&x,&mut b2)?;
    b2.axpy(-1.0, &b)?;
    let r_norm = b2.norm(NormType::NORM_INFINITY)?;
    petsc_println!(petsc.world(), "Residual norm: {:.5e}", r_norm)?;

    let iters = ksp.get_iteration_number()?;
    petsc_println!(petsc.world(), "Iters {}", iters)?;

    //ksp.view_with(Some(&petsc.viewer_create_ascii_stdout()?))?;
    x.view_with(Some(&petsc.viewer_create_ascii_stdout()?))?;
    petsc_println_all!(petsc.world(), "Process [{}]\n{:.5e}", petsc.world().rank(), *ksp.get_dm()?.da_vec_view(&x)?)?;

    // return
    Ok(())
}
