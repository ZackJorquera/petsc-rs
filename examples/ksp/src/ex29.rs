//! This file will show how to do the ksp ex29 example in rust using the petsc-rs bindings.
//!
//! Concepts: KSP^solving a system of linear equations
//! Concepts: KSP^Laplacian, 2d
//! Processors: n
//!
//! Inhomogeneous Laplacian in 2D. Modeled by the partial differential equation
//!    $$ -div \rho grad u = f,  0 < x,y < 1 $$,
//! with forcing function
//!    $$ f = e^{-x^2/\nu} e^{-y^2/\nu} $$
//! with Dirichlet boundary conditions
//!    $$ u = f(x,y) for x = 0, x = 1, y = 0, y = 1 $$
//! or pure Neumman boundary conditions
//! This uses multigrid to solve the linear system
//!
//! To run:
//! ```test
//! $ cargo build --bin ex29
//! $ mpiexec -n 1 target/debug/ex29
//! $ mpiexec -n 2 target/debug/ex29
//! ```

static HELP_MSG: &str = "Solves 2D inhomogeneous Laplacian using multigrid.\n\n";

use petsc_rs::prelude::*;
use structopt::StructOpt;
use ndarray::prelude::*;

use std::fmt;

mod opt;
use opt::*;

#[derive(Copy, Clone, Debug, PartialEq)]
enum BCType { DIRICHLET, NEUMANN }

impl std::str::FromStr for BCType {
    type Err = std::io::Error;
    fn from_str(input: &str) -> Result<BCType, std::io::Error> {
        if input.to_uppercase() == "NEUMANN" {
            Ok(BCType::NEUMANN)
        } else {
            Ok(BCType::DIRICHLET)
        }
    }
}

impl fmt::Display for BCType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BCType::NEUMANN => write!(f, "neumann"),
            _ => write!(f, "dirichlet"),
        }
    }
}

#[derive(Debug, StructOpt)]
#[structopt(name = "ex29", about = "Options for the inhomogeneous Poisson equation")]
struct Opt {
    /// The conductivity
    #[structopt(short, long, default_value = "1.0")]
    rho: f64, // real

    /// The width of the Gaussian source
    #[structopt(short, long, default_value = "0.1")]
    nu: f64, // real

    /// Type of boundary condition
    #[structopt(short, long, default_value = "DIRICHLET")]
    bc_type: BCType,

    /// Run solver multiple times, useful for performance studies of solver
    #[structopt(short, long)]
    test_solver: bool,

    /// use `-- -help` for petsc help
    #[structopt(subcommand)]
    sub: Option<PetscOpt>,
}

fn main() -> petsc_rs::Result<()> {
    let Opt {rho, nu, bc_type, test_solver: _, sub: ext_args} = Opt::from_args();
    let petsc_args = PetscOpt::petsc_args(ext_args); // Is there an easier way to do this
    let check_matis = false;

    let petsc = Petsc::builder()
        .args(petsc_args)
        .help_msg(HELP_MSG)
        .init()?;

    petsc_println!(petsc.world(), "(petsc_println!) Hello parallel world of {} processes!", petsc.world().size() )?;

    let mut ksp = petsc.ksp_create()?;
    let mut da = DM::da_create_2d(petsc.world(), DMBoundaryType::DM_BOUNDARY_NONE, DMBoundaryType::DM_BOUNDARY_NONE,
        DMDAStencilType::DMDA_STENCIL_STAR, 3, 3, None, None, 1, 1, None, None)?;
    da.set_from_options()?;
    da.set_up()?;
    da.da_set_uniform_coordinates(0.0, 1.0, 0.0, 1.0, 0.0, 0.0)?;
    da.da_set_feild_name(0, "Pressure")?;

    let mut x = da.create_global_vector()?;

    ksp.set_dm(da)?;
    ksp.set_compute_rhs(|_ksp, dm, b| {
        // We will define the forcing function $f = e^{-x^2/\nu} e^{-y^2/\nu}$
        let (_, mx, my, _, _, _, _, _, _, _, _, _, _) = dm.da_get_info()?;
        let (hx, hy) = (1.0 / (mx as f64 - 1.0), 1.0 / (my as f64 - 1.0));
        let (xs, ys, _, _xm, _ym, _) = dm.da_get_corners()?;

        {
            let mut b_view = dm.da_vec_view_mut(b)?;

            b_view.indexed_iter_mut().map(|(pat, v)| { 
                    let s = pat.slice();
                    (((s[0]+xs as usize) as f64, (s[1]+ys as usize) as f64), v) 
                })
                .for_each(|((i,j), v)| {
                    *v = f64::exp(-(i*hx)*(i*hx)/nu)*f64::exp(-(j*hy)*(j*hy)/nu)*hx*hy;
                });
        }

        b.assemble()?;

        if bc_type == BCType::NEUMANN {
            let nullspace = NullSpace::create(petsc.world(), true, [])?;
            nullspace.remove_from(b)?;
        }

        Ok(())
    })?;

    ksp.set_compute_operators(|_ksp, dm, _, jac| {
        let (_, mx, my, _, _, _, _, _, _, _, _, _, _) = dm.da_get_info()?;
        let (hx, hy) = (1.0 / (mx as f64 - 1.0), 1.0 / (my as f64 - 1.0));
        let (hxdhy, hydhx) = (hx/hy, hy/hx);
        let (xs, ys, _, xm, ym, _) = dm.da_get_corners()?;

        jac.assemble_with_stencil((ys..ys+ym).map(|j| (xs..xs+xm).map(move |i| (i,j))).flatten()
                .map(|(i,j)| {
                    let row = MatStencil { i, j, c: 0, k: 0 };
                    let rho = compute_rho(i, j, mx, my, rho);
                    if i == 0 || j == 0 || i == mx-1 || j == my-1 {
                        if bc_type == BCType::DIRICHLET {
                            vec![(row, row, 2.0*rho*(hxdhy + hydhx))]
                        } else {
                            let mut vec = vec![];
                            let mut numx = 0; let mut numy = 0;
                            if j != 0 { vec.push((row, MatStencil { i, j: j-1, c: 0, k: 0 }, -rho*hxdhy)); numy += 1; }
                            if i != 0 { vec.push((row, MatStencil { i: i-1, j, c: 0, k: 0 }, -rho*hydhx)); numx += 1; }
                            if i != mx-1 { vec.push((row, MatStencil { i: i+1, j, c: 0, k: 0 }, -rho*hydhx)); numx += 1; }
                            if j != my-1 { vec.push((row, MatStencil { i, j: j+1, c: 0, k: 0 }, -rho*hxdhy)); numy += 1; }
                            vec.push((row, row, numx as f64 *rho*hydhx + numy as f64*rho*hxdhy));
                            vec
                        }
                    } else {
                        vec![
                            (row, MatStencil { i, j: j-1, c: 0, k: 0 },-rho*hxdhy),
                            (row, MatStencil { i: i-1, j, c: 0, k: 0 },-rho*hydhx),
                            (row, row, 2.0*rho*(hxdhy + hydhx)),
                            (row, MatStencil { i: i+1, j, c: 0, k: 0 },-rho*hydhx),
                            (row, MatStencil { i, j: j+1, c: 0, k: 0 },-rho*hxdhy),
                        ]
                    }
                }).flatten(), 
            InsertMode::INSERT_VALUES, MatAssemblyType::MAT_FINAL_ASSEMBLY)?;

        if check_matis {
            todo!()
        }

        if bc_type == BCType::NEUMANN {
            let nullspace = NullSpace::create(petsc.world(), true, [])?;
            jac.set_nullspace(Some(std::rc::Rc::new(nullspace)))?;
        }

        Ok(())
    })?;

    ksp.set_from_options()?;
    ksp.set_up()?;

    ksp.solve(None, &mut x)?;

    let iters = ksp.get_iteration_number()?;
    petsc_println!(petsc.world(), "Iters {}", iters)?;

    //ksp.view_with(Some(&petsc.viewer_create_ascii_stdout()?))?;
    x.view_with(Some(&petsc.viewer_create_ascii_stdout()?))?;
    petsc_println_all!(petsc.world(), "Process [{}]\n{:.5e}", petsc.world().rank(), *ksp.get_dm()?.da_vec_view(&x)?)?;

    // return
    Ok(())
}

fn compute_rho(i: i32, j: i32, mx: i32, my: i32, center_rho: f64) -> f64 {
    if (i as f64) > (mx as f64/3.0) && (i as f64) < (2.0*mx as f64/3.0)
        && (j as f64) > (my as f64/3.0) && (j as f64) < (2.0 * (my as f64) / 3.0) {
      center_rho
    } else {
      1.0
    }
}
