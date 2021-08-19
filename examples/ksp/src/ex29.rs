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
//! $ cargo build --bin ksp-ex29
//! $ mpiexec -n 1 target/debug/ksp-ex29
//! $ mpiexec -n 2 target/debug/ksp-ex29
//! ```

static HELP_MSG: &str = "Solves 2D inhomogeneous Laplacian using multigrid.\n\n";

use mpi::traits::*;
use ndarray::prelude::*;
use petsc_rs::prelude::*;

use std::fmt;

#[derive(Copy, Clone, Debug, PartialEq)]
enum BCType {
    DIRICHLET,
    NEUMANN,
}

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

impl Default for BCType {
    fn default() -> Self {
        BCType::DIRICHLET
    }
}

struct Opt {
    /// The conductivity
    rho: PetscReal,

    /// The width of the Gaussian source
    nu: PetscReal,

    /// Type of boundary condition
    bc_type: BCType,

    /// Run solver multiple times, useful for performance studies of solver
    test_solver: bool,

    check_matis: bool,
}

impl PetscOpt for Opt {
    fn from_petsc_opt_builder(pob: &mut PetscOptBuilder) -> petsc_rs::Result<Self> {
        let rho = pob.options_real("-rho", "", "ksp-ex29", 1.0)?;
        let nu = pob.options_real("-nu", "", "ksp-ex29", 0.1)?;
        let bc_type = pob.options_from_string("-bc_type", "", "ksp-ex29", BCType::DIRICHLET)?;
        let test_solver = pob.options_bool("-test_solver", "", "ksp-ex29", false)?;
        let check_matis = pob.options_bool("-check_matis", "", "ksp-ex29", false)?;
        Ok(Opt {
            rho,
            nu,
            bc_type,
            test_solver,
            check_matis,
        })
    }
}

fn main() -> petsc_rs::Result<()> {
    let petsc = Petsc::builder()
        .args(std::env::args())
        .help_msg(HELP_MSG)
        .init()?;

    let Opt {
        rho,
        nu,
        bc_type,
        test_solver: _test_solver,
        check_matis,
    } = petsc.options_get()?;

    petsc_println!(
        petsc.world(),
        "(petsc_println!) Hello parallel world of {} processes!",
        petsc.world().size()
    )?;

    let mut ksp = petsc.ksp_create()?;
    let mut da = DM::da_create_2d(
        petsc.world(),
        DMBoundaryType::DM_BOUNDARY_NONE,
        DMBoundaryType::DM_BOUNDARY_NONE,
        DMDAStencilType::DMDA_STENCIL_STAR,
        3,
        3,
        None,
        None,
        1,
        1,
        None,
        None,
    )?;
    da.set_from_options()?;
    da.set_up()?;
    da.da_set_uniform_coordinates(0.0, 1.0, 0.0, 1.0, 0.0, 0.0)?;
    da.da_set_feild_name(0, "Pressure")?;

    let mut x = da.create_global_vector()?;

    ksp.set_dm(da)?;
    ksp.set_compute_rhs(|ksp, b| {
        // We will define the forcing function $f = e^{-x^2/\nu} e^{-y^2/\nu}$
        let dm = ksp.try_get_dm().unwrap();
        let (_, mx, my, _, _, _, _, _, _, _, _, _, _) = dm.da_get_info()?;
        let (hx, hy) = (
            1.0 / (PetscScalar::from(mx as PetscReal) - 1.0),
            1.0 / (PetscScalar::from(my as PetscReal) - 1.0),
        );
        let (xs, ys, _, _xm, _ym, _) = dm.da_get_corners()?;

        {
            let mut b_view = dm.da_vec_view_mut(b)?;

            b_view
                .indexed_iter_mut()
                .map(|(pat, v)| {
                    let s = pat.slice();
                    (
                        (
                            (s[0] + xs as usize) as PetscReal,
                            (s[1] + ys as usize) as PetscReal,
                        ),
                        v,
                    )
                })
                .for_each(|((i, j), v)| {
                    *v = PetscScalar::exp(-(i * hx) * (i * hx) / nu)
                        * PetscScalar::exp(-(j * hy) * (j * hy) / nu)
                        * hx
                        * hy;
                });
        }

        b.assemble()?;

        if bc_type == BCType::NEUMANN {
            let nullspace = NullSpace::create(petsc.world(), true, [])?;
            nullspace.remove_from(b)?;
        }

        Ok(())
    })?;

    ksp.set_compute_operators(|ksp, _, jac| {
        let dm = ksp.try_get_dm().unwrap();
        let (_, mx, my, _, _, _, _, _, _, _, _, _, _) = dm.da_get_info()?;
        let (hx, hy) = (
            1.0 / (PetscScalar::from(mx as PetscReal) - 1.0),
            1.0 / (PetscScalar::from(my as PetscReal) - 1.0),
        );
        let (hxdhy, hydhx) = (hx / hy, hy / hx);
        let (xs, ys, _, xm, ym, _) = dm.da_get_corners()?;

        jac.assemble_with_stencil(
            (ys..ys + ym)
                .map(|j| (xs..xs + xm).map(move |i| (i, j)))
                .flatten()
                .map(|(i, j)| {
                    let row = MatStencil { i, j, c: 0, k: 0 };
                    let rho = compute_rho(i, j, mx, my, rho);
                    if i == 0 || j == 0 || i == mx - 1 || j == my - 1 {
                        if bc_type == BCType::DIRICHLET {
                            vec![(row, row, 2.0 * rho * (hxdhy + hydhx))]
                        } else {
                            let mut vec = vec![];
                            let mut numx = 0;
                            let mut numy = 0;
                            if j != 0 {
                                vec.push((
                                    row,
                                    MatStencil {
                                        i,
                                        j: j - 1,
                                        c: 0,
                                        k: 0,
                                    },
                                    -rho * hxdhy,
                                ));
                                numy += 1;
                            }
                            if i != 0 {
                                vec.push((
                                    row,
                                    MatStencil {
                                        i: i - 1,
                                        j,
                                        c: 0,
                                        k: 0,
                                    },
                                    -rho * hydhx,
                                ));
                                numx += 1;
                            }
                            if i != mx - 1 {
                                vec.push((
                                    row,
                                    MatStencil {
                                        i: i + 1,
                                        j,
                                        c: 0,
                                        k: 0,
                                    },
                                    -rho * hydhx,
                                ));
                                numx += 1;
                            }
                            if j != my - 1 {
                                vec.push((
                                    row,
                                    MatStencil {
                                        i,
                                        j: j + 1,
                                        c: 0,
                                        k: 0,
                                    },
                                    -rho * hxdhy,
                                ));
                                numy += 1;
                            }
                            vec.push((
                                row,
                                row,
                                numx as PetscReal * rho * hydhx + numy as PetscReal * rho * hxdhy,
                            ));
                            vec
                        }
                    } else {
                        vec![
                            (
                                row,
                                MatStencil {
                                    i,
                                    j: j - 1,
                                    c: 0,
                                    k: 0,
                                },
                                -rho * hxdhy,
                            ),
                            (
                                row,
                                MatStencil {
                                    i: i - 1,
                                    j,
                                    c: 0,
                                    k: 0,
                                },
                                -rho * hydhx,
                            ),
                            (row, row, 2.0 * rho * (hxdhy + hydhx)),
                            (
                                row,
                                MatStencil {
                                    i: i + 1,
                                    j,
                                    c: 0,
                                    k: 0,
                                },
                                -rho * hydhx,
                            ),
                            (
                                row,
                                MatStencil {
                                    i,
                                    j: j + 1,
                                    c: 0,
                                    k: 0,
                                },
                                -rho * hxdhy,
                            ),
                        ]
                    }
                })
                .flatten(),
            InsertMode::INSERT_VALUES,
            MatAssemblyType::MAT_FINAL_ASSEMBLY,
        )?;

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

fn compute_rho(
    i: PetscInt,
    j: PetscInt,
    mx: PetscInt,
    my: PetscInt,
    center_rho: PetscReal,
) -> PetscReal {
    if (i as PetscReal) > (mx as PetscReal / 3.0)
        && (i as PetscReal) < (2.0 * mx as PetscReal / 3.0)
        && (j as PetscReal) > (my as PetscReal / 3.0)
        && (j as PetscReal) < (2.0 * (my as PetscReal) / 3.0)
    {
        center_rho
    } else {
        1.0
    }
}
