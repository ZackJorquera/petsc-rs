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

#![allow(dead_code)]

static HELP_MSG: &str = "Poisson Problem in 2d and 3d with simplicial finite elements.\n\
    We solve the Poisson problem in a rectangular\n\
    domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\
    This example supports discretized auxiliary fields (conductivity) as well as\n\
    multilevel nonlinear solvers.\n\n\n";

use core::slice;
use std::{fmt::{self, Display}, io::{Error, ErrorKind}, rc::Rc, str::FromStr};

use petsc_rs::prelude::*;
use mpi::topology::UserCommunicator;
use mpi::traits::*;
use rand::prelude::*;

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[allow(non_camel_case_types)]
enum BCType { NEUMANN, DIRICHLET, NONE }
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[allow(non_camel_case_types)]
enum RunType { RUN_FULL, RUN_EXACT, RUN_TEST, RUN_PERF }
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[allow(non_camel_case_types)]
enum CoeffType { COEFF_NONE, COEFF_ANALYTIC, COEFF_FIELD, COEFF_NONLINEAR, COEFF_CIRCLE, COEFF_CROSS, COEFF_CHECKERBOARD_0, COEFF_CHECKERBOARD_1 }

impl FromStr for BCType {
    type Err = Error;
    fn from_str(input: &str) -> Result<BCType, Error> {
        match input.to_uppercase().as_str() {
            "NEUMANN" => Ok(BCType::NEUMANN),
            "DIRICHLET" => Ok(BCType::DIRICHLET),
            "NONE" => Ok(BCType::NONE),
            _ => Err(Error::new(ErrorKind::InvalidInput, format!("{}, is not valid", input)))
        }
    }
}

impl Display for BCType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BCType::NEUMANN => write!(f, "Neumann"),
            BCType::DIRICHLET => write!(f, "Dirichlet"),
            _ => write!(f, "None"),
        }
    }
}

impl Default for BCType {
    fn default() -> Self { BCType::DIRICHLET }
}

impl FromStr for RunType {
    type Err = Error;
    fn from_str(input: &str) -> Result<RunType, Error> {
        match input.to_uppercase().as_str() {
            "RUN_FULL" => Ok(RunType::RUN_FULL),
            "RUN_EXACT" => Ok(RunType::RUN_EXACT),
            "RUN_TEST" => Ok(RunType::RUN_TEST),
            "RUN_PERF" => Ok(RunType::RUN_PERF),
            _ => Err(Error::new(ErrorKind::InvalidInput, format!("{}, is not valid", input)))
        }
    }
}

impl Display for RunType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RunType::RUN_FULL => write!(f, "run_full"),
            RunType::RUN_EXACT => write!(f, "run_exact"),
            RunType::RUN_TEST => write!(f, "run_test"),
            _ => write!(f, "run_perf"),
        }
    }
}

impl Default for RunType {
    fn default() -> Self { RunType::RUN_FULL }
}

impl FromStr for CoeffType {
    type Err = Error;
    fn from_str(input: &str) -> Result<CoeffType, Error> {
        match input.to_uppercase().as_str() {
            "COEFF_NONE" => Ok(CoeffType::COEFF_NONE),
            "COEFF_ANALYTIC" => Ok(CoeffType::COEFF_ANALYTIC),
            "COEFF_FIELD" => Ok(CoeffType::COEFF_FIELD),
            "COEFF_NONLINEAR" => Ok(CoeffType::COEFF_NONLINEAR),
            "COEFF_CIRCLE" => Ok(CoeffType::COEFF_CIRCLE),
            "COEFF_CROSS" => Ok(CoeffType::COEFF_CROSS),
            "COEFF_CHECKERBOARD_0" => Ok(CoeffType::COEFF_CHECKERBOARD_0),
            "COEFF_CHECKERBOARD_1" => Ok(CoeffType::COEFF_CHECKERBOARD_1),
            _ => Err(Error::new(ErrorKind::InvalidInput, format!("{}, is not valid", input)))
        }
    }
}

impl Display for CoeffType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CoeffType::COEFF_NONE => write!(f, "COEFF_NONE"),
            CoeffType::COEFF_ANALYTIC => write!(f, "COEFF_ANALYTIC"),
            CoeffType::COEFF_FIELD => write!(f, "COEFF_FIELD"),
            CoeffType::COEFF_NONLINEAR => write!(f, "COEFF_NONLINEAR"),
            CoeffType::COEFF_CIRCLE => write!(f, "COEFF_CIRCLE"),
            CoeffType::COEFF_CROSS => write!(f, "COEFF_CROSS"),
            CoeffType::COEFF_CHECKERBOARD_0 => write!(f, "COEFF_CHECKERBOARD_0"),
            CoeffType::COEFF_CHECKERBOARD_1 => write!(f, "COEFF_CHECKERBOARD_1"),
        }
    }
}

impl Default for CoeffType {
    fn default() -> Self { CoeffType::COEFF_NONE }
}

#[derive(Debug)]
struct Opt {
    // debug: PetscInt,
    run_type: RunType,
    // dim: PetscInt,
    // periodicity: (DMBoundaryType, DMBoundaryType, DMBoundaryType),
    // cells: (PetscInt, PetscInt, PetscInt),
    // file_name: Option<String>,
    // interpolate: bool,
    // refinement_limit: PetscReal,
    bc_type: BCType,
    variable_coefficient: CoeffType,
    field_bc: bool,
    jacobian_mf: bool,
    show_initial: bool,
    show_solution: bool,
    restart: bool,
    // view_hierarchy: bool,
    // simplex: bool,
    quiet: bool,
    nonz_init: bool,
    bd_integral: bool,
    check_ksp: bool,
    div: PetscInt,
    k: PetscInt,
    rand: bool,
    dm_view: bool,
    guess_vec_view: bool,
    show_opts: bool,
}

impl PetscOpt for Opt {
    fn from_petsc(petsc: &Petsc) -> petsc_rs::Result<Self> {
        Ok(Opt { 
            // debug: petsc.options_try_get_int("-debug")?.unwrap_or(0),
            run_type: petsc.options_try_get_from_string("-run_type")?.unwrap_or(RunType::RUN_FULL),
            // dim: petsc.options_try_get_int("-dim")?.unwrap_or(2),
            // periodicity: (
            //     petsc.options_try_get_from_string("-x_periodicity")?.unwrap_or(DMBoundaryType::DM_BOUNDARY_NONE),
            //     petsc.options_try_get_from_string("-y_periodicity")?.unwrap_or(DMBoundaryType::DM_BOUNDARY_NONE),
            //     petsc.options_try_get_from_string("-z_periodicity")?.unwrap_or(DMBoundaryType::DM_BOUNDARY_NONE)
            // ),
            // cells: (
            //     petsc.options_try_get_int("-cell0")?.unwrap_or(2),
            //     petsc.options_try_get_int("-cell1")?.unwrap_or(2),
            //     petsc.options_try_get_int("-cell2")?.unwrap_or(2),
            // ),
            // file_name: petsc.options_try_get_string("-file_name")?,
            // interpolate: petsc.options_try_get_bool("-interpolate")?.unwrap_or(true),
            // refinement_limit: petsc.options_try_get_real("-refinement_limit")?.unwrap_or(0.0),
            bc_type: petsc.options_try_get_from_string("-bc_type")?.unwrap_or(BCType::DIRICHLET),
            variable_coefficient: petsc.options_try_get_from_string("-variable_coefficient")?.unwrap_or(CoeffType::COEFF_NONE),
            field_bc: petsc.options_try_get_bool("-field_bc")?.unwrap_or(false),
            jacobian_mf: petsc.options_try_get_bool("-jacobian_mf")?.unwrap_or(false),
            show_initial: petsc.options_try_get_bool("-show_initial")?.unwrap_or(false),
            show_solution: petsc.options_try_get_bool("-show_solution")?.unwrap_or(false),
            restart: petsc.options_try_get_bool("-restart")?.unwrap_or(false),
            // view_hierarchy: petsc.options_try_get_bool("-view_hierarchy")?.unwrap_or(false),
            // simplex: petsc.options_try_get_bool("-simplex")?.unwrap_or(true),
            quiet: petsc.options_try_get_bool("-quiet")?.unwrap_or(false),
            nonz_init: petsc.options_try_get_bool("-nonz_init")?.unwrap_or(false),
            bd_integral: petsc.options_try_get_bool("-bd_integral")?.unwrap_or(false),
            check_ksp: petsc.options_try_get_bool("-check_ksp")?.unwrap_or(false),
            div: petsc.options_try_get_int("-div")?.unwrap_or(4),
            k: petsc.options_try_get_int("-k")?.unwrap_or(1),
            rand: petsc.options_try_get_bool("-rand")?.unwrap_or(false),
            dm_view: petsc.options_try_get_bool("-dm_view")?.unwrap_or(false),
            guess_vec_view: petsc.options_try_get_bool("-guess_vec_view")?.unwrap_or(false),
            show_opts: petsc.options_try_get_bool("-show_opts")?.unwrap_or(false),
        })
    }
}

fn create_mesh<'a, 'b>(world: &'a UserCommunicator, opt: &Opt) -> petsc_rs::Result<(DM<'a, 'b>, Option<Vec<PetscInt>>)> {
    // let dm = if let Some(filename) = opt.file_name {
    //     let mut dm = DM::plex_create_from_file(world, &filename, opt.interpolate)?;
    //     dm.plex_set_refinement_uniform(false)?;
    //     dm
    // } else {
    //     let mut dm = DM::plex_create_box_mesh(world, opt.dim, opt.simplex, opt.cells, None, None, opt.periodicity, opt.interpolate)?;
    //     dm.set_name("Mesh")?;
    //     dm
    // };

    let mut dm = DM::plex_create(world)?;
    dm.set_name("Mesh")?;
    dm.set_from_options()?;

    // TODO: if `-dm_plex_convert_type` flag set

    if opt.dm_view {
        dm.view_with(None)?;
    }
    let kgrid = if opt.rand {
        let dim = dm.get_dimension()?;
        let n = opt.div.pow(dim as u32);
        let mut rng = StdRng::seed_from_u64(1973);
        Some((0..n).map(|_| 1 + rng.gen_range(0..opt.k)).collect())
    } else {
        None
    };

    Ok((dm, kgrid))
}

// For now, this api does not exist in `petsc-rs` so we have to use petsc-sys
fn petsc_ds_set_residual(ds: &mut DS, f0: unsafe extern "C" fn(PetscInt, PetscInt, PetscInt,
    *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
    *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
    PetscReal, *const PetscReal, PetscInt, *const PetscScalar, *mut PetscScalar), f1: unsafe extern "C" fn(PetscInt, PetscInt, PetscInt,
    *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
    *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
    PetscReal, *const PetscReal, PetscInt, *const PetscScalar, *mut PetscScalar)) -> petsc_rs::Result<()>
{
    let ierr = unsafe { petsc_sys::PetscDSSetResidual(ds.as_raw(), 0, Some(f0), Some(f1)) };
    if ierr == 0 {
        Ok(())
    } else {
        Petsc::set_error(ds.world(), PetscErrorKind::PETSC_ERR_ARG_CORRUPT, "PetscDSSetResidual failed")
    }
}

// For now, this api does not exist in `petsc-rs` so we have to use petsc-sys
fn petsc_weak_form_set_index_bd_residual(wf: &mut WeakForm, label: &DMLabel, id: PetscInt, f0: unsafe extern "C" fn(PetscInt, PetscInt, PetscInt,
    *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
    *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
    PetscReal, *const PetscReal, *const PetscReal, PetscInt, *const PetscScalar, *mut PetscScalar)) -> petsc_rs::Result<()>
{
    let ierr = unsafe { petsc_sys::PetscWeakFormSetIndexBdResidual(wf.as_raw(), label.as_raw(), id, 0, 0, 0, Some(f0), 0, None) };
    if ierr == 0 {
        Ok(())
    } else {
        Petsc::set_error(wf.world(), PetscErrorKind::PETSC_ERR_ARG_CORRUPT, "PetscWeakFormSetIndexBdResidual failed")
    }
}

// For now, this api does not exist in `petsc-rs` so we have to use petsc-sys
fn petsc_ds_set_jacobian(ds: &mut DS, g3: unsafe extern "C" fn(PetscInt, PetscInt, PetscInt,
    *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
    *const PetscInt, *const PetscInt, *const PetscScalar, *const PetscScalar, *const PetscScalar,
    PetscReal, PetscReal, *const PetscReal, PetscInt, *const PetscScalar, *mut PetscScalar)) -> petsc_rs::Result<()>
{
    let ierr = unsafe { petsc_sys::PetscDSSetJacobian(ds.as_raw(), 0, 0, None, None, None, Some(g3)) };
    if ierr == 0 {
        Ok(())
    } else {
        Petsc::set_error(ds.world(), PetscErrorKind::PETSC_ERR_ARG_CORRUPT, "PetscDSSetJacobian failed")
    }
}

fn setup_problem(dm: &mut DM, opt: &Opt) -> petsc_rs::Result<fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> petsc_rs::Result<()>> {
    let dim = dm.get_dimension()? as usize;
    let periodicity = {
        if let Some((_, _, p)) = dm.get_periodicity()? {
            let mut periodicity = vec![DMBoundaryType::DM_BOUNDARY_NONE; dim];
            periodicity.clone_from_slice(p);
            Some(periodicity)
        } else {
            None
        }
    };
    {
        let mut ds = dm.try_get_ds_mut().unwrap();

        match opt.variable_coefficient {
            CoeffType::COEFF_NONE => {
                if periodicity.is_some() && periodicity.as_ref().unwrap()[0] != DMBoundaryType::DM_BOUNDARY_NONE {
                    if periodicity.as_ref().unwrap()[1] != DMBoundaryType::DM_BOUNDARY_NONE {
                        petsc_ds_set_residual(&mut ds, f0_xytrig_u, f1_u)?;
                        petsc_ds_set_jacobian(&mut ds, g3_uu)?;
                    } else {
                        petsc_ds_set_residual(&mut ds, f0_xtrig_u, f1_u)?;
                        petsc_ds_set_jacobian(&mut ds, g3_uu)?;
                    }
                } else {
                    petsc_ds_set_residual(ds, f0_u, f1_u)?;
                    petsc_ds_set_jacobian(ds, g3_uu)?;
                }
            },
            CoeffType::COEFF_ANALYTIC => {
                petsc_ds_set_residual(ds, f0_analytic_u, f1_analytic_u)?;
                petsc_ds_set_jacobian(ds, g3_analytic_uu)?;
            },
            CoeffType::COEFF_FIELD => {
                petsc_ds_set_residual(ds, f0_analytic_u, f1_field_u)?;
                petsc_ds_set_jacobian(ds, g3_field_uu)?;
            },
            CoeffType::COEFF_NONLINEAR => {
                petsc_ds_set_residual(ds, f0_analytic_nonlinear_u, f1_analytic_nonlinear_u)?;
                petsc_ds_set_jacobian(ds, g3_analytic_nonlinear_uu)?;
            },
            CoeffType::COEFF_CIRCLE => {
                petsc_ds_set_residual(ds, f0_circle_u, f1_u)?;
                petsc_ds_set_jacobian(ds, g3_uu)?;
            },
            CoeffType::COEFF_CROSS => {
                petsc_ds_set_residual(ds, f0_cross_u, f1_u)?;
                petsc_ds_set_jacobian(ds, g3_uu)?;
            },
            CoeffType::COEFF_CHECKERBOARD_0 => {
                petsc_ds_set_residual(ds, f0_checkerboard_0_u, f1_field_u)?;
                petsc_ds_set_jacobian(ds, g3_field_uu)?;
            },
            _ => {
                Petsc::set_error(ds.world(), PetscErrorKind::PETSC_ERR_ARG_WRONG, format!("Invalid variable coefficient type {:?}", opt.variable_coefficient))?;
            },
        }
    } // drop ds

    let exact_func: fn (PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> petsc_rs::Result<()>;

    let id = 1;
    match dim {
        2 => {
            match opt.variable_coefficient {
                CoeffType::COEFF_CIRCLE => { exact_func = circle_u_2d; },
                CoeffType::COEFF_CROSS => { exact_func = cross_u_2d; },
                CoeffType::COEFF_CHECKERBOARD_0 => { exact_func = zero; },
                _ => {
                    if periodicity.is_some() && periodicity.as_ref().unwrap()[0] != DMBoundaryType::DM_BOUNDARY_NONE {
                        if periodicity.as_ref().unwrap()[1] != DMBoundaryType::DM_BOUNDARY_NONE {
                            exact_func = xytrig_u_2d;
                        } else {
                            exact_func = xtrig_u_2d;
                        }
                    } else {
                        exact_func = quadratic_u_2d;
                        // exact_field = quadratic_u_field_2d;
                    }
                }
            }

            if opt.bc_type == BCType::NEUMANN {
                let label = dm.get_label("boundary")?.unwrap();
                let bd = dm.add_boundary_natural("wall", &label, slice::from_ref(&id), 0, &[])?;
                let (mut wf, _, _, _, _, _, _) = dm.try_get_ds_mut().unwrap().get_boundary_info(bd)?;
                petsc_weak_form_set_index_bd_residual(&mut wf, &label, id, f0_bd_u)?;
            }
        },
        3 => {
            exact_func = quadratic_u_3d;

            if opt.bc_type == BCType::NEUMANN {
                let label = dm.get_label("boundary")?.unwrap();
                let bd = dm.add_boundary_natural("wall", &label, slice::from_ref(&id), 0, &[])?;
                let (mut wf, _, _, _, _, _, _) = dm.try_get_ds_mut().unwrap().get_boundary_info(bd)?;
                petsc_weak_form_set_index_bd_residual(&mut wf, &label, id, f0_bd_u)?;
            }
        },
        _ =>  {
            exact_func = xtrig_u_2d;
            Petsc::set_error(dm.world(), PetscErrorKind::PETSC_ERR_ARG_OUTOFRANGE, format!("Invalid dimension {}", dim))?;
        },
    }

    if opt.variable_coefficient == CoeffType::COEFF_CHECKERBOARD_0 {
        dm.try_get_ds_mut().unwrap().set_constants(&[opt.div, opt.k])?;
    }

    dm.try_get_ds_mut().unwrap().set_exact_solution(0, exact_func.clone())?;

    if opt.bc_type == BCType::DIRICHLET {
        if let Some(label) = dm.get_label("marker")? {
            if opt.field_bc { Petsc::set_error(dm.world(), PetscErrorKind::PETSC_ERR_ARG_WRONG, "`field_bd` can't be true")?; }
            let _ = dm.add_boundary_essential("wall", &label, slice::from_ref(&id), 0, &[], exact_func.clone())?;
        } else {
            todo!();
        }

    }

    Ok(exact_func)
}

fn setup_discretization(dm: &mut DM, opt: &Opt, kgrid: &Option<Vec<i32>>) -> petsc_rs::Result<fn(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> petsc_rs::Result<()>> {
    let dim = dm.get_dimension()?;
    // DMConvert(dm, DMPLEX, &plex);
    // DMPlexIsSimplex(plex, &simplex);
    // for now we are just doing this (we assume the dm is a DMPlex)
    let simplex = dm.plex_is_simplex()?;

    let mut fe = Field::create_default(dm.world(), dim, 1, simplex, None, None)?;
    fe.set_name("potential")?;
    
    let fe_aux = if opt.variable_coefficient == CoeffType::COEFF_FIELD || opt.variable_coefficient == CoeffType::COEFF_CHECKERBOARD_1 {
        let mut fe_aux = Field::create_default(dm.world(), dim, 1, simplex, "mat_", None)?;
        fe_aux.set_name("coefficient")?;
        fe_aux.copy_quadrature_from(&fe)?;
        Some(fe_aux)
    } else if opt.field_bc {
        let mut fe_aux = Field::create_default(dm.world(), dim, 1, simplex, "bc_", None)?;
        fe_aux.copy_quadrature_from(&fe)?;
        Some(fe_aux)
    } else {
        None//fe_aux = Field::create_default(dm.world(), dim, 1, simplex, "none_", None)?;
    };

    //let _ = dm.clone_without_closures()?;
    dm.add_field(None, fe)?;
    dm.create_ds()?;
    
    let exact_func = setup_problem(dm, opt)?;

    let _dm_cloned = dm.clone_without_closures()?;
    let mut cdm = dm;
    loop {
        setup_aux_dm(cdm, &fe_aux, opt, kgrid)?;
        if opt.bc_type == BCType::DIRICHLET {
            if !cdm.has_label("marker")? {
                create_bc_label(cdm, "marker", opt)?;
            }
        }
        // TODO: why does this remove the features?
        //cdm.copy_disc_from(&dm_cloned)?;
        if let Some(this_cdm) = cdm.get_coarse_dm_mut()? {
            cdm = this_cdm;
        } else {
            break;
        }
    }

    Ok(exact_func)
}

fn create_bc_label(dm: &mut DM, labelname: &str, _opt: &Opt) -> petsc_rs::Result<()> {
    dm.create_label(labelname)?;
    let mut label = dm.get_label(labelname)?.unwrap();
    // TODO: do convert
    dm.plex_mark_boundary_faces(1, &mut label)
}

fn setup_aux_dm(dm: &mut DM, fe_aux: &Option<Field>, opt: &Opt, kgrid: &Option<Vec<i32>>) -> petsc_rs::Result<()> {
    let dim = dm.get_dimension()?;
    let simplex = dm.plex_is_simplex()?;
    let coord_dm = dm.get_coordinate_dm()?.clone();
    if let Some(fe_aux) = fe_aux {
        // clone the fe_aux
        let this_fe_aux = if opt.variable_coefficient == CoeffType::COEFF_FIELD || opt.variable_coefficient == CoeffType::COEFF_CHECKERBOARD_1 {
            let mut this_fe_aux = Field::create_default(dm.world(), dim, 1, simplex, "mat_", None)?;
            this_fe_aux.set_name("coefficient")?;
            this_fe_aux.copy_quadrature_from(fe_aux)?;
            this_fe_aux
        } else if opt.field_bc {
            let mut this_fe_aux = Field::create_default(dm.world(), dim, 1, simplex, "bc_", None)?;
            this_fe_aux.copy_quadrature_from(fe_aux)?;
            this_fe_aux
        } else {
            return Ok(()); // this_fe_aux = Field::create_default(dm.world(), dim, 0, simplex, "none_", None)?;
        };

        let vec;
        {
            let mut dm_aux = dm.clone_shallow()?;
            dm_aux.set_coordinate_dm(coord_dm)?;
            dm_aux.add_field(None, this_fe_aux)?;
            dm_aux.create_ds()?;
            vec = if opt.field_bc { setup_bc(&mut dm_aux, opt)? }
            else            { setup_material(&mut dm_aux, opt, kgrid)? };
        }

        dm.set_auxiliary_vec(None, 0, vec)?;
    }

    Ok(())
}

fn setup_material<'a>(dm_aux: &mut DM<'a, '_>, opt: &Opt, kgrid: &Option<Vec<i32>>) -> petsc_rs::Result<Vector<'a>> {
    let mut nu = dm_aux.create_local_vector()?;
    nu.set_name("Coefficient")?;
    // Rust has trouble knowing we want Box<dyn _> (and not Box<_>) without the explicate type signature.
    // Thus we need to define funcs outside of the `project_function_local` function call.
    let funcs: [Box<dyn FnMut(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> petsc_rs::Result<()>>; 1]
        = [Box::new(|a,b,c,d,e| {
            if opt.variable_coefficient == CoeffType::COEFF_CHECKERBOARD_0 {
                checkerboard_coeff(a, b, c, d, e, (opt, kgrid))
            } else {
                nu_2d(a,b,c,d,e)
            }
        })];
    dm_aux.project_function_local(0.0, InsertMode::INSERT_ALL_VALUES, &mut nu, funcs)?;
    //dm.set_auxiliary_vec(None, 0, nu)?;

    Ok(nu)
}

fn setup_bc<'a>(dm_aux: &mut DM<'a, '_>, _opt: &Opt) -> petsc_rs::Result<Vector<'a>> {
    let mut uexact = dm_aux.create_local_vector()?;
    let funcs: [Box<dyn FnMut(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> petsc_rs::Result<()>>; 1]
        = [Box::new(|dim,b,c,d,e| {
            if dim == 2 {
                quadratic_u_2d(dim, b, c, d, e)
            } else {
                quadratic_u_3d(dim, b, c, d, e)
            }
        })];
    dm_aux.project_function_local(0.0, InsertMode::INSERT_ALL_VALUES, &mut uexact, funcs)?;
    //dm.set_auxiliary_vec(None, 0, uexact)?;

    Ok(uexact)
}

fn main() -> petsc_rs::Result<()> {
    let petsc = Petsc::builder()
        .args(std::env::args())
        .help_msg(HELP_MSG)
        .init()?;
    
    let opt = Opt::from_petsc(&petsc)?;

    if opt.show_opts {
        petsc_println!(petsc.world(), "Opts: {:#?}", opt)?;
    }

    let (mut dm, kgrid) = create_mesh(petsc.world(), &opt)?;

    let exact_func = setup_discretization(&mut dm, &opt, &kgrid)?;

    let mut u = dm.create_global_vector()?;
    u.set_name("potential")?;

    #[allow(non_snake_case)]
    let J = dm.create_matrix()?;

    #[allow(non_snake_case)]
    let mut A = if opt.jacobian_mf {
        let (mg, ng) = J.get_global_size()?;
        let (m, n) = J.get_local_size()?;
        #[allow(non_snake_case)]
        let mut A = Mat::create(J.world())?;
        A.set_sizes(m, n, mg, ng)?;
        A.set_type_str("shell")?;
        A.set_up()?;

        let mut u_loc = dm.create_local_vector()?;
        let exact_funcs: [Box<dyn FnMut(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> petsc_rs::Result<()>>; 1]
            = [Box::new(exact_func)];
        if opt.field_bc { todo!(); }
        else            { dm.project_function_local(0.0, InsertMode::INSERT_BC_VALUES, &mut u_loc, exact_funcs)?; }

        // MatShellSetContext(A, &userJ);
        // todo!();

        Some(A)
    } else {
        None
    };

    let _nullspace = if opt.bc_type == BCType::DIRICHLET {
        let ns = Rc::new(NullSpace::create(dm.world(), true, vec![])?);
        if let Some(a_mat) = A.as_mut() {
            a_mat.set_nullspace(ns.clone())?;
        }
        Some(ns)
    } else {
        None
    };

    let exact_funcs: [Box<dyn FnMut(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> petsc_rs::Result<()>>; 1]
        = [Box::new(exact_func)];
    if opt.field_bc { todo!(); }
    else            { dm.project_function(0.0, InsertMode::INSERT_BC_VALUES, &mut u, exact_funcs)?; }

    if opt.show_initial {
        let mut local = dm.get_local_vector()?;
        dm.global_to_local(&u, InsertMode::INSERT_VALUES, &mut local)?;
        petsc_println!(petsc.world(), "Local Function:")?;
        petsc_println_all!(petsc.world(), "[process {}]\n{:.5}", petsc.world().rank(), *u.view()?)?;
        u.view_with(None)?;
    }

    let mut snes = SNES::create(petsc.world())?;
    snes.set_dm(dm)?;
    snes.set_from_options()?;

    // TODO: do we want to do: DMPlexSetSNESLocalFEM
    if let Some(a_mat) = A {
        snes.set_jacobian(a_mat, J, |_, _, _, _| { Ok (()) })?;
    } else {
        snes.set_jacobian_single_mat(J, |_, _, _| { Ok (()) })?;
    }

    if opt.run_type == RunType::RUN_FULL || opt.run_type == RunType::RUN_EXACT {
        let initial_guess: Box<dyn FnMut(PetscInt, PetscReal, &[PetscReal], PetscInt, &mut [PetscScalar]) -> petsc_rs::Result<()>>
            = if opt.nonz_init { Box::new(ecks) } else { Box::new(zero) };
        if opt.run_type == RunType::RUN_FULL {
            snes.get_dm_mut()?.project_function(0.0, InsertMode::INSERT_VALUES, &mut u, [initial_guess])?;
        }
        if opt.guess_vec_view {
            u.view_with(None)?;
        }
        snes.solve(None, &mut u)?;
        let soln = snes.get_solution()?;
        let dm = snes.get_dm_mut()?;

        if opt.show_solution {
            let mut local = dm.get_local_vector()?;
            dm.global_to_local(&soln, InsertMode::INSERT_VALUES, &mut local)?;
            petsc_println!(petsc.world(), "Solution:")?;
            petsc_println_all!(petsc.world(), "[process {}]\n{:.5}", petsc.world().rank(), *u.view()?)?;
            u.view_with(None)?;
        }
    } else {
        todo!();
    }

    // TODO:
    /*
    ierr = VecViewFromOptions(u, NULL, "-vec_view");CHKERRQ(ierr);
    {
        Vec nu;

        ierr = DMGetAuxiliaryVec(dm, NULL, 0, &nu);CHKERRQ(ierr);
        if (nu) {ierr = VecViewFromOptions(nu, NULL, "-coeff_view");CHKERRQ(ierr);}
    }

    if (user.bdIntegral) {
        DMLabel   label;
        PetscInt  id = 1;
        PetscScalar bdInt = 0.0;
        PetscReal   exact = 3.3333333333;

        ierr = DMGetLabel(dm, "marker", &label);CHKERRQ(ierr);
        ierr = DMPlexComputeBdIntegral(dm, u, label, 1, &id, bd_integral_2d, &bdInt, NULL);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Solution boundary integral: %.4g\n", (double) PetscAbsScalar(bdInt));CHKERRQ(ierr);
        if (PetscAbsReal(PetscAbsScalar(bdInt) - exact) > PETSC_SQRT_MACHINE_EPSILON) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Invalid boundary integral %g != %g", (double) PetscAbsScalar(bdInt), (double)exact);
    }
    */



    Ok(())
}

const PETSC_PI: PetscReal = std::f64::consts::PI as PetscReal;

fn zero(_dim: PetscInt, _time: PetscReal, _x: &[PetscReal], _nc: PetscInt, u: &mut [PetscScalar]) -> petsc_rs::Result<()>
{
    u[0] = 0.0;
    Ok(())
}

fn ecks(_dim: PetscInt, _time: PetscReal, x: &[PetscReal], _nc: PetscInt, u: &mut [PetscScalar]) -> petsc_rs::Result<()>
{
    u[0] = x[0];
    Ok(())
}

/*
  In 2D for Dirichlet conditions, we use exact solution:

    u = x^2 + y^2
    f = 4

  so that

    -\Delta u + f = -4 + 4 = 0

  For Neumann conditions, we have

    -\nabla u \cdot -\hat y |_{y=0} =  (2y)|_{y=0} =  0 (bottom)
    -\nabla u \cdot  \hat y |_{y=1} = -(2y)|_{y=1} = -2 (top)
    -\nabla u \cdot -\hat x |_{x=0} =  (2x)|_{x=0} =  0 (left)
    -\nabla u \cdot  \hat x |_{x=1} = -(2x)|_{x=1} = -2 (right)

  Which we can express as

    \nabla u \cdot  \hat n|_\Gamma = {2 x, 2 y} \cdot \hat n = 2 (x + y)

  The boundary integral of this solution is (assuming we are not orienting the edges)

    \int^1_0 x^2 dx + \int^1_0 (1 + y^2) dy + \int^1_0 (x^2 + 1) dx + \int^1_0 y^2 dy = 1/3 + 4/3 + 4/3 + 1/3 = 3 1/3
*/
fn quadratic_u_2d(_dim: PetscInt, _time: PetscReal, x: &[PetscReal], _nc: PetscInt, u: &mut [PetscScalar]) -> petsc_rs::Result<()>
{
    u[0] = x[0]*x[0] + x[1]*x[1];
    Ok(())
}

unsafe extern "C" fn quadratic_u_field_2d(_dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: *const PetscInt, _u_off_x: *const PetscInt, _u: *const PetscScalar, _u_t: *const PetscScalar, _u_x: *const PetscScalar,
    _a_off: *const PetscInt, _a_off_x: *const PetscInt, a: *const PetscScalar, _a_t: *const PetscScalar, _a_x: *const PetscScalar,
    _t: PetscReal, _x: *const PetscReal, _nc: PetscInt, _consts: *const PetscScalar, uexact: *mut PetscScalar)
{
    *uexact = *a;
}

fn circle_u_2d(_dim: PetscInt, _time: PetscReal, x: &[PetscReal], _nc: PetscInt, u: &mut [PetscScalar]) -> petsc_rs::Result<()>
{
    let alpha   = 500.0;
    let radius2 = PetscReal::powi(0.15, 2);
    let r2      = (x[0] - 0.5).powi(2) + (x[1] - 0.5).powi(2);
    let xi      = alpha*(radius2 - r2);

    u[0] = PetscScalar::tanh(xi) + 1.0;
    Ok(())
}

fn cross_u_2d(_dim: PetscInt, _time: PetscReal, x: &[PetscReal], _nc: PetscInt, u: &mut [PetscScalar]) -> petsc_rs::Result<()>
{
    let alpha = 50.0*4.0;
    let xy    = (x[0]-0.5)*(x[1]-0.5);
    u[0] = PetscReal::sin(alpha*xy) * if alpha*xy.abs() < 2.0 * PETSC_PI { 1.0 } else { 0.01 };
    Ok(())
}

unsafe extern "C" fn f0_u(_dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: *const PetscInt, _u_off_x: *const PetscInt, _u: *const PetscScalar, _u_t: *const PetscScalar, _u_x: *const PetscScalar,
    _a_off: *const PetscInt, _a_off_x: *const PetscInt, _a: *const PetscScalar, _a_t: *const PetscScalar, _a_x: *const PetscScalar,
    _t: PetscReal, _x: *const PetscReal, _nc: PetscInt, _consts: *const PetscScalar, f0: *mut PetscScalar)
{
    *f0 = 4.0;
}

unsafe extern "C" fn f0_circle_u(dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: *const PetscInt, _u_off_x: *const PetscInt, _u: *const PetscScalar, _u_t: *const PetscScalar, _u_x: *const PetscScalar,
    _a_off: *const PetscInt, _a_off_x: *const PetscInt, _a: *const PetscScalar, _a_t: *const PetscScalar, _a_x: *const PetscScalar,
    _t: PetscReal, x: *const PetscReal, _nc: PetscInt, _consts: *const PetscScalar, f0: *mut PetscScalar)
{
    let f0 = slice::from_raw_parts_mut(f0, dim as usize);
    let x = slice::from_raw_parts(x, dim as usize);

    let alpha   = 500.0;
    let radius2 = PetscReal::powi(0.15, 2);
    let r2      = (x[0] - 0.5).powi(2) + (x[1] - 0.5).powi(2);
    let xi      = alpha*(radius2 - r2);

    f0[0] = (-4.0*alpha - 8.0*alpha.powi(2)*r2*PetscReal::tanh(xi)) * (1.0/PetscReal::cosh(xi)).powi(2);
}

unsafe extern "C" fn f0_cross_u(dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: *const PetscInt, _u_off_x: *const PetscInt, _u: *const PetscScalar, _u_t: *const PetscScalar, _u_x: *const PetscScalar,
    _a_off: *const PetscInt, _a_off_x: *const PetscInt, _a: *const PetscScalar, _a_t: *const PetscScalar, _a_x: *const PetscScalar,
    _t: PetscReal, x: *const PetscReal, _nc: PetscInt, _consts: *const PetscScalar, f0: *mut PetscScalar)
{
    let f0 = slice::from_raw_parts_mut(f0, dim as usize);
    let x = slice::from_raw_parts(x, dim as usize);

    let alpha = 50.0*4.0;
    let xy    = (x[0]-0.5)*(x[1]-0.5);

    f0[0] = PetscReal::sin(alpha*xy) * if alpha*xy.abs() < 2.0 * PETSC_PI { 1.0 } else { 0.01 };
}

unsafe extern "C" fn f0_checkerboard_0_u(dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: *const PetscInt, _u_off_x: *const PetscInt, _u: *const PetscScalar, _u_t: *const PetscScalar, _u_x: *const PetscScalar,
    _a_off: *const PetscInt, _a_off_x: *const PetscInt, _a: *const PetscScalar, _a_t: *const PetscScalar, _a_x: *const PetscScalar,
    _t: PetscReal, x: *const PetscReal, _nc: PetscInt, _consts: *const PetscScalar, f0: *mut PetscScalar)
{
    let f0 = slice::from_raw_parts_mut(f0, dim as usize);
    let x = slice::from_raw_parts(x, dim as usize);

    f0[0] = -20.0*PetscReal::exp(-((x[0] - 0.5).powi(2) + (x[1] - 0.5).powi(2)));
}

unsafe extern "C" fn f0_bd_u(dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: *const PetscInt, _u_off_x: *const PetscInt, _u: *const PetscScalar, _u_t: *const PetscScalar, _u_x: *const PetscScalar,
    _a_off: *const PetscInt, _a_off_x: *const PetscInt, _a: *const PetscScalar, _a_t: *const PetscScalar, _a_x: *const PetscScalar,
    _t: PetscReal, x: *const PetscReal, n: *const PetscReal, _nc: PetscInt, _consts: *const PetscScalar, f0: *mut PetscScalar)
{
    let f0 = slice::from_raw_parts_mut(f0, dim as usize);
    let x = slice::from_raw_parts(x, dim as usize);
    let n = slice::from_raw_parts(n, dim as usize);

    f0[0] = (0..dim as usize).fold(0.0, |res, d| res + -n[d]*2.0*x[d]);
}

/* gradU[comp*dim+d] = {u_x, u_y} or {u_x, u_y, u_z} */
unsafe extern "C" fn f1_u(dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: *const PetscInt, _u_off_x: *const PetscInt, _u: *const PetscScalar, _u_t: *const PetscScalar, u_x: *const PetscScalar,
    _a_off: *const PetscInt, _a_off_x: *const PetscInt, _a: *const PetscScalar, _a_t: *const PetscScalar, _a_x: *const PetscScalar,
    _t: PetscReal, _x: *const PetscReal, _nc: PetscInt, _consts: *const PetscScalar, f1: *mut PetscScalar)
{
    let f1 = slice::from_raw_parts_mut(f1, dim as usize);
    let u_x = slice::from_raw_parts(u_x, dim as usize);

    for d in 0..dim as usize { f1[d] = u_x[d]; }
}

/* < \nabla v, \nabla u + {\nabla u}^T >
   This just gives \nabla u, give the perdiagonal for the transpose */
unsafe extern "C" fn g3_uu(dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: *const PetscInt, _u_off_x: *const PetscInt, _u: *const PetscScalar, _u_t: *const PetscScalar, _u_x: *const PetscScalar,
    _a_off: *const PetscInt, _a_off_x: *const PetscInt, _a: *const PetscScalar, _a_t: *const PetscScalar, _a_x: *const PetscScalar,
    _t: PetscReal, _u_t_shift: PetscReal, _x: *const PetscReal, _nc: PetscInt, _consts: *const PetscScalar, g3: *mut PetscScalar)
{
    let g3 = slice::from_raw_parts_mut(g3, (dim * dim) as usize);

    for d in 0..dim as usize { g3[d*dim as usize+d] = 1.0; }
}

/*
  In 2D for x periodicity and y Dirichlet conditions, we use exact solution:

    u = sin(2 pi x)
    f = -4 pi^2 sin(2 pi x)

  so that

    -\Delta u + f = 4 pi^2 sin(2 pi x) - 4 pi^2 sin(2 pi x) = 0
*/
fn xtrig_u_2d(_dim: PetscInt, _time: PetscReal, x: &[PetscReal], _nc: PetscInt, u: &mut [PetscScalar]) -> petsc_rs::Result<()>
{
    u[0] = PetscReal::sin(2.0*PETSC_PI*x[0]);
    Ok(())
}

unsafe extern "C" fn f0_xtrig_u(dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: *const PetscInt, _u_off_x: *const PetscInt, _u: *const PetscScalar, _u_t: *const PetscScalar, _u_x: *const PetscScalar,
    _a_off: *const PetscInt, _a_off_x: *const PetscInt, _a: *const PetscScalar, _a_t: *const PetscScalar, _a_x: *const PetscScalar,
    _t: PetscReal, x: *const PetscReal, _nc: PetscInt, _consts: *const PetscScalar, f0: *mut PetscScalar)
{
    let f0 = slice::from_raw_parts_mut(f0, dim as usize);
    let x = slice::from_raw_parts(x, dim as usize);

    f0[0] = -4.0*PETSC_PI.powi(2)*PetscReal::sin(2.0*PETSC_PI*x[0]);
}

/*
  In 2D for x-y periodicity, we use exact solution:

    u = sin(2 pi x) sin(2 pi y)
    f = -8 pi^2 sin(2 pi x)

  so that

    -\Delta u + f = 4 pi^2 sin(2 pi x) sin(2 pi y) + 4 pi^2 sin(2 pi x) sin(2 pi y) - 8 pi^2 sin(2 pi x) = 0
*/
fn xytrig_u_2d(_dim: PetscInt, _time: PetscReal, x: &[PetscReal], _nc: PetscInt, u: &mut [PetscScalar]) -> petsc_rs::Result<()>
{
    u[0] = PetscReal::sin(2.0*PETSC_PI*x[0])*PetscReal::sin(2.0*PETSC_PI*x[1]);
    Ok(())
}

unsafe extern "C" fn f0_xytrig_u(_dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: *const PetscInt, _u_off_x: *const PetscInt, _u: *const PetscScalar, _u_t: *const PetscScalar, _u_x: *const PetscScalar,
    _a_off: *const PetscInt, _a_off_x: *const PetscInt, _a: *const PetscScalar, _a_t: *const PetscScalar, _a_x: *const PetscScalar,
    _t: PetscReal, x: *const PetscReal, _nc: PetscInt, _consts: *const PetscScalar, f0: *mut PetscScalar)
{
    *f0 = -8.0*PETSC_PI.powi(2)*PetscReal::sin(2.0*PETSC_PI*(*x));
}

/*
  In 2D for Dirichlet conditions with a variable coefficient, we use exact solution:

    u  = x^2 + y^2
    f  = 6 (x + y)
    nu = (x + y)

  so that

    -\div \nu \grad u + f = -6 (x + y) + 6 (x + y) = 0
*/
fn nu_2d(_dim: PetscInt, _time: PetscReal, x: &[PetscReal], _nc: PetscInt, u: &mut [PetscScalar]) -> petsc_rs::Result<()>
{
  u[0] = x[0] + x[1];
  Ok(())
}

fn checkerboard_coeff(dim: PetscInt, _time: PetscReal, x: &[PetscReal], _nc: PetscInt, u: &mut [PetscScalar], ctx: (&Opt, &Option<Vec<PetscInt>>)) -> petsc_rs::Result<()>
{
    let (opt, kgrid) = ctx;
    let k;
    let mask = (0..dim as usize).fold(0, |mask, d| (mask + (x[d] * opt.div as PetscReal) as PetscInt) % 2);
    if let Some(kgrid) = kgrid {
        let ind = (0..dim as usize).fold(0, |mut ind, d| {
            if d > 0 { ind *= dim }
            ind + (x[d] * opt.div as PetscReal) as PetscInt
        });
        k = kgrid[ind as usize];
    } else {
        k = opt.k;
    }
    u[0] = if mask == 1 { PetscScalar::from(1.0) } else { PetscScalar::from(10.0).powi(-k) };
    
    Ok(())
}

unsafe extern "C" fn f0_analytic_u(dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: *const PetscInt, _u_off_x: *const PetscInt, _u: *const PetscScalar, _u_t: *const PetscScalar, _u_x: *const PetscScalar,
    _a_off: *const PetscInt, _a_off_x: *const PetscInt, _a: *const PetscScalar, _a_t: *const PetscScalar, _a_x: *const PetscScalar,
    _t: PetscReal, x: *const PetscReal, _nc: PetscInt, _consts: *const PetscScalar, f0: *mut PetscScalar)
{
    let f0 = slice::from_raw_parts_mut(f0, dim as usize);
    let x = slice::from_raw_parts(x, dim as usize);

    f0[0] = 6.0*(x[0] + x[1]);
}

/* gradU[comp*dim+d] = {u_x, u_y} or {u_x, u_y, u_z} */
unsafe extern "C" fn f1_analytic_u(dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: *const PetscInt, _u_off_x: *const PetscInt, _u: *const PetscScalar, _u_t: *const PetscScalar, u_x: *const PetscScalar,
    _a_off: *const PetscInt, _a_off_x: *const PetscInt, _a: *const PetscScalar, _a_t: *const PetscScalar, _a_x: *const PetscScalar,
    _t: PetscReal, x: *const PetscReal, _nc: PetscInt, _consts: *const PetscScalar, f1: *mut PetscScalar)
{
    let f1 = slice::from_raw_parts_mut(f1, dim as usize);
    let x = slice::from_raw_parts(x, dim as usize);
    let u_x = slice::from_raw_parts(u_x, dim as usize);

    for d in 0..dim as usize { f1[d] = (x[0] + x[1])*u_x[d]; }
}

unsafe extern "C" fn f1_field_u(dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: *const PetscInt, _u_off_x: *const PetscInt, _u: *const PetscScalar, _u_t: *const PetscScalar, u_x: *const PetscScalar,
    _a_off: *const PetscInt, _a_off_x: *const PetscInt, a: *const PetscScalar, _a_t: *const PetscScalar, _a_x: *const PetscScalar,
    _t: PetscReal, _x: *const PetscReal, _nc: PetscInt, _consts: *const PetscScalar, f1: *mut PetscScalar)
{
    let f1 = slice::from_raw_parts_mut(f1, dim as usize);
    let a = slice::from_raw_parts(a, dim as usize);
    let u_x = slice::from_raw_parts(u_x, dim as usize);

    for d in 0..dim as usize { f1[d] = a[0]*u_x[d]; }
}

/* < \nabla v, \nabla u + {\nabla u}^T >
   This just gives \nabla u, give the perdiagonal for the transpose */
unsafe extern "C" fn g3_analytic_uu(dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: *const PetscInt, _u_off_x: *const PetscInt, _u: *const PetscScalar, _u_t: *const PetscScalar, _u_x: *const PetscScalar,
    _a_off: *const PetscInt, _a_off_x: *const PetscInt, _a: *const PetscScalar, _a_t: *const PetscScalar, _a_x: *const PetscScalar,
    _t: PetscReal, _u_t_shift: PetscReal, x: *const PetscReal, _nc: PetscInt, _consts: *const PetscScalar, g3: *mut PetscScalar)
{
    let g3 = slice::from_raw_parts_mut(g3, (dim * dim) as usize);
    let x = slice::from_raw_parts(x, dim as usize);

    for d in 0..dim as usize { g3[d*dim as usize+d] = x[0] + x[1]; }
}

unsafe extern "C" fn g3_field_uu(dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: *const PetscInt, _u_off_x: *const PetscInt, _u: *const PetscScalar, _u_t: *const PetscScalar, _u_x: *const PetscScalar,
    _a_off: *const PetscInt, _a_off_x: *const PetscInt, a: *const PetscScalar, _a_t: *const PetscScalar, _a_x: *const PetscScalar,
    _t: PetscReal, _u_t_shift: PetscReal, _x: *const PetscReal, _nc: PetscInt, _consts: *const PetscScalar, g3: *mut PetscScalar)
{
    let g3 = slice::from_raw_parts_mut(g3, (dim * dim) as usize);
    let a = slice::from_raw_parts(a, dim as usize);

    for d in 0..dim as usize { g3[d*dim as usize+d] = a[0]; }
}

/*
  In 2D for Dirichlet conditions with a nonlinear coefficient (p-Laplacian with p = 4), we use exact solution:

    u  = x^2 + y^2
    f  = 16 (x^2 + y^2)
    nu = 1/2 |grad u|^2

  so that

    -\div \nu \grad u + f = -16 (x^2 + y^2) + 16 (x^2 + y^2) = 0
*/
unsafe extern "C" fn f0_analytic_nonlinear_u(dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: *const PetscInt, _u_off_x: *const PetscInt, _u: *const PetscScalar, _u_t: *const PetscScalar, _u_x: *const PetscScalar,
    _a_off: *const PetscInt, _a_off_x: *const PetscInt, _a: *const PetscScalar, _a_t: *const PetscScalar, _a_x: *const PetscScalar,
    _t: PetscReal, x: *const PetscReal, _nc: PetscInt, _consts: *const PetscScalar, f0: *mut PetscScalar)
{
    let f0 = slice::from_raw_parts_mut(f0, dim as usize);
    let x = slice::from_raw_parts(x, dim as usize);

    f0[0] = 16.0*(x[0]*x[0] + x[1]*x[1]);
}

/* gradU[comp*dim+d] = {u_x, u_y} or {u_x, u_y, u_z} */
unsafe extern "C" fn f1_analytic_nonlinear_u(dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: *const PetscInt, _u_off_x: *const PetscInt, _u: *const PetscScalar, _u_t: *const PetscScalar, u_x: *const PetscScalar,
    _a_off: *const PetscInt, _a_off_x: *const PetscInt, _a: *const PetscScalar, _a_t: *const PetscScalar, _a_x: *const PetscScalar,
    _t: PetscReal, _x: *const PetscReal, _nc: PetscInt, _consts: *const PetscScalar, f1: *mut PetscScalar)
{
    let f1 = slice::from_raw_parts_mut(f1, dim as usize);
    let u_x = slice::from_raw_parts(u_x, dim as usize);

    let nu = (0..dim as usize).fold(0.0, |nu, d| nu + u_x[d]*u_x[d]);
    for d in 0..dim as usize { f1[d] = 0.5*nu*u_x[d]; }
}

/*
  grad (u + eps w) - grad u = eps grad w

  1/2 |grad (u + eps w)|^2 grad (u + eps w) - 1/2 |grad u|^2 grad u
= 1/2 (|grad u|^2 + 2 eps <grad u,grad w>) (grad u + eps grad w) - 1/2 |grad u|^2 grad u
= 1/2 (eps |grad u|^2 grad w + 2 eps <grad u,grad w> grad u)
= eps (1/2 |grad u|^2 grad w + grad u <grad u,grad w>)
*/
unsafe extern "C" fn g3_analytic_nonlinear_uu(dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: *const PetscInt, _u_off_x: *const PetscInt, _u: *const PetscScalar, _u_t: *const PetscScalar, u_x: *const PetscScalar,
    _a_off: *const PetscInt, _a_off_x: *const PetscInt, _a: *const PetscScalar, _a_t: *const PetscScalar, _a_x: *const PetscScalar,
    _t: PetscReal, _u_t_shift: PetscReal, _x: *const PetscReal, _nc: PetscInt, _consts: *const PetscScalar, g3: *mut PetscScalar)
{
    let g3 = slice::from_raw_parts_mut(g3, (dim * dim) as usize);
    let u_x = slice::from_raw_parts(u_x, dim as usize);

    let nu = (0..dim as usize).fold(0.0, |nu, d| nu + u_x[d]*u_x[d]);
    for d in 0..dim as usize {
        g3[d*dim as usize+d] = 0.5*nu;
        for e in 0..dim as usize {
            g3[d*dim as usize+e] += u_x[d]*u_x[e];
        }
    }
}

/*
  In 3D for Dirichlet conditions we use exact solution:

    u = 2/3 (x^2 + y^2 + z^2)
    f = 4

  so that

    -\Delta u + f = -2/3 * 6 + 4 = 0

  For Neumann conditions, we have

    -\nabla u \cdot -\hat z |_{z=0} =  (2z)|_{z=0} =  0 (bottom)
    -\nabla u \cdot  \hat z |_{z=1} = -(2z)|_{z=1} = -2 (top)
    -\nabla u \cdot -\hat y |_{y=0} =  (2y)|_{y=0} =  0 (front)
    -\nabla u \cdot  \hat y |_{y=1} = -(2y)|_{y=1} = -2 (back)
    -\nabla u \cdot -\hat x |_{x=0} =  (2x)|_{x=0} =  0 (left)
    -\nabla u \cdot  \hat x |_{x=1} = -(2x)|_{x=1} = -2 (right)

  Which we can express as

    \nabla u \cdot  \hat n|_\Gamma = {2 x, 2 y, 2z} \cdot \hat n = 2 (x + y + z)
*/
fn quadratic_u_3d(_dim: PetscInt, _time: PetscReal, x: &[PetscReal], _nc: PetscInt, u: &mut [PetscScalar]) -> petsc_rs::Result<()>
{
  u[0] = 2.0*(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])/3.0;
  Ok(())
}

fn quadratic_u_field_3d(_dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: &[PetscInt], _u_off_x: &[PetscInt], _u: &[PetscScalar], _u_t: &[PetscScalar], _u_x: &[PetscScalar],
    _a_off: &[PetscInt], _a_off_x: &[PetscInt], a: &[PetscScalar], _a_t: &[PetscScalar], _a_x: &[PetscScalar],
    _t: PetscReal, _x: &[PetscReal], _nc: PetscInt, _consts: &[PetscScalar], uexact: &mut [PetscScalar])
{
    uexact[0] = a[0];
}

fn bd_integral_2d(_dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: &[PetscInt], _u_off_x: &[PetscInt], u: &[PetscScalar], _u_t: &[PetscScalar], _u_x: &[PetscScalar],
    _a_off: &[PetscInt], _a_off_x: &[PetscInt], _a: &[PetscScalar], _a_t: &[PetscScalar], _a_x: &[PetscScalar],
    _t: PetscReal, _x: &[PetscReal], _n: &[PetscReal], _nc: PetscInt, _consts: &[PetscScalar], uint: &mut [PetscScalar])
{
    uint[0] = u[0];
}