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

use std::{
    fmt::{self, Display},
    io::{Error, ErrorKind},
    str::FromStr
};

use petsc_rs::prelude::*;
use mpi::topology::UserCommunicator;
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

fn setup_problem(dm: &mut DM, opt: &Opt) -> petsc_rs::Result<()> {
    let dim = dm.get_dimension()? as usize;
    let periodicity = {
        let (_, _, _, p) = dm.get_periodicity()?;
        let mut periodicity = vec![DMBoundaryType::DM_BOUNDARY_NONE; dim];
        periodicity.clone_from_slice(p);
        periodicity
    };
    let ds = dm.try_get_ds_mut().unwrap();

    match opt.variable_coefficient {
        CoeffType::COEFF_NONE => {
            if periodicity[0] != DMBoundaryType::DM_BOUNDARY_NONE {
                if periodicity[1] != DMBoundaryType::DM_BOUNDARY_NONE {
                    todo!()
                } else {
                    todo!()
                }
            } else {
                todo!()
            }
        },
        _ => todo!()
    }

    todo!()
}

fn SetupDiscretization(dm: &mut DM, opt: &Opt) -> petsc_rs::Result<()> {
    let dim = dm.get_dimension()?;
    // DMConvert(dm, DMPLEX, &plex);
    // DMPlexIsSimplex(plex, &simplex);
    // for now we are just doing this (we assume the dm is a DMPlex)
    let simplex = dm.plex_is_simplex()?;

    let mut fe = Field::create_default(dm.world(), dim, 1, simplex, None, None)?;
    fe.set_name("potential")?;
    if opt.variable_coefficient == CoeffType::COEFF_FIELD || opt.variable_coefficient == CoeffType::COEFF_CHECKERBOARD_1 {
        let mut fe_aux = Field::create_default(dm.world(), dim, 1, simplex, "mat_", None)?;
        fe_aux.set_name("coefficient")?;
        fe_aux.copy_quadrature_from(&fe)?;
    } else if opt.field_bc {
        let mut fe_aux = Field::create_default(dm.world(), dim, 1, simplex, "bc_", None)?;
        fe_aux.copy_quadrature_from(&fe)?;
    }

    dm.add_field(None, fe)?;
    dm.create_ds()?;

    setup_problem(dm, opt)?;

    // TODO: SetupAuxDM

    todo!()
}

fn main() -> petsc_rs::Result<()> {
    let petsc = Petsc::builder()
        .args(std::env::args())
        .help_msg(HELP_MSG)
        .init()?;
    
    let opt = Opt::from_petsc(&petsc)?;

    let (dm, kgrid) = create_mesh(petsc.world(), &opt)?;

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

fn quadratic_u_field_2d(_dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: &[PetscInt], _u_off_x: &[PetscInt], _u: &[PetscScalar], _u_t: &[PetscScalar], _u_x: &[PetscScalar],
    _a_off: &[PetscInt], _a_off_x: &[PetscInt], a: &[PetscScalar], _a_t: &[PetscScalar], _a_x: &[PetscScalar],
    _t: PetscReal, _x: &[PetscReal], _nc: PetscInt, _consts: &[PetscScalar], uexact: &mut [PetscScalar])
{
    uexact[0] = a[0];
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

fn f0_u(_dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: &[PetscInt], _u_off_x: &[PetscInt], _u: &[PetscScalar], _u_t: &[PetscScalar], _u_x: &[PetscScalar],
    _a_off: &[PetscInt], _a_off_x: &[PetscInt], _a: &[PetscScalar], _a_t: &[PetscScalar], _a_x: &[PetscScalar],
    _t: PetscReal, _x: &[PetscReal], _nc: PetscInt, _consts: &[PetscScalar], f0: &mut [PetscScalar])
{
    f0[0] = 4.0;
}

fn f0_circle_u(_dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: &[PetscInt], _u_off_x: &[PetscInt], _u: &[PetscScalar], _u_t: &[PetscScalar], _u_x: &[PetscScalar],
    _a_off: &[PetscInt], _a_off_x: &[PetscInt], _a: &[PetscScalar], _a_t: &[PetscScalar], _a_x: &[PetscScalar],
    _t: PetscReal, x: &[PetscReal], _nc: PetscInt, _consts: &[PetscScalar], f0: &mut [PetscScalar])
{
    let alpha   = 500.0;
    let radius2 = PetscReal::powi(0.15, 2);
    let r2      = (x[0] - 0.5).powi(2) + (x[1] - 0.5).powi(2);
    let xi      = alpha*(radius2 - r2);

    f0[0] = (-4.0*alpha - 8.0*alpha.powi(2)*r2*PetscReal::tanh(xi)) * (1.0/PetscReal::cosh(xi)).powi(2);
}

fn f0_cross_u(_dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: &[PetscInt], _u_off_x: &[PetscInt], _u: &[PetscScalar], _u_t: &[PetscScalar], _u_x: &[PetscScalar],
    _a_off: &[PetscInt], _a_off_x: &[PetscInt], _a: &[PetscScalar], _a_t: &[PetscScalar], _a_x: &[PetscScalar],
    _t: PetscReal, x: &[PetscReal], _nc: PetscInt, _consts: &[PetscScalar], f0: &mut [PetscScalar])
{
    let alpha = 50.0*4.0;
    let xy    = (x[0]-0.5)*(x[1]-0.5);

    f0[0] = PetscReal::sin(alpha*xy) * if alpha*xy.abs() < 2.0 * PETSC_PI { 1.0 } else { 0.01 };
}

fn f0_checkerboard_0_u(_dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: &[PetscInt], _u_off_x: &[PetscInt], _u: &[PetscScalar], _u_t: &[PetscScalar], _u_x: &[PetscScalar],
    _a_off: &[PetscInt], _a_off_x: &[PetscInt], _a: &[PetscScalar], _a_t: &[PetscScalar], _a_x: &[PetscScalar],
    _t: PetscReal, x: &[PetscReal], _nc: PetscInt, _consts: &[PetscScalar], f0: &mut [PetscScalar])
{
    f0[0] = -20.0*PetscReal::exp(-((x[0] - 0.5).powi(2) + (x[1] - 0.5).powi(2)));
}

fn f0_bd_u(dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: &[PetscInt], _u_off_x: &[PetscInt], _u: &[PetscScalar], _u_t: &[PetscScalar], _u_x: &[PetscScalar],
    _a_off: &[PetscInt], _a_off_x: &[PetscInt], _a: &[PetscScalar], _a_t: &[PetscScalar], _a_x: &[PetscScalar],
    _t: PetscReal, x: &[PetscReal], n: &[PetscReal], _nc: PetscInt, _consts: &[PetscScalar], f0: &mut [PetscScalar])
{
    f0[0] = (0..dim as usize).fold(0.0, |res, d| res + -n[d]*2.0*x[d]);
}

/* gradU[comp*dim+d] = {u_x, u_y} or {u_x, u_y, u_z} */
fn f1_u(dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: &[PetscInt], _u_off_x: &[PetscInt], _u: &[PetscScalar], _u_t: &[PetscScalar], u_x: &[PetscScalar],
    _a_off: &[PetscInt], _a_off_x: &[PetscInt], _a: &[PetscScalar], _a_t: &[PetscScalar], _a_x: &[PetscScalar],
    _t: PetscReal, _x: &[PetscReal], _nc: PetscInt, _consts: &[PetscScalar], f1: &mut [PetscScalar])
{
    for d in 0..dim as usize { f1[d] = u_x[d]; }
}

/* < \nabla v, \nabla u + {\nabla u}^T >
   This just gives \nabla u, give the perdiagonal for the transpose */
fn g3_uu(dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: &[PetscInt], _u_off_x: &[PetscInt], _u: &[PetscScalar], _u_t: &[PetscScalar], _u_x: &[PetscScalar],
    _a_off: &[PetscInt], _a_off_x: &[PetscInt], _a: &[PetscScalar], _a_t: &[PetscScalar], _a_x: &[PetscScalar],
    _t: PetscReal, _u_t_shift: PetscReal, _x: &[PetscReal], _nc: PetscInt, _consts: &[PetscScalar], g3: &mut [PetscScalar])
{
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

fn f0_xtrig_u(_dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: &[PetscInt], _u_off_x: &[PetscInt], _u: &[PetscScalar], _u_t: &[PetscScalar], _u_x: &[PetscScalar],
    _a_off: &[PetscInt], _a_off_x: &[PetscInt], _a: &[PetscScalar], _a_t: &[PetscScalar], _a_x: &[PetscScalar],
    _t: PetscReal, x: &[PetscReal], _nc: PetscInt, _consts: &[PetscScalar], f0: &mut [PetscScalar])
{
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

fn f0_xytrig_u(_dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: &[PetscInt], _u_off_x: &[PetscInt], _u: &[PetscScalar], _u_t: &[PetscScalar], _u_x: &[PetscScalar],
    _a_off: &[PetscInt], _a_off_x: &[PetscInt], _a: &[PetscScalar], _a_t: &[PetscScalar], _a_x: &[PetscScalar],
    _t: PetscReal, x: &[PetscReal], _nc: PetscInt, _consts: &[PetscScalar], f0: &mut [PetscScalar])
{
    f0[0] = -8.0*PETSC_PI.powi(2)*PetscReal::sin(2.0*PETSC_PI*x[0]);
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

fn f0_analytic_u(_dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: &[PetscInt], _u_off_x: &[PetscInt], _u: &[PetscScalar], _u_t: &[PetscScalar], _u_x: &[PetscScalar],
    _a_off: &[PetscInt], _a_off_x: &[PetscInt], _a: &[PetscScalar], _a_t: &[PetscScalar], _a_x: &[PetscScalar],
    _t: PetscReal, x: &[PetscReal], _nc: PetscInt, _consts: &[PetscScalar], f0: &mut [PetscScalar])
{
  f0[0] = 6.0*(x[0] + x[1]);
}

/* gradU[comp*dim+d] = {u_x, u_y} or {u_x, u_y, u_z} */
fn f1_analytic_u(dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: &[PetscInt], _u_off_x: &[PetscInt], _u: &[PetscScalar], _u_t: &[PetscScalar], u_x: &[PetscScalar],
    _a_off: &[PetscInt], _a_off_x: &[PetscInt], _a: &[PetscScalar], _a_t: &[PetscScalar], _a_x: &[PetscScalar],
    _t: PetscReal, x: &[PetscReal], _nc: PetscInt, _consts: &[PetscScalar], f1: &mut [PetscScalar])
{
    for d in 0..dim as usize { f1[d] = (x[0] + x[1])*u_x[d]; }
}

fn f1_field_u(dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: &[PetscInt], _u_off_x: &[PetscInt], _u: &[PetscScalar], _u_t: &[PetscScalar], u_x: &[PetscScalar],
    _a_off: &[PetscInt], _a_off_x: &[PetscInt], a: &[PetscScalar], _a_t: &[PetscScalar], _a_x: &[PetscScalar],
    _t: PetscReal, _x: &[PetscReal], _nc: PetscInt, _consts: &[PetscScalar], f1: &mut [PetscScalar])
{
    for d in 0..dim as usize { f1[d] = a[0]*u_x[d]; }
}

/* < \nabla v, \nabla u + {\nabla u}^T >
   This just gives \nabla u, give the perdiagonal for the transpose */
fn g3_analytic_uu(dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: &[PetscInt], _u_off_x: &[PetscInt], _u: &[PetscScalar], _u_t: &[PetscScalar], _u_x: &[PetscScalar],
    _a_off: &[PetscInt], _a_off_x: &[PetscInt], _a: &[PetscScalar], _a_t: &[PetscScalar], _a_x: &[PetscScalar],
    _t: PetscReal, _u_t_shift: PetscReal, x: &[PetscReal], _nc: PetscInt, _consts: &[PetscScalar], g3: &mut [PetscScalar])
{
    for d in 0..dim as usize { g3[d*dim as usize+d] = x[0] + x[1]; }
}

fn g3_field_uu(dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: &[PetscInt], _u_off_x: &[PetscInt], _u: &[PetscScalar], _u_t: &[PetscScalar], _u_x: &[PetscScalar],
    _a_off: &[PetscInt], _a_off_x: &[PetscInt], a: &[PetscScalar], _a_t: &[PetscScalar], _a_x: &[PetscScalar],
    _t: PetscReal, _u_t_shift: PetscReal, _x: &[PetscReal], _nc: PetscInt, _consts: &[PetscScalar], g3: &mut [PetscScalar])
{
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
fn f0_analytic_nonlinear_u(_dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: &[PetscInt], _u_off_x: &[PetscInt], _u: &[PetscScalar], _u_t: &[PetscScalar], _u_x: &[PetscScalar],
    _a_off: &[PetscInt], _a_off_x: &[PetscInt], _a: &[PetscScalar], _a_t: &[PetscScalar], _a_x: &[PetscScalar],
    _t: PetscReal, x: &[PetscReal], _nc: PetscInt, _consts: &[PetscScalar], f0: &mut [PetscScalar])
{
  f0[0] = 16.0*(x[0]*x[0] + x[1]*x[1]);
}

/* gradU[comp*dim+d] = {u_x, u_y} or {u_x, u_y, u_z} */
fn f1_analytic_nonlinear_u(dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: &[PetscInt], _u_off_x: &[PetscInt], _u: &[PetscScalar], _u_t: &[PetscScalar], u_x: &[PetscScalar],
    _a_off: &[PetscInt], _a_off_x: &[PetscInt], _a: &[PetscScalar], _a_t: &[PetscScalar], _a_x: &[PetscScalar],
    _t: PetscReal, _x: &[PetscReal], _nc: PetscInt, _consts: &[PetscScalar], f1: &mut [PetscScalar])
{
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
fn g3_analytic_nonlinear_uu(dim: PetscInt, _nf: PetscInt, _nf_aux: PetscInt,
    _u_off: &[PetscInt], _u_off_x: &[PetscInt], _u: &[PetscScalar], _u_t: &[PetscScalar], u_x: &[PetscScalar],
    _a_off: &[PetscInt], _a_off_x: &[PetscInt], _a: &[PetscScalar], _a_t: &[PetscScalar], _a_x: &[PetscScalar],
    _t: PetscReal, _u_t_shift: PetscReal, _x: &[PetscReal], _nc: PetscInt, _consts: &[PetscScalar], g3: &mut [PetscScalar])
{
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
