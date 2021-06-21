//! This file will show how to do the dm ex1 example in rust using the petsc-rs bindings.
//!
//! Concepts: DM^creating vectors with a DMDA
//! Processors: n
//!
//!   [`Vec::view_with()`] on DMDA vectors first puts the Vec elements into global natural ordering before printing (or plotting)
//! them. In 2d 5 by 2 DMDA this means the numbering is
//!
//!      5  6   7   8   9
//!      0  1   2   3   4
//!
//! Now the default split across 2 processors with the DM  is (by rank)
//!
//!     0  0   0  1   1
//!     0  0   0  1   1
//!
//! So the global PETSc ordering is
//!
//!     3  4  5   8  9
//!     0  1  2   6  7
//!
//! Use the options
//!      -da_grid_x <nx> - number of grid points in x direction, if M < 0
//!      -da_grid_y <ny> - number of grid points in y direction, if N < 0
//!      -da_processors_x <MX> number of processors in x direction
//!      -da_processors_y <MY> number of processors in x direction
//!
//! To run:
//! ```text
//! $ cargo build --bin ex1
//! $ target/debug/ex1
//! ```

// TODO: this is not the example ex1.c (remove all code i added to test with)

static HELP_MSG: &str = "Tests VecView() contour plotting for 2d DMDAs.\n\n";

use petsc_rs::prelude::*;

fn main() -> petsc_rs::Result<()> {
    let star_stencil = false;
    let view_global = true;

    let (m, n) = (10, 8);

    // optionally initialize mpi
    let univ = mpi::initialize().unwrap();
    // init with options
    let petsc = Petsc::builder()
        .args(std::env::args())
        .help_msg(HELP_MSG)
        .init()?;
    
    let viewer = petsc.viewer_create_ascii_stdout()?;

    /* Create distributed array and get vectors */
    let stype = if star_stencil {DMDAStencilType::DMDA_STENCIL_STAR} else {DMDAStencilType::DMDA_STENCIL_BOX};
    let mut dm = DM::da_create_2d(petsc.world(), DMBoundaryType::DM_BOUNDARY_NONE, DMBoundaryType::DM_BOUNDARY_NONE,
        stype, m, n, None, None, 1, 1, None, None)?;
    dm.set_from_options()?;
    dm.set_up()?;

    let mut global = dm.create_global_vector()?;
    let mut local = dm.create_local_vector()?;

    let gs = global.get_global_size()?;
    let osr = global.get_ownership_range()?;
    global.assemble_with((0..gs)
            .filter(|i| osr.contains(i))
            .map(|i| (i, i as f64)),
        InsertMode::INSERT_VALUES)?;

    //global.set_all(1.0)?;

    { dm.da_vec_view_mut(&mut global)?[[1,2]] = 0.0; }

    petsc_println!(petsc.world(), "global vec:");
    for i in 0..petsc.world().size() {
        univ.world().barrier();
        if petsc.world().rank() == i {
            println!("(rank: {})\n{:?}", i, dm.da_vec_view(&global)?);
        }
        univ.world().barrier();
    }
    univ.world().barrier();

    global.view_with(&viewer)?;

    dm.global_to_local(&global, InsertMode::INSERT_VALUES, &mut local)?;

    local.scale(petsc.world().rank() as f64 + 1.0)?;
    dm.local_to_global(&local, InsertMode::ADD_VALUES, &mut global)?;

    dm.view_with(&viewer)?;
    if view_global {
        petsc_println!(petsc.world(), "(rank: {}) global vec: {:?}", petsc.world().rank(), global.view()?);

        petsc_println!(petsc.world(), "global vec:");
        for i in 0..petsc.world().size() {
            petsc_println!(petsc.world(), "(rank: {})", i);
            if petsc.world().rank() == i {
                println!("{:?}", dm.da_vec_view(&global)?);
            }
            univ.world().barrier();
        }

        global.view_with(&viewer)?;
    }

    // return
    Ok(())
}
