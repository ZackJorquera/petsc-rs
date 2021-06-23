//! This file will show how to do the dm ex1 example in rust using the petsc-rs bindings.
//!
//! Concepts: DM^creating vectors with a DMDA
//! Processors: n
//!
//!   [`Vec::view_with()`] on DMDA vectors first puts the Vec elements into global natural ordering before printing (or plotting)
//! them. In 2d 5 by 2 DMDA this means the numbering is
//!
//!      5   6   7   8   9        0   5
//!      0   1   2   3   4        1   6
//!                          or   2   7
//!                               3   8
//!                               4   9
//!
//! Now the default split across 2 processors with the DM  is (by rank)
//!
//!     0  0   0  1   1           0   0
//!     0  0   0  1   1           0   0
//!                          or   0   0
//!                               1   1
//!                               1   1
//!
//! So the global PETSc ordering is
//!
//!     3  4  5   8  9            0   3
//!     0  1  2   6  7            1   4
//!                          or   2   5
//!                               6   8
//!                               7   9
//!
//! If we filled the vector with global PETSc ordering, i.e. the above, when we use [`Vec::view_with()`]
//! we would see the following order
//!
//!     Process [0]
//!     0 1 2 6 7 3
//!     Process [1]
//!     4 5 8 9
//!
//! We can also print out the correct 2d DMDA vector using [`DM::da_vec_view()`] in the following way:
//! `petsc_println_all!(petsc.world(), "(Process: {}) global vec:\n{:?}", petsc.world().rank(), dm.da_vec_view(&global)?);`
//!
//! ```text
//! (Process: 0) global vec: 
//! [[0.0, 3.0],
//!  [1.0, 4.0],
//!  [2.0, 5.0]], shape=[3, 2], strides=[1, 3], layout=Ff (0xa), dynamic ndim=2
//! (Process: 1) global vec:
//! [[9.0, 8.0],
//!  [7.0, 9.0]], shape=[2, 2], strides=[1, 2], layout=Ff (0xa), dynamic ndim=2
//! ```
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

static HELP_MSG: &str = "Tests VecView() contour plotting for 2d DMDAs.\n\n";

use petsc_rs::prelude::*;

fn main() -> petsc_rs::Result<()> {
    let star_stencil = false;
    let view_global = true;

    let (m, n) = (10, 8);

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

    global.set_all(-3.0)?;

    dm.global_to_local(&global, InsertMode::INSERT_VALUES, &mut local)?;

    local.scale(petsc.world().rank() as f64 + 1.0)?;
    dm.local_to_global(&local, InsertMode::ADD_VALUES, &mut global)?;

    dm.view_with(Some(&viewer))?;
    if view_global {
        petsc_println_all!(petsc.world(), "(Process: {}) global vec (flat):\n{:?}", petsc.world().rank(), global.view()?);

        petsc_println_all!(petsc.world(), "(Process: {}) global vec:\n{:?}", petsc.world().rank(), dm.da_vec_view(&global)?);

        // Note, this might print the vector in a different order than the above two
        global.view_with(Some(&viewer))?;
    }

    // return
    Ok(())
}
