//! This file will show how to do the vec ex1 example in rust using the petsc-rs bindings.
//!
//! Concepts: vectors^assembling vectors;
//! Processors: n
//!
//! Use "petsc_rs::prelude::*" to get direct access to all important petsc-rs bindings
//!     and mpi traits which allow you to call things like `world.size()`.
//!
//! To run:
//! ```text
//! $ cargo build --bin vec-ex2
//! $ mpiexec -n 1 target/debug/vec-ex2
//! $ mpiexec -n 5 target/debug/vec-ex2
//! ```

static HELP_MSG: &'static str = "Builds a parallel vector with 1 component on the first processor, 2 on the second, etc.\n\
    Then each processor adds one to all elements except the last rank.\n\n";

use petsc_rs::prelude::*;
use mpi::traits::*;

fn main() -> petsc_rs::Result<()> {
    // optionally initialize mpi
    // let _univ = mpi::initialize().unwrap();
    // init with options
    let petsc = Petsc::builder()
        .args(std::env::args())
        .help_msg(HELP_MSG)
        .init()?;

    // or init with no options
    // let petsc = Petsc::init_no_args()?;

    let rank = petsc.world().rank() as PetscInt;

    let mut x = petsc.vec_create()?;
    x.set_sizes(Some(rank+1), None)?;
    x.set_from_options()?;
    let size = x.get_global_size()?;
    x.set_all(PetscScalar::from(1.0))?;
    // x.set_all(PetscScalar {re: 1.0, im: 1.0})?;

    x.assemble_with((0..size-rank).map(|i| (i, PetscScalar::from(1.0))), InsertMode::ADD_VALUES)?;

    let viewer = Viewer::create_ascii_stdout(petsc.world())?;
    x.view_with(Some(&viewer))?;

    // return
    Ok(())
}
