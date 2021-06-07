// This file will show how to do the kps ex1 example in rust using the raw petsc bindings.
// As well as some higher level bindings to handle errors and stuff.
//
// Concepts: KSP^solving a system of linear equations
// Processors: 1
//
// TODO: finish disc
// Use "petsc_rs::prelude::*"  so that we can use mpi.
// Include "petsc_rs::petsc_raw" so that we can raw bindings.  Note that will
// automatically includes all petsc function from petsc.h

static HELP_MSG: &'static str = "Solves a tridiagonal linear system with KSP.\n\n";

use petsc_rs::prelude::*;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "ex1", about = HELP_MSG)]
struct Opt {
    /// Size of the vector and matrix
    #[structopt(short, long, default_value = "10")]
    num_elems: i32,

    /// use -help for petsc help
    #[structopt(subcommand)]
    sub: Option<PetscOpt>,
}

#[derive(Debug, PartialEq, StructOpt)]
enum PetscOpt {
    /// use -help for petsc help
    #[structopt(name = "Petsc Args", external_subcommand)]
    PetscArgs(Vec<String>),
}

impl PetscOpt
{
    fn petsc_args(self_op: Option<Self>) -> Vec<String>
    {
        match self_op
        {
            Some(PetscOpt::PetscArgs(mut vec)) => {
                vec.push(std::env::args().next().unwrap());
                vec.rotate_right(1);
                vec
            },
            _ => vec![std::env::args().next().unwrap()]
        }
    }
}

fn main() -> petsc_rs::Result<()> {
    let Opt {num_elems: _n, sub: ext_args} = Opt::from_args();
    let petsc_args = PetscOpt::petsc_args(ext_args); // Is there an easier way to do this

    // optionally initialize mpi
    let _univ = mpi::initialize().unwrap();
    let petsc = Petsc::builder()
        .args(petsc_args)
        .help_msg(HELP_MSG)
        .init()?;

    petsc_println!(petsc,
        "Hello parallel world from process {} of {}!",
        petsc.world().rank(),
        petsc.world().size()
    );

    todo!()
}
