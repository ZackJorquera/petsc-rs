use structopt::StructOpt;

#[derive(Debug, PartialEq, StructOpt)]
pub enum PetscOpt {
    /// use `-- -help` for petsc help
    #[structopt(name = "Petsc Args", external_subcommand)]
    PetscArgs(Vec<String>),
}

impl PetscOpt
{
    pub fn petsc_args(self_op: Option<Self>) -> Vec<String>
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
