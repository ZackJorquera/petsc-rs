# petsc-rs: PETSc rust bindings

PETSc, pronounced PET-see (/ˈpɛt-siː/), is a suite of data structures and routines for the scalable (parallel) solution of scientific applications modeled by partial differential equations. It supports MPI, ~~and GPUs through CUDA or OpenCL, as well as hybrid MPI-GPU parallelism~~. ~~PETSc (sometimes called PETSc/TAO) also contains the TAO optimization software library~~. (I crossed these out because petsc-rs does not support them).

PETSc is intended for use in large-scale application projects, many ongoing computational science projects are built around the PETSc libraries. PETSc is easy to use for beginners. Moreover, its careful design allows advanced users to have detailed control over the solution process. petsc-rs includes a large suite of parallel linear, nonlinear equation solvers and ODE integrators that are easily used in application codes written in Rust. PETSc provides many of the mechanisms needed within parallel application codes, such as simple parallel matrix and vector assembly routines that allow the overlap of communication and computation. In addition, PETSc includes support for parallel distributed arrays useful for finite difference methods.

## Usage

To call petsc-rs from a Rust package, the following Cargo.toml can be used.
```toml
petsc-rs = { git = "https://github.com/ZackJorquera/petsc-rs/", branch = "main" }
```

In order for this to work correctly, you need to [download PETSc](https://www.mcs.anl.gov/petsc/download/index.html) (version 3.15 or above). Then you need to [configure and install PETSc](https://www.mcs.anl.gov/petsc/documentation/installation.html). I haven't tested all the different ways to install PETSc, but the following works. Note, it is required that you install an MPI library globally and not have PETSc install it for you. Im using `openmpi`, which gives me `mpicc` and `mpicxx`.
```text
./configure --with-cc=mpicc --with-cxx=mpicxx --download-f2cblaslapack --with-fc=0
make all check
```

Then you must set the environment variables `PETSC_DIR` and `PETSC_ARCH` to where you installed PETSc.

Note, for the linking on the Rust side to work, you will also need to install a version of LLVM/Clang.

From here, you should be able to compile your projects using petsc-rs. However, if you get linking errors when you run a program, you might need to set the `LD_LIBRARY_PATH` environment variable to include `$PETSC_DIR/$PETSC_ARCH/lib`.

### Optional Build Parameters

If you want to use a PETSc with non-standard precisions for floats or integers, or complex numbers you can include something like the following in your Cargo.toml.
```toml
petsc-rs = { git = "https://github.com/ZackJorquera/petsc-rs/", branch = "main", default-features = false, features = ["petsc-real-f32", "petsc-int-i64"] }
```

If you want to have a release build you will have to set the `PETSC_ARCH_RELEASE` environment variable with the directory in `PETSC_DIR` where the release build is. Then when compile with release mode, the PETSc release build will be used. 

### Features

PETSc has support for multiple different sizes of scalars and integers. To expose this
to rust, we require you set different features. The following are all the features that
can be set. Note, you are required to have exactly one scalar feature set and exactly
one integer feature set. And it must match the PETSc install.
- **`petsc-real-f64`** *(enabled by default)* — Sets the real type, `PetscReal`, to be `f64`.
Also sets the complex type, `PetscComplex`, to be `Complex<f64>`.
- **`petsc-real-f32`** — Sets the real type, `PetscReal` to be `f32`.
Also sets the complex type, `PetscComplex`, to be `Complex<f32>`.
- **`petsc-use-complex`** *(disabled by default)* *(experimental only)* - Sets the scalar type, `PetscScalar`, to
be the complex type, `PetscComplex`. If disabled then the scalar type is the real type, `PetscReal`.
You must be using the `complex-scalar` branch to enable this feature.
- **`petsc-int-i32`** *(enabled by default)* — Sets the integer type, `PetscInt`, to be `i32`.
- **`petsc-int-i64`** — Sets the integer type, `PetscInt`, to be `i64`.

## Running PETSc Programs

All cargo projects can be built with 
```text
cargo build
```
Note, cargo normally puts the binary at `target/debug/petsc_program_name`.

All PETSc programs use the MPI (Message Passing Interface) standard for message-passing communication [[For94](https://petsc.org/release/documentation/manual/getting_started/#id205)]. Thus, to execute PETSc programs, users must know the procedure for beginning MPI jobs on their selected computer system(s). For instance, when using the MPICH implementation of MPI and many others, the following command initiates a program that uses eight processors:

```text
mpiexec -n 8 target/debug/petsc_program_name [petsc_options]
```

## Getting Started Example

To help the user start using PETSc immediately, we begin with a simple uniprocessor example that solves the one-dimensional Laplacian problem with finite differences. This sequential code, which can be found in [`examples/ksp/src/ex1.rs`](examples/ksp/src/ex1.rs), illustrates the solution of a linear system with KSP, the interface to the preconditioners, Krylov subspace methods, and direct linear solvers of PETSc.

```rust
//! This file will show how to do the kps ex1 example in rust using the petsc-rs bindings.
//!
//! Concepts: KSP^solving a system of linear equations
//! Processors: 1
//!
//! Use "petsc_rs::prelude::*" to get direct access to all important petsc-rs bindings
//!     and mpi traits which allow you to call things like `world.size()`.
//!
//! To run:
//! ```text
//! $ cargo build --bin ksp-ex1
//! $ target/debug/ksp-ex1
//! Norm of error 2.41202e-15, Iters 5
//! $ mpiexec -n 1 target/debug/ksp-ex1
//! Norm of error 2.41202e-15, Iters 5
//! $ target/debug/ksp-ex1 -n 100
//! Norm of error 1.14852e-2, Iters 318
//! ```
//!
//! Note:  The corresponding parallel example is ex23.rs

static HELP_MSG: &str = "Solves a tridiagonal linear system with KSP.\n\n";

use petsc_rs::prelude::*;

fn main() -> petsc_rs::Result<()> {
    let petsc = Petsc::builder()
        .args(std::env::args())
        .help_msg(HELP_MSG)
        .init()?;

    let n = petsc.options_try_get_int("-n")?.unwrap_or(10);

    if petsc.world().size() != 1 {
        Petsc::set_error(petsc.world(), PetscErrorKind::PETSC_ERR_WRONG_MPI_SIZE,
            "This is a uniprocessor example only!")?;
    }

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //   Compute the matrix and right-hand-side vector that define
    //   the linear system, Ax = b.
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    // Create vectors.  Note that we form 1 vector from scratch and
    // then duplicate as needed.
    let mut x = petsc.vec_create()?;
    x.set_name("Solution")?;
    x.set_sizes(None, Some(n))?;
    x.set_from_options()?;
    let mut b = x.clone();
    let mut u = x.clone();

    #[allow(non_snake_case)]
    let mut A = petsc.mat_create()?;
    A.set_sizes(None, None, Some(n), Some(n))?;
    A.set_from_options()?;
    A.set_up()?;

    // Assemble matrix:
    // Note, `PetscScalar` could be a complex number, so best practice is to instead of giving
    // float literals (i.e. `1.5`) when a function takes a `PetscScalar` wrap in in a `from`
    // call. E.x. `PetscScalar::from(1.5)`. This will do nothing if `PetscScalar` in a real number,
    // but if `PetscScalar` is complex it will construct a complex value which the imaginary part being
    // set to `0`.
    A.assemble_with((0..n).map(|i| (-1..=1).map(move |j| (i,i+j))).flatten()
            .filter(|&(i, j)| i < n && j < n) // we could also filter out negatives, but assemble_with does that for us
            .map(|(i,j)| if i == j { (i, j, PetscScalar::from(2.0)) }
                         else { (i, j, PetscScalar::from(-1.0)) }),
        InsertMode::INSERT_VALUES, MatAssemblyType::MAT_FINAL_ASSEMBLY)?;

    // Set exact solution; then compute right-hand-side vector.
    u.set_all(PetscScalar::from(1.0))?;
    Mat::mult(&A, &u, &mut b)?;

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //          Create the linear solver and set various options
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    let mut ksp = petsc.ksp_create()?;

    // Set operators. Here the matrix that defines the linear system
    // also serves as the matrix that defines the preconditioner.
    #[allow(non_snake_case)]
    let rc_A = std::rc::Rc::new(A);
    ksp.set_operators(Some(rc_A.clone()), Some(rc_A.clone()))?;

    // Set linear solver defaults for this problem (optional).
    // - By extracting the KSP and PC contexts from the KSP context,
    //     we can then directly call any KSP and PC routines to set
    //     various options.
    // - The following four statements are optional; all of these
    //     parameters could alternatively be specified at runtime via
    //     `KSP::set_from_options()`.
    let pc = ksp.get_pc_mut()?;
    pc.set_type(PCType::PCJACOBI)?;
    ksp.set_tolerances(Some(1.0e-5), None, None, None)?;

    // Set runtime options, e.g.,
    //     `-ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>`
    // These options will override those specified above as long as
    // `KSP::set_from_options()` is called _after_ any other customization
    // routines.
    ksp.set_from_options()?;

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //                  Solve the linear system
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ksp.solve(Some(&b), &mut x)?;

    // View solver info; we could instead use the option -ksp_view to
    // print this info to the screen at the conclusion of `KSP::solve()`.
    let viewer = Viewer::create_ascii_stdout(petsc.world())?;
    ksp.view_with(Some(&viewer))?;

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //                Check the solution and clean up
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    x.axpy(PetscScalar::from(-1.0), &u)?;
    let x_norm = x.norm(NormType::NORM_2)?;
    let iters = ksp.get_iteration_number()?;
    petsc_println!(petsc.world(), "Norm of error {:.5e}, Iters {}", x_norm, iters)?;

    // All PETSc objects are automatically destroyed when they are no longer needed.
    // PetscFinalize() is also automatically called.

    // return
    Ok(())
}
```

## Examples

More examples can be found in [`examples/`](examples/)

## C API Documentation

- [Getting Started](https://petsc.org/release/documentation/manual/getting_started/)

- [Programming with PETSc/TAO](https://petsc.org/release/documentation/manual/programming/)
