# `petsc-rs`: PETSc rust bindings

PETSc, pronounced PET-see (/ˈpɛt-siː/), is a suite of data structures and routines for the scalable (parallel) solution of scientific applications modeled by partial differential equations. It supports MPI, ~~and GPUs through CUDA or OpenCL, as well as hybrid MPI-GPU parallelism~~. ~~PETSc (sometimes called PETSc/TAO) also contains the TAO optimization software library~~. (I crossed these out because `petsc-rs` does not support them yet).

PETSc is intended for use in large-scale application projects, many ongoing computational science projects are built around the PETSc libraries. PETSc is easy to use for beginners. Moreover, its careful design allows advanced users to have detailed control over the solution process. `petsc-rs` includes a large suite of parallel linear, nonlinear equation solvers and ODE integrators that are easily used in application codes written in Rust. PETSc provides many of the mechanisms needed within parallel application codes, such as simple parallel matrix and vector assembly routines that allow the overlap of communication and computation. In addition, PETSc includes support for parallel distributed arrays useful for finite difference methods.

Note, `petsc-rs` is a work in progress, so expect that a lot of functionality will be missing. Furthermore, many existing functions and structs may be incorrect wrappers to the C API and/or subject to change in the future.

## Usage

To use `petsc-rs` from a Rust package, the following can be put in your `Cargo.toml`. Note, `petsc-rs` is supported for rust 1.54 and above.
```toml
[dependencies]
petsc-rs = { git = "https://gitlab.com/petsc/petsc-rs/", branch = "main" }
```

In order for `petsc-rs` to work correctly, you need to [download PETSc](https://petsc.org/release/download/). Note, `petsc-rs` requires PETSc version `3.15` or the main branch (prerelease version `3.16-dev.0`). Using the main branch is unstable as new breaking changes could be added (`petsc-rs` has been tested using commit [`7e766471`](https://gitlab.com/petsc/petsc/tree/7e76647156e152455000fa90dacb50cc4a1f8f17) of PETSc). Regardless, `petsc-rs` will automatically detect what version of PETSc you are using and build the correct wrappers. If the version of PETSc you are using is not supported, then `petsc-rs` will fail to build.

Next, you need to [configure and install PETSc](https://petsc.org/release/install/). I haven't tested all the different ways to install PETSc, but the following I know works for `petsc-rs`. Note, it is required that you install an MPI library globally and not have PETSc install it for you. This is needed by the [rsmpi](https://github.com/rsmpi/rsmpi) crate (look at its [requirements](https://github.com/rsmpi/rsmpi#requirements) for more information). I'm using `openmpi` 3.1.3, which gives me `mpicc` and `mpicxx`.
```text
./configure --with-cc=mpicc --with-cxx=mpicxx --download-f2cblaslapack --with-fc=0
make all check
```

Then you must set the environment variables `PETSC_DIR` and `PETSC_ARCH` to where you installed PETSc.

Note, for the linking on the Rust side to work, you will also need to install `libclang`. See the [bindgen project's requirements](https://rust-lang.github.io/rust-bindgen/requirements.html) for more information.

From here, you should be able to compile your projects using cargo. However, if you get linking errors when you run a program, you might need to set the `LD_LIBRARY_PATH` environment variable to include `$PETSC_DIR/$PETSC_ARCH/lib`. This is automatically done when you use any cargo command such as `cargo run`, but might not be set when you manually run the binary.

### Optional Build Parameters

If you want to use a PETSc with non-standard precisions for floats or integers, or for complex numbers (unsafe) you can include something like the following in your Cargo.toml.
```toml
[dependencies.petsc-rs]
git = "https://gitlab.com/petsc/petsc-rs/"
branch = "main"
default-features = false  # note, default turns on "petsc-real-f64" and "petsc-int-i32"
features = ["petsc-real-f32", "petsc-int-i64"]
```

Note, you will have to build PETSc with the same settings that you wish to use for `petsc-rs`. When building, the `petsc-sys` package will validate that the PETSc install matches the requested feature flags.

If you want to have a release build you will have to set the `PETSC_ARCH_RELEASE` environment variable with the directory in `PETSC_DIR` where the release build is. Then when compiling with release mode, the PETSc release build will be used.

### Features

PETSc has support for multiple different sizes of scalars and integers. To expose this
to rust, we require you set different features. The following are all the features that
can be set. Note, you are required to have exactly one scalar feature set and exactly
one integer feature set. And it must match the PETSc install.
- **`petsc-real-f64`** *(enabled by default)* — Sets the real type, `PetscReal`, to be `f64`.
Also sets the complex type, `PetscComplex`, to be `Complex<f64>`.
- **`petsc-real-f32`** — Sets the real type, `PetscReal` to be `f32`.
Also sets the complex type, `PetscComplex`, to be `Complex<f32>`.
- **`petsc-use-complex-unsafe`** *(disabled by default)* *(unsafe)* - Sets the scalar type, `PetscScalar`, to
be the complex type, `PetscComplex`. If disabled then the scalar type is the real type, `PetscReal`.
This is unsafe because `petsc-rs` makes no guarantees about following the correct calling convention across the FFI boundary (read gitlab [issue #1](https://gitlab.com/petsc/petsc-rs/-/issues/1) for more information).
- **`petsc-int-i32`** *(enabled by default)* — Sets the integer type, `PetscInt`, to be `i32`.
- **`petsc-int-i64`** — Sets the integer type, `PetscInt`, to be `i64`.

### Using `petsc-sys`

If you wish to use raw bindings from `petsc-sys` in the same crate that you are using `petsc-rs` you can import the `petsc-sys` crate with the following line in your `Cargo.toml`. An example of using both `petsc-rs` and `petsc-sys` can be found in [`examples/snes/src/ex12.rs`](https://gitlab.com/petsc/petsc-rs/-/blob/main/examples/snes/src/ex12.rs).

```toml
[dependencies]
petsc-sys = { git = "https://gitlab.com/petsc/petsc-rs/", branch = "main", default-features = false }
```

Note, `petsc-sys` has the same type related feature flags as `petsc-rs`, and `petsc-rs` will pass its flags to `petsc-sys`. To avoid conflicts you should use `default-features = false` when importing `petsc-sys` so that you don't accidentally enable any additional flags.

### Using `mpi`

If you want to use `mpi` in your project, you MUST use `rsmpi v0.6` or above. Currently, this isn't available on [crates.io](https://crates.io/) so you will have to get it from the GitHub repo. You can do that by adding the following to your `Cargo.toml`.

```toml
[dependencies]
mpi = { git = "https://github.com/rsmpi/rsmpi.git", branch = "main" }
```

Or to be consistent with `petsc-rs` you can use `rev = "82e1d35"` instead of `branch = "main"`.

## Running PETSc Programs

All cargo projects can be built with 
```text
cargo build
```
Note, cargo normally puts the binary at `target/debug/petsc_program_name`.

All PETSc programs use the MPI (Message Passing Interface) standard for message-passing communication [[For94](https://petsc.org/release/docs/manual/getting_started/#id205)]. Thus, to execute PETSc programs, users must know the procedure for beginning MPI jobs on their selected computer system(s). For instance, when using the MPICH implementation of MPI and many others, the following command initiates a program that uses eight processors:

```text
mpiexec -n 8 target/debug/petsc_program_name [petsc_options]
```

## Getting Started Example

To help the user start using PETSc immediately, we begin with a simple uniprocessor example that solves the one-dimensional Laplacian problem with finite differences. This sequential code, which can be found in [`examples/ksp/src/ex1.rs`](https://gitlab.com/petsc/petsc-rs/-/blob/main/examples/ksp/src/ex1.rs), illustrates the solution of a linear system with KSP, the interface to the preconditioners, Krylov subspace methods, and direct linear solvers of PETSc. Note, to compile and run this code, you should be in the [`examples/`](https://gitlab.com/petsc/petsc-rs/-/tree/main/examples/) directory.

```rust
//! This file will show how to do the kps ex1 example in rust using the petsc-rs bindings.
//!
//! Concepts: KSP^solving a system of linear equations
//! Processors: 1
//!
//! Use "petsc_rs::prelude::*" to get direct access to all important petsc-rs bindings.
//! Use "mpi::traits::*" to get access to all mpi traits which allow you to call things like `world.size()`.
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
//! To build for complex you can use the flag `--features petsc-use-complex-unsafe`
//!
//! Note:  The corresponding parallel example is ex23.rs

static HELP_MSG: &str = "Solves a tridiagonal linear system with KSP.\n\n";

use petsc_rs::prelude::*;
use mpi::traits::*;

fn main() -> petsc_rs::Result<()> {
    let petsc = Petsc::builder()
        .args(std::env::args())
        .help_msg(HELP_MSG)
        .init()?;

    let n = petsc.options_try_get_int("-n")?.unwrap_or(10);
    let show_solution = petsc.options_try_get_bool("-show_solution")?.unwrap_or(false);

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
    x.set_sizes(None, n)?;
    x.set_from_options()?;
    let mut b = x.clone();
    let mut u = x.clone();

    #[allow(non_snake_case)]
    let mut A = petsc.mat_create()?;
    A.set_sizes(None, None, n, n)?;
    A.set_from_options()?;
    A.set_up()?;

    // Assemble matrix:
    A.assemble_with((0..n).map(|i| (-1..=1).map(move |j| (i,i+j))).flatten()
            // we could also filter out negatives, but `assemble_with` does that for us
            .filter(|&(i, j)| i < n && j < n)
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
    ksp.set_operators(&A, &A)?;

    // Set linear solver defaults for this problem (optional).
    // - By extracting the KSP and PC contexts from the KSP context,
    //     we can then directly call any KSP and PC routines to set
    //     various options.
    // - The following statements are optional; all of these
    //     parameters could alternatively be specified at runtime via
    //     `KSP::set_from_options()`.
    let pc = ksp.get_pc_or_create()?;
    pc.set_type(PCType::PCJACOBI)?;
    ksp.set_tolerances(1.0e-5, None, None, None)?;

    // Set runtime options, e.g.,
    //     `-ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>`
    // These options will override those specified above as long as
    // `KSP::set_from_options()` is called _after_ any other customization
    // routines.
    ksp.set_from_options()?;

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //                  Solve the linear system
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ksp.solve(&b, &mut x)?;

    // View solver info; we could instead use the option -ksp_view to
    // print this info to the screen at the conclusion of `KSP::solve()`.
    let viewer = Viewer::create_ascii_stdout(petsc.world())?;
    viewer.view(&ksp)?;

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //                Check the solution and clean up
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if show_solution {
        viewer.view(&x)?;
        // Or we can do the following. Note, in a multi-process comm
        // world we should instead use `petsc_println_sync!`.
        println!("{}: {:.2}", x.get_name()?, *x.view()?);
    }
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

More examples can be found in [`examples/`](https://gitlab.com/petsc/petsc-rs/-/tree/main/examples/)

## Documentation

Currently, the `petsc-rs` documentation is not hosted anywhere. However, you can build the documentation with:
```text
cargo doc
```
You can use the `--open` flag to open it in your browser.

## C API Documentation

- [Getting Started](https://petsc.org/release/docs/manual/getting_started/)

- [Programming with PETSc/TAO](https://petsc.org/release/docs/manual/)
