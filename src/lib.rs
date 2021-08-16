#![warn(rustdoc::broken_intra_doc_links)]
#![warn(missing_docs)]

//! # [petsc-rs](#): PETSc rust bindings
//!
//! PETSc, pronounced PET-see (/ˈpɛt-siː/), is a suite of data structures and routines for the scalable
//! (parallel) solution of scientific applications modeled by partial differential equations. PETSc supports
//! MPI through the `mpi` crate from [`rsmpi`](https://github.com/rsmpi/rsmpi).
//!
//! # Basic Usage
//! 
//! First, you will need to add `petsc-rs` to your `Cargo.toml`.
//! ```toml
//! [dependencies]
//! petsc-rs = { git = "https://gitlab.com/petsc/petsc-rs/", branch = "main" }
//! ```
//! Next, to get access to all the important traits and types you can use `use petsc_rs::prelude::*`.
//! Some of the important types that are included are:
//!
//! * Index sets ([`IS`](indexset::IS)), including permutations, for indexing into vectors, renumbering, etc
//! * Vectors ([`Vector`](vector::Vector))
//! * Matrices ([`Mat`](mat::Mat)) (generally sparse)
//! * Krylov subspace methods ([`KSP`](ksp::KSP))
//! * Preconditioners ([`PC`](pc::PC))
//! * Nonlinear solvers ([`SNES`](snes::SNES))
//! * Managing interactions between mesh data structures and vectors, matrices, and solvers ([`DM`](dm::DM))
//!
//! Most PETSc programs begin by initializing PETSc which can be done with [`PetscBuilder::init()`]
//! or [`Petsc::init_no_args()`].
//!
//! ```
//! use petsc_rs::prelude::*;
//! fn main() -> petsc_rs::Result<()> {
//!     let p = Petsc::init_no_args()?;
//!     petsc_println!(p.world(), "Hello, PETSc!")?;
//!     Ok(())
//! }
//! ```
//! 
//! All PETSc routines return a [`petsc_rs::Result`](crate::Result), which is a wrapper around the
//! standard [`Result`](std::result::Result) type which indicates whether an error has occurred during
//! the call.
//!
//! # Features
//! 
//! PETSc has support for multiple different sizes of scalars and integers. This is exposed
//! to rust with different features that you can set. The following are all the features that
//! can be set. Note, you are required to have exactly one scalar feature set and exactly
//! one integer feature set. And it must match the PETSc install.
//!
//! - **`petsc-real-f64`** *(enabled by default)* — Sets the real type, [`PetscReal`], to be `f64`.
//! Also sets the complex type, [`PetscComplex`], to be `Complex<f64>`.
//! - **`petsc-real-f32`** — Sets the real type, [`PetscReal`] to be `f32`.
//! Also sets the complex type, [`PetscComplex`], to be `Complex<f32>`.
//! - **`petsc-use-complex`** *(disabled by default)* *(experimental only)* - Sets the scalar type, [`PetscScalar`], to
//! be the complex type, [`PetscComplex`]. If disabled then the scalar type is the real type, [`PetscReal`].
//! You must be using the `complex-scalar` branch to enable this feature.
//! - **`petsc-int-i32`** *(enabled by default)* — Sets the integer type, [`PetscInt`], to be `i32`.
//! - **`petsc-int-i64`** — Sets the integer type, [`PetscInt`], to be `i64`.
//!
//! If you want to use a PETSc with non-standard precisions for floats or integers, or for complex numbers
//! (experimental only) you can include something like the following in your Cargo.toml.
//! ```toml
//! [dependencies.petsc-rs]
//! git = "https://gitlab.com/petsc/petsc-rs/"
//! branch = "main"  # for complex numbers use the "complex-scalar" branch
//! default-features = false  # note, default turns on "petsc-real-f64" and "petsc-int-i32"
//! features = ["petsc-real-f32", "petsc-int-i64"]
//! ```
//!
//! # Further Reading
//! 
//! - [C API Getting Started](https://petsc.org/release/documentation/manual/getting_started/)
//!
//! - [C API Programming with PETSc/TAO](https://petsc.org/release/documentation/manual/programming/)
//!
//! - [`petsc-rs` GitLab page](https://gitlab.com/petsc/petsc-rs/)

// useful doc page: https://petsc.org/release/docs/manualpages/singleindex.html

use std::fmt::Display;
use std::marker::PhantomData;
use std::ops::{Deref, Bound, RangeBounds};
use std::os::raw::{c_char, c_int};
use std::vec;

/// The raw C language PETSc API
///
/// Documented on the [PETSc Documentation page](https://petsc.org/release/documentation/).
// TODO: should this be public? Should we rename to `ffi`? Or should we just reexport petsc-sys?
// Or just keep it the same?
pub(crate) mod petsc_raw {
    pub use petsc_sys::*;
}

pub use petsc_raw::{PetscInt, PetscReal};
pub use petsc_raw::NormType;

use mpi::{self, traits::*};
use std::mem::{MaybeUninit, ManuallyDrop};
use std::ffi::{CString, CStr, };
use mpi::topology::UserCommunicator;

pub(crate) mod internal_macros;

pub mod vector;
pub mod mat;
pub mod ksp;
#[path = "preconditioner.rs"] pub mod pc; // TODO: or should i just rename the file
pub mod viewer;
pub mod snes;
pub mod dm;
pub mod indexset;
pub mod spaces;

use vector::Vector;
use mat::Mat;
use ksp::KSP;
use snes::SNES;
use viewer::Viewer;

pub mod prelude {
    //! Commonly used items.
    pub use crate::{
        Petsc,
        PetscErrorKind,
        PetscError,
        InsertMode,
        PetscInt,
        PetscScalar,
        PetscReal,
        PetscComplex,
        PetscAsRaw,
        PetscAsRawMut,
        PetscObject,
        PetscOptBuilder,
        petsc_println,
        petsc_println_sync,
        petsc_print,
        petsc_print_sync,
        petsc_panic,
        vector::{Vector, VecOption, VectorType, },
        mat::{Mat, MatAssemblyType, MatOption, MatDuplicateOption, MatStencil, NullSpace, MatType,
            MatOperation, },
        ksp::{KSP, KSPType, },
        snes::{SNES, DomainOrPetscError::DomainErr, SNESType, },
        pc::{PC, PCType, },
        dm::{DM, DMBoundaryType, DMDAStencilType, DMType, FEDisc, DS, WeakForm, DMLabel,
            DMBoundaryConditionType, FVDisc, DMField, },
        indexset::{IS, ISType, },
        viewer::{Viewer, PetscViewerFormat, ViewerType, PetscViewable, },
        spaces::{Space, DualSpace, SpaceType, DualSpaceType},
        NormType,
        PetscOpt,
    };
}

#[cfg(feature = "petsc-use-complex")]
use num_complex::Complex;

/// Prints to standard out with a new line, only from the first processor in the communicator.
///
/// Calls from other processes are ignored.
///
/// Note, this macro creates a block that evaluates to a [`petsc_rs::Result`](Result), so the try operator, `?`,
/// can and should be used. 
///
/// # Example
///
/// ```
/// # use petsc_rs::prelude::*;
/// # use mpi::traits::*;
/// # fn main() -> petsc_rs::Result<()> {
/// let petsc = petsc_rs::Petsc::init_no_args().unwrap();
///
/// // will print once no matter how many processes there are
/// petsc_println!(petsc.world(), "Hello parallel world of {} processes!", petsc.world().size())?;
/// // This will print just a new line
/// petsc_println!(petsc.world())?;
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! petsc_println {
    ($world:expr) => ( Petsc::print($world, "\n") );
    ($world:expr, $($arg:tt)*) => ({
        let s = format!("{}\n", format_args!($($arg)*));
        Petsc::print($world, s)
    })
}

/// Prints to standard out without a new line, only from the first processor in the communicator.
///
/// Calls from other processes are ignored.
///
/// Note, this macro creates a block that evaluates to a [`petsc_rs::Result`](Result), so the try operator, `?`,
/// can and should be used.
///
/// Also look at [`petsc_println!`].
#[macro_export]
macro_rules! petsc_print {
    ($world:expr, $($arg:tt)*) => ({
        let s = ::std::fmt::format(format_args!($($arg)*));
        Petsc::print($world, s)
    })
}

/// Prints synchronized output from several processors with a new line.
///
/// Output of the first processor is followed by that of the second, etc.
///
/// Will automatically call `PetscSynchronizedFlush` after.
///
/// Note, this macro creates a block that evaluates to a [`petsc_rs::Result`](Result), so the try operator, `?`,
/// can and should be used.
///
/// # Example
///
/// ```
/// # use petsc_rs::prelude::*;
/// # use mpi::traits::*;
/// # fn main() -> petsc_rs::Result<()> {
/// let petsc = petsc_rs::Petsc::init_no_args().unwrap();
///
/// // will print multiple times, once for each processor
/// Petsc::print_sync(petsc.world(), format!("Hello parallel world of {} processes from process {}!\n",
///     petsc.world().size(), petsc.world().rank()))?;
/// // or use:
/// petsc_println_sync!(petsc.world(), "Hello parallel world of {} processes from process {}!", 
///     petsc.world().size(), petsc.world().rank())?;
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! petsc_println_sync {
    ($world:expr) => ( Petsc::print_sync($world, "\n") );
    ($world:expr, $($arg:tt)*) => ({
        let s = format!("{}\n", format_args!($($arg)*));
        Petsc::print_sync($world, s)
    })
}

/// Prints synchronized output from several processors with out a new line.
///
/// Output of the first processor is followed by that of the second, etc.
///
/// Will automatically call `PetscSynchronizedFlush` after.
///
/// Note, this macro creates a block that evaluates to a [`petsc_rs::Result`](Result), so the try operator, `?`,
/// can and should be used.
///
/// Also look at [`petsc_println_sync!`].
#[macro_export]
macro_rules! petsc_print_sync {
    ($world:expr, $($arg:tt)*) => ({
        let s = ::std::fmt::format(format_args!($($arg)*));
        Petsc::print_sync($world, s)
    })
}

/// Aborts the PETSc program with an error code.
///
/// Similar to [`panic!`], but will call [`Petsc::set_error()`] and [`Petsc::abort()`],
/// and not rust's panic.
///
/// If you `petsc_panic!` with a message, than PETSc must be initialized. Otherwise,
/// `petsc_panic!` with out a message only requires MPI to be initialized.
///
/// # Example
///
/// ```should_panic
/// # use petsc_rs::prelude::*;
/// # use mpi::traits::*;
/// # fn main() -> petsc_rs::Result<()> {
/// let petsc = petsc_rs::Petsc::init_no_args().unwrap();
///
/// petsc_panic!(petsc.world(), PetscErrorKind::PETSC_ERR_USER);
/// petsc_panic!(petsc.world(), PetscErrorKind::PETSC_ERR_USER,
///     "this is a terrible mistake!");
/// petsc_panic!(petsc.world(), PetscErrorKind::PETSC_ERR_USER,
///     "this is a {} {message}", "fancy", message = "message");
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! petsc_panic {
    ($world:expr, $err_kind:expr) => ({
        Petsc::abort($world, $err_kind)
    });
    ($world:expr, $err_kind:expr, $($arg:tt)*) => ({
        let s = ::std::fmt::format(format_args!($($arg)*));
        let _ = Petsc::set_error($world, $err_kind, s).unwrap_err();
        Petsc::abort($world, $err_kind)
    })
}

/// PETSc result
pub type Result<T> = std::result::Result<T, PetscError>;

/// PETSc Error type.
///
/// You can create an error with [`Petsc::set_error()`].
#[derive(Debug)]
pub struct PetscError {
    pub(crate) kind: PetscErrorKind,
    pub(crate) error: Box<dyn std::error::Error + Send + Sync>,
}

impl Display for PetscError {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(fmt)
    }
}

/// A list specifying types of PETSc errors.
pub type PetscErrorType = petsc_raw::PetscErrorType;
/// A list specifying kinds of PETSc errors.
pub type PetscErrorKind = petsc_raw::PetscErrorCodeEnum;
pub use petsc_raw::InsertMode;

/// Helper struct which allows you to call [`PetscInitialize`] with optional parameters.
///
/// Must call [`PetscBuilder::init()`] to get the [`Petsc`] object.
///
/// # Examples
///
/// ```no_run
/// # use petsc_rs::prelude::*;
/// # use mpi::traits::*;
/// let univ = mpi::initialize().unwrap();
/// let petsc = Petsc::builder()
///     .args(std::env::args())
///     // Note, if we don't split the comm world and just use the default world
///     // then there is no need to use the world method as we do here.
///     .world(univ.world().duplicate())
///     .help_msg("Hello, this is a help message\n")
///     .file("path/to/database/file")
///     .init().unwrap();
/// ```
///
/// Look at [`PetscBuilder::world()`] for more info on setting the comm world.
///
/// Note `Petsc::builder().init()` is the same as [`Petsc::init_no_args()`].
///
/// [`PetscInitialize`]: petsc_raw::PetscInitialize
#[derive(Default)]
pub struct PetscBuilder
{
    world: Option<UserCommunicator>,
    args: Option<Vec<String>>,
    file: Option<String>,
    help_msg: Option<String>,

}

impl PetscBuilder
{
    /// Calls [`PetscInitialize`] with the options given.
    ///
    /// Initializes the PETSc database and MPI. Will also call `MPI_Init()` if that has
    /// yet to be called, so this routine should always be called near the beginning
    /// of your program -- usually the very first line!
    ///
    /// [`PetscInitialize`]: petsc_raw::PetscInitialize
    pub fn init(self) -> Result<Petsc>
    {
        // Note the argc/argv data we give to `PetscInitialize` is used internally and so it  must
        // live longer than the call to `PetscFinalize`. In other words, it must outlive the `Petsc`
        // type this method creates.

        // Note, we can drop argc when we are done using the pointer
        let mut argc_boxed;
        let c_argc_p = if let Some(ref args) = self.args {
            argc_boxed = Box::new(args.len() as c_int);
            &mut *argc_boxed as *mut c_int
        } else {
            argc_boxed = Box::new(0);
            std::ptr::null_mut()
        };

        // We only need to drop the following 3 objects to clean up (and also argc)
        let cstr_args_owned = self.args.as_ref().map_or(vec![], |args| 
            args.iter().map(|arg| CString::new(arg.deref()).expect("CString::new failed"))
                .collect::<Vec<CString>>());
        let mut c_args_owned = cstr_args_owned.iter().map(|arg| arg.as_ptr() as *mut _)
            .collect::<Vec<*mut c_char>>();
        c_args_owned.push(std::ptr::null_mut());
        let mut c_args_boxed = Box::new(c_args_owned.as_mut_ptr());

        let c_args_p = self.args.as_ref().map_or(std::ptr::null_mut(), |_| &mut *c_args_boxed as *mut _);

        // Note, the file string does not need to outlive the `Petsc` type
        let file_cstring = self.file.map(|ref f| CString::new(f.deref()).ok()).flatten();
        let file_c_str = file_cstring.as_ref().map_or_else(|| std::ptr::null(), |v| v.as_ptr());

        // We dont have to leak the file string
        let help_cstring = self.help_msg.map(|ref h| CString::new(h.deref()).ok()).flatten();
        let help_c_str = help_cstring.as_ref().map_or_else(|| std::ptr::null(), |v| v.as_ptr());

        let drop_world_first;
        let ierr;
        // We pass in the args data so that we can reconstruct the vec to free all the memory.
        let petsc = Petsc { world: match self.world { 
                Some(world) => {
                    // Note, in this case MPI has already initialized

                    // SAFETY: Nothing should use the global variable `PETSC_COMM_WORLD` directly
                    // everything should access it through the `Petsc.world()` method which is only
                    // accessible after this (at least on the rust side of things). 
                    // Additional info on using this variable can be found here:
                    // https://petsc.org/release/docs/manualpages/Sys/PETSC_COMM_WORLD.html
                    unsafe { petsc_raw::PETSC_COMM_WORLD = world.as_raw(); }
                    drop_world_first = false;
                    ierr = unsafe { petsc_raw::PetscInitialize(c_argc_p, c_args_p, file_c_str, help_c_str) };

                    ManuallyDrop::new(world)
                }, 
                _ => {
                    // Note, in this case MPI has not been initialized, it will be initialized by PETSc
                    ierr = unsafe { petsc_raw::PetscInitialize(c_argc_p, c_args_p, file_c_str, help_c_str) };
                    drop_world_first = true;
                    ManuallyDrop::new(mpi::topology::SystemCommunicator::world().duplicate())
                }
            },
            _arg_data: self.args.as_ref().map(|_| (argc_boxed, cstr_args_owned, c_args_owned, c_args_boxed)),
            drop_world_first
        };
        unsafe { chkerrq!(petsc.world(), ierr) }?;

        Ok(petsc)
    }

    /// The command line arguments
    ///
    /// Must start with the name of the program (the first `String` of `std::env::args()`).
    /// Most of the time just use `std::env::args()` as input.
    pub fn args<T>(mut self, args: T) -> Self
    where
        T: std::iter::IntoIterator<Item = String>
    {
        self.args = Some(args.into_iter().map(|e| e).collect());
        self
    }

    /// Sets the [`PETSC_COMM_WORLD`](https://petsc.org/release/docs/manualpages/Sys/PETSC_COMM_WORLD.html#PETSC_COMM_WORLD)
    /// variable which represents all the processes that PETSc knows about.
    ///
    /// By default `PETSC_COMM_WORLD` and `MPI_COMM_WORLD` ([`mpi::topology::SystemCommunicator::world()`])
    /// are identical unless you wish to run PETSc on ONLY a subset of `MPI_COMM_WORLD`. That is where this
    /// method can be use. Note, you must initialize mpi (with [`mpi::initialize()`]).
    ///
    /// This method takes in a [`UserCommunicator`]. If you have a different type of communicator,
    /// use `.into()` To convert to a [`UserCommunicator`] or duplicate the communicator with [`Communicator::duplicate()`].
    ///
    /// Note, if no communicator is supplied then the system communicator will be used (duplicated as a [`UserCommunicator`]).
    ///
    /// After you call [`PetscBuilder::init()`], the value returned by [`Petsc::world()`] is the value set here.
    pub fn world(mut self, world: UserCommunicator) -> Self
    {
        self.world = Some(world);
        self
    }

    /// Help message to print
    pub fn help_msg<T: ToString>(mut self, help_msg: T) -> Self
    {
        self.help_msg = Some(help_msg.to_string());
        self
    }

    /// PETSc database file.
    ///
    /// Append ":yaml" to filename to specify YAML options format. 
    /// Use empty string (or don't call this method) to not check for code specific file.
    /// Also checks ~/.petscrc, .petscrc and petscrc. Use -skip_petscrc in the code specific
    /// file (or command line) to skip ~/.petscrc, .petscrc and petscrc files
    pub fn file<T: ToString>(mut self, file: T) -> Self
    {
        self.file = Some(file.to_string());
        self
    }
}

/// Struct that facilitates setting command line arguments.
///
/// Wraps a set of queries on the options database that are related and should be displayed
/// on the same window of a GUI that allows the user to set the options interactively.
///
/// This replaces [`PetscOptionsBegin`](https://petsc.org/release/docs/manualpages/Sys/PetscOptionsBegin.html)
/// and [`PetscOptionsEnd`](https://petsc.org/release/docs/manualpages/Sys/PetscOptionsEnd.html) from the C API.
pub struct PetscOptBuilder<'pl, 'pool, 'strlt> {
    petsc: &'pl Petsc,
    petsc_opt_obj: &'pool mut MaybeUninit<petsc_sys::PetscOptionItems>,
    _str_phantom: PhantomData<&'strlt CStr> // There are string held in the petsc_opt_obj
}

impl<'pl, 'pool, 'strlt> PetscOptBuilder<'pl, 'pool, 'strlt> {
    fn new(petsc: &'pl Petsc, petsc_opt_obj: &'pool mut MaybeUninit<petsc_sys::PetscOptionItems>,
        prefix_cs: Option<&'strlt CStr>, mess_cs: &'strlt CStr, sec_cs: Option<&'strlt CStr>) -> Result<Self>
    {
        let ierr = unsafe { petsc_sys::PetscOptionsBegin_Private(
            petsc_opt_obj.as_mut_ptr(), petsc.world().as_raw(),
            prefix_cs.map_or(std::ptr::null() , |cs| cs.as_ptr()),
            mess_cs.as_ptr(),
            sec_cs.map_or(std::ptr::null() , |cs| cs.as_ptr())) };
        unsafe { chkerrq!(petsc.world(), ierr) }?;
        Ok(Self { petsc, petsc_opt_obj, _str_phantom: PhantomData })
    }

    /// Gets the integer value for a particular option in the database (over a range).
    pub fn options_int_range(&mut self, opt: &str, text: &str, man: &str, default: PetscInt, range: impl RangeBounds<PetscInt>) -> Result<PetscInt> {
        let opt_cs = CString::new(opt).expect("`CString::new` failed");
        let text_cs = CString::new(text).expect("`CString::new` failed");
        let man_cs = CString::new(man).expect("`CString::new` failed");
        let mut opt_val = MaybeUninit::uninit();
        let mut set = MaybeUninit::uninit();
        let lb = match range.start_bound() {
            Bound::Unbounded => PetscInt::MIN,
            Bound::Included(&lb) => lb,
            Bound::Excluded(&lb) => lb + 1,
        };
        let ub = match range.end_bound() {
            Bound::Unbounded => PetscInt::MAX,
            Bound::Included(&ub) => ub,
            Bound::Excluded(&ub) => ub - 1,
        };
        let ierr = unsafe { 
            petsc_raw::PetscOptionsInt_Private(self.petsc_opt_obj.as_mut_ptr(), opt_cs.as_ptr(),
            text_cs.as_ptr(), man_cs.as_ptr(), default, opt_val.as_mut_ptr(), set.as_mut_ptr(),
            lb, ub) };
        unsafe { chkerrq!(self.petsc.world(), ierr) }?;

        Ok(if unsafe { set.assume_init().into() } { unsafe { opt_val.assume_init() } } 
            else { default } )
    }

    /// Gets the integer value for a particular option in the database.
    pub fn options_int(&mut self, opt: &str, text: &str, man: &str, default: PetscInt) -> Result<PetscInt> {
        let opt_cs = CString::new(opt).expect("`CString::new` failed");
        let text_cs = CString::new(text).expect("`CString::new` failed");
        let man_cs = CString::new(man).expect("`CString::new` failed");
        let mut opt_val = MaybeUninit::uninit();
        let mut set = MaybeUninit::uninit();
        let ierr = unsafe { 
            petsc_raw::PetscOptionsInt_Private(self.petsc_opt_obj.as_mut_ptr(), opt_cs.as_ptr(),
            text_cs.as_ptr(), man_cs.as_ptr(), default, opt_val.as_mut_ptr(), set.as_mut_ptr(),
            PetscInt::MIN, PetscInt::MAX) };
        unsafe { chkerrq!(self.petsc.world(), ierr) }?;

        Ok(if unsafe { set.assume_init().into() } { unsafe { opt_val.assume_init() } } 
            else { default } )
    }

    /// Gets the Logical (true or false) value for a particular option in the database.
    ///
    /// Note, TRUE, true, YES, yes, no string, and 1 all translate to `true`.
    /// FALSE, false, NO, no, and 0 all translate to `false`
    pub fn options_bool(&mut self, opt: &str, text: &str, man: &str, default: bool) -> Result<bool> {
        let opt_cs = CString::new(opt).expect("`CString::new` failed");
        let text_cs = CString::new(text).expect("`CString::new` failed");
        let man_cs = CString::new(man).expect("`CString::new` failed");
        let mut opt_val = MaybeUninit::uninit();
        let mut set = MaybeUninit::uninit();
        let ierr = unsafe { 
            petsc_raw::PetscOptionsBool_Private(self.petsc_opt_obj.as_mut_ptr(), opt_cs.as_ptr(),
            text_cs.as_ptr(), man_cs.as_ptr(), default.into(), opt_val.as_mut_ptr(), set.as_mut_ptr()) };
        unsafe { chkerrq!(self.petsc.world(), ierr) }?;

        Ok(if unsafe { set.assume_init().into() } { unsafe { opt_val.assume_init().into() } } 
            else { default } )
    }

    /// Gets the floating point value for a particular option in the database..
    pub fn options_real(&mut self, opt: &str, text: &str, man: &str, default: PetscReal) -> Result<PetscReal> {
        let opt_cs = CString::new(opt).expect("`CString::new` failed");
        let text_cs = CString::new(text).expect("`CString::new` failed");
        let man_cs = CString::new(man).expect("`CString::new` failed");
        let mut opt_val = MaybeUninit::uninit();
        let mut set = MaybeUninit::uninit();
        let ierr = unsafe { 
            petsc_raw::PetscOptionsReal_Private(self.petsc_opt_obj.as_mut_ptr(), opt_cs.as_ptr(),
            text_cs.as_ptr(), man_cs.as_ptr(), default, opt_val.as_mut_ptr(), set.as_mut_ptr()) };
        unsafe { chkerrq!(self.petsc.world(), ierr) }?;

        Ok(if unsafe { set.assume_init().into() } { unsafe { opt_val.assume_init() } } 
            else { default } )
    }

    /// Gets the string value for a particular option in the database.
    ///
    /// Gets, at most, 127 characters.
    pub fn options_string(&mut self, opt: &str, text: &str, man: &str, default: &str) -> Result<String> {
        let opt_cs = CString::new(opt).expect("`CString::new` failed");
        let text_cs = CString::new(text).expect("`CString::new` failed");
        let man_cs = CString::new(man).expect("`CString::new` failed");
        let default_cs = CString::new(default).expect("`CString::new` failed");
        const BUF_LEN: usize = 128;
        let mut buf = [0 as u8; BUF_LEN];
        let mut set = MaybeUninit::uninit();
        let ierr = unsafe { 
            petsc_raw::PetscOptionsString_Private(self.petsc_opt_obj.as_mut_ptr(), opt_cs.as_ptr(),
            text_cs.as_ptr(), man_cs.as_ptr(), default_cs.as_ptr(), buf.as_mut_ptr() as *mut _,
            BUF_LEN as u64, set.as_mut_ptr()) };
        unsafe { chkerrq!(self.petsc.world(), ierr) }?;

        Ok(if unsafe { set.assume_init().into() } {
            let nul_term = buf.iter().position(|&v| v == 0).unwrap();
            let c_str: &CStr = CStr::from_bytes_with_nul(&buf[..nul_term+1]).unwrap();
            let str_slice: &str = c_str.to_str().unwrap();
            let string: String = str_slice.to_owned();
            string
        } else {
            default.to_owned()
        })
    }

    /// Gets an option in the database (as a string), then converts to type `E`.
    ///
    /// Note, If the [`from_str`](std::str::FromStr::from_str()) fails, then
    /// [`default()`](Default::default()) is used.
    pub fn options_from_string<E>(&mut self, opt: &str, text: &str, man: &str, default: E) -> Result<E>
    where
        E: Default + std::str::FromStr + Display,
    {
        let as_str = self.options_string(opt, text, man, &default.to_string())?;
        // TODO: or should we error instead of default
        Ok(E::from_str(as_str.as_str()).unwrap_or(E::default()))
    }
}

impl Drop for PetscOptBuilder<'_, '_, '_> {
    fn drop(&mut self) {
        let ierr = unsafe { petsc_raw::PetscOptionsEnd_Private(self.petsc_opt_obj.as_mut_ptr()) };
        let _ = unsafe { chkerrq!(self.petsc.world(), ierr) }; // TODO: should I unwrap or what idk?
    }
}

/// A Petsc is a wrapper around PETSc initialization and Finalization.
///
/// Also stores a reference to the the `MPI_COMM_WORLD`/`PETSC_COMM_WORLD` variable.
pub struct Petsc {
    // This is functionally the same as `PETSC_COMM_WORLD` in the C api
    // Note, just because it is an option doesn't mean it will ever be None
    // It is only set to none in the drop function (everywhere else it is some).
    pub(crate) world: ManuallyDrop<UserCommunicator>,

    // This is used to drop the argc/args data when Petsc is dropped, we never actually use it
    // on the rust side.
    _arg_data: Option<(Box<c_int>, Vec<CString>, Vec<*mut c_char>, Box<*mut *mut c_char>)>,

    drop_world_first: bool,
}

// Destructor
impl Drop for Petsc {
    fn drop(&mut self) {
        // SAFETY: PetscFinalize might call MPI_FINALIZE, which means we need to make sure our 
        // comm world is dropped before that if that is the case. Otherwise, we want to drop our
        // comm world after. Also `ManuallyDrop::drop` is only called once and then the zombie
        // value is never used again.
        unsafe {
            if self.drop_world_first {
                ManuallyDrop::drop(&mut self.world);
            }
            petsc_raw::PetscFinalize();
            if !self.drop_world_first {
                ManuallyDrop::drop(&mut self.world);
            }
        }
    }
}

impl Petsc {
    /// Major version of the PETSc library being used.
    ///
    /// Same as [`petsc_sys::PETSC_VERSION_MAJOR`].
    pub const VERSION_MAJOR: usize = petsc_raw::PETSC_VERSION_MAJOR as usize;

    /// Minor version of the PETSc library being used.
    ///
    /// Same as [`petsc_sys::PETSC_VERSION_MINOR`].
    ///
    /// This value is only "correct" if PETSc is using a release version. If, however,
    /// the version of PETSc was `v3.16-dev.0`, this value would still be `15` not `16`.
    /// This is because when the version is calculated the minor version is increase by
    /// one if the version is pre-release.
    pub const VERSION_MINOR: usize = petsc_raw::PETSC_VERSION_MINOR as usize;

    /// Subminor/patch version of the PETSc library being used.
    ///
    /// Same as [`petsc_sys::PETSC_VERSION_SUBMINOR`].
    ///
    /// This value is only "correct" if PETSc is using a release version. If the version
    /// of PETSc is pre-release then this value should be `0`, but wont necessarily be `0`.
    pub const VERSION_SUBMINOR: usize = petsc_raw::PETSC_VERSION_SUBMINOR as usize;

    /// If the PETSc library is release (`true`) or pre-release (`false`).
    ///
    /// Same as [`petsc_sys::PETSC_VERSION_RELEASE`] but casted to a `bool`.
    pub const VERSION_RELEASE: bool = petsc_raw::PETSC_VERSION_RELEASE != 0;

    /// Creates a [`PetscBuilder`] which allows you to specify arguments when calling [`PetscInitialize`](petsc_raw::PetscInitialize).
    pub fn builder() -> PetscBuilder {
        PetscBuilder::default()
    }

    /// Calls [`PetscInitialize`] without the command line arguments.
    ///
    /// If you want to pass in Arguments use [`Petsc::builder()`].
    ///
    /// ```
    /// let petsc = petsc_rs::Petsc::init_no_args().unwrap();
    /// ```
    ///
    /// [`PetscInitialize`]: petsc_raw::PetscInitialize
    pub fn init_no_args() -> Result<Self> {
        let ierr = unsafe { petsc_raw::PetscInitializeNoArguments() };
        let petsc = Self { world: ManuallyDrop::new(mpi::topology::SystemCommunicator::world().duplicate()),
            _arg_data: None, drop_world_first: true };
        unsafe { chkerrq!(petsc.world(), ierr) }?;

        Ok(petsc)
    }

    /// Gets a reference to the PETSc comm world. 
    ///
    /// This is effectively equivalent to [`mpi::topology::SystemCommunicator::world()`]
    /// if you haven't set the comm world to something other that the system communicator 
    /// during petsc initialization using a [`PetscBuilder`].
    ///
    /// The value is functionally the same as the `PETSC_COMM_WORLD` global in the C
    /// API. If you want to use a different comm world, then you have to define that outside
    /// of the [`Petsc`] object. Read docs for [`PetscBuilder::world()`] for more information.
    pub fn world(&self) -> &UserCommunicator {
        self.world.deref()
    }

    /// Internal error checker
    ///
    /// replacement for the CHKERRQ macro in the C api
    ///
    /// unsafe because `ierr` MUST be memory equivalent to a valid [`petsc_raw::PetscErrorCodeEnum`] or `0`.
    #[doc(hidden)]
    pub(crate) unsafe fn check_error<C: Communicator>(world: &C, line: i32, func_name: &str, file_name: &str, ierr: petsc_raw::PetscErrorCode) -> Result<()> {
        // Return early if code is clean
        if ierr == 0 {
            return Ok(());
        }

        // SAFETY: This should be safe as we expect the errors to be valid. All inputs are generated from
        // Petsc functions, not user input. But we can't guarantee that they are all valid.
        // We also create the `PetscErrorCodeEnum` enum to have the same size and alignment as `i32`.
        // Which is what petsc_raw::PetscErrorCode is.
        let error_kind = std::mem::transmute(ierr);
        let error = PetscError { kind: error_kind, error: "".into() };

        let c_s_r = CString::new(error.error.to_string());
        let file_cs = CString::new(file_name).expect("`CString::new` failed");
        let func_cs = CString::new(func_name).expect("`CString::new` failed");

        let _ = petsc_raw::PetscError(world.as_raw(), line, func_cs.as_ptr(), 
            file_cs.as_ptr(), ierr, PetscErrorType::PETSC_ERROR_REPEAT,
            c_s_r.as_ref().map(|s| s.as_ptr()).unwrap_or(std::ptr::null()));

        return Err(error);
    }

    /// Function to call when an error has been detected.
    ///
    /// replacement for the SETERRQ macro in the C api.
    ///
    /// Note, this will always return an `Err`.
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # use mpi::traits::*;
    /// let petsc = petsc_rs::Petsc::init_no_args().unwrap();
    /// if petsc.world().size() != 1 {
    ///     // note, cargo wont run tests with mpi so this will never be reached
    ///     assert!(Petsc::set_error(petsc.world(), PetscErrorKind::PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!").is_err());
    /// }
    /// ```
    ///
    /// Same as [`Petsc::set_error2()`] but sets `line`, `func_name`, and `file_name` to `None`.
    pub fn set_error<C: Communicator, E>(world: &C, error_kind: PetscErrorKind, err_msg: E) -> Result<()>
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>
    {
        Petsc::set_error2(world, None, None, None, error_kind, err_msg)
    }

    /// Same as [`Petsc::set_error()`] but allows you to set the line number, function name, and file name.
    pub fn set_error2<C: Communicator, E>(world: &C, line: Option<i32>, func_name: Option<&str>, file_name: Option<&str>, error_kind: PetscErrorKind, err_msg: E) -> Result<()>
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>
    {
        let error = PetscError { kind: error_kind, error: err_msg.into() };

        let c_s_r = CString::new(error.error.to_string());
        let file_ocs = file_name.map(|file_name| CString::new(file_name).expect("`CString::new` failed"));
        let func_ocs = func_name.map(|func_name| CString::new(func_name).expect("`CString::new` failed"));

        // TODO: It would be nice to get line number, func name, and file name from the caller,
        // however, it might be better for us to use results to build a stack trace, and
        // then it wouldn't really matter what we do here.
        // Regardless, the error handling needs a lot of work to be as functional as C PETSc.
        unsafe {
            let _ = petsc_raw::PetscError(world.as_raw(), line.unwrap_or(-1), func_ocs.as_ref().map_or(std::ptr::null(), |cs| cs.as_ptr()), 
                file_ocs.as_ref().map_or(std::ptr::null(), |cs| cs.as_ptr()), error_kind as petsc_raw::PetscErrorCode,
                PetscErrorType::PETSC_ERROR_INITIAL,
                c_s_r.as_ref().map(|s| s.as_ptr()).unwrap_or(std::ptr::null()));
        }

        return Err(error);
    }

    /// Aborts PETSc program execution.
    ///
    /// Note, this does not require PETSc to be initialized.
    ///
    /// Same as [`mpi::topology::Communicator::abort()`].
    #[inline]
    #[track_caller]
    pub fn abort<C: Communicator>(world: &C, error_kind: PetscErrorKind) -> ! {
        // TODO: the c code also does this: `if (petscindebugger) abort();`
        world.abort(error_kind as i32)
    }

    /// Internal unwrap but calls [`petsc_panic!`].
    #[inline]
    #[track_caller]
    pub(crate) fn unwrap_or_abort<T, C: Communicator>(res: Result<T>, world: &C) -> T {
        // TODO: we should make this work for non PetscError error types
        match res {
            Ok(t) => t,
            Err(e) => petsc_panic!(world, e.kind, "called `Petsc::unwrap_or_abort()` on an `Err` value: {}", &e),
        }
    }

    /// replacement for the `PetscPrintf` function in the C api. 
    ///
    /// You can also use the [`petsc_println!`] macro to have string formatting.
    ///
    /// Prints to standard out, only from the first processor in the communicator.
    /// Calls from other processes are ignored.
    ///
    /// # Example
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # use mpi::traits::*;
    /// # fn main() -> petsc_rs::Result<()> {
    /// let petsc = petsc_rs::Petsc::init_no_args().unwrap();
    ///
    /// Petsc::print(petsc.world(), format!("Hello parallel world of {} processes!\n", petsc.world().size()))?;
    /// // or use:
    /// petsc_println!(petsc.world(), "Hello parallel world of {} processes!", petsc.world().size())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn print<C: Communicator, T: ToString>(world: &C, msg: T) -> Result<()> {
        let msg_cs = ::std::ffi::CString::new(msg.to_string()).expect("`CString::new` failed");

        // The first entry needs to be `%s` so that this function is not susceptible to printf injections.
        let ps = CString::new("%s").unwrap();

        let ierr = unsafe { petsc_raw::PetscPrintf(world.as_raw(), ps.as_ptr(), msg_cs.as_ptr()) };
        unsafe { chkerrq!(world, ierr) }
    }

    /// Replacement for the `PetscSynchronizedPrintf` function in the C api.
    ///
    /// You can also use the [`petsc_println_sync!`] macro to have rust string formatting.
    ///
    /// Prints synchronized output from several processors. Output of the first processor is followed by
    /// that of the second, etc.
    ///
    /// Will automatically call `PetscSynchronizedFlush` after.
    ///
    /// # Example
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # use mpi::traits::*;
    /// # fn main() -> petsc_rs::Result<()> {
    /// let petsc = petsc_rs::Petsc::init_no_args().unwrap();
    ///
    /// Petsc::print_sync(petsc.world(), format!("Hello parallel world of {} processes from process {}!\n",
    ///     petsc.world().size(), petsc.world().rank()))?;
    /// // or use:
    /// petsc_println_sync!(petsc.world(), "Hello parallel world of {} processes from process {}!", 
    ///     petsc.world().size(), petsc.world().rank())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn print_sync<C: Communicator, T: ToString>(world: &C, msg: T) -> Result<()> {
        let msg_cs = ::std::ffi::CString::new(msg.to_string()).expect("`CString::new` failed");

        // The first entry needs to be `%s` so that this function is not susceptible to printf injections.
        let ps = CString::new("%s").unwrap();

        let ierr = unsafe { petsc_raw::PetscSynchronizedPrintf(world.as_raw(), ps.as_ptr(), msg_cs.as_ptr()) };
        unsafe { chkerrq!(world, ierr) }?;

        let ierr = unsafe { petsc_raw::PetscSynchronizedFlush(world.as_raw(), petsc_raw::PETSC_STDOUT) };
        unsafe { chkerrq!(world, ierr) }
    }

    /// Gets the integer value for a particular option in the database.
    pub fn options_try_get_int(&self, name: &str) -> Result<Option<PetscInt>> {
        let name_cs = CString::new(name).expect("`CString::new` failed");
        let mut opt_val = MaybeUninit::uninit();
        let mut set = MaybeUninit::uninit();
        let ierr = unsafe { 
            petsc_raw::PetscOptionsGetInt(std::ptr::null_mut(), std::ptr::null(),
            name_cs.as_ptr(), opt_val.as_mut_ptr(), set.as_mut_ptr()) };
        unsafe { chkerrq!(self.world(), ierr) }?;

        Ok(if unsafe { set.assume_init().into() } { Some(unsafe { opt_val.assume_init() }) } 
            else { None } )
    }

    /// Gets the Logical (true or false) value for a particular option in the database.
    ///
    /// Note, TRUE, true, YES, yes, no string, and 1 all translate to `true`.
    /// FALSE, false, NO, no, and 0 all translate to `false`
    pub fn options_try_get_bool(&self, name: &str) -> Result<Option<bool>> {
        let name_cs = CString::new(name).expect("`CString::new` failed");
        let mut opt_val = MaybeUninit::uninit();
        let mut set = MaybeUninit::uninit();
        let ierr = unsafe { 
            petsc_raw::PetscOptionsGetBool(std::ptr::null_mut(), std::ptr::null(),
            name_cs.as_ptr(), opt_val.as_mut_ptr(), set.as_mut_ptr()) };
        unsafe { chkerrq!(self.world(), ierr) }?;

        Ok(if unsafe { set.assume_init().into() } { Some(unsafe { opt_val.assume_init().into() }) } 
            else { None } )
    }

    /// Gets the floating point value for a particular option in the database..
    pub fn options_try_get_real(&self, name: &str) -> Result<Option<PetscReal>> {
        let name_cs = CString::new(name).expect("`CString::new` failed");
        let mut opt_val = MaybeUninit::uninit();
        let mut set = MaybeUninit::uninit();
        let ierr = unsafe { 
            petsc_raw::PetscOptionsGetReal(std::ptr::null_mut(), std::ptr::null(),
            name_cs.as_ptr(), opt_val.as_mut_ptr(), set.as_mut_ptr()) };
        unsafe { chkerrq!(self.world(), ierr) }?;

        Ok(if unsafe { set.assume_init().into() } { Some(unsafe { opt_val.assume_init() }) } 
            else { None } )
    }

    /// Gets the string value for a particular option in the database.
    ///
    /// Gets, at most, 127 characters.
    pub fn options_try_get_string(&self, name: &str) -> Result<Option<String>> {
        let name_cs = CString::new(name).expect("`CString::new` failed");
        // TODO: is this big enough
        const BUF_LEN: usize = 128;
        let mut buf = [0 as u8; BUF_LEN];
        let mut set = MaybeUninit::uninit();
        let ierr = unsafe { 
            petsc_raw::PetscOptionsGetString(std::ptr::null_mut(), std::ptr::null(),
            name_cs.as_ptr(), buf.as_mut_ptr() as *mut _, BUF_LEN as u64, set.as_mut_ptr()) };
        unsafe { chkerrq!(self.world(), ierr) }?;

        Ok(if unsafe { set.assume_init().into() } {
            let nul_term = buf.iter().position(|&v| v == 0).unwrap();
            let c_str: &CStr = CStr::from_bytes_with_nul(&buf[..nul_term+1]).unwrap();
            let str_slice: &str = c_str.to_str().unwrap();
            let string: String = str_slice.to_owned();
            Some(string)
        } else {
            None
        })
    }

    /// Gets an option in the database (as a string), then converts to type `E`.
    ///
    /// Note, If the [`from_str`](std::str::FromStr::from_str()) fails, then
    /// [`default()`](Default::default()) is used.
    pub fn options_try_get_from_string<E>(&self, name: &str) -> Result<Option<E>>
    where
        E: Default + std::str::FromStr,
    {
        let as_str = self.options_try_get_string(name)?;
        if let Some(as_str) = as_str {
            Ok(Some(E::from_str(as_str.as_str()).unwrap_or(E::default())))
        } else {
            Ok(None)
        }
    }

    /// Gets multiple values from the options database.
    ///
    /// Uses [`PetscOpt::from_petsc_opt_builder()`].
    ///
    /// Is the same as [`Petsc::options_build()`] but with all option arguments being `None` and
    /// `title` being `"Program Option"`.
    pub fn options_get<T: PetscOpt>(&self) -> Result<T> {
        let title_cs = CString::new("Program Option").expect("`CString::new` failed");
        let mut petsc_opt_obj = MaybeUninit::<petsc_sys::PetscOptionItems>::zeroed();
        unsafe { &mut *petsc_opt_obj.as_mut_ptr() }.count = 1;
        let mut pob = PetscOptBuilder::new(self, &mut petsc_opt_obj, None, &title_cs, None)?;
        PetscOpt::from_petsc_opt_builder(&mut pob)
    }

    /// Creates a [`PetscOptBuilder`] to facilitates setting command line arguments.
    pub fn options_build<'strlt1, 'strlt2, T: PetscOpt>(&self, prefix: impl Into<Option<&'strlt1 str>>,
        title: &str, man_section: impl Into<Option<&'strlt2 str>>) -> Result<T>
    {
        let prefix_cs = prefix.into().map(|prefix| CString::new(prefix).expect("`CString::new` failed"));
        let prefix_csr = prefix_cs.as_ref().map(|cs| cs.deref());
        let title_cs = CString::new(title).expect("`CString::new` failed");
        let sec_cs = man_section.into().map(|man_section| CString::new(man_section).expect("`CString::new` failed"));
        let sec_csr = sec_cs.as_ref().map(|cs| cs.deref());

        let mut petsc_opt_obj = MaybeUninit::<petsc_sys::PetscOptionItems>::zeroed();

        for c in if unsafe { petsc_sys::PetscOptionsPublish }.into() {-1} else {1} .. 1 {
            unsafe { &mut *petsc_opt_obj.as_mut_ptr() }.count = c;
            let mut pob = PetscOptBuilder::new(self, &mut petsc_opt_obj, prefix_csr, &title_cs, sec_csr)?;

            let _ = T::from_petsc_opt_builder(&mut pob)?;
        }

        unsafe { &mut *petsc_opt_obj.as_mut_ptr() }.count = 1;
        let mut pob = PetscOptBuilder::new(self, &mut petsc_opt_obj, prefix_csr, &title_cs, sec_csr)?;
        PetscOpt::from_petsc_opt_builder(&mut pob)
    }

    /// Creates an empty vector object.
    ///
    /// Note, it will use the default comm world from [`Petsc::world()`].
    ///
    /// The type can then be set with [`Vector::set_type`], or [`Vector::set_from_options`].
    ///
    /// Note, this is the same as using [`Vector::create(petsc.world())`](Vector::create).
    ///
    /// # Example
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # use mpi::traits::*;
    /// let petsc = Petsc::init_no_args().unwrap();
    /// let vec = petsc.vec_create().unwrap();
    /// ```
    pub fn vec_create(&self) -> Result<crate::Vector> {
        crate::Vector::create(self.world())
    }

    /// Creates an empty matrix object. 
    ///
    /// Note, it will use the default comm world from [`Petsc::world()`].
    ///
    /// Note, this is the same as using [`Mat::create(petsc.world())`](Mat::create).
    ///
    /// # Example
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # use mpi::traits::*;
    /// let petsc = Petsc::init_no_args().unwrap();
    /// let mat = petsc.mat_create().unwrap();
    /// ```
    pub fn mat_create(&self) -> Result<crate::Mat> {
        crate::Mat::create(self.world())
    }

    /// Creates the default KSP context.
    ///
    /// Note, it will use the default comm world from [`Petsc::world()`].
    ///
    /// Note, this is the same as using [`KSP::create(petsc.world())`](KSP::create).
    ///
    /// # Example
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # use mpi::traits::*;
    /// let petsc = Petsc::init_no_args().unwrap();
    /// let ksp = petsc.ksp_create().unwrap();
    /// ```
    pub fn ksp_create(&self) -> Result<crate::KSP> {
        crate::KSP::create(self.world())
    }

    /// Creates the default SNES context.
    ///
    /// Note, it will use the default comm world from [`Petsc::world()`].
    ///
    /// Note, this is the same as using [`SNES::create(petsc.world())`](SNES::create).
    ///
    /// # Example
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # use mpi::traits::*;
    /// let petsc = Petsc::init_no_args().unwrap();
    /// let snes = petsc.snes_create().unwrap();
    /// ```
    pub fn snes_create(&self) -> Result<crate::SNES> {
        crate::SNES::create(self.world())
    }

    /// Creates a viewer context the prints to stdout
    ///
    /// A replacement the the C API's `PETSC_VIEWER_STDOUT_WORLD`.
    ///
    /// Note, this is the same as using [`Viewer::create_ascii_stdout(petsc.world())`](Viewer::create_ascii_stdout()).
    pub fn viewer_create_ascii_stdout(&self) -> Result<crate::Viewer> {
        Viewer::create_ascii_stdout(self.world())
    }
}

/// A rust type than can identify as a raw value understood by the PETSc C API.
pub unsafe trait PetscAsRaw {
    /// The raw PETSc C API type
    type Raw;
    /// The raw value
    fn as_raw(&self) -> Self::Raw;
}

unsafe impl<'a, T: PetscAsRaw + 'a> PetscAsRaw for &'a T {
    type Raw = <T as PetscAsRaw>::Raw;
    fn as_raw(&self) -> Self::Raw {
        (*self).as_raw()
    }
}

// We do this impl so that we want use it in the `wrap_simple_petsc_member_funcs!` macro.
unsafe impl<PT, T: PetscAsRaw<Raw = *mut PT>> PetscAsRaw for Option<T> {
    type Raw = <T as PetscAsRaw>::Raw;

    fn as_raw(&self) -> Self::Raw {
        self.as_ref().map_or(std::ptr::null_mut(), |inner| inner.as_raw())
    }
}

/// A rust type than can provide a mutable pointer to a raw value understood by the PETSc C API.
pub unsafe trait PetscAsRawMut: PetscAsRaw {
    /// A mutable pointer to the raw value
    fn as_raw_mut(&mut self) -> *mut <Self as PetscAsRaw>::Raw;
}

/// The trait that is implemented for any PETSc Object [`petsc_rs::vector::Vector`](Vector), 
/// [`petsc_rs::mat::Mat`](Mat), [`petsc_rs::ksp::KSP`](KSP), etc.
pub trait PetscObject<'a, PT>: PetscAsRaw<Raw = *mut PT> {
    /// Gets the MPI communicator world for any [`PetscObject`] regardless of type;
    fn world(&self) -> &'a UserCommunicator;

    /// Sets a string name associated with a PETSc object.
    fn set_name(&mut self, name: &str) -> crate::Result<()> {
        let name_cs = ::std::ffi::CString::new(name).expect("`CString::new` failed");
        
        let ierr = unsafe { crate::petsc_raw::PetscObjectSetName(self.as_raw() as *mut crate::petsc_raw::_p_PetscObject, name_cs.as_ptr()) };
        unsafe { chkerrq!(self.world(), ierr) }
    }

    /// Gets a string name associated with a PETSc object.
    fn get_name(&self) -> crate::Result<String> {
        let mut c_buf = ::std::mem::MaybeUninit::<*const ::std::os::raw::c_char>::uninit();
        
        let ierr = unsafe { crate::petsc_raw::PetscObjectGetName(self.as_raw() as *mut crate::petsc_raw::_p_PetscObject, c_buf.as_mut_ptr()) };
        unsafe { chkerrq!(self.world(), ierr) }?;

        let c_str = unsafe { ::std::ffi::CStr::from_ptr(c_buf.assume_init()) };
        crate::Result::Ok(c_str.to_string_lossy().to_string())
    }

    /// Determines whether a PETSc object is of a particular type (given as a string).
    ///
    /// Some types might also implement `type_compare` which takes in the PETSc object specific type enum.
    fn type_compare_str(&self, type_name: &str) -> Result<bool> {
        let type_name_cs = ::std::ffi::CString::new(type_name).expect("`CString::new` failed");
        let mut tmp = ::std::mem::MaybeUninit::<crate::petsc_raw::PetscBool>::uninit();

        let ierr = unsafe { crate::petsc_raw::PetscObjectTypeCompare(
            self.as_raw() as *mut _, type_name_cs.as_ptr(),
            tmp.as_mut_ptr()
        )};
        unsafe { chkerrq!(self.world(), ierr) }?;

        crate::Result::Ok(unsafe { tmp.assume_init() }.into())
    }

    /// Sets the prefix used for searching for all options of PetscObjectType in the database. 
    fn set_options_prefix(&mut self, prefix: &str) -> crate::Result<()> {
        let name_cs = ::std::ffi::CString::new(prefix).expect("`CString::new` failed");
        
        let ierr = unsafe { crate::petsc_raw::PetscObjectSetOptionsPrefix(self.as_raw() as *mut crate::petsc_raw::_p_PetscObject, name_cs.as_ptr()) };
        unsafe { chkerrq!(self.world(), ierr) }
    }
}

/// These are loose wrappers that are only intended to be accessed internally.
pub(crate) trait PetscObjectPrivate<'a, PT>: PetscObject<'a, PT> {
    wrap_simple_petsc_member_funcs! {
        PetscObjectReference, reference, takes mut, is unsafe, #[doc = "Indicates to any PetscObject that it is being referenced by another PetscObject. This increases the reference count for that object by one."];
        PetscObjectDereference, dereference, takes mut, is unsafe, #[doc = "Indicates to any PetscObject that it is being referenced by one less PetscObject. This decreases the reference count for that object by one."];
        PetscObjectGetReference, get_reference_count, output PetscInt, cnt, #[doc = "Gets the current reference count for any PETSc object."];
        PetscObjectGetClassId, get_class_id, output petsc_raw::PetscClassId, id, #[doc = "Gets the classid for any PetscObject"];
    }
}

/// This is a internal template struct that is used when an object could have multiple types.
///
/// For example this is used in [`dm::DM::get_field_from_c_struct()`]
// TODO: Should we impl Drop for it?
struct PetscObjectStruct<'a> {
    pub(crate) world: &'a UserCommunicator,
    pub(crate) po_p: *mut petsc_raw::_p_PetscObject,
}

impl_petsc_object_traits! { PetscObjectStruct, po_p, petsc_raw::_p_PetscObject, PetscObjectView, PetscObjectDestroy; }

// Because the `view_with` function created is private it will yell at us
// for not using it so until then I will comment this out:
// impl_petsc_view_func!{ PetscObjectStruct, PetscObjectView }

// TODO: make into a derive macro
/// This trait is used to define how to get the options in a struct from the petsc object.
///
/// # Example
///
/// ```
/// # use petsc_rs::prelude::*;
/// # use mpi::traits::*;
/// struct Opt {
///     m: PetscInt,
///     n: PetscInt,
///     view_exact_sol: bool,
/// }
/// 
/// impl PetscOpt for Opt {
///     fn from_petsc_opt_builder(pob: &mut PetscOptBuilder) -> petsc_rs::Result<Self> {
///         let m = pob.options_int("-m", "The size `m`", "doc-test", 8)?;
///         let n = pob.options_int("-n", "The size `n`", "doc-test", 7)?;
///         let view_exact_sol = pob.options_bool("-view_exact_sol", "Output the solution for verification", "doc-test", false)?;
///         Ok(Opt { m, n, view_exact_sol })
///     }
/// }
///
/// # fn main() -> petsc_rs::Result<()> {
/// let petsc = Petsc::builder()
///     .args(std::env::args())
///     .init()?;
///
/// let Opt { m, n, view_exact_sol } = petsc.options_get()?;
/// # Ok(())
/// # }
/// ```
pub trait PetscOpt
where
    Self: Sized
{
    /// Builds the struct from a [`PetscOptBuilder`] object.
    fn from_petsc_opt_builder(petsc: &mut PetscOptBuilder) -> Result<Self>;
}

/// PETSc type that represents a complex number with precision matching that of PetscReal.
#[cfg(any(feature = "petsc-use-complex", feature = "petsc-sys/petsc-use-complex"))]
pub type PetscComplex = Complex<PetscReal>;
/// PETSc type that represents a complex number with precision matching that of PetscReal.
#[cfg(not(any(feature = "petsc-use-complex", feature = "petsc-sys/petsc-use-complex")))]
pub type PetscComplex = petsc_sys::PetscComplex;

/// PETSc scalar type.
///
/// Can represent either a real or complex number in varying levels of precision. The specific 
/// representation can be set by features for [`petsc-sys`](crate).
///
/// Note, `PetscScalar` could be a complex number, so best practice is to instead of giving
/// float literals (i.e. `1.5`) when a function takes a `PetscScalar` wrap in in a `from`
/// call. E.x. `PetscScalar::from(1.5)`. This will do nothing if `PetscScalar` in a real number,
/// but if `PetscScalar` is complex it will construct a complex value with the imaginary part being
/// set to `0`.
///
/// # Example
///
/// ```
/// # use petsc_rs::prelude::*;
/// # use mpi::traits::*;
/// // This will always work
/// let a = PetscScalar::from(1.5);
/// ```
#[cfg(not(any(feature = "petsc-use-complex", feature = "petsc-sys/petsc-use-complex")))]
pub type PetscScalar = PetscReal;

/// PETSc scalar type.
///
/// Can represent either a real or complex number in varying levels of precision. The specific 
/// representation can be set by features for [`petsc-sys`](crate).
///
/// Note, `PetscScalar` could be a complex number, so best practice is to instead of giving
/// float literals (i.e. `1.5`) when a function takes a `PetscScalar` wrap in in a `from`
/// call. E.x. `PetscScalar::from(1.5)`. This will do nothing if `PetscScalar` in a real number,
/// but if `PetscScalar` is complex it will construct a complex value with the imaginary part being
/// set to `0`.
///
/// # Example
///
/// ```
/// # use petsc_rs::prelude::*;
/// # use mpi::traits::*;
/// // This will always work
/// let a = PetscScalar::from(1.5);
/// ```
#[cfg(any(feature = "petsc-use-complex", feature = "petsc-sys/petsc-use-complex"))]
pub type PetscScalar = Complex<PetscReal>;

#[cfg(doctest)]
mod readme_doctest {
    // This will run the doc tests in README.md
    #[doc = include_str!("../README.md")]
    extern {}
}
