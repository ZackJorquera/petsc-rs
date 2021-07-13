#![warn(rustdoc::broken_intra_doc_links)]
#![warn(missing_docs)]

//! # [petsc-rs](#): PETSc rust bindings
//!
//! PETSc, pronounced PET-see (/ˈpɛt-siː/), is a suite of data structures and routines for the scalable
//! (parallel) solution of scientific applications modeled by partial differential equations. PETSc supports
//! MPI through the [rsmpi](https://github.com/rsmpi/rsmpi) crate.
//!
//! # Basic Usage
//! 
//! First, you will need to add `petsc-rs` to your `Cargo.toml`. Next, to get access to all the important
//! traits and types you can use `use petsc_rs::prelude::*`. Some of the important types that are included are:
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
//! As an example, if you wanted to use a petsc install that uses PETSc with 64-bit integers,
//! 32-bit floats (single precision), and real numbers for scalars you would put the following
//! in your `Cargo.toml`
//! ```text
//! petsc-rs = { version = "*", default-features = false, features = ["petsc-real-f32", "petsc-int-i64"] }
//! ```
//!
//! # Further Reading
//! 
//! - [C API Getting Started](https://petsc.org/release/documentation/manual/getting_started/)
//!
//! - [C API Programming with PETSc/TAO](https://petsc.org/release/documentation/manual/programming/)
//!
//! - [`petsc-rs` github page](https://github.com/ZackJorquera/petsc-rs)

use std::ops::Deref;
use std::os::raw::{c_char, c_int};
use std::vec;

pub(crate) mod petsc_raw {
    pub use petsc_sys::*;
}

pub use petsc_raw::{PetscInt, PetscReal};
pub use petsc_raw::NormType;

use mpi::{self, traits::*};
use std::mem::{MaybeUninit, ManuallyDrop};
use std::ffi::{CString, CStr, };
use mpi::topology::UserCommunicator;

pub(crate) mod macros;

pub mod vector;
pub mod mat;
pub mod ksp;
#[path = "preconditioner.rs"] pub mod pc; // TODO: or should i just rename the file
pub mod viewer;
pub mod snes;
pub mod dm;
pub mod indexset;

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
        petsc_println,
        petsc_println_all,
        vector::{Vector, VecOption, },
        mat::{Mat, MatAssemblyType, MatOption, MatDuplicateOption, MatStencil, NullSpace },
        ksp::{KSP, },
        snes::{SNES, DomainOrPetscError::DomainErr, },
        pc::{PC, PCType, },
        dm::{DM, DMBoundaryType, DMDAStencilType, DMType, },
        indexset::{IS, },
        viewer::{Viewer, PetscViewerFormat, },
        NormType,
        PetscOpt,
    };
}

#[cfg(feature = "petsc-use-complex")]
use num_complex::Complex;

// https://petsc.org/release/docs/manualpages/Sys/index.html

/// Prints to standard out, only from the first processor in the communicator.
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

/// Prints synchronized output from several processors. Output of the first processor is followed by
/// that of the second, etc.
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
/// Petsc::print_all(petsc.world(), format!("Hello parallel world of {} processes from process {}!\n",
///     petsc.world().size(), petsc.world().rank()))?;
/// // or use:
/// petsc_println_all!(petsc.world(), "Hello parallel world of {} processes from process {}!", 
///     petsc.world().size(), petsc.world().rank())?;
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! petsc_println_all {
    ($world:expr) => ( Petsc::print_all($world, "\n") );
    ($world:expr, $($arg:tt)*) => ({
        let s = format!("{}\n", format_args!($($arg)*));
        Petsc::print_all($world, s)
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

pub use petsc_raw::PetscErrorType as PetscErrorType;
pub use petsc_raw::PetscErrorCodeEnum as PetscErrorKind;
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
            args.iter().map(|arg| CString::new(arg.to_string()).expect("CString::new failed"))
                .collect::<Vec<CString>>());
        let mut c_args_owned = cstr_args_owned.iter().map(|arg| arg.as_ptr() as *mut _)
            .collect::<Vec<*mut c_char>>();
        c_args_owned.push(std::ptr::null_mut());
        let mut c_args_boxed = Box::new(c_args_owned.as_mut_ptr());

        let c_args_p = self.args.as_ref().map_or(std::ptr::null_mut(), |_| &mut *c_args_boxed as *mut _);

        // Note, the file string does not need to outlive the `Petsc` type
        let file_cstring = self.file.map(|ref f| CString::new(f.to_string()).ok()).flatten();
        let file_c_str = file_cstring.as_ref().map_or_else(|| std::ptr::null(), |v| v.as_ptr());

        // We dont have to leak the file string
        let help_cstring = self.help_msg.map(|ref h| CString::new(h.to_string()).ok()).flatten();
        let help_c_str = help_cstring.as_ref().map_or_else(|| std::ptr::null(), |v| v.as_ptr());

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

                ierr = unsafe { petsc_raw::PetscInitialize(c_argc_p, c_args_p, file_c_str, help_c_str) };

                ManuallyDrop::new(world)
            }, 
            _ => {
                // Note, in this case MPI has not been initialized, it will be initialized by PETSc
                ierr = unsafe { petsc_raw::PetscInitialize(c_argc_p, c_args_p, file_c_str, help_c_str) };
                ManuallyDrop::new(mpi::topology::SystemCommunicator::world().duplicate())
            }
        }, _arg_data: self.args.as_ref().map(|_| (argc_boxed, cstr_args_owned, c_args_owned, c_args_boxed)) };
        Petsc::check_error(petsc.world(), ierr)?;

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
    _arg_data: Option<(Box<c_int>, Vec<CString>, Vec<*mut c_char>, Box<*mut *mut c_char>)>
}

// Destructor
impl Drop for Petsc {
    fn drop(&mut self) {
        // SAFETY: PetscFinalize can call MPI_FINALIZE, which means we need to make sure our 
        // comm world is dropped before that. Also after `ManuallyDrop::drop` is called `Petsc`
        // is dropped so the zombie value is never used again
        unsafe {
            ManuallyDrop::drop(&mut self.world);
            petsc_raw::PetscFinalize();
        }
    }
}

impl Petsc {
    /// Creates a [`PetscBuilder`] which allows you to specify arguments when calling [`PetscInitialize`](petsc_raw::PetscInitialize).
    pub fn builder() -> PetscBuilder
    {
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
            _arg_data: None };
        Petsc::check_error(petsc.world(), ierr)?;

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
    /// replacement for the CHKERRQ macro in the C api
    #[doc(hidden)]
    pub(crate) fn check_error<C: Communicator>(world: &C, ierr: petsc_raw::PetscErrorCode) -> Result<()> {
        // Return early if code is clean
        if ierr == 0 {
            return Ok(());
        }

        // SAFETY: This should be safe as we expect the errors to be valid. All inputs are generated from
        // Petsc functions, not user input. But we can't guarantee that they are all valid. 
        // Note, there is no way to make sure PetscErrorKind uses `u32` under the hood, but it should
        // use `u32` as long as there are no negative numbers and all variants fit in a u32 (which, right 
        // now is the case). Also note, `petsc_raw::PetscErrorCodeEnum` is defined with the #[repr(u32)] 
        // when it is create from bindgen, but this is subject to change if the c type changes.
        let error_kind = unsafe { std::mem::transmute(ierr) }; // TODO: make not use unsafe 
        let error = PetscError { kind: error_kind, error: "".into() };

        let c_s_r = CString::new(error.error.to_string());

        // TODO: add file macro and line macro if possible
        // might not matter because in debug rust will give stack trace
        unsafe {
            let _ = petsc_raw::PetscError(world.as_raw(), -1, std::ptr::null(), 
                std::ptr::null(), ierr, PetscErrorType::PETSC_ERROR_REPEAT,
                c_s_r.as_ref().map(|s| s.as_ptr()).unwrap_or(std::ptr::null()));
        }

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
    pub fn set_error<C: Communicator, E>(world: &C, error_kind: PetscErrorKind, err_msg: E) -> Result<()>
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>
    {
        let error = PetscError { kind: error_kind, error: err_msg.into() };

        let c_s_r = CString::new(error.error.to_string());

        // TODO: add file petsc func, and line if possible
        // might not matter because in debug rust will give stack trace
        unsafe {
            let _ = petsc_raw::PetscError(world.as_raw(), -1, std::ptr::null(), 
                std::ptr::null(), error_kind as petsc_raw::PetscErrorCode,
                PetscErrorType::PETSC_ERROR_INITIAL,
                c_s_r.as_ref().map(|s| s.as_ptr()).unwrap_or(std::ptr::null()));
        }

        return Err(error);
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
        Petsc::check_error(world, ierr)
    }

    /// Replacement for the `PetscSynchronizedPrintf` function in the C api.
    ///
    /// You can also use the [`petsc_println_all!`] macro to have rust string formatting.
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
    /// Petsc::print_all(petsc.world(), format!("Hello parallel world of {} processes from process {}!\n",
    ///     petsc.world().size(), petsc.world().rank()))?;
    /// // or use:
    /// petsc_println_all!(petsc.world(), "Hello parallel world of {} processes from process {}!", 
    ///     petsc.world().size(), petsc.world().rank())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn print_all<C: Communicator, T: ToString>(world: &C, msg: T) -> Result<()> {
        let msg_cs = ::std::ffi::CString::new(msg.to_string()).expect("`CString::new` failed");

        // The first entry needs to be `%s` so that this function is not susceptible to printf injections.
        let ps = CString::new("%s").unwrap();

        let ierr = unsafe { petsc_raw::PetscSynchronizedPrintf(world.as_raw(), ps.as_ptr(), msg_cs.as_ptr()) };
        Petsc::check_error(world, ierr)?;

        let ierr = unsafe { petsc_raw::PetscSynchronizedFlush(world.as_raw(), petsc_raw::PETSC_STDOUT) };
        Petsc::check_error(world, ierr)
    }

    /// Gets the integer value for a particular option in the database.
    pub fn options_try_get_int<T: ToString>(&self, name: T) -> Result<Option<PetscInt>> {
        let name_cs = CString::new(name.to_string()).expect("`CString::new` failed");
        let mut opt_val = MaybeUninit::uninit();
        let mut set = MaybeUninit::uninit();
        let ierr = unsafe { 
            petsc_raw::PetscOptionsGetInt(std::ptr::null_mut(), std::ptr::null(),
            name_cs.as_ptr(), opt_val.as_mut_ptr(), set.as_mut_ptr()) };
        Petsc::check_error(self.world(), ierr)?;

        Ok(if unsafe { set.assume_init().into() } { Some(unsafe { opt_val.assume_init() }) } 
            else { None } )
    }

    /// Gets the Logical (true or false) value for a particular option in the database.
    ///
    /// Note, TRUE, true, YES, yes, no string, and 1 all translate to `true`.
    /// FALSE, false, NO, no, and 0 all translate to `false`
    pub fn options_try_get_bool<T: ToString>(&self, name: T) -> Result<Option<bool>> {
        let name_cs = CString::new(name.to_string()).expect("`CString::new` failed");
        let mut opt_val = MaybeUninit::uninit();
        let mut set = MaybeUninit::uninit();
        let ierr = unsafe { 
            petsc_raw::PetscOptionsGetBool(std::ptr::null_mut(), std::ptr::null(),
            name_cs.as_ptr(), opt_val.as_mut_ptr(), set.as_mut_ptr()) };
        Petsc::check_error(self.world(), ierr)?;

        Ok(if unsafe { set.assume_init().into() } { Some(unsafe { opt_val.assume_init().into() }) } 
            else { None } )
    }

    /// Gets the floating point value for a particular option in the database..
    pub fn options_try_get_real<T: ToString>(&self, name: T) -> Result<Option<PetscReal>> {
        let name_cs = CString::new(name.to_string()).expect("`CString::new` failed");
        let mut opt_val = MaybeUninit::uninit();
        let mut set = MaybeUninit::uninit();
        let ierr = unsafe { 
            petsc_raw::PetscOptionsGetReal(std::ptr::null_mut(), std::ptr::null(),
            name_cs.as_ptr(), opt_val.as_mut_ptr(), set.as_mut_ptr()) };
        Petsc::check_error(self.world(), ierr)?;

        Ok(if unsafe { set.assume_init().into() } { Some(unsafe { opt_val.assume_init() }) } 
            else { None } )
    }

    /// Gets the string value for a particular option in the database.
    ///
    /// Gets, at most, 127 characters.
    pub fn options_try_get_string<T: ToString>(&self, name: T) -> Result<Option<String>> {
        let name_cs = CString::new(name.to_string()).expect("`CString::new` failed");
        // TODO: is this big enough
        const BUF_LEN: usize = 128;
        let mut buf = [0 as u8; BUF_LEN];
        let mut set = MaybeUninit::uninit();
        let ierr = unsafe { 
            petsc_raw::PetscOptionsGetString(std::ptr::null_mut(), std::ptr::null(),
            name_cs.as_ptr(), buf.as_mut_ptr() as *mut _, BUF_LEN as u64, set.as_mut_ptr()) };
        Petsc::check_error(self.world(), ierr)?;

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
    pub fn options_try_get_from_string<T: ToString, E>(&self, name: T) -> Result<Option<E>>
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
    /// Uses [`PetscOpt::from_petsc()`].
    pub fn options_try_get<T: PetscOpt>(&self) -> Result<T> {
        PetscOpt::from_petsc(self)
    }

    /// Creates an empty vector object.
    ///
    /// Note, it will use the default comm world from [`Petsc::world()`].
    ///
    /// The type can then be set with [`Vector::set_type`](#), or [`Vector::set_from_options`].
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

/// A rust type than can provide a mutable pointer to a raw value understood by the PETSc C API.
pub unsafe trait PetscAsRawMut: PetscAsRaw {
    /// A mutable pointer to the raw value
    fn as_raw_mut(&mut self) -> *mut <Self as PetscAsRaw>::Raw;
}

/// The trait that is implemented for any PETSc Object [`petsc-rs::vector::Vector`](Vector), 
/// [`petsc-rs::mat::Mat`](Mat), [`petsc-rs::ksp::KSP`](KSP), etc.
pub trait PetscObject<'a, PT>: PetscAsRaw<Raw = *mut PT> {
    /// Gets the MPI communicator world for any [`PetscObject`] regardless of type;
    fn world(&self) -> &'a UserCommunicator;

    /// Sets a string name associated with a PETSc object.
    fn set_name<T: ::std::string::ToString>(&mut self, name: T) -> crate::Result<()> {
        let name_cs = ::std::ffi::CString::new(name.to_string()).expect("`CString::new` failed");
        
        let ierr = unsafe { crate::petsc_raw::PetscObjectSetName(self.as_raw() as *mut crate::petsc_raw::_p_PetscObject, name_cs.as_ptr()) };
        Petsc::check_error(self.world(), ierr)
    }

    /// Gets a string name associated with a PETSc object.
    fn get_name(&self) -> crate::Result<String> {
        let mut c_buf = ::std::mem::MaybeUninit::<*const ::std::os::raw::c_char>::uninit();
        
        let ierr = unsafe { crate::petsc_raw::PetscObjectGetName(self.as_raw() as *mut crate::petsc_raw::_p_PetscObject, c_buf.as_mut_ptr()) };
        Petsc::check_error(self.world(), ierr)?;

        let c_str = unsafe { ::std::ffi::CStr::from_ptr(c_buf.assume_init()) };
        crate::Result::Ok(c_str.to_string_lossy().to_string())
    }

    /// Determines whether a PETSc object is of a particular type (given as a string). 
    fn type_compare<T: ToString>(&self, type_name: T) -> Result<bool> {
        let type_name_cs = ::std::ffi::CString::new(type_name.to_string()).expect("`CString::new` failed");
        let mut tmp = ::std::mem::MaybeUninit::<crate::petsc_raw::PetscBool>::uninit();

        let ierr = unsafe { crate::petsc_raw::PetscObjectTypeCompare(
            self.as_raw() as *mut _, type_name_cs.as_ptr(),
            tmp.as_mut_ptr()
        )};
        Petsc::check_error(self.world(), ierr)?;

        crate::Result::Ok(unsafe { tmp.assume_init() }.into())
    }
}

/// These are loose wrappers that are only intended to be accessed internally.
pub(crate) trait PetscObjectPrivate<'a, PT>: PetscObject<'a, PT> {
    wrap_simple_petsc_member_funcs! {
        // TODO: should these be unsafe? for is it fine not to because the are internal?
        PetscObjectReference, reference, takes mut, is unsafe, #[doc = "Indicates to any PetscObject that it is being referenced by another PetscObject. This increases the reference count for that object by one."];
        PetscObjectDereference, dereference, takes mut, is unsafe, #[doc = "Indicates to any PetscObject that it is being referenced by one less PetscObject. This decreases the reference count for that object by one."];
        PetscObjectGetReference, get_reference_count, output PetscInt, cnt, #[doc = "Gets the current reference count for any PETSc object."];
    }
}

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
///     fn from_petsc(petsc: &Petsc) -> petsc_rs::Result<Self> {
///         let m = petsc.options_try_get_int("-m")?.unwrap_or(8);
///         let n = petsc.options_try_get_int("-n")?.unwrap_or(7);
///         let view_exact_sol = petsc.options_try_get_bool("-view_exact_sol")?.unwrap_or(false);
///         Ok(Opt { m, n, view_exact_sol })
///     }
/// }
///
/// # fn main() -> petsc_rs::Result<()> {
/// let petsc = Petsc::builder()
///     .args(std::env::args())
///     .init()?;
///
/// let Opt { m, n, view_exact_sol } = Opt::from_petsc(&petsc)?;
/// # Ok(())
/// # }
/// ```
pub trait PetscOpt
where
    Self: Sized
{
    /// Builds the struct from a [`Petsc`] object.
    fn from_petsc(petsc: &Petsc) -> Result<Self>;
}

// We want to expose the complex type using the num-complex Complex type
// which has the same memory layout as the one bindgen creates, `__BindgenComplex`.
// TODO: is this the best way to do this? If we are just going to transmute to convert 
// why dont we ignore these types from bindgen a manually define them our selves like
// we did for MPI_Comm.

/// PETSc type that represents a complex number with precision matching that of PetscReal.
#[cfg(any(feature = "petsc-use-complex", feature = "petsc-sys/petsc-use-complex"))]
pub type PetscComplex = Complex<PetscReal>;
/// PETSc type that represents a complex number with precision matching that of PetscReal.
#[cfg(not(any(feature = "petsc-use-complex", feature = "petsc-sys/petsc-use-complex")))]
pub type PetscComplex = petsc_sys::PetscComplex;

// TODO: I dont like how i have to do the doc string twice
/// PETSc scalar type.
///
/// Can represent either a real or complex number in varying levels of precision. The specific 
/// representation can be set by features for [`petsc-sys`](crate).
///
/// Note, `PetscScalar` could be a complex number, so best practice is to instead of giving
/// float literals (i.e. `1.5`) when a function takes a `PetscScalar` wrap in in a `from`
/// call. E.x. `PetscScalar::from(1.5)`. This will do nothing if `PetscScalar` in a real number,
/// but if `PetscScalar` is complex it will construct a complex value which the imaginary part being
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
/// but if `PetscScalar` is complex it will construct a complex value which the imaginary part being 
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

// This is a hack so that we can get around the arbitrary expressions in
// key-value attributes issue. This wont be needed in rust v1.54
#[cfg(doctest)]
mod readme_doctest {
    macro_rules! doctest_readme {
        ($s:expr) => {
            #[doc = $s]
            extern {}
        };
    }
    // This will run the doc tests in README.md
    doctest_readme!(include_str!("../README.md"));
}
