#![warn(rustdoc::broken_intra_doc_links)]
#![warn(missing_docs)]

//! # petsc-rs: PETSc rust bindings
//!
//! read <https://petsc.org/release/documentation/manual/getting_started>

use std::os::raw::c_char;
use std::vec;

pub(crate) mod petsc_raw {
    pub use petsc_sys::*;
}

pub(crate) mod macros;

pub mod vector;
pub mod mat;
pub mod ksp;
#[path = "preconditioner.rs"] pub mod pc; // TODO: or should i just rename the file
pub mod viewer;
pub mod snes;

pub mod prelude {
    //! Commonly used items.
    pub use crate::{
        Petsc,
        PetscErrorKind,
        InsertMode,
        petsc_println,
        vector::{Vector, NormType, VecOption, },
        mat::{Mat, MatAssemblyType, MatOption, MatDuplicateOption, },
        ksp::{KSP, },
        snes::{SNES, },
        pc::{PC, PCType, },
        viewer::{Viewer, PetscViewerFormat, },
    };
    pub use mpi::traits::*;
    pub(crate) use crate::Result;
    pub(crate) use mpi::{self, traits::*};
    pub(crate) use crate::petsc_raw;
    pub(crate) use std::mem::MaybeUninit;
    pub(crate) use std::ffi::CString;
    pub(crate) use std::rc::Rc;
}

use prelude::*;

// https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/index.html

// TODO: Would it make sense to only have PetscInitializeNoArguments and have rust deal with all the 
// options database.

// TODO: should all Petsc types be reference counted, are should we force functions that create refrences 
// (i.e. call `PetscObjectReference`) to take `Rc`s (or `Arc`s, idk if it needs to be thread safe (i would 
// guess not)). Or should it all be handled under the hood. It seems like this is not really a public
// facing function. And it is unclear how to remove a reference count, like it seems that calling destroy
// or PetscObjectDereference will decrement the count.

// TODO: we should accept PetscScalar, but we will just assume that it is always a f64 for now.
// We need to add support for PetscScalar being complex, or being a different size float.
// It seems like this is a compile time thing so it isn't as important to add right now.
// https://petsc.org/release/docs/manualpages/Sys/PetscScalar.html#PetscScalar

/// Prints to standard out, only from the first processor in the communicator.
/// Calls from other processes are ignored.
///
/// Note, the macro internally uses the try operator, `?`, so it can only be used
/// in functions that return a [`petsc_rs::Result`](Result).
///
/// # Example
///
/// ```
/// # use petsc_rs::prelude::*;
/// # fn main() -> petsc_rs::Result<()> {
/// let petsc = petsc_rs::Petsc::init_no_args().unwrap();
///
/// petsc_println!(petsc.world(), "Hello parallel world of {} processes!", petsc.world().size());
/// // This will print just a new line
/// petsc_println!(petsc.world());
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! petsc_println {
    ($world:expr) => (Petsc::print($world, "\n")?);
    ($world:expr, $($arg:tt)*) => ({
        Petsc::print($world, format_args!($($arg)*))?;
        Petsc::print($world, "\n")?;
    })
}

/// PETSc result
pub type Result<T> = std::result::Result<T, PetscError>;

/// PETSc Error type.
///
/// Can be used with [`Petsc::set_error`].
///
/// [`Petsc::set_error`]: Petsc::set_error
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
/// ```
/// # use petsc_rs::prelude::*;
/// let univ = mpi::initialize().unwrap();
/// let petsc = Petsc::builder()
///     .args(std::env::args())
///     // Note, if we don't split the comm world and just use the default world
///     // then there is no need to use the world method as we do here.
///     .world(Box::new(univ.world()))
///     .help_msg("Hello, this is a help message\n")
///     .init().unwrap();
/// ```
///
/// Note `Petsc::builder().init()` is the same as [`Petsc::init_no_args()`].
///
/// [`PetscInitialize`]: petsc_raw::PetscInitialize
#[derive(Default)]
pub struct PetscBuilder
{
    world: Option<Box<dyn Communicator>>,
    args: Option<Vec<String>>,
    file: Option<String>,
    help_msg: Option<String>,

}

impl PetscBuilder
{
    /// Calls [`PetscInitialize`] with the options given.
    /// Initializes the PETSc database and MPI. Will also call MPI_Init() if that has
    /// yet to be called, so this routine should always be called near the beginning
    /// of your program -- usually the very first line!
    ///
    /// [`PetscInitialize`]: petsc_raw::PetscInitialize
    pub fn init(self) -> Result<Petsc>
    {
        let mut argc;
        let c_argc_p = if let Some(ref args) = self.args { 
            argc = args.len() as i32;
            &mut argc as *mut i32
        } else { 
            argc = 0;
            std::ptr::null_mut()
        };

        // I think we want to leak args because Petsc stores it as a global
        // At least we need to make sure it lives past the call to `PetscFinalize`
        // We heap allocate args with a Vec and then leak it
        let mut args_array = vec![std::ptr::null_mut(); argc as usize];
        let vec_cap = args_array.capacity();
        let mut c_args = std::ptr::null_mut();
        let c_args_p = if let Some(ref args) = self.args {
            for (arg, c_arg_p) in args.iter().zip(args_array.iter_mut()) {
                // `CString::into_raw` will leak the data.
                *c_arg_p = CString::new(arg.to_string()).expect("CString::new failed").into_raw();
            }

            c_args = args_array.leak().as_mut_ptr() as *mut *mut c_char;

            &mut c_args as *mut *mut *mut c_char
        } else {
            std::ptr::null_mut()
        };
        // Note: we leak each arg (as CString) and args_array.

        // We dont have to leak the file string
        let file_cstring = self.file.map(|ref f| CString::new(f.to_string()).ok()).flatten();
        let file_c_str = file_cstring.as_ref().map_or_else(|| std::ptr::null(), |v| v.as_ptr());

        // We dont have to leak the file string
        let help_cstring = self.help_msg.map(|ref h| CString::new(h.to_string()).ok()).flatten();
        let help_c_str = help_cstring.as_ref().map_or_else(|| std::ptr::null(), |v| v.as_ptr());

        let ierr = unsafe { petsc_raw::PetscInitialize(c_argc_p, c_args_p, file_c_str, help_c_str) };
        // We pass in the args data so that we can reconstruct the vec to free all the memory.
        let petsc = Petsc { world: match self.world { 
            Some(world) => {
                unsafe { petsc_raw::PETSC_COMM_WORLD = world.as_raw(); }
                world
            }, 
            _ => Box::new(mpi::topology::SystemCommunicator::world()) 
        }, raw_args_vec_data: self.args.map(|_| (c_args, argc as usize, vec_cap)) };
        Petsc::check_error(petsc.world(), ierr)?;

        Ok(petsc)
    }

    /// The command line arguments
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
    /// After you call [`PetscBuilder::init()`], the value returned by [`Petsc::world()`] is the value set here.
    pub fn world(mut self, world: Box<dyn Communicator>) -> Self
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
/// Also stores a reference to the the `MPI_COMM_WORLD`/`PETSC_COMM_WORLD` variable.
pub struct Petsc {
    // This is functionally the same as `PETSC_COMM_WORLD` in the C api
    pub(crate) world: Box<dyn Communicator>,

    // This is used to drop the args data
    raw_args_vec_data: Option<(*mut *mut c_char, usize, usize)>,
}

// Destructor
impl Drop for Petsc {
    fn drop(&mut self) {
        unsafe {
            petsc_raw::PetscFinalize();
        }

        if let Some((ptr, len, cap)) = self.raw_args_vec_data
        {
            // SAFETY: The vec ptr was created with `Vec::leak` and the str_ptr's were created
            // with `CString::into_raw`. Everything should be valid.
            let vec = unsafe { Vec::from_raw_parts(ptr, len, cap) };
            vec.iter().for_each(|str_ptr| { let _ = unsafe { CString::from_raw(*str_ptr) }; });
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
    /// Will call [`PetscFinalize`] on drop.
    ///
    /// Same as [`PetscInitializeNoArguments`]
    ///
    /// If you want to pass in Arguments use [`Petsc::builder`].
    ///
    /// ```
    /// let petsc = petsc_rs::Petsc::init_no_args().unwrap();
    /// ```
    ///
    /// [`PetscInitialize`]: petsc_raw::PetscInitialize
    /// [`PetscFinalize`]: petsc_raw::PetscFinalize
    /// [`PetscInitializeNoArguments`]: petsc_raw::PetscInitializeNoArguments
    /// [`Petsc::builder`]: Petsc::builder
    pub fn init_no_args() -> Result<Self> {
        let ierr = unsafe { petsc_raw::PetscInitializeNoArguments() };
        let petsc = Self { world: Box::new(mpi::topology::SystemCommunicator::world()), raw_args_vec_data: None };
        Petsc::check_error(petsc.world(), ierr)?;

        Ok(petsc)
    }

    /// Gets a reference to the PETSc comm world. 
    ///
    /// This is effectively equivalent to  [`mpi::topology::SystemCommunicator::world()`]
    /// if you haven't set the comm world to something other that the system communicator 
    /// during petsc initialization using a [`PetscBuilder`].
    ///
    /// The value is functionally the same as the `PETSC_COMM_WORLD` global in the C
    /// API. If you want to use a different comm world, then you have to define that outside
    /// of the [`Petsc`] object. Read docs for [`PetscBuilder::world()`] for more information.
    ///
    /// [`mpi::topology::SystemCommunicator::world`]: mpi::topology::SystemCommunicator::world
    pub fn world<'a>(&'a self) -> &'a dyn Communicator {
        self.world.as_ref()
    }

    // TODO: move out of `Petsc` maybe
    /// Internal error checker
    /// replacement for the CHKERRQ macro in the C api
    #[doc(hidden)]
    pub(crate) fn check_error(world: &dyn Communicator, ierr: petsc_raw::PetscErrorCode) -> Result<()> {
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

    // TODO: move out of `Petsc` maybe
    /// Function to call when an error has been detected.
    /// replacement for the SETERRQ macro in the C api.
    /// Will always return an `Err`.
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// let petsc = petsc_rs::Petsc::init_no_args().unwrap();
    /// if petsc.world().size() != 1 {
    ///     // note, cargo wont run tests with mpi so this will never be reached
    ///     assert!(Petsc::set_error(petsc.world(), PetscErrorKind::PETSC_ERROR_WRONG_MPI_SIZE, "This is a uniprocessor example only!").is_err());
    /// }
    /// ```
    ///
    pub fn set_error<E>(world: &dyn Communicator, error_kind: PetscErrorKind, err_msg: E) -> Result<()>
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

    // TODO: move out of `Petsc` maybe
    /// replacement for the `PetscPrintf` function in the C api. You can also use the [`petsc_println`] macro
    /// to have string formatting.
    /// Prints to standard out, only from the first processor in the communicator. Calls from other processes are ignored.
    ///
    /// # Example
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # fn main() -> petsc_rs::Result<()> {
    /// let petsc = petsc_rs::Petsc::init_no_args().unwrap();
    ///
    /// Petsc::print(petsc.world(), format!("Hello parallel world of {} processes!\n", petsc.world().size()))?;
    /// // or use:
    /// petsc_println!(petsc.world(), "Hello parallel world of {} processes!", petsc.world().size());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [`petsc_println`]: petsc_println
    //#[doc(hidden)]
    pub fn print<T: ToString>(world: &dyn Communicator, msg: T) -> Result<()> {
        let msg_cs = ::std::ffi::CString::new(msg.to_string()).expect("`CString::new` failed");

        // The first entry needs to be `%s` so that this function is not susceptible to printf injections.
        let ps = CString::new("%s").unwrap();

        let ierr = unsafe { petsc_raw::PetscPrintf(world.as_raw(), ps.as_ptr(), msg_cs.as_ptr()) };
        Petsc::check_error(world, ierr)
    }

    /// Creates an empty vector object.
    ///
    /// Note, it will use the default comm world from [`Petsc::world()`].
    ///
    /// The type can then be set with [`Vector::set_type`](#), or [`Vector::set_from_options`].
    /// Same as [`Vector::create`].
    ///
    /// # Example
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
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
    /// # Example
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
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
    /// # Example
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
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
    /// # Example
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// let petsc = Petsc::init_no_args().unwrap();
    /// let snes = petsc.snes_create().unwrap();
    /// ```
    pub fn snes_create(&self) -> Result<crate::SNES> {
        crate::SNES::create(self.world())
    }
}
