// TODO: write macros to do some easy stuff
#![macro_use]

// TODO: make macro use `::std::*` or `crate::*` for everything

/// Internal macro used to make wrappers of "simple" Petsc function
///
/// This macro wraps a "simple" PETSc function that takes any number of inputs and returns any number of outputs.
/// You can also set if the function takes a mutable reference to self or immutable.
/// These can be repeated multiple times to define multiple methods in one macro call.
/// Also supports doc strings
/// It is "simple" if it no type conversion is need between the petsc-rs types and the ones used by petsc-raw (or petsc-sys)
/// 
/// # Usage
///
/// Say we have a petsc function like `VecNorm` which is defined to be `pub unsafe fn VecNorm(arg1: Vec, arg2: NormType, arg3: *mut PetscReal) -> PetscErrorCode`
/// and we want our rust wrapper to be `pub fn norm(&self, norm_type: NormType) -> crate::Result<f64>`.
/// We can then use this macro in the following way:
///
/// ```ignore
/// wrap_simple_petsc_member_funcs! {
///     VecNorm, norm, vec_p, input NormType, norm_type, output f64, tmp1, #[doc = "Computes the vector norm."];
/// }
/// ```
///
/// For a more general case lets say we have a Petsc function called `TestSetABRetCD`
/// that is defined as follows: 
/// `pub unsafe fn TestSetABRetCD(arg1: Test, arg2: PetscInt, arg3: PetscReal, arg4: *mut PetscReal, arg5: *mut PetscInt) -> PetscErrorCode`
/// It takes in two inputs `arg2` and `arg3` and returns two outputs with `arg4` and
/// `arg5`. It acts on the made-up Petsc type `Test` that in rust we define as follows:
/// ```ignore
/// pub struct Test<'a> {
///     pub(crate) world: &'a dyn Communicator,
///     pub(crate) test_p: petsc_raw::Test,
/// }
/// ```
/// Note, you need to have a member `world: &'a dyn Communicator` for this macro to work.
///
/// We can then using the macro in the following way to create the function 
/// `pub fn set_ab_ret_cd(&mut self, a: i32, b: f64) -> crate::Result<(f64, i32)>`
/// 
/// ```ignore
/// impl<'a> Test<'a> {
///     wrap_simple_petsc_member_funcs! {
///         TestSetABRetCD, set_ab_ret_cd, test_p, input i32, a, input f64, b, 
/// //         ^                  ^          ^        ^
/// //      Petsc func name       |   pointer member  |- Then for each input
/// //                       rust func name          put `input type, param_name,`
///             output f64, c, output i32, d, takes mut, #[doc = "doc-string"];
/// //           ^                              ^
/// //           |- for each output             |- If you want the method to take
/// //          put `output type, tmp_name`     a `&mut self` put `takes mut,`
///     }
/// }
/// ```
/// Note, the number of input, the number of outputs, and if it takes a mutable
/// reference to self is all variable/optional and can be set to what even you need.
macro_rules! wrap_simple_petsc_member_funcs {
    // This is the most general case for the wrapper macro. It wraps a PETSc function that takes any number of input
    // and returns any number one output. You can also set if the function takes a mutable reference or not
    // These can be repeated multiple times to define multiple methods.
    // TODO: should i switch the order of input out put to take `input $param_name:ident: $param_type:ty,`
    // There are couple of ways to make this macro more readable, I could also add for parentheses.
    {$(
        $raw_func:ident, $new_func:ident, $raw_ptr_var:ident, $(input $param_type:ty, $param_name:ident,)* $(output $ret_type:ty, $tmp_ident:ident,)* $(takes $mut_tag:tt,)? #[$doc:meta];
    )*} => {
$(
    #[$doc]
    #[allow(unused_parens)]
    pub fn $new_func(& $($mut_tag)? self, $( $param_name: $param_type ),*) -> crate::Result<($( $ret_type ),*)>
    {
        $(
            let mut $tmp_ident = ::std::mem::MaybeUninit::<$ret_type>::uninit();
        )*
        let ierr = unsafe { crate::petsc_raw::$raw_func(self.$raw_ptr_var, $( $param_name.into(), )* $( $tmp_ident.as_mut_ptr() as *mut _ ),* )};
        Petsc::check_error(self.world, ierr)?;

        #[allow(unused_unsafe)]
        crate::Result::Ok(unsafe { ( $( $tmp_ident.assume_init() ),* ) })
    }
)*
    };
}


/// This macro is used specifically to wrap PETSc preallocate functions. It cover all the different 
/// input patterns for that. 
/// These can be repeated multiple times to define multiple like methods.
// TODO: make into one case like `wrap_simple_petsc_member_funcs!` if we can
macro_rules! wrap_prealloc_petsc_member_funcs {
    {$(
        $raw_func:ident, $new_func:ident, $raw_ptr_var:ident, $arg1:ident, $arg2:ident, #[$doc:meta];
    )*} => {
$(
    #[$doc]
    pub fn $new_func(&mut self, $arg1: crate::PetscInt, $arg2: ::std::option::Option<&[crate::PetscInt]>) -> crate::Result<()> {
        let ierr = unsafe { crate::petsc_raw::$raw_func(self.$raw_ptr_var, $arg1, 
            $arg2.map(|o| o.as_ptr()).unwrap_or(::std::ptr::null())) };
        Petsc::check_error(self.world, ierr)
    }
)*
    };
    {$(
        $raw_func:ident, $new_func:ident, $raw_ptr_var:ident, $arg1:ident, $arg2:ident, $arg3:ident, $arg4:ident, #[$doc:meta];
    )*} => {
$(
    #[$doc]
    pub fn $new_func(&mut self, $arg1: crate::PetscInt, $arg2: ::std::option::Option<&[crate::PetscInt]>, $arg3: crate::PetscInt, $arg4: ::std::option::Option<&[crate::PetscInt]>) -> crate::Result<()> {
        let ierr = unsafe { crate::petsc_raw::$raw_func(self.$raw_ptr_var, $arg1, 
            $arg2.map(|o| o.as_ptr()).unwrap_or(::std::ptr::null()), $arg3,
            $arg4.map(|o| o.as_ptr()).unwrap_or(::std::ptr::null())) };
        Petsc::check_error(self.world, ierr)
    }
)*
    };
    {$(
        $raw_func:ident, $new_func:ident, $raw_ptr_var:ident, $arg1:ident, $arg2:ident, $arg3:ident, #[$doc:meta];
    )*} => {
$(
    #[$doc]
    pub fn $new_func(&mut self, $arg1: crate::PetscInt, $arg2: crate::PetscInt, $arg3: ::std::option::Option<&[crate::PetscInt]>) -> crate::Result<()> {
        let ierr = unsafe { crate::petsc_raw::$raw_func(self.$raw_ptr_var, $arg1, $arg2,
            $arg3.map(|o| o.as_ptr()).unwrap_or(::std::ptr::null())) };
        Petsc::check_error(self.world, ierr)
    }
)*
    };
    {$(
        $raw_func:ident, $new_func:ident, $raw_ptr_var:ident, $arg1:ident, $arg2:ident, $arg3:ident, $arg4:ident, $arg5:ident, #[$doc:meta];
    )*} => {
$(
    #[$doc]
    pub fn $new_func(&mut self, $arg1: crate::PetscInt, $arg2: crate::PetscInt, $arg3: ::std::option::Option<&[crate::PetscInt]>, $arg4: crate::PetscInt, $arg5: ::std::option::Option<&[crate::PetscInt]>) -> crate::Result<()> {
        let ierr = unsafe { crate::petsc_raw::$raw_func(self.$raw_ptr_var, $arg1, $arg2,
            $arg3.map(|o| o.as_ptr()).unwrap_or(::std::ptr::null()), $arg4,
            $arg5.map(|o| o.as_ptr()).unwrap_or(::std::ptr::null())) };
        Petsc::check_error(self.world, ierr)
    }
)*
    };
}

/// Defines `set_name`, `get_name`, and TODO: add more maybe
macro_rules! impl_petsc_object_funcs {
    ($struct_name:ident, $raw_ptr_var:ident $(, $add_lt:lifetime)*) => {
        impl<'a> $struct_name<'a, $( $add_lt ),*>
        {
            /// Sets a string name associated with a PETSc object.
            pub fn set_name<T: ::std::string::ToString>(&mut self, name: T) -> crate::Result<()> {
                let name_cs = ::std::ffi::CString::new(name.to_string()).expect("`CString::new` failed");
                
                let ierr = unsafe { crate::petsc_raw::PetscObjectSetName(self.$raw_ptr_var as *mut crate::petsc_raw::_p_PetscObject, name_cs.as_ptr()) };
                Petsc::check_error(self.world, ierr)
            }

            /// Gets a string name associated with a PETSc object.
            pub fn get_name(&self) -> crate::Result<String> {
                let mut c_buf = ::std::mem::MaybeUninit::<*const ::std::os::raw::c_char>::uninit();
                
                let ierr = unsafe { crate::petsc_raw::PetscObjectGetName(self.$raw_ptr_var as *mut crate::petsc_raw::_p_PetscObject, c_buf.as_mut_ptr()) };
                Petsc::check_error(self.world, ierr)?;

                let c_str = unsafe { ::std::ffi::CStr::from_ptr(c_buf.assume_init()) };
                crate::Result::Ok(c_str.to_string_lossy().to_string())
            }

            // TODO: add PetscObjectRef and PetscObjectDeref, but make them unsafe for now
        }
    };
}

// defines `view_with`
macro_rules! impl_petsc_view_func {
    ($struct_name:ident, $raw_ptr_var:ident, $raw_view_func:ident $(, $add_lt:lifetime)*) => {
        impl<'a> $struct_name<'a, $( $add_lt ),*>
        {
            /// Views the object with a viewer
            pub fn view_with(&self, viewer: &crate::viewer::Viewer) -> crate::Result<()> {
                
                let ierr = unsafe { crate::petsc_raw::$raw_view_func(self.$raw_ptr_var, viewer.viewer_p) };
                Petsc::check_error(self.world, ierr)
            }
        }
    };
}
