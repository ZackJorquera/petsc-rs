// TODO: write macros to do some easy stuff
#![macro_use]

// TODO: make macro use `::std::*` or `crate::*` for everything

/// Internal macro used to make wrappers of "simple" Petsc function
///
/// Note, using this macro can be unsafe, the methods that it creates use unsafe code and thus if you
/// incorrectly use the macro you can create code that does not work (but might compile).
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
/// For a more general case lets say we have a Petsc function called `VecSetABRetCD`
/// that is defined as follows (from bindgen): 
/// `pub unsafe fn VecSetABRetCD(arg1: Vec, arg2: PetscInt, arg3: PetscReal, arg4: *mut PetscReal, arg5: *mut PetscInt) -> PetscErrorCode`
/// It takes in two inputs `arg2` and `arg3` and returns two outputs with `arg4` and
/// `arg5` (through pointers). For refrence the rust `petsc-rs::Vector` type is defined as the following:
/// ```ignore
/// pub struct Vector<'a> {
///     pub(crate) world: &'a dyn Communicator,
///     pub(crate) test_p: petsc_raw::Vec,
/// }
/// ```
/// Note, you need to have a member `world: &'a dyn Communicator` and some pointer type to the C petsc type
/// for this macro to work.
///
/// We can then using the macro in the following way to create the function 
/// `pub fn set_ab_ret_cd(&mut self, a: i32, b: f64) -> crate::Result<(f64, i32)>`
/// Note, you should use `PetscInt` and `PetscReal` instead of `i32` and `f64`.
/// ```ignore
/// impl Vec<'_> {
///     wrap_simple_petsc_member_funcs! {
///         VecSetABRetCD, pub set_ab_ret_cd, vec_p, input i32, a, input f64, b, 
/// //         ^            ^         ^          ^        ^
/// //    Petsc func name   └- vis    |   pointer member  |- Then for each input
/// //                           rust func name          put `input type, param_name,`
///             output f64, c, output i32, d, takes mut, #[doc = "doc-string"];
/// //           ^                              ^
/// //           |- for each output             |- If you want the method to take
/// //          put `output type, tmp_name`     a `&mut self` put `takes mut,`
///     }
/// }
/// ```
/// Note, the number of input, the number of outputs, and if it takes a mutable
/// reference to self is all variable/optional and can be set to what even you need.
///
/// ## More Advanced Input
///
/// `PetscScalar` can be a complex type, in the pets-rc side we use a different Complex type
/// than is used by the raw function so some automatic conversion is done for you to accommodate
/// this for both in input type and the output type.
///
/// Note, the input type can differ from the raw input type if `.into()` can be use
/// for conversion. This is done automatically. If the inputs to the rust wrapper function
/// is a struct, like `Vector` you can also use the macro to get a member value.
/// ```ignore
/// impl Vec<'_> {
///     wrap_simple_petsc_member_funcs! {
///         VecAXPY, pub axpy, vec_p, input PetscScalar, alpha, input &Vector,
///             other .vec_p, #[doc = "doc-string"];
/// //                ^ just add `.member_name` after the param_name
///     }
/// }
/// ```
/// 
/// If the rust wrapper output type shares the same memory layout as the type used by the
/// raw Petsc function, than nothing needs to be done as a pointer cast is done
/// automatically. If you with to do a conversion that requires an into you can do something
/// like the following.
/// ```ignore
/// impl NullSpace<'_> {
///     wrap_simple_petsc_member_funcs! {
///         MatNullSpaceTest, pub test, ns_p, input &Mat, vec .mat_p,
///         output bool, is_null .into from petsc_raw::PetscBool, #[doc = "doc-string"];
/// //                           ^      ^      ^
/// //                    add `.into`   |      |
/// //                            Then add `from <original_type>`
///     }
/// }
/// ```
/// Note, the `original_type` is the type that is given to the raw petsc function.
/// It most likely will be the same type that the raw function wants, but only must
/// shares the same memory layout as the type used by the raw Petsc function, as a
/// pointer cast is done automatically.
macro_rules! wrap_simple_petsc_member_funcs {
    {$(
        $raw_func:ident, $vis_par:vis $new_func:ident, $raw_ptr_var:ident,
        $(input $param_type:ty, $param_name:ident $(.$member_name:ident)? ,)*
        $(output $ret_type:ty, $tmp_ident:ident $(.$into_fn:ident from $raw_ret_type:ty)? ,)*
        $(takes $mut_tag:tt,)? #[$doc:meta];
    )*} => {
$(
    #[$doc]
    #[allow(unused_parens)]
    $vis_par fn $new_func(& $($mut_tag)? self, $( $param_name: $param_type ),*)
        -> crate::Result<($( $ret_type ),*)>
    {
        $(
            let mut $tmp_ident = ::std::mem::MaybeUninit $(::<$raw_ret_type>)? ::uninit();
        )*
        let ierr = unsafe { crate::petsc_raw::$raw_func(
            self.$raw_ptr_var,
            $( $param_name $(.$member_name)?.into() , )*
            $( $tmp_ident.as_mut_ptr() as *mut _ ),*
        )};
        Petsc::check_error(self.world, ierr)?;

        #[allow(unused_unsafe)]
        crate::Result::Ok(unsafe { ( $( $tmp_ident.assume_init() $(.$into_fn())? ),* ) })
    }
)*
    };
}


/// This macro is used specifically to wrap PETSc preallocate functions. It cover all the different 
/// input patterns for that. 
/// These can be repeated multiple times to define multiple like methods.
macro_rules! wrap_prealloc_petsc_member_funcs {
    {$(
        $raw_func:ident, $new_func:ident, $raw_ptr_var:ident, $(block $arg1:ident,)? $(nz $arg2:ident, $arg3:ident,)+ #[$doc:meta];
    )*} => {
$(
    #[$doc]
    pub fn $new_func(&mut self, $($arg1: PetscInt,)? $($arg2: PetscInt, $arg3: ::std::option::Option<&[PetscInt]>),+) -> crate::Result<()> {
        let ierr = unsafe { crate::petsc_raw::$raw_func(
            self.$raw_ptr_var, 
            $( $arg1, )?
            $(
                $arg2,
                $arg3.map(|o| o.as_ptr()).unwrap_or(::std::ptr::null()) 
            ),+ ) };
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

            /// Determines whether a PETSc object is of a particular type (given as a string). 
            pub fn type_compare<T: ToString>(&self, type_name: T) -> Result<bool> {
                let type_name_cs = ::std::ffi::CString::new(type_name.to_string()).expect("`CString::new` failed");
                let mut tmp = ::std::mem::MaybeUninit::<crate::petsc_raw::PetscBool>::uninit();

                let ierr = unsafe { crate::petsc_raw::PetscObjectTypeCompare(
                    self.$raw_ptr_var as *mut _, type_name_cs.as_ptr(),
                    tmp.as_mut_ptr()
                )};
                Petsc::check_error(self.world, ierr)?;

                #[allow(unused_unsafe)]
                crate::Result::Ok(unsafe { tmp.assume_init() }.into())
            }
        }
    };
}

// defines `view_with`
macro_rules! impl_petsc_view_func {
    ($struct_name:ident, $raw_ptr_var:ident, $raw_view_func:ident $(, $add_lt:lifetime)*) => {
        impl<'a> $struct_name<'a, $( $add_lt ),*>
        {
            /// Views the object with a viewer
            pub fn view_with(&self, viewer: Option<&crate::viewer::Viewer>) -> crate::Result<()> {
                let owned_viewer;
                let viewer = if let Some(viewer) = viewer {
                    viewer
                } else {
                    owned_viewer = Some(crate::viewer::Viewer::create_ascii_stdout(self.world)?);
                    owned_viewer.as_ref().unwrap()
                };
                let ierr = unsafe { crate::petsc_raw::$raw_view_func(self.$raw_ptr_var, viewer.viewer_p) };
                Petsc::check_error(self.world, ierr)
            }
        }
    };
}
