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
///     VecNorm, norm, input NormType, norm_type, output f64, tmp1, #[doc = "Computes the vector norm."];
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
///     pub(crate) world: &'a UserCommunicator,
///     pub(crate) vec_p: petsc_raw::Vec,
/// }
/// ```
/// Note, for the macro to work, the type must implement `crate::PetscAsRaw` and `crate::PetscObject`.
/// This can be done with the [`impl_petsc_object_traits!`] macro. Once a wrapper type implements
/// [`PetscAsRaw`](crate::PetscAsRaw), so does `Option<T>` where the `None` case because null ptr.
///
/// We can then using the macro in the following way to create the function 
/// `pub fn set_ab_ret_cd(&mut self, a: i32, b: f64) -> crate::Result<(f64, i32)>`
/// Note, you should use `PetscInt` and `PetscReal` instead of `i32` and `f64`.
/// ```ignore
/// impl Vec<'_> {
///     wrap_simple_petsc_member_funcs! {
///         VecSetABRetCD, pub set_ab_ret_cd, input i32, a, input f64, b, 
/// //         ^            ^         ^            ^
/// //    Petsc func name   └- vis    |            |- Then for each input
/// //                           rust func name   put `input type, param_name,`
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
///         VecAXPY, pub axpy, input PetscScalar, alpha, input &Vector,
///             other .as_raw, #[doc = "doc-string"];
/// //                ^ just add `.as_raw` after the param_name
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
///         MatNullSpaceTest, pub test, input &Mat, vec .as_raw,
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
///
/// If you want the function to consume an input, you can use `consume .<member_name>` in
/// the following way. Note, this requires `member_name` to be a member of the struct type,
/// and it must be an [`Option`].
/// ```ignore
/// pub struct KSP<'a> {
///     world: &'a UserCommunicator,
///     ksp_p: petsc_raw::KSP,
///     owned_dm: Option<DM<'a>>,
/// }
///
/// impl KSP<'_> {
///     wrap_simple_petsc_member_funcs! {
///         KSPSetDM, pub set_dm, input DM, dm .as_raw consume .owned_dm,
/// //                                            ^      ^         ^
/// //                     Note, `as_raw` is only ┘      |         |
/// //                     used with raw function      Add `consume .<member_name>`
///             takes mut, #[doc = "doc-string"];
///     }
/// }
/// ```
/// Note this will drop the member value before it sets it.
///
/// # Real Examples
///
/// Almost every file in `src/` uses this macro at the bottom of the file.
macro_rules! wrap_simple_petsc_member_funcs {
    {$(
        $raw_func:ident, $vis_par:vis $new_func:ident,
        $(input $param_type:ty, $param_name:ident $(.$as_raw_fn:ident)? $(consume .$member:ident)? ,)*
        $(output $ret_type:ty, $tmp_ident:ident $(.$into_fn:ident from $raw_ret_type:ty)? ,)*
        $(takes $mut_tag:tt,)? $(is $is_unsafe:ident,)? $( #[$att:meta] )+;
    )*} => {
$(
    $( #[$att] )+
    #[allow(unused_parens)]
    $vis_par $($is_unsafe)? fn $new_func(& $($mut_tag)? self, $( $param_name: $param_type ),*)
        -> crate::Result<($( $ret_type ),*)>
    {
        $(
            let mut $tmp_ident = ::std::mem::MaybeUninit $(::<$raw_ret_type>)? ::uninit();
        )*
        #[allow(unused_unsafe)]
        let ierr = unsafe { crate::petsc_raw::$raw_func(
            self.as_raw() as *mut _,
            $( $param_name $(.$as_raw_fn())?.into() , )*
            $( $tmp_ident.as_mut_ptr() as *mut _ ),*
        )};
        Petsc::check_error(self.world(), ierr)?;

        $($( 
            let _ = self.$member.take();
            self.$member = Some($param_name);
        )?)*

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
        $raw_func:ident, $new_func:ident, $(block $arg1:ident,)? $(nz $arg2:ident, $arg3:ident,)+ $( #[$att:meta] )+;
    )*} => {
$(
    $( #[$att] )+
    pub fn $new_func(&mut self, $($arg1: PetscInt,)? $($arg2: PetscInt, $arg3: ::std::option::Option<&[PetscInt]>),+) -> crate::Result<()> {
        let ierr = unsafe { crate::petsc_raw::$raw_func(
            self.as_raw(), 
            $( $arg1, )?
            $(
                $arg2,
                $arg3.map(|o| o.as_ptr()).unwrap_or(::std::ptr::null()) 
            ),+ ) };
        Petsc::check_error(self.world(), ierr)
    }
)*
    };
}

macro_rules! impl_petsc_object_traits {
    ($struct_name:ident, $raw_ptr_var:ident, $raw_ptr_ty:ty $(, $add_lt:lifetime)*) => {
        unsafe impl<'a> crate::PetscAsRaw for $struct_name<'a, $( $add_lt ),*> {
            type Raw = *mut $raw_ptr_ty;

            #[inline]
            fn as_raw(&self) -> Self::Raw {
                self.$raw_ptr_var
            }
        } 

        unsafe impl<'a> crate::PetscAsRawMut for $struct_name<'a, $( $add_lt ),*> {
            #[inline]
            fn as_raw_mut(&mut self) -> *mut Self::Raw {
                &mut self.$raw_ptr_var as *mut _
            }
        } 

        impl<'a> crate::PetscObject<'a, $raw_ptr_ty> for $struct_name<'a, $( $add_lt ),*> {
            #[inline]
            fn world(&self) -> &'a mpi::topology::UserCommunicator {
                self.world
            }
        }

        impl<'a> crate::PetscObjectPrivate<'a, $raw_ptr_ty> for $struct_name<'a, $( $add_lt ),*> { }
    };
}

// defines `view_with`
macro_rules! impl_petsc_view_func {
    ($struct_name:ident, $raw_view_func:ident $(, $add_lt:lifetime)*) => {
        impl $struct_name<'_, $( $add_lt ),*>
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
                let ierr = unsafe { crate::petsc_raw::$raw_view_func(self.as_raw(), viewer.viewer_p) };
                Petsc::check_error(self.world, ierr)
            }
        }
    };
}
