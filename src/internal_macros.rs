//! Macros for internal use.
//!
//! This could be for generating bindings or for things that shouldn't be exposed to crate users.

#![macro_use]

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
///     VecNorm, pub norm, input NormType, norm_type,
///         output PetscReal, tmp1, #[doc = "Computes the vector norm."];
/// }
/// ```
///
/// Note, for the raw PETSc function, `VecNorm` in this case, do not include the `petsc_raw::` as it is done internally.
///
/// For a more general case lets say we have a Petsc function called `VecSetABRetCD`
/// that is defined as follows (from bindgen):
/// `pub unsafe fn VecSetABRetCD(arg1: Vec, arg2: PetscInt, arg3: PetscReal, arg4: *mut PetscReal, arg5: *mut PetscInt) -> PetscErrorCode`
/// It takes in two inputs `arg2` and `arg3` and returns two outputs with `arg4` and
/// `arg5` (through pointers). For reference the rust `petsc_rs::Vector` type is defined as the following:
/// ```ignore
/// pub struct Vector<'a> {
///     pub(crate) world: &'a UserCommunicator,
///     pub(crate) vec_p: *mut petsc_raw::_p_Vec,
/// }
///
/// impl_petsc_object_traits! { Vector, vec_p, petsc_raw::_p_Vec, VecView, VecDestroy; }
/// ```
/// Note, for the macro to work, the type must implement `crate::PetscAsRaw` and `crate::PetscObject`.
/// This can be done with the [`impl_petsc_object_traits!`] macro. Once a wrapper type implements
/// [`PetscAsRaw`](crate::PetscAsRaw), so does `Option<T>` where the `None` case becomes null.
///
/// We can then using the macro in the following way to create the function
/// `pub fn set_ab_ret_cd(&mut self, a: i32, b: f64) -> crate::Result<(f64, i32)>`
/// Note, you should use `PetscInt` and `PetscReal` instead of `i32` and `f64`.
/// I'm just using `i32` and `f64` to save horizontal space in this example.
/// ```ignore
/// impl Vec<'_> {
///     wrap_simple_petsc_member_funcs! {
///         VecSetABRetCD, pub set_ab_ret_cd, input i32, a, input f64, b,
/// //         ^            ^         ^            ^
/// //    PETSc func name   └─ vis    │            ├─ Then for each input
/// //                           rust func name   put `input <type>, <param_name>,`
///             output f64, c, output i32, d, takes mut, #[doc = "doc-string"];
/// //           ^                              ^
/// //           ├─ for each output             └─ If you want the method to take
/// //          put `output <type>, <tmp_name>,`   a `&mut self` put `takes mut,`
///     }
/// }
/// ```
/// Note, the number of input, the number of outputs, and if it takes a mutable
/// reference to self is all variable/optional and can be set to what even you need.
///
/// ## More Advanced Input
///
/// Note, the input type can differ from the raw input type if `.into()` can be use
/// for conversion. This is done automatically. If the inputs to the rust wrapper function
/// is a struct, like `Vector` you can also use the macro to apply a simple method, in this
/// case we apply `.as_raw()`.
/// ```ignore
/// impl Vec<'_> {
///     wrap_simple_petsc_member_funcs! {
///         VecAXPY, pub axpy, input PetscScalar, alpha, input &Vector,
///             other .as_raw, takes mut #[doc = "doc-string"];
///       //          ^
///     } //          └ just add `.<as_raw_fn>` after the param_name
/// }
/// ```
/// Note, simple method you give, `as_raw` in this case, is applied before the `.into()`.
///
/// Like inputs, the output types can differ from the raw output types as long as `.into()` can
/// be use for conversion. The following is an example returns a `bool` but the raw function
/// returns a `petsc_raw::PetscBool`. Currently this macro can not be use to return anything that
/// requires a more complicated conversion like a creating a wrapper struct (like `Vector` or `Mat`).
/// ```ignore
/// impl NullSpace<'_> {
///     wrap_simple_petsc_member_funcs! {
///         MatNullSpaceTest, pub test, input &Mat, mat .as_raw,
///         output bool, is_null , #[doc = "doc-string"];
///     }
/// }
/// ```
///
/// Because both the input type and output type automaticly apply `.into()`, `PetscScalar` can
/// be a complex with no extra cost. On the petsc-rc side we use a different Complex type
/// than is used by the raw function, however, `Into` and `From` are implemented between them.
///
/// If you want the function to consume an input, you can use `consume .<member_name>` in
/// the following way. Note, this requires `member_name` to be a member of the struct type,
/// and it must be an [`Option`].
/// ```ignore
/// pub struct KSP<'a, 'tl, 'bl> {
///     world: &'a UserCommunicator
///     ksp_p: petsc_raw::KSP,
///     owned_dm: Option<DM<'a, 'tl>>,
///     // some fields omitted
/// }
///
/// impl<'a, 'tl, 'bl> KSP<'a, 'tl, 'bl> { // ┌ You need the lifetime parameters
///     wrap_simple_petsc_member_funcs! { //  v
///         KSPSetDM, pub set_dm, input DM<'a, 'tl>, dm .as_raw consume .owned_dm,
/// //                                                     ^      ^         ^
/// //                         Note, `as_raw` is only used ┘      │         │
/// //                         when calling the raw function.     │         │
///             takes mut, #[doc = "doc-string"]; //     Add `consume .<member_name>`
///     }
/// }
/// ```
///
/// If you want the rust function to be `unsafe` you can add `is unsafe` at the end, before the doc-string.
///
/// # Real Examples
///
/// The examples show in this doc-string exist in `petsc-rs`, except for the `VecSetABRetCD` one.
///
/// Almost every file in `src/` uses this macro at the bottom of the file.
macro_rules! wrap_simple_petsc_member_funcs {
    {$(
        $raw_func:ident, $vis_par:vis $new_func:ident,
        $(input $param_type:ty, $param_name:ident $(.$as_raw_fn:ident)? $(consume .$member:ident)? ,)*
        $(output $ret_type:ty, $tmp_ident:ident,)*
        $(takes $mut_tag:tt,)? $(is $is_unsafe:ident,)? $( #[$att:meta] )+;
    )*} => {
$(
    $( #[$att] )+
    #[allow(unused_parens)]
    $vis_par $($is_unsafe)? fn $new_func(& $($mut_tag)? self, $( $param_name: $param_type ),*)
        -> crate::Result<($( $ret_type ),*)>
    {
        $(
            let mut $tmp_ident = ::std::mem::MaybeUninit::uninit();
        )*
        #[allow(unused_unsafe)]
        let ierr = unsafe { crate::petsc_raw::$raw_func(
            self.as_raw() as *mut _,
            $( $param_name $(.$as_raw_fn())?.into() , )*
            $( $tmp_ident.as_mut_ptr() ),*
        )};
        #[allow(unused_unsafe)]
        unsafe { chkerrq!(self.world(), ierr) }?;

        $($(
            let _ = self.$member.take();
            self.$member = Some($param_name);
        )?)*

        #[allow(unused_unsafe)]
        crate::Result::Ok(unsafe { ( $( $tmp_ident.assume_init() .into() ),* ) })
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
    pub fn $new_func(&mut self, $($arg1: crate::PetscInt,)? $($arg2: crate::PetscInt, $arg3: ::std::option::Option<&[PetscInt]>),+) -> crate::Result<()> {
        let ierr = unsafe { crate::petsc_raw::$raw_func(
            self.as_raw(),
            $( $arg1, )?
            $(
                $arg2,
                $arg3.map(|o| o.as_ptr()).unwrap_or(::std::ptr::null())
            ),+ ) };
        unsafe { chkerrq!(self.world(), ierr) }
    }
)*
    };
}

/// Implements [`PetscAsRaw`](crate::PetscAsRaw), [`PetscAsRawMut`](crate::PetscAsRawMut), [`PetscObject`](crate::PetscObject),
/// [`PetscObjectPrivate`](crate::PetscObjectPrivate), [`viewer::PetscViewable`](crate::viewer::PetscViewable),
/// and [`Drop`](::std::ops::Drop).
///
/// Note, you should not include the `petsc_raw::` on the raw function names, but you should on the raw pointer type.
///
/// You can run this macro on multiple structs at a time:
/// ```ignore
/// impl_petsc_object_traits! {
///     DM, dm_p, petsc_raw::_p_DM, DMView, DMDestroy, '_;
///     DMLabel, dml_p, petsc_raw::_p_DMLabel, DMLabelView, DMLabelDestroy;
///     FEDisc, fe_p, petsc_raw::_p_PetscFE, PetscFEView, PetscFEDestroy, '_;
///     FVDisc, fv_p, petsc_raw::_p_PetscFV, PetscFVView, PetscFVDestroy;
///     DMField, field_p, petsc_raw::_p_DMField, DMFieldView, DMFieldDestroy;
///     DS, ds_p, petsc_raw::_p_PetscDS, PetscDSView, PetscDSDestroy, '_;
///     WeakForm, wf_p, petsc_raw::_p_PetscWeakForm, PetscWeakFormView, PetscWeakFormDestroy;
/// }
/// ```
macro_rules! impl_petsc_object_traits {
    {$(
        $struct_name:ident, $raw_ptr_var:ident, $raw_ptr_ty:ty, $raw_view_func:ident, $raw_destroy_func:ident $(, $add_lt:lifetime)* ;
    )*} => {
    $(
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
                self.world // should we make this a macro input or is it safe to assume that is will always be `.world`
            }
        }

        impl<'a> crate::PetscObjectPrivate<'a, $raw_ptr_ty> for $struct_name<'a, $( $add_lt ),*> { }

        impl crate::viewer::PetscViewable for $struct_name<'_, $( $add_lt ),*> {
            /// Views the object with a viewer
            fn view_with<'vl, 'val: 'vl>(&self, viewer: impl Into<Option<&'vl crate::viewer::Viewer<'val>>>) -> crate::Result<()> {
                let owned_viewer;
                let viewer = if let Some(viewer) = viewer.into() {
                    viewer
                } else {
                    // Or should we just pass NULL into the raw_view_func
                    owned_viewer = Some(crate::viewer::Viewer::create_ascii_stdout(self.world())?);
                    owned_viewer.as_ref().unwrap()
                };
                let ierr = unsafe { crate::petsc_raw::$raw_view_func(self.as_raw(), viewer.as_raw()) };
                unsafe { chkerrq!(self.world(), ierr) }
            }
        }

        impl ::std::ops::Drop for $struct_name<'_, $( $add_lt ),*> {
            fn drop(&mut self) {
                let ierr = unsafe { crate::petsc_raw::$raw_destroy_func(&mut self.as_raw() as *mut _) };
                let _ = unsafe { chkerrq!(self.world(), ierr) }; // TODO: should I unwrap or what idk?
            }
        }
    )*
    };
}

/// This macro returns the name of the enclosing function. As the internal
/// implementation is based on the [`std::any::type_name`], this macro
/// derives all the limitations of this function.
///
/// Rust doesn't have a built in way of doing this yet: <https://github.com/rust-lang/rfcs/issues/1743>.
///
/// The code for this macro is from: <https://stackoverflow.com/a/40234666/9664285>
/// and <https://docs.rs/stdext/0.3.1/src/stdext/macros.rs.html#63-74>
// TODO: should we just use the stdext crate? it is light weight.
macro_rules! function_name {
    () => {{
        // IDK why rust thinks these functions are never used.
        #[allow(dead_code)]
        fn f() {}
        #[allow(dead_code)]
        fn type_name_of<T>(_: T) -> &'static str {
            ::std::any::type_name::<T>()
        }
        let name = type_name_of(f);
        &name[..name.len() - 3]
    }};
}

/// Calls [`Petsc::check_error()`](crate::Petsc::check_error()) with the line number, function name, and file name added.
///
/// Because [`Petsc::check_error`](crate::Petsc::check_error()) and [`function_name!`] are not exposed to create users
/// this macro is only intended for internal use.
///
/// Note, this is unsafe to use because [`Petsc::check_error()`](crate::Petsc::check_error()) is unsafe.
macro_rules! chkerrq {
    ($world:expr, $ierr_code:expr) => {{
        #[allow(unused_unsafe)]
        crate::Petsc::check_error(
            $world,
            line!() as i32,
            function_name!(),
            file!(),
            $ierr_code,
        )
    }};
}

/// Calls [`Petsc::set_error2()`](crate::Petsc::set_error2()) with the line number, function name, and file name added.
///
/// For now, this macro is only intended for internal use.
macro_rules! seterrq {
    ($world:expr, $err_kind:expr, $err_msg:expr) => {{
        #[allow(unused_unsafe)]
        crate::Petsc::set_error2(
            $world,
            Some(line!() as i32),
            Some(function_name!()),
            Some(file!()),
            $err_kind,
            $err_msg,
        )
    }};
}
