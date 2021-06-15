// TODO: write macros to do some easy stuff
#![macro_use]

// TODO: make macro use `::std::*` or `crate::*` for everything
macro_rules! wrap_simple_petsc_member_funcs {
    // This is the most simple of the wrapper macros; it is for a PETSc function that takes no input
    // and has no outputs. These can be repeated multiple times to define multiple like methods.
    // `$new_func` will take a mutable reference to self
    // Note, we always return a result, but this returns a `Result<()>`.
    {$(
        $raw_func:ident, $new_func:ident, $raw_ptr_var:ident, #[$doc:meta];
    )*} => {
$(
    #[$doc]
    pub fn $new_func(&mut self) -> crate::Result<()> {
        let ierr = unsafe { crate::petsc_raw::$raw_func(self.$raw_ptr_var) };
        Petsc::check_error(self.world, ierr)
    }
)*
    };

    // This wrapper macro is used for a PETSc function that takes a single input and have no outputs.
    // It also takes mutable access of the PETSc object
    // `$new_func` will take a mutable reference to self
    // These can be repeated multiple times to define multiple like methods.
    {$(
        $raw_func:ident, $new_func:ident, $raw_ptr_var:ident, $param_name:ident, $param_type:ty, #[$doc:meta];
    )*} => {
$(
    #[$doc]
    pub fn $new_func(&mut self, $param_name: $param_type) -> crate::Result<()>
    {
        let ierr = unsafe { crate::petsc_raw::$raw_func(self.$raw_ptr_var, $param_name) };
        Petsc::check_error(self.world, ierr)
    }
)*
    };

    // This wrapper macro is used for a PETSc function that take no input and has one output.
    // These can be repeated multiple times to define multiple like methods.
    // `$new_func` will take an immutable reference to self
    // Note, it return `Result<$ret_type>`
    {$(
        $raw_func:ident, $new_func:ident, $raw_ptr_var:ident, $ret_type:ty, #[$doc:meta];
    )*} => {
$(
    #[$doc]
    pub fn $new_func(&self) -> crate::Result<$ret_type>
    {
        let mut res = ::std::mem::MaybeUninit::<$ret_type>::uninit();
        let ierr = unsafe { crate::petsc_raw::$raw_func(self.$raw_ptr_var, res.as_mut_ptr()) };
        Petsc::check_error(self.world, ierr)?;

        crate::Result::Ok(unsafe { res.assume_init() })
    }
)*
    };
    // This is the most general case for the wrapper macro. It wraps a PETSc function that takes any number of input
    // and returns any number one output. You can also set if the function takes a mutable reference or not
    // These can be repeated multiple times to define multiple methods.
    // TODO: Make everything use this
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
        let ierr = unsafe { crate::petsc_raw::$raw_func(self.$raw_ptr_var, $( $param_name, )* $( $tmp_ident.as_mut_ptr() ),* )};
        Petsc::check_error(self.world, ierr)?;

        #[allow(unused_unsafe)]
        crate::Result::Ok(unsafe { ( $( $tmp_ident.assume_init() ),* ) })
    }
)*
    };
    // TODO: add simple return type one
}


/// This macro is used specifically to wrap PETSc preallocate functions. It cover all the different 
/// input patterns for that. 
/// These can be repeated multiple times to define multiple like methods.
macro_rules! wrap_prealloc_petsc_member_funcs {
    {$(
        $raw_func:ident, $new_func:ident, $raw_ptr_var:ident, $arg1:ident, $arg2:ident, #[$doc:meta];
    )*} => {
$(
    #[$doc]
    pub fn $new_func(&mut self, $arg1: i32, $arg2: ::std::option::Option<&[i32]>) -> crate::Result<()> {
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
    pub fn $new_func(&mut self, $arg1: i32, $arg2: ::std::option::Option<&[i32]>, $arg3: i32, $arg4: ::std::option::Option<&[i32]>) -> crate::Result<()> {
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
    pub fn $new_func(&mut self, $arg1: i32, $arg2: i32, $arg3: ::std::option::Option<&[i32]>) -> crate::Result<()> {
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
    pub fn $new_func(&mut self, $arg1: i32, $arg2: i32, $arg3: ::std::option::Option<&[i32]>, $arg4: i32, $arg5: ::std::option::Option<&[i32]>) -> crate::Result<()> {
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
