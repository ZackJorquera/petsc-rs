// TODO: write macros to do some easy stuff
#![macro_use]

// TODO: make macro use `::std::*` or `crate::*` for everything
macro_rules! wrap_simple_petsc_member_funcs {
    {$(
        $raw_func:ident, $new_func:ident, $raw_ptr_var:ident, #[$doc:meta];
    )*} => {
$(
    #[$doc]
    pub fn $new_func(&mut self) -> crate::Result<()> {
        let ierr = unsafe { crate::petsc_raw::$raw_func(self.$raw_ptr_var) };
        self.petsc.check_error(ierr)
    }
)*
    };
    {$(
        $raw_func:ident, $new_func:ident, $raw_ptr_var:ident, $param_name:ident, $param_type:ty, #[$doc:meta];
    )*} => {
$(
    #[$doc]
    pub fn $new_func(&mut self, $param_name: $param_type) -> crate::Result<()>
    {
        let ierr = unsafe { crate::petsc_raw::$raw_func(self.$raw_ptr_var, $param_name) };
        self.petsc.check_error(ierr)
    }
)*
    };
    {$(
        $raw_func:ident, $new_func:ident, $raw_ptr_var:ident, $ret_type:ty, #[$doc:meta];
    )*} => {
$(
    #[$doc]
    pub fn $new_func(&self) -> crate::Result<$ret_type>
    {
        let mut res = ::std::mem::MaybeUninit::<$ret_type>::uninit();
        let ierr = unsafe { crate::petsc_raw::$raw_func(self.$raw_ptr_var, res.as_mut_ptr()) };
        self.petsc.check_error(ierr)?;

        crate::Result::Ok(unsafe { res.assume_init() })
    }
)*
    };
    // TODO: add simple return type one
}

#[allow(unused_macros)]
macro_rules! wrap_prealloc_petsc_member_funcs {
    {$(
        $raw_func:ident, $new_func:ident, $raw_ptr_var:ident, $arg1:ident, $arg2:ident, #[$doc:meta];
    )*} => {
$(
    #[$doc]
    pub fn $new_func(&mut self, $arg1: i32, $arg2: ::std::option::Option<&[i32]>) -> crate::Result<()> {
        let ierr = unsafe { crate::petsc_raw::$raw_func(self.$raw_ptr_var, $arg1, 
            $arg2.map(|o| o.as_ptr()).unwrap_or(::std::ptr::null())) };
        self.petsc.check_error(ierr)
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
        self.petsc.check_error(ierr)
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
        self.petsc.check_error(ierr)
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
        self.petsc.check_error(ierr)
    }
)*
    };
}

macro_rules! impl_petsc_object_funcs {
    ($struct_name:ident, $raw_ptr_var:ident) => {
        impl<'a> $struct_name<'a>
        {
            /// Sets a string name associated with a PETSc object.
            pub fn set_name<T: ::std::string::ToString>(&mut self, name: T) -> crate::Result<()> {
                let name_cs = ::std::ffi::CString::new(name.to_string()).expect("`CString::new` failed");
                
                let ierr = unsafe { crate::petsc_raw::PetscObjectSetName(self.$raw_ptr_var as *mut crate::petsc_raw::_p_PetscObject, name_cs.as_ptr()) };
                self.petsc.check_error(ierr)
            }

            /// Gets a string name associated with a PETSc object.
            pub fn get_name(&self) -> crate::Result<String> {
                let mut c_buf = ::std::mem::MaybeUninit::<*const ::std::os::raw::c_char>::uninit();
                
                let ierr = unsafe { crate::petsc_raw::PetscObjectGetName(self.$raw_ptr_var as *mut crate::petsc_raw::_p_PetscObject, c_buf.as_mut_ptr()) };
                self.petsc.check_error(ierr)?;

                let c_str = unsafe { ::std::ffi::CStr::from_ptr(c_buf.assume_init()) };
                crate::Result::Ok(c_str.to_string_lossy().to_string())
            }

            // TODO: add PetscObjectRef and PetscObjectDeref, but make them unsafe for now
        }
    };
}
