// TODO: write macros to do some easy stuff
#![macro_use]

macro_rules! wrap_simple_petsc_member_funcs {
    {$(
        $raw_func:ident, $new_func:ident, $raw_ptr_var:ident, #[$doc:meta];
    )*} => {
$(
    #[$doc]
    pub fn $new_func(&mut self) -> Result<()> {
        let ierr = unsafe { petsc_raw::$raw_func(self.$raw_ptr_var) };
        self.petsc.check_error(ierr)
    }
)*
    };
    {$(
        $raw_func:ident, $new_func:ident, $raw_ptr_var:ident, $param_name:ident, $param_type:ty, #[$doc:meta];
    )*} => {
$(
    #[$doc]
    pub fn $new_func(&mut self, $param_name: $param_type) -> Result<()>
    {
        let ierr = unsafe { petsc_raw::$raw_func(self.$raw_ptr_var, $param_name) };
        self.petsc.check_error(ierr)
    }
)*
    };
    {$(
        $raw_func:ident, $new_func:ident, $raw_ptr_var:ident, $ret_type:ty, #[$doc:meta];
    )*} => {
$(
    #[$doc]
    pub fn $new_func(&self) -> Result<$ret_type>
    {
        let mut res = MaybeUninit::<$ret_type>::uninit();
        let ierr = unsafe { petsc_raw::$raw_func(self.$raw_ptr_var, res.as_mut_ptr()) };
        self.petsc.check_error(ierr)?;

        Ok(unsafe { res.assume_init() })
    }
)*
    };
    // TODO: add simple return type one
}

macro_rules! impl_petsc_object_funcs {
    ($struct_name:ident, $raw_ptr_var:ident) => {
        impl<'a> $struct_name<'a>
        {
            /// Sets a string name associated with a PETSc object.
            pub fn set_name<T: ToString>(&mut self, name: T) -> Result<()> {
                let name_cs = ::std::ffi::CString::new(name.to_string()).expect("`CString::new` failed");
                
                let ierr = unsafe { crate::petsc_raw::PetscObjectSetName(self.$raw_ptr_var as *mut crate::petsc_raw::_p_PetscObject, name_cs.as_ptr()) };
                self.petsc.check_error(ierr)
            }

            /// Gets a string name associated with a PETSc object.
            pub fn get_name(&self) -> Result<String> {
                let mut c_buf = ::std::mem::MaybeUninit::<*const ::std::os::raw::c_char>::uninit();
                
                let ierr = unsafe { crate::petsc_raw::PetscObjectGetName(self.$raw_ptr_var as *mut crate::petsc_raw::_p_PetscObject, c_buf.as_mut_ptr()) };
                self.petsc.check_error(ierr)?;

                let c_str = unsafe { ::std::ffi::CStr::from_ptr(c_buf.assume_init()) };
                Ok(c_str.to_string_lossy().to_string())
            }
        }
    };
}
