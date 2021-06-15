//! PETSc vectors (Vec objects) are used to store the field variables in PDE-based (or other) simulations.
//!
//! PETSc C API docs: <https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/index.html>

use std::ops::{Deref, DerefMut};

use crate::prelude::*;

// TODO: should we add a builder type so that you have to call set_type or set_from_options in order to use the vector
/// Abstract PETSc vector object
pub struct Vector<'a> {
    pub(crate) world: &'a dyn Communicator,

    pub(crate) vec_p: *mut petsc_raw::_p_Vec, // I could use Vec which is the same thing, but i think using a pointer is more clear
}

/// A immutable view of a Vector with Deref to slice.
pub struct VectorView<'a, 'b> {
    vec: &'b Vector<'a>,
    array: *const f64,
}

/// A mutable view of a Vector with Deref to slice.
pub struct VectorViewMut<'a, 'b> {
    vec: &'b mut Vector<'a>,
    array: *mut f64,
}

impl<'a> Drop for Vector<'a> {
    fn drop(&mut self) {
        unsafe {
            let ierr = petsc_raw::VecDestroy(&mut self.vec_p as *mut *mut petsc_raw::_p_Vec);
            let _ = Petsc::check_error(self.world, ierr); // TODO should i unwrap or what idk?
        }
    }
}

pub use petsc_raw::NormType;
pub use petsc_raw::VecOption;

impl<'a> Vector<'a> {
    /// Creates an empty vector object. The type can then be set with [`Vector::set_type`](#), or [`Vector::set_from_options`].
    /// Same as [`Petsc::vec_create`].
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// let petsc = Petsc::init_no_args().unwrap();
    ///
    /// Vector::create(petsc.world()).unwrap();
    /// ```
    pub fn create(world: &'a dyn Communicator) -> Result<Self> {
        let mut vec_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::VecCreate(world.as_raw(), vec_p.as_mut_ptr()) };
        Petsc::check_error(world, ierr)?;

        Ok(Vector { world, vec_p: unsafe { vec_p.assume_init() } })
    }

    /// Creates a new vector of the same type as an existing vector.
    ///
    /// [`duplicate`](Vector::duplicate) DOES NOT COPY the vector entries, but rather 
    /// allocates storage for the new vector. Use [`Vector::copy_values`](#) to copy a vector.
    pub fn duplicate(&self) -> Result<Self> {
        let mut vec2_p = MaybeUninit::uninit();
        let ierr = unsafe { petsc_raw::VecDuplicate(self.vec_p, vec2_p.as_mut_ptr()) };
        Petsc::check_error(self.world, ierr)?;

        Ok(Vector { world: self.world, vec_p: unsafe { vec2_p.assume_init() } })
    }

    ///  Assembles the vector by calling [`Vector::assembly_begin()`] then [`Vector::assembly_end()`]
    pub fn assemble(&mut self) -> Result<()>
    {
        self.assembly_begin()?;
        // TODO: what would even go here?
        self.assembly_end()
    }

    /// Sets the local and global sizes, and checks to determine compatibility
    ///
    /// The inputs can be `None` to have PETSc decide the size.
    /// `local_size` and `global_size` cannot be both `None`. If one processor calls this with
    /// `global_size` of `None` then all processors must, otherwise the program will hang.
    pub fn set_sizes(&mut self, local_size: Option<i32>, global_size: Option<i32>) -> Result<()> {
        let ierr = unsafe { petsc_raw::VecSetSizes(
            self.vec_p, local_size.unwrap_or(petsc_raw::PETSC_DECIDE_INTEGER), 
            global_size.unwrap_or(petsc_raw::PETSC_DECIDE_INTEGER)) };
        Petsc::check_error(self.world, ierr)
    }

    /// Computes self += alpha * other
    pub fn axpy(&mut self, alpha: f64, other: &Vector) -> Result<()>
    {
        let ierr = unsafe { petsc_raw::VecAXPY(self.vec_p, alpha, other.vec_p) };
        Petsc::check_error(self.world, ierr)
    }

    /// Sets an option for controling a vector's behavior.
    pub fn set_option(&mut self, option: VecOption, flg: bool) -> Result<()>
    {
        let ierr = unsafe { petsc_raw::VecSetOption(self.vec_p, 
            option, if flg {petsc_raw::PetscBool::PETSC_TRUE} else {petsc_raw::PetscBool::PETSC_FALSE}) };
        Petsc::check_error(self.world, ierr)
    }

    /// Inserts or adds values into certain locations of a vector.
    ///
    /// `ix` and `v` must be the same length or `set_values` will panic.
    ///
    /// If you call `x.set_option(VecOption::VEC_IGNORE_NEGATIVE_INDICES, true)`, negative indices
    /// may be passed in ix. These rows are simply ignored. This allows easily inserting element
    /// load matrices with homogeneous Dirchlet boundary conditions that you don't want represented
    /// in the vector.
    ///
    /// These values may be cached, so [`Vector::assembly_begin()`] and [`Vector::assembly_end()`] MUST be
    /// called after all calls to [`Vector::set_values()`] have been completed.
    ///
    /// You might find [`Vector::assemble_with()`] more useful and more idiomatic.
    ///
    /// Parameters.
    /// 
    /// * `ix` - indices where to add
    /// * `v` - array of values to be added
    /// * `iora` - Either [`INSERT_VALUES`](InsertMode::INSERT_VALUES) or [`ADD_VALUES`](InsertMode::ADD_VALUES), 
    /// where [`ADD_VALUES`](InsertMode::ADD_VALUES) adds values to any existing entries, and 
    /// [`INSERT_VALUES`](InsertMode::INSERT_VALUES) replaces existing entries with new values.
    ///
    /// # Example
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # let petsc = Petsc::init_no_args().unwrap();
    /// if petsc.world().size() != 1 {
    ///     // note, cargo wont run tests with mpi so this will never be reached,
    ///     // but this example will only work in a uniprocessor comm world
    ///     Petsc::set_error(petsc.world(), PetscErrorKind::PETSC_ERROR_WRONG_MPI_SIZE, "This is a uniprocessor example only!").unwrap();
    /// }
    ///
    /// let mut v = petsc.vec_create().unwrap();
    /// v.set_sizes(None, Some(10)).unwrap(); // create vector of size 10
    /// v.set_from_options().unwrap();
    ///
    /// v.set_values(&[0, 3, 7, 9], &[1.1, 2.2, 3.3, 4.4], InsertMode::INSERT_VALUES).unwrap();
    /// // You MUST assemble before you can use 
    /// v.assembly_begin().unwrap();
    /// v.assembly_end().unwrap();
    ///
    /// assert_eq!(&v.get_values(0..10).unwrap()[..], &[1.1,0.0,0.0,2.2,0.0,0.0,0.0,3.3,0.0,4.4]);
    ///
    /// v.set_values(&vec![0, 2, 8, 9], &vec![1.0, 2.0, 3.0, 4.0], InsertMode::ADD_VALUES).unwrap();
    /// // You MUST assemble before you can use 
    /// v.assembly_begin().unwrap();
    /// v.assembly_end().unwrap();
    /// assert_eq!(&v.get_values(0..10).unwrap()[..], &[2.1,0.0,2.0,2.2,0.0,0.0,0.0,3.3,3.0,8.4]);
    /// ```
    pub fn set_values(&mut self, ix: &[i32], v: &[f64], iora: InsertMode) -> Result<()>
    {
        // TODO: should I do these asserts?
        assert!(iora == InsertMode::INSERT_VALUES || iora == InsertMode::ADD_VALUES);
        assert_eq!(ix.len(), v.len());

        let ni = ix.len() as i32;
        let ierr = unsafe { petsc_raw::VecSetValues(self.vec_p, ni, ix.as_ptr(), v.as_ptr(), iora) };
        Petsc::check_error(self.world, ierr)
    }

    /// Allows you to give an iter that will be use to make a series of calls to [`Vector::set_values()`].
    /// Then is followed by both [`Vector::assembly_begin()`] and [`Vector::assembly_end()`].
    ///
    /// [`assemble_with()`](Vector::assemble_with()) will short circuit on the first error
    /// from [`Vector::set_values()`], returning it.
    ///
    /// # Example
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # use std::slice::from_ref;
    /// # fn main() -> petsc_rs::Result<()> {
    /// # let petsc = Petsc::init_no_args()?;
    /// if petsc.world().size() != 1 {
    ///     // note, cargo wont run tests with mpi so this will never be reached,
    ///     // but this example will only work in a uniprocessor comm world
    ///     Petsc::set_error(petsc.world(), PetscErrorKind::PETSC_ERROR_WRONG_MPI_SIZE, "This is a uniprocessor example only!")?;
    /// }
    ///
    /// let mut v = petsc.vec_create()?;
    /// v.set_sizes(None, Some(10))?; // create vector of size 10
    /// v.set_from_options()?;
    ///
    /// v.assemble_with([0, 3, 7, 9].iter().cloned()
    ///         .zip([1.1, 2.2, 3.3, 4.4]), InsertMode::INSERT_VALUES)?;
    /// assert_eq!(&v.get_values(0..10)?[..], &[1.1,0.0,0.0,2.2,0.0,0.0,0.0,3.3,0.0,4.4]);
    ///
    /// v.assemble_with([(0, 1.0), (2, 2.0), (8, 3.0), (9, 4.0)], InsertMode::ADD_VALUES)?;
    /// assert_eq!(&v.get_values(0..10)?[..], &[2.1,0.0,2.0,2.2,0.0,0.0,0.0,3.3,3.0,8.4]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn assemble_with<I>(&mut self, iter_builder: I, iora: InsertMode) -> Result<()>
    where
        I: IntoIterator<Item = (i32, f64)>
    {
        // We don't actually care about the num_inserts value, we just need something that
        // implements `Sum` so we can use the sum method and `()` does not.
        let _num_inserts = iter_builder.into_iter().map(|(ix, v)| {
            // TODO: check the arrays are valid
            self.set_values(std::slice::from_ref(&ix),
                std::slice::from_ref(&v), iora).map(|_| 1)
        }).sum::<Result<i32>>()?;
        // Note, `sum()` will short-circuit the iterator if an error is encountered.

        self.assembly_begin()?;
        self.assembly_end()
    }

    /// Gets values from certain locations of a vector.
    ///
    /// Currently can only get values on the same processor.
    ///
    /// Most of the time creating a vector view will be more useful: [`Vector::view()`] or [`Vector::view_mut()`].
    ///
    /// # Example
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # let petsc = Petsc::init_no_args().unwrap();
    /// if petsc.world().size() != 1 {
    ///     // note, cargo wont run tests with mpi so this will never be reached,
    ///     // but this example will only work in a uniprocessor comm world
    ///     Petsc::set_error(petsc.world(), PetscErrorKind::PETSC_ERROR_WRONG_MPI_SIZE, "This is a uniprocessor example only!").unwrap();
    /// }
    ///
    /// let mut v = petsc.vec_create().unwrap();
    /// v.set_sizes(None, Some(10)).unwrap(); // create vector of size 10
    /// v.set_from_options().unwrap();
    ///
    /// let ix = [0, 2, 7, 9];
    /// v.set_values(&ix, &[1.1, 2.2, 3.3, 4.4], InsertMode::INSERT_VALUES).unwrap();
    ///
    /// assert_eq!(&v.get_values(ix).unwrap()[..], &[1.1, 2.2, 3.3, 4.4]);
    /// assert_eq!(&v.get_values(vec![2, 0, 9, 7]).unwrap()[..], &[2.2, 1.1, 4.4, 3.3]);
    /// assert_eq!(&v.get_values(0..10).unwrap()[..], &[1.1,0.0,2.2,0.0,0.0,0.0,0.0,3.3,0.0,4.4]);
    /// assert_eq!(&v.get_values((0..5).map(|v| v*2)).unwrap()[..], &[1.1,2.2,0.0,0.0,0.0]);
    /// ```
    pub fn get_values<T>(&self, ix: T) -> Result<Vec<f64>>
    where
        T: IntoIterator<Item = i32>,
        <T as IntoIterator>::IntoIter: ExactSizeIterator
    {
        // TODO: ix can't be a &[i32; N] because it impl IntoIterator with Item=&i32 (same with &Vec<i32>)
        // This might not be an issue because [i32; N] implements copy.

        // TODO: make this return a slice (like how libceed has the VectorView type)
        // For now im going to return a vector because this is a temporary testing function
        // Really we need to use `VecView` which can get data from the whole vector

        // TODO: is this good, it feels like it would be better to just accept &[i32] and then we dont 
        // need to do the collect. Although, it is nice to accept ranges or iters as input.

        // TODO: i added Vector::view which returns a slice, do we still need this method?

        let ix_iter = ix.into_iter();
        let ni = ix_iter.len();
        let ix_array = ix_iter.collect::<Vec<_>>();
        let mut out_vec = vec![f64::default();ni];

        let ierr = unsafe { petsc_raw::VecGetValues(self.vec_p, ni as i32, ix_array.as_ptr(), out_vec[..].as_mut_ptr()) };
        Petsc::check_error(self.world, ierr)?;

        Ok(out_vec)
    }

    /// Returns the range of indices owned by this processor.
    ///
    /// This method assumes that the vectors are laid
    /// out with the first n1 elements on the first processor, next n2 elements on the second, etc.
    /// For certain parallel layouts this range may not be well defined.
    pub fn get_ownership_range(&self) -> Result<std::ops::Range<i32>> {
        let mut low = MaybeUninit::<i32>::uninit();
        let mut high = MaybeUninit::<i32>::uninit();
        let ierr = unsafe { petsc_raw::VecGetOwnershipRange(self.vec_p, low.as_mut_ptr(), high.as_mut_ptr()) };
        Petsc::check_error(self.world, ierr)?;

        Ok(unsafe { low.assume_init()..high.assume_init() })
    }

    /// Returns the range of indices owned by EACH processor.
    ///
    /// This method assumes that the vectors are laid
    /// out with the first n1 elements on the first processor, next n2 elements on the second, etc.
    /// For certain parallel layouts this range may not be well defined.
    pub fn get_ownership_ranges(&self) -> Result<Vec<std::ops::Range<i32>>> {
        let mut array = MaybeUninit::<*const i32>::uninit();
        let ierr = unsafe { petsc_raw::VecGetOwnershipRanges(self.vec_p, array.as_mut_ptr()) };
        Petsc::check_error(self.world, ierr)?;

        // SAFETY: Petsc says it is an array of length size+1
        let slice_from_array = unsafe { 
            std::slice::from_raw_parts(array.assume_init(), self.world.size() as usize + 1) };
        let array_iter = slice_from_array.iter();
        let mut slice_iter_p1 = slice_from_array.iter();
        let _ = slice_iter_p1.next();
        Ok(array_iter.zip(slice_iter_p1).map(|(s,e)| *s..*e).collect())
    }

    /// Create an immutable view of the vector.
    ///
    /// # Implementation Note
    ///
    /// Standard PETSc vectors use contiguous storage so that this routine does not copy the data.
    /// Other vector implementations may require to copy the data, but must such implementations
    /// should cache the contiguous representation so that only one copy is performed when this routine
    /// is called multiple times in sequence.
    ///
    /// # Example
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # fn main() -> petsc_rs::Result<()> {
    /// # let petsc = Petsc::init_no_args()?;
    /// let mut v = petsc.vec_create()?;
    /// v.set_sizes(None, Some(10))?; // create vector of size 10
    /// v.set_from_options()?;
    /// 
    /// v.assemble_with([0, 3, 7, 9].iter().cloned()
    ///         .zip([1.1, 2.2, 3.3, 4.4]), InsertMode::INSERT_VALUES)?;
    /// assert_eq!(&v.get_values(0..10)?[..], &[1.1,0.0,0.0,2.2,0.0,0.0,0.0,3.3,0.0,4.4]);
    ///
    /// {
    ///     let mut v_view = v.view()?;
    ///     assert_eq!(&v_view[..], &[1.1,0.0,0.0,2.2,0.0,0.0,0.0,3.3,0.0,4.4]);
    /// }
    ///
    /// v.assemble_with([(0, 1.0), (2, 2.0), (8, 3.0), (9, 4.0)], InsertMode::ADD_VALUES)?;
    ///
    /// // It is valid to have multiple immutable views
    /// let v_view = v.view()?;
    /// let v_view2 = v.view()?;
    /// assert_eq!(&v_view[..], &v_view2[..]);
    /// assert_eq!(&v_view2[..], &[2.1,0.0,2.0,2.2,0.0,0.0,0.0,3.3,3.0,8.4]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn view<'b>(&'b self) -> Result<VectorView<'a, 'b>>
    {
        VectorView::new(self)
    }

    /// Create an mutable view of the vector.
    ///
    /// # Implementation Note
    ///
    /// Returns a slice to a contiguous array that contains this processor's portion of the vector data. 
    /// For the standard PETSc vectors, [`view_mut()`](Vector::view_mut()) returns a pointer to the local
    /// data array and does not use any copies. If the underlying vector data is not stored in a contiguous
    /// array this routine will copy the data to a contiguous array and return a slice to that. You MUST
    /// drop the returned [`VectorViewMut`] when you no longer need access to the slice.
    ///
    /// # Example
    ///
    /// ```
    /// # use petsc_rs::prelude::*;
    /// # fn main() -> petsc_rs::Result<()> {
    /// # let petsc = Petsc::init_no_args()?;
    /// let mut v = petsc.vec_create()?;
    /// v.set_sizes(None, Some(10))?; // create vector of size 10
    /// v.set_from_options()?;
    /// v.set_all(1.5)?;
    ///
    /// {
    ///     let mut v_view = v.view_mut()?;
    ///     assert_eq!(&v_view[..], &[1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]);
    ///     v_view[2] = 9.0;
    /// }
    ///
    /// let v_view = v.view()?;
    /// assert_eq!(&v_view[..], &[1.5, 1.5, 9.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn view_mut<'b>(&'b mut self) -> Result<VectorViewMut<'a, 'b>>
    {
        VectorViewMut::new(self)
    }

    // TODO: add `from_array`/`from_slice` and maybe also `set_slice`
}

impl Drop for VectorViewMut<'_, '_> {
    fn drop(&mut self) {
        unsafe {
            let ierr = petsc_raw::VecRestoreArray(self.vec.vec_p, &mut self.array as *mut _);
            let _ = Petsc::check_error(self.vec.world, ierr); // TODO should i unwrap or what idk?
        }
    }
}

impl Drop for VectorView<'_, '_> {
    fn drop(&mut self) {
        unsafe {
            let ierr = petsc_raw::VecRestoreArrayRead(self.vec.vec_p, &mut self.array as *mut _);
            let _ = Petsc::check_error(self.vec.world, ierr); // TODO should i unwrap or what idk?
        }
    }
}

impl<'a, 'b> VectorViewMut<'a, 'b> {
    /// Constructs a VectorViewMut from a Vector reference
    fn new(vec: &'b mut Vector<'a>) -> Result<Self> {
        let mut array = MaybeUninit::<*mut f64>::uninit();
        let ierr = unsafe { petsc_raw::VecGetArray(vec.vec_p, array.as_mut_ptr()) };
        Petsc::check_error(vec.world, ierr)?;

        Ok(Self { vec, array: unsafe { array.assume_init() } })
    }
}

impl<'a, 'b> VectorView<'a, 'b> {
    /// Constructs a VectorViewMut from a Vector reference
    fn new(vec: &'b Vector<'a>) -> Result<Self> {
        let mut array = MaybeUninit::<*const f64>::uninit();
        let ierr = unsafe { petsc_raw::VecGetArrayRead(vec.vec_p, array.as_mut_ptr()) };
        Petsc::check_error(vec.world, ierr)?;

        Ok(Self { vec, array: unsafe { array.assume_init() } })
    }
}

impl Deref for VectorViewMut<'_, '_> {
    type Target = [f64];
    fn deref(&self) -> &[f64] {
        unsafe { std::slice::from_raw_parts(self.array, self.vec.get_global_size().unwrap() as usize) }
    }
}

impl DerefMut for VectorViewMut<'_, '_> {
    fn deref_mut(&mut self) -> &mut [f64] {
        unsafe { std::slice::from_raw_parts_mut(self.array, self.vec.get_global_size().unwrap() as usize) }
    }
}

impl Deref for VectorView<'_, '_> {
    type Target = [f64];
    fn deref(&self) -> &[f64] {
        unsafe { std::slice::from_raw_parts(self.array, self.vec.get_global_size().unwrap() as usize) }
    }
}

// macro impls
impl<'a> Vector<'a> {
    wrap_simple_petsc_member_funcs! {
        VecSetFromOptions, set_from_options, vec_p, #[doc = "Configures the vector from the options database."];
        VecSetUp, set_up, vec_p, #[doc = "Sets up the internal vector data structures for the later use."];
        VecAssemblyBegin, assembly_begin, vec_p, #[doc = "Begins assembling the vector. This routine should be called after completing all calls to VecSetValues()."];
        VecAssemblyEnd, assembly_end, vec_p, #[doc = "Completes assembling the vector. This routine should be called after VecAssemblyBegin()."];
    }

    wrap_simple_petsc_member_funcs! {
        VecSet, set_all, vec_p, alpha, f64, #[doc = "Sets all components of a vector to a single scalar value.\n\nYou CANNOT call this after you have called [`Vector::set_values()`]."];
    }

    wrap_simple_petsc_member_funcs! {
        VecGetLocalSize, get_local_size, vec_p, i32, #[doc = "Returns the number of elements of the vector stored in local memory."];
        VecGetSize, get_global_size, vec_p, i32, #[doc = "Returns the global number of elements of the vector."];
    }

    wrap_simple_petsc_member_funcs! {
        VecNorm, norm, vec_p, input NormType, norm_type, output f64, tmp1, #[doc = "Computes the vector norm."];
    }
}

impl_petsc_object_funcs!{ Vector, vec_p }

impl_petsc_view_func!{ Vector, vec_p, VecView }
