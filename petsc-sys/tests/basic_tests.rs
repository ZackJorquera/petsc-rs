#[test]
fn can_initialize() {
    let ierr = unsafe { petsc_sys::PetscInitializeNoArguments() };
    assert_eq!(ierr, 0);
}