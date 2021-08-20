# `petsc-rs` Contribute

This file is not really intended to tell you how to contribute like most `CONTRIBUTING.md` files but instead will serve as an explanation of what work has already been done and what work still needs to be done. Most of this will talk about issues I faced and decisions I made. I don't believe that a lot of the decisions I made are the best solutions, but I think there is value in explaining why I made them.

Contributing to `petsc-rs` is encouraged. For a better guide of how to contribute, read the [PETSc `CONTRIBUTING` file](https://gitlab.com/petsc/petsc/-/blob/main/CONTRIBUTING).

Before you commit, make sure all the doc-tests and examples still work as expected. This includes testing for both real and complex numbers as well as using PETSc `v3.15` and the main branch.

## Library structure

### Crates

The `petsc-rs` project is separated into 3 crates: `petsc-rs`, `petsc-sys`, and `build-probe-petsc`.

#### `petsc-rs`

The goal of this crates is to prove a high-level wrapper around the C API that follows Rust rules and provides an easy-to-use rusty API. This crate is where most of the work will be done.

This crate validates the version of the PETSc library being used under the hood. If the version of PETSc being used is not supported, then `petsc-rs` will fail to build. This is because some of the C API changes from version to version and `petsc-rs` wrappers need to be added to support these changes, whereas `petsc-sys` does not as `bindgen` dynamically creates the bindings.

#### `petsc-sys`

This crate provides direct bindings to the C API. The bindings are created using [bindgen](https://github.com/rust-lang/rust-bindgen). For the most part, very little work is done on top of the bindings except for some small trait implementations as well as creating enums of types that were created with `#define` in C. Bindgen already turns these variables into rust consts, but we take it a step further and turn them into enums. We only create these enums if the `generate-enums` feature is enabled (note, this feature is enabled by `petsc-rs`). 

An example of this is the `PetscErrorCodeEnum` enum. We programmatically create it using the [`#define PETSC_ERR_* ##` values](https://petsc.org/release/include/petscerror.h.html). Note, for many of these enums, we rename them when we use them in `petsc-rs`.

This crate validates that PETSc was built with the correct types, i.e. the size of ints, the precision of floats, and if it uses complex numbers. If the features don't match the PETSc library then this crate will fail to compile. This compiler failure is done artificially and could be removed in the future. We could do it the same way we do the versioning in `petsc-rs`, by creating config variables internally instead of using features. However, the main reason this is not done is that the plan was to incorporate some way of building PETSc in the future using the feature flags. Maybe we could meet in the middle and add a `petsc-type-detect` feature.

#### `build-probe-petsc`

This crate is used to probe for the petsc library. It also provides us with version information and build information stored in some of the header files, like the size of ints and floats, if we are using complex or real numbers, and version information. The main reason why this is separated from petsc-sys is that it is useful for `petsc-rs` to know about some of this information.

### features

The only types of features that we are using are features to specify types, i.e., if integers are 32 bit vs 64, the precision of floats, and if we are using complex numbers. These features are only used to verify the build of PETSc. If the build does not match the features, then the build of `petsc-sys` will fail. In the future, they could also be used to build PETSc. 

### `build.rs`

Most of the script work is in the `petsc-sys` crate. This is to manage bindgen and create the enum types. An example of one of the enum types that is created is the `PetscErrorCodeEnum` enum. We programmatically create it in the [`create_enum_from_consts`](https://gitlab.com/petsc/petsc-rs/-/blob/71594342/petsc-sys/build.rs#L44-88) function using the [`#define PETSC_ERR_* ##` values](https://petsc.org/release/include/petscerror.h.html) as reference. Another example is all the `XXXTypeEnum` enums like `MatTypeEnum` and `DMTypeEnum`. We create these in two steps, first, we create an enum using the `#define` idents, then we create a table of the `#define` values which are C string. This is done in the [`create_type_enum_and_table`](https://gitlab.com/petsc/petsc-rs/-/blob/71594342/petsc-sys/build.rs#L90-135) function.

Right now, this is all done in the `build.rs` for `petsc-sys`, but I think it would make more sense to move this to the `petsc-rs` side. The main reason this isn't in `petsc-rs` is that it would be a lot harder to get the `#define`s (or rust consts) needed to create the enums without just re-running bindgen a second time. So, for now, this is done in `petsc-sys`. In the end, I don't think this is that big of an issue as I don't see the distinction between the two crates as being that important.

## Design of `petsc-rs`

### Complex numbers

While complex numbers are unsafe, I've still done a lot of work to support complex numbers as safely as possible. The main thing is that `petsc-rs` uses the [`num-complex` `Complex` type](https://docs.rs/num-complex/0.4.0/num_complex/struct.Complex.html), but `petsc-sys` uses the complex type that bindgen creates. To convert between the two I implemented `Into` and `From` that uses a `mem::transmute` under the hood. Or, if you have a pointer to a complex number the conversion can be a simple pointer cast.

### lifetimes and struct references

You'll notice that most of the rust wrapper structs have one to three generic lifetime parameters (`'a`, `'tl`, and `'bl`). This is something I'm not very happy about because it feels like it adds unnecessary bloat to the types, however, it is necessary. The first lifetime, named `'a`, everywhere refers to the comm world lifetime, this just ensures that all the types live longer than the MPI/PETSc initialization and finalization. An alternative could be to have each type own its own comm world and we clone them for each new PETSc object, but this would add a lot of actual bloat to the structs. The second lifetime, named `'tl`, refers to the trampoline lifetime, this is attached to every closure used as a callback. It ensures that the closures live at least as long as the struct they are attached to. One minor problem with this is since all closures use the same lifetime you can get some weird issues, but they seem to be easily fixable most of the time. The last lifetime, named `'bl`, refers to the borrow lifetime, this is attached to any objects that the struct needs to use but only stores as a reference. An example of this is in the struct `SNESFunctionTrampolineData` which stores a reference to a `Vector`. The reason why this is a reference and not an owned version in the first place is because we want to accept types that can't be owned by the `SNES`, like `BorrowVector`. This is done in snes-ex28 when we use `DM::composite_get_access_mut()` to separate a vector into two parts and give one of the parts to `SNES::set_function()`. The main goal is to make sure the lifetime of the borrowed vector lives at least as long as the `SNES`. An alternative could be to write a method that takes a `BorrowVector` or a regular `Vector` using a trait or an enum, but this might cause other problems.

### Callbacks

One thing we have to do a lot in the `petsc-rs` wrappers is adding support for callbacks as many PETSc functions use callbacks. For the most part, we do them all the same way: Put the closure into a trampoline data struct that is then Pinned using `Box::pin`, then pass the trampoline data to the C function through the context parameter. We also create a trampoline function that is C ABI compatible and calls the rust closure from the trampoline data passed in through the context. We also do some conversion from raw C types to the rust wrapper type in the trampoline function so the caller can use the callbacks seamlessly within their rust code. An interesting workaround we do is for when we create the temporary rust types in the trampoline function. Because we want the caller to interact with the rust types, we have to create new rust wrapper types that are different from the original rust type that wraps the C pointer. As far as C is concerned they are the same object because they share the same pointer. acknowledging that they are basically the same object, on the rust side we wrap all of the rust types in a `ManuallyDrop` because we don't increment its reference count and we don't want to drop it. Also, this means that nothing the needs to be dropped should be added to the type unless we manually drop it or it also doesn't need to be dropped. 

This causes one major problem, the closure gives mutable access to a rust wrapper type we must ensure that nothing is added/changed only on the rust side because when the trampoline function exits, all of that information is lost. This isn't a problem in most cases as we mainly just give mutable access to `Vector`s which contain no fields other than the raw C pointer. This will be a problem if we ever give mutable access to a `SNES`, a `DM`, or a `KSP`. It currently is a problem for `MatShell` as some operations are given mutable access to the `MatShell` in the closure so they can edit the inner data. But, we prevent the above-mentioned issue by doing two things: The data that can be edited is stored in the trampoline data so it is essentially given to the trampoline function by reference. We then store this reference in the `MatShell` separately from the trampoline data variable so we don't have to set the trampoline data. And two, we enable a variable, `in_operation_closure`, that will prevent, at runtime, any methods that would edit the trampoline data, other than the mat data, as that would not be allowed. 

### The `wrap_simple_petsc_member_funcs!` macro

A lot of the wrappers that need to be created for many of the PETSc functions are very boilerplate, so I created a macro to try and do a lot of the more tedious stuff for us. I think I did a good job explaining how to use it in its doc-string, so I won't mention that here.

## What still needs to be done

There is a bunch of little things that still need to be implemented. [Here is a list of all PETSc things](https://petsc.org/release/docs/manualpages/singleindex.html); as an end goal, I would like to have something implemented for most of them. We could look at `petsc4py` to see what they have and haven't implemented wrappers for.

Also, I'm certain that I made a bunch of mistakes with the wrappers. There are probably many circumstances where I didn't follow rust's rule or where I incorrectly wrapped a PETSc function. I tried to make `TODO` comments in cases that I was unsure of, but I probably missed many others. The best way to fix these is to just use the library and fix errors and bugs that show up.

### Goals/TODOs (*not comprehensive*).

Most of these are just things I made as I was making `petsc-rs` so some might be bad ideas or not applicable to the current `petsc-rs`. There is no order to them either.

There are also a lot of `TODO` comments throughout the repository that I try to keep up to date.

- [ ] should DMDA and other types of DM be their own struct. Then DM can be a trait or something. This would make it safer, i.e. you can't call da method unless you are a DMDA. This same question exists for all PETSc wrapper types. However, I think what we have right now works fine so I'm not super motivated to try and implement this right now. Also, look at the [`FieldDisc` type](https://gitlab.com/petsc/petsc-rs/-/blob/95cb741c/src/dm.rs#L122-128), maybe we could make `DM` an enum that can be different types of DMs.
- [x] Do we want wrappers of `DMSNESSetFunction`, `DMKSPSetComputeOperators`, and the such. If so it would make sense to move the trampoline data to the DM, i.e. `SNES::set_function` would just call `DM::snes_set_function` and that would deal with the closure.
- [ ] Add a wrapper for `DMGetCoordinates` - returns a vector with a different type than PetscScalar. should we use generics on vector, or we could make a new struct CoordVec or something?
- [ ] make Mat builder an option for construction. I think what we have now works well enough, but you shouldn't be able to use the mat until you call `set_up` so it would make sense to have that be a consuming builder.
  - [ ] make Vector builder an option for construction
- [ ] add comments to raw bindings (would have to edit the bindgen results) (the PETSc code has comments on all the functions (in `*.c` file), It would be cool if we could grab it) (or just link to the online C API docs). I don't know how to do this or if we even can. I've tried a couple of things but have had no success. In the end, it probably isn't a big deal because the user can always just search the PETSc C API docs.
- [ ] make build.rs build petsc if a feature to do that is on
  - [ ] add features on how to install blas (maybe) (look into blas-lapack-rs) - I don't know if this is necessary. Also, we don't use blas directly anywhere is so IDK if this even matters.
  - [ ] I suspect that if Petsc installed mpi, then we could use that version, i.e., we need to find the mpi installed on the system (look at rsmpi to see what they do in this case).
  - [ ] In the same vein we should support a static install of petsc (I think the only reason it doesn't work for me rn is that the static install I have is 3.10 not 3.15)
  - [ ] I foresee this being an issue in the future, when we install petsc in petsc-sys build.rs, we will want to make sure that the petsc-rs build.rs is not run until the install has finished, or have some way of petsc-rs waiting for the install. Note, cargo runs them in parallel right now so we can guarantee anything about what is run when.
- [x] make all unwraps in docstrings us try `?` instead
- [ ] add PF bindings (https://petsc.org/release/docs/manualpages/PF/index.html)
- [ ] add better MatStencil type for petsc-rs (maybe add a bunch of types that all implement `Into<MatStencil>` or a bunch of `new_Xd` functions.)
- [ ] add wrapper for `MatNullSpaceSetFunction`
- [ ] it would be cool to make a macro that functions sort of like stuctopt for the PetscOpt trait
- [ ] work on complex numbers more. The api, while ok, still has problems. An annoying thing is the fact that complex numbers have no `.abs()` method and real numbers have no `.norm()` method. This make supporting both real and complex numbers a pain. We have to use a `#[cfg(feature = "petsc-use-complex-unsafe")]` (look at snes-ex3 for an example of this).
- [ ] it would make sense to move a lot of the high-level functionality to the rust side (like we did for `da_vec_view`)
  - [ ] we could do this for viewers
  - [ ] Also add GUI viewer, look how OpenCV rust bindings do it
  - [ ] we could maybe do this for error handling (i.e. rewrite PetscError).
- [ ] make DMDA use const generic to store the dimensions - this is in the same vein as having DMDA be its own struct type
- [ ] make set_sizes take an enum that contains both local and global so you can't give two Nones and get an error - IDK if this is that important, I kind of like the current API.
- [ ] create wrapper macro to synchronize method calls, for example, we could do something like `sync! { petsc.world(), println!("hello process: {}", petsc.world().rank()) }` in place of `petsc_println_all!(petsc.world(), "hello process: {}", petsc.world().rank())`. Would this even work?
- [ ] should we rename the `create` methods to be `new` (make things rustier). We would have to change the name of the existing private `new` methods to something else though.
- [x] make it so `panic!` aborts nicely (i.e. it calls PetscAbort and MPI_Abort), maybe we have to make a petsc_abort!.
  - [ ] Make all uses of panic! use petsc_abort! instead. this includes unwrap, assert_eq, and others probably. I started this to test out the API, but never finished it.
- [ ] add Quadrature type (https://petsc.org/release/docs/manualpages/FE/PetscQuadrature.html)
  - [ ] do `PetscDTStroudConicalQuadrature`
- [ ] do `https://petsc.org/release/docs/manualpages/DMPLEX/DMPlexSetClosurePermutationTensor.html`
- [ ] should the examples be in `src/` folders or should we put them in the root with the `Cargo.toml`. Or should they just all be in examples/ so that cargo build --examples works.
- [ ] Add a proc macro or something that does the same thing as PetscFunctionBeginUser and PetscFunctionReturn. Or incorporate this into the try `?` operator
- [ ] add `DMPlexInsertBoundaryValues`
- [ ] Generate code for types like DMBoundaryType, right now we manually write the impls for it.
- [ ] Generate FromStr for some of the generated enums
- [x] Move the `boundary_trampoline_data` from DM to DS also move all methods to the DS and have wrapper ones replace existing ones. This should be easy-ish, look at KSP::set_operators and how it calls PC::set_operators.
- [x] when we take a file path as an input we should probably take an `AsRef<Path>`, for the `PetscBuilder::file` we might want to separate the ':yaml' thing from the file path, at least as far as the caller is concerned.  
- [ ] in a lot of the doc test examples we use `set_from_options` and other stuff from the options database. This is a problem because the test relies on the options being specific values. We should instead manually set the values. Using the options database in the full examples should be fine though.
- [ ] Some functions return `Rc`s to inner types, but maybe shouldn't. Unless we store it as an Rc, then it's probably fine.
- [ ] Should `get_local_vector` take `&mut self`. This is because it does mutate the DM a little, however, it might also be possible to just use some interior mutability construct. Or maybe we store the `localin`/`localout` array in the rust side as a slice and put a RefCell around that. Or maybe it doesn't even have to be around the arrays and could be a different variable that acts as a mutability lock handled all internally, but at this point, it would be useless as it is already safe. Or maybe we just make it take `&mut self` and also return `&self`/`&mut self` you aren't locked out of doing anything because of the lifetime in `BorrowVector`. IDK what to do here, as it seems like it is fine already, but it also doesn't seem fine.
- [ ] right now we block versions of PETSc that aren't supported, but maybe we should have a feature to disable this so users can use unsupported versions if they actually do work.
- [ ] I added `build-probe-petsc` because I wanted to get PETSc version/build info in petsc-rs build.rs, but it might also work to make petsc-sys a build dependency for petsc-rs, and then we can just use the consts from that.
- [ ] Add TS wrappers (<https://petsc.org/release/docs/manualpages/TS/index.html>)
- [ ] Add Tao wrappers (<https://petsc.org/release/docs/manualpages/Tao/index.html>)
- [ ] DMLabel should be `Rc<DMLable>` or something. At least when we return the label from a function. Its a little weird to `get_label` then modify it and then drop it because it is reference counted on the C side.
- [ ] Add better tests, maybe run them in a `gitlab-ci.yml`.
