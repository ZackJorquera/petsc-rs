#ifndef PETSC_FFI_WRAPPER_H
#define PETSC_FFI_WRAPPER_H

// This include file contains all information on how PETSc was build
#include <petscconf.h>

// This include file allows you to use ANY public PETSc function
#include <petsc.h>

// These include files are additional private headers
#include <petsc/private/petscdsimpl.h>
#include <petsc/private/dmimpl.h>
#include <petsc/private/dmlabelimpl.h>
#include <petsc/private/matimpl.h>

#endif //PETSC_FFI_WRAPPER_H