#ifndef PETSC_FFI_WRAPPER_H
#define PETSC_FFI_WRAPPER_H

// This include file allows you to use ANY public PETSc function
#include <petsc.h>


// TODO: move this enum to the rust side, the rhs value are visable in rust
// TODO: do we even want this to be an enum
/// PETSc Error Codes
typedef enum {
    /// should always be one less then the smallest value 
    PETSC_ERROR_MIN_VALUE        = PETSC_ERR_MIN_VALUE,
    /// unable to allocate requested memory 
    PETSC_ERROR_MEM              = PETSC_ERR_MEM,
    /// no support for requested operation 
    PETSC_ERROR_SUP              = PETSC_ERR_SUP,
    /// no support for requested operation on this computer system 
    PETSC_ERROR_SUP_SYS          = PETSC_ERR_SUP_SYS,
    /// operation done in wrong order 
    PETSC_ERROR_ORDER            = PETSC_ERR_ORDER,
    /// signal received 
    PETSC_ERROR_SIG              = PETSC_ERR_SIG,
    /// floating point exception 
    PETSC_ERROR_FP               = PETSC_ERR_FP,
    /// corrupted PETSc object 
    PETSC_ERROR_COR              = PETSC_ERR_COR,
    /// error in library called by PETSc 
    PETSC_ERROR_LIB              = PETSC_ERR_LIB,
    /// PETSc library generated inconsistent data 
    PETSC_ERROR_PLIB             = PETSC_ERR_PLIB,
    /// memory corruption 
    PETSC_ERROR_MEMC             = PETSC_ERR_MEMC,
    /// iterative method (KSP or SNES) failed 
    PETSC_ERROR_CONV_FAILED      = PETSC_ERR_CONV_FAILED,
    /// user has not provided needed function 
    PETSC_ERROR_USER             = PETSC_ERR_USER,
    /// error in system call 
    PETSC_ERROR_SYS              = PETSC_ERR_SYS,
    /// pointer does not point to valid address 
    PETSC_ERROR_POINTER          = PETSC_ERR_POINTER,
    /// MPI library at runtime is not compatible with MPI user compiled with 
    PETSC_ERROR_MPI_LIB_INCOMP   = PETSC_ERR_MPI_LIB_INCOMP,
    /// nonconforming object sizes used in operation 
    PETSC_ERROR_ARG_SIZ          = PETSC_ERR_ARG_SIZ,
    /// two arguments not allowed to be the same 
    PETSC_ERROR_ARG_IDN          = PETSC_ERR_ARG_IDN,
    /// wrong argument (but object probably ok) 
    PETSC_ERROR_ARG_WRONG        = PETSC_ERR_ARG_WRONG,
    /// null or corrupted PETSc object as argument 
    PETSC_ERROR_ARG_CORRUPT      = PETSC_ERR_ARG_CORRUPT,
    /// input argument, out of range 
    PETSC_ERROR_ARG_OUTOFRANGE   = PETSC_ERR_ARG_OUTOFRANGE,
    /// invalid pointer argument 
    PETSC_ERROR_ARG_BADPTR       = PETSC_ERR_ARG_BADPTR,
    /// two args must be same object type 
    PETSC_ERROR_ARG_NOTSAMETYPE  = PETSC_ERR_ARG_NOTSAMETYPE,
    /// two args must be same communicators 
    PETSC_ERROR_ARG_NOTSAMECOMM  = PETSC_ERR_ARG_NOTSAMECOMM,
    /// object in argument is in wrong state, e.g. unassembled mat 
    PETSC_ERROR_ARG_WRONGSTATE   = PETSC_ERR_ARG_WRONGSTATE,
    /// the type of the object has not yet been set 
    PETSC_ERROR_ARG_TYPENOTSET   = PETSC_ERR_ARG_TYPENOTSET,
    /// two arguments are incompatible 
    PETSC_ERROR_ARG_INCOMP       = PETSC_ERR_ARG_INCOMP,
    /// argument is null that should not be 
    PETSC_ERROR_ARG_NULL         = PETSC_ERR_ARG_NULL,
    /// type name doesn't match any registered type 
    PETSC_ERROR_ARG_UNKNOWN_TYPE = PETSC_ERR_ARG_UNKNOWN_TYPE,
    /// unable to open file 
    PETSC_ERROR_FILE_OPEN        = PETSC_ERR_FILE_OPEN,
    /// unable to read from file 
    PETSC_ERROR_FILE_READ        = PETSC_ERR_FILE_READ,
    /// unable to write to file 
    PETSC_ERROR_FILE_WRITE       = PETSC_ERR_FILE_WRITE,
    /// unexpected data in file 
    PETSC_ERROR_FILE_UNEXPECTED  = PETSC_ERR_FILE_UNEXPECTED,
    /// detected a zero pivot during LU factorization 
    PETSC_ERROR_MAT_LU_ZRPVT     = PETSC_ERR_MAT_LU_ZRPVT,
    /// detected a zero pivot during Cholesky factorization 
    PETSC_ERROR_MAT_CH_ZRPVT     = PETSC_ERR_MAT_CH_ZRPVT,
    /// 
    PETSC_ERROR_INT_OVERFLOW     = PETSC_ERR_INT_OVERFLOW,
    /// 
    PETSC_ERROR_FLOP_COUNT       = PETSC_ERR_FLOP_COUNT,
    /// solver did not converge 
    PETSC_ERROR_NOT_CONVERGED    = PETSC_ERR_NOT_CONVERGED,
    /// MatGetFactor() failed 
    PETSC_ERROR_MISSING_FACTOR   = PETSC_ERR_MISSING_FACTOR,
    /// attempted to over write options which should not be changed 
    PETSC_ERROR_OPT_OVERWRITE    = PETSC_ERR_OPT_OVERWRITE,
    /// example/application run with number of MPI ranks it does not support 
    PETSC_ERROR_WRONG_MPI_SIZE   = PETSC_ERR_WRONG_MPI_SIZE,
    /// missing or incorrect user input 
    PETSC_ERROR_USER_INPUT       = PETSC_ERR_USER_INPUT,
    /// unable to load a GPU resource, for example cuBLAS 
    PETSC_ERROR_GPU_RESOURCE     = PETSC_ERR_GPU_RESOURCE,
    /// An error from a GPU call, this may be due to lack of resources on the GPU or a true error in the call 
    PETSC_ERROR_GPU              = PETSC_ERR_GPU,
    /// general MPI error 
    PETSC_ERROR_MPI              = PETSC_ERR_MPI,
    /// this is always the one more than the largest error code 
    PETSC_ERROR_MAX_VALUE        = PETSC_ERR_MAX_VALUE,
} PetscErrorCodeEnum;

#endif //PETSC_FFI_WRAPPER_H