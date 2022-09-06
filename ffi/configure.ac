AC_INIT([libpythonbody], [0.1.0], [uli@shad0wfax.de])
AC_PROG_CC
AC_LANG(C)

AM_INIT_AUTOMAKE
AC_OPENMP
AC_CONFIG_MACRO_DIRS([m4])
AC_CONFIG_FILES([Makefile])

dnl check for omp
AS_IF([test x$enable_openmp != "xno"], [
AC_CHECK_HEADERS([omp.h],[enable_omp=yes])
])
AM_CONDITIONAL([OMP], [test "x$enable_omp" = "xyes"])

dnl check for OpenCL
AC_CHECK_HEADERS([CL/opencl.h],[enable_opencl=yes])
AM_CONDITIONAL([OPENCL], [test "x$enable_opencl" = "xyes"])

AC_CHECK_PROG(enable_cuda,nvcc,yes)
AM_CONDITIONAL([NVCC], [test "x$enable_cuda" = "xyes"])

LT_INIT
AC_OUTPUT
