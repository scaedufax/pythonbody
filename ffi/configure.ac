AC_INIT([libpythonbody], [0.1.0], [uli@shad0wfax.de])
AC_PROG_CC
AC_LANG(C)

AM_INIT_AUTOMAKE
AC_OPENMP
AC_CONFIG_MACRO_DIR([m4])
m4_include([m4/ax_check_compile_flag.m4])
AC_CONFIG_FILES([Makefile])

AC_PROG_GREP

dnl cuda
AC_ARG_WITH([cuda],
            AS_HELP_STRING([--with-cuda=/path/to/cuda], [Specify path to cuda])
	   )

dnl debug
AC_ARG_ENABLE([debug],
    AS_HELP_STRING([--enable-debug], ["enable debugging info"]))
AM_CONDITIONAL([DEBUG], [test "x$enable_omp" = "xyes"])

dnl check for omp
AS_IF([test x$enable_openmp != "xno"], [
AC_CHECK_HEADERS([omp.h],[enable_omp=yes])
])
AM_CONDITIONAL([OMP], [test "x$enable_omp" = "xyes"])

dnl check for optimization flags
AX_CHECK_COMPILE_FLAG([-O3],
					  [CFLAGS_OPTIMISATION="-O3"],
					  [CFLAGS_OPTIMISATION=""])
AC_SUBST([CFLAGS_OPTIMISATION])

dnl check -std=c99 flag
AX_CHECK_COMPILE_FLAG([-std=c99],
					  [],
				      [exit 1])

dnl check for AVX
AC_ARG_ENABLE([avx],
	      AS_HELP_STRING([--enable-avx], ["enable avx"])
	      )
AS_IF([test "x$enable_avx" != "xno"],
      [AC_CHECK_FILE(["/proc/cpuinfo"],
		   [AC_MSG_CHECKING([wheter CPU supports AVX])
			AS_IF([test `grep -c avx /proc/cpuinfo`],      
			      [enable_avx="yes"],
			      [enable_avx="no"],
			     )
		    AC_MSG_RESULT([$enable_avx])],
		   [enable_avx="no"])
      ])

AS_IF([test "x$enable_avx" != "xno"],
	  [AC_CHECK_HEADER([immintrin.h],
					   [enable_avx="yes"],
					   [enable_avx="no"])
	  ])
AS_IF([test "x$enable_avx" != "xno"],
	  [AX_CHECK_COMPILE_FLAG([-mavx],
							 [enable_avx="yes"],
							 [enable_avx="no"])
	  ]
  )
AC_MSG_CHECKING([for AVX])
AS_IF([test "x$enable_avx" != "xno"],
	  [CFLAGS_AVX="-mavx -DAVX"
	   AC_SUBST([CFLAGS_AVX])
	   AC_MSG_RESULT([yes])],
      [CFLAGS_AVX=""
	   AC_MSG_RESULT([no])])

dnl check for OpenCL and whether to look within cuda
AS_IF([test "x$with_cuda" != "x"],
 	  [CFLAGS="$CFLAGS -I$with_cuda/include"
	   CUDA_CFLAGS="-I$with_cuda/include"
	   CUDA_LDFLAGS="-L$with_cuda/lib64"],
	  []
	  )

AC_CHECK_HEADERS([CL/opencl.h],
				[enable_opencl=yes],
				[enable_opencl=no])

AM_CONDITIONAL([OPENCL], [test "x$enable_opencl" = "xyes"])
AM_CONDITIONAL([WITH_CUDA], [test "x$with_cuda" != "x"])

AC_SUBST([CUDA_CFLAGS])
AC_SUBST([CUDA_LDFLAGS])

LT_INIT
AC_OUTPUT
