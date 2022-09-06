ACLOCAL_AMFLAGS = -I m4

AUTOMAKE_OPTIONS = foreign subdir-objects
lib_LTLIBRARIES = libgrav_pot.la libcummean.la
libgrav_pot_la_SOURCES = src/grav_pot.c
libgrav_pot_la_LDFLAGS = $(OPENMP_LDFLAGS) 
libgrav_pot_la_CFLAGS = $(OPENMP_CFLAGS)
if OPENCL
libgrav_pot_la_LDFLAGS += -lOpenCL  
endif
libcummean_la_SOURCES = src/cummean.c
libcummean_la_LDFLAGS = $(OPENMP_LDFLAGS) 
libcummean_la_CFLAGS = $(OPENMP_CFLAGS)


if NVCC
nvcc --compiler-options '-fPIC' -o .libs/grav_pot_cuda.so --shared src/grav_pot.cu
endif
