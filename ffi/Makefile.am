ACLOCAL_AMFLAGS = -I m4

CFLAGS= -std=c99

AUTOMAKE_OPTIONS = foreign subdir-objects
AM_CFLAGS = -I./include/
lib_LTLIBRARIES = libpythonbody.la #libgrav_pot.la libcummean.la libpythonbody.la
libpythonbody_la_SOURCES = src/ocl.c src/cummean.c src/grav_pot.c src/neighbour_density.c
libpythonbody_la_CFLAGS = -I./include 
libpythonbody_la_LDFLAGS = -lm 

if DEBUG
libpythonbody_la_CFLAGS += -g
endif

if OMP
libpythonbody_la_CFLAGS += $(OPENMP_CFLAGS) 
libpythonbody_la_LDFLAGS += $(OPENMP_LDFLAGS) 
endif

if OPENCL
libpythonbody_la_LDFLAGS += -lOpenCL 
bin_PROGRAMS = tests
tests_SOURCES = src/tests.c src/ocl.c src/cummean.c src/grav_pot.c src/neighbour_density.c
tests_CFLAGS = -I./include 
tests_LDFLAGS = -lm
tests_CFLAGS += $(OPENMP_CFLAGS) 
tests_LDFLAGS += $(OPENMP_LDFLAGS)
if DEBUG
tests_CFLAGS += -g
endif
tests_LDFLAGS += -lOpenCL 
endif

