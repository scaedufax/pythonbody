ACLOCAL_AMFLAGS = -I m4

AUTOMAKE_OPTIONS = foreign subdir-objects
AM_CFLAGS = -I./include/
lib_LTLIBRARIES = libpythonbody.la #libgrav_pot.la libcummean.la libpythonbody.la
bin_PROGRAMS = tests
libpythonbody_la_SOURCES = src/ocl.c src/cummean.c src/grav_pot.c
libpythonbody_la_CFLAGS = -I./include 
libpythonbody_la_LDFLAGS = -lm 
tests_SOURCES = src/tests.c src/ocl.c src/cummean.c src/grav_pot.c
tests_CFLAGS = -I./include 
tests_LDFLAGS = -lm

if DEBUG
libpythonbody_la_CFLAGS += -g
tests_CFLAGS += -g
endif

if OMP
libpythonbody_la_CFLAGS += $(OPENMP_CFLAGS) 
libpythonbody_la_LDFLAGS += $(OPENMP_LDFLAGS) 
tests_CFLAGS += $(OPENMP_CFLAGS) 
tests_LDFLAGS += $(OPENMP_LDFLAGS) 
endif

if OPENCL
libpythonbody_la_LDFLAGS += -lOpenCL 
tests_LDFLAGS += -lOpenCL 
endif

