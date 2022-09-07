ACLOCAL_AMFLAGS = -I m4

AUTOMAKE_OPTIONS = foreign subdir-objects
lib_LTLIBRARIES = libgrav_pot.la libcummean.la
libgrav_pot_la_SOURCES = src/grav_pot.c
libgrav_pot_la_LDFLAGS =
libgrav_pot_la_CFLAGS =
libcummean_la_LDFLAGS =
libcummean_la_CFLAGS =
if OMP
libgrav_pot_la_LDFLAGS += $(OPENMP_LDFLAGS) 
libgrav_pot_la_CFLAGS += $(OPENMP_CFLAGS)
libcummean_la_LDFLAGS += $(OPENMP_LDFLAGS) 
libcummean_la_CFLAGS += $(OPENMP_CFLAGS)
endif

if OPENCL
libgrav_pot_la_LDFLAGS += -lOpenCL 
endif

libcummean_la_SOURCES = src/cummean.c
