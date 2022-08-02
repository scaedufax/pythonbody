CFLAGS=-fPIC -std=c99 -fopenmp

all: grav_pot_cuda.so #grav_pot_ocl.so #grav_pot_cuda.so grav_pot.so


grav_pot.so: grav_pot.o
	gcc -fopenmp -shared -o grav_pot.so grav_pot.o
prav_pot.o:
	gcc -fopenmp -c -fPIC -o grav_pot.o grav_pot.c
grav_pot_cuda.so: #grav_pot_cuda.o
	nvcc --compiler-options '-fPIC' -o grav_pot_cuda.so --shared grav_pot.cu
	#g++ -shared -L/opt/cuda/lib64 -lcudadevrt -lcudart -o grav_pot_cuda.so grav_pot_cuda.o 

#grav_pot_cuda.o:
#	nvcc -Xcompiler '-fPIC' -rdc=true -dc grav_pot.cu
#	nvcc -Xcompiler '-fPIC' -rdc=true -dlink grav_pot.o -o grav_pot_cuda.o
grav_pot_ocl.so:
	gcc -lOpenCL -lm -shared -fPIC grav_pot_cl.c -o grav_pot_ocl.so
grav_pot_ocl.o:
	gcc -lOpenCL -lm -c -fPIC -o grav_pot_ocl.o grav_pot_cl.c

clean:
	test -f grav_pot.o && rm grav_pot.o || true
	test -f grav_pot.so && rm grav_pot.so || true
	test -f grav_pot_cuda.o && rm grav_pot_cuda.o || true
	test -f grav_pot_cuda.so && rm grav_pot_cuda.so || true
	test -f grav_pot_ocl.o && rm grav_pot_ocl.o || true
	test -f grav_pot_ocl.so && rm grav_pot_ocl.so || true
