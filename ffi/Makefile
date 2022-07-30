CFLAGS=-fPIC -std=c99 -fopenmp

grav_pot.so: grav_pot.o
	gcc -fopenmp -shared -o grav_pot.so grav_pot.o
prav_pot.o:
	gcc -fopenmp -c -fPIC -o grav_pot.o grav_pot.c

clean:
	test -f grav_pot.o && rm grav_pot.o || true
	test -f grav_pot.so && rm grav_pot.so || true
