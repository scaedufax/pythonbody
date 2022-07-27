CFLAGS=-fPIC -std=c99

grav_pot.so: grav_pot.o
	gcc -shared -o grav_pot.so grav_pot.o
prav_pot.o:
	gcc -c -fPIC -o grav_pot.o grav_pot.c

clean:
	rm grav_pot.o
	rm grav_pot.so
