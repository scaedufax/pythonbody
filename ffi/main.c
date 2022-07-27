#include <stdio.h>
#include "grav_pot.h"

int main (void) {
	double a = 1.2;
	double b = 2.2;

	printf("a = %f, b = %f, a+b=%f\n",a,b,a+b);
	printf("a + b = %f\n", grav_pot(a,b));
	printf("a = %f, b = %f, a+b=%f\n",a,b,a+b);
	return 0;
}
