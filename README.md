# SSGA
Schönhage-Strassen-Gallot algorithm  

## About

SSG algorithm is an improvement of Schönhage-Strassen fast multiplication algorithm for large integers.  
Inputs/output vectors are not weighted/unweighted and the recursion is not a Fast Fourier Transform but a Z-transform.  

The fast polynomial multiplication is based on the Chinese remainder theorem over polynomials.  

The remainders *P*(*x*) (mod *x*<sup>*n*</sup> - *r*) and *P*(*x*) (mod *x*<sup>*n*</sup> + *r*) can be calculated from *P*(*x*) (mod *x*<sup>2*n*</sup> - *r*<sup>2</sup>).  
Let *P*(*x*) (mod *x*<sup>2*n*</sup> - *r*<sup>2</sup>) = *a*<sub>0</sub> + *a*<sub>1</sub> *x* + *a*<sub>2</sub> *x*<sup>2</sup> + ... + a<sub>2*n*-1</sub> *x*<sup>2*n*-1</sup>. We have  
*P*(*x*) (mod *x*<sup>*n*</sup> - *r*) = (*a*<sub>0</sub> + *r* *a*<sub>*n*</sub>) + (*a*<sub>1</sub> + *r* *a*<sub>*n*+1</sub>) *x* + ... + (*a*<sub>*n*-1</sub> + *r* *a*<sub>2*n*-1</sub>) *x*<sup>*n*-1</sup>,  
*P*(*x*) (mod *x*<sup>*n*</sup> + *r*) = (*a*<sub>0</sub> - *r* *a*<sub>*n*</sub>) + (*a*<sub>1</sub> - *r* *a*<sub>*n*+1</sub>) *x* + ... + (*a*<sub>*n*-1</sub> - *r* *a*<sub>2*n*-1</sub>) *x*<sup>*n*-1</sup>.  

These relations are easy to invert. If *b*<sub>*i*</sub> = *a*<sub>*i*</sub> + *r* *a*<sub>*n*+*i*</sub> and *b*<sub>*n*+*i*</sub> = *a*<sub>*i*</sub> - *r* *a*<sub>*n*+*i*</sub>, we have  
*a*<sub>*i*</sub> = (*b*<sub>*i*</sub> + *b*<sub>*n*+*i*</sub>) / 2,  
*a*<sub>*n*+*i*</sub> = (*b*<sub>*i*</sub> - *b*<sub>*n*+*i*</sub>) / 2*r*.  
Rather than halving each component at each step, outputs can be multiplied by *l*^-1.

If *n* is a power of two and the roots of *r* exist then the relations can be applied recursively.  
Because the roots are themselves some powers of two in Schönhage-Strassen algorithm, the multiplications in the forward and inverse relations are bit shifts.

Schönhage and Strassen used a Fermat number 2<sup>*N*</sup> + 1 for top level ring and another smaller Fermat number 2<sup>*n*</sup> + 1 is the modulus of the ring of integers of the transform such that the algorithm can be evaluated recursively.  
But a Fermat number is not needed for top level and a Mersenne number 2<sup>*N*</sup> - 1 is a better choice because one is a root of *x*<sup>*N*</sup> - 1 and this avoids some bit shifts.  

The transform is invariably based on the Fermat ring 2<sup>*n*</sup> + 1.  
The function **SSG_mul_Fermat** implements *x* &times; *y* mod 2<sup>*N*</sup> + 1 and **SSG_mul_Mersenne** implements *x* &times; *y* mod 2<sup>*N*</sup> - 1.  **SSG_mul** makes use of a Mersenne number to split the product of a *N*-bit number into *l* products of *n* bits, where *l* ~ sqrt(*N*) and *n* ~ 3 sqrt(*N*).  
