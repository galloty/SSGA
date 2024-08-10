# SSGA
Schönhage-Strassen-Gallot algorithm  

## About

SSG algorithm is an improvement of Schönhage-Strassen fast multiplication algorithm for large integers.  
Coefficient vectors are not weighted and the recursion is not a Fast Fourier Transform but a Z-transform.  

The fast polynomial multiplication is based on the Chinese remainder theorem over polynomials.  

The remainders *P*(*x*) (mod *x*<sup>*n*</sup> - *r*) and *P*(*x*) (mod *x*<sup>*n*</sup> + *r*) can be calculated from *P*(*x*) (mod *x*<sup>2*n*</sup> - *r*<sup>2</sup>).  

Let *P*(*x*) (mod *x*<sup>2*n*</sup> - *r*<sup>2</sup>) = *a*<sub>0</sub> + *a*<sub>1</sub> *x* + *a*<sub>2</sub> *x*<sup>2</sup> + ... + a<sub>2*n*-1</sub> *x*<sup>2*n*-1</sup>. We have  
*P*(*x*) (mod *x*<sup>*n*</sup> - *r*) = (*a*<sub>0</sub> + *r* *a*<sub>*n*</sub>) + (*a*<sub>1</sub> + *r* *a*<sub>*n*+1</sub>) *x* + ... + (*a*<sub>*n*-1</sub> + *r* *a*<sub>2*n*-1</sub>) *x*<sup>*n*-1</sup>,  
*P*(*x*) (mod *x*<sup>*n*</sup> + *r*) = (*a*<sub>0</sub> - *r* *a*<sub>*n*</sub>) + (*a*<sub>1</sub> - *r* *a*<sub>*n*+1</sub>) *x* + ... + (*a*<sub>*n*-1</sub> - *r* *a*<sub>2*n*-1</sub>) *x*<sup>*n*-1</sup>,  


These relations are easy to invert. If *b*<sub>*i*</sub> = *a*<sub>*i*</sub> + *r* *a*<sub>*n*+*i*</sub> and *b*<sub>*n*+*i*</sub> = *a*<sub>*i*</sub> - *r* *a*<sub>*n*+*i*</sub>, we have  
*a*<sub>*i*</sub> = (*b*<sub>*i*</sub> + *b*<sub>*n*+*i*</sub>) / 2,  
*a*<sub>*n*+*i*</sub> = (*b*<sub>*i*</sub> - *b*<sub>*n*+*i*</sub>) / 2*r*.

If *n* is a power of two and the roots of *r* exist then the relations can be applied recursively.  
Because the roots are themselves some powers of two in Schönhage-Strassen algorithm, the multiplications in the forward and inverse relations are simple bit shifts.
