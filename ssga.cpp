/*
Copyright 2024, Yves Gallot

SSGA is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <cstdint>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

#include <gmp.h>

// modulo 2^n + 1
class Mod
{
private:
	mpz_t _z;
	static unsigned int _n;
	static mpz_t _F;

	void _mod()
	{
		Mod hi; mpz_fdiv_q_2exp(hi._z, _z, _n); mpz_fdiv_r_2exp(_z, _z, _n);
		*this = *this - hi;
		if (mpz_sgn(_z) < 0) mpz_add(_z, _z, _F);
	}

public:
	static void init() { mpz_init(_F); }
	static void clear() { mpz_clear(_F); }

	static void set_n(const unsigned int n) { _n = n; mpz_set_ui(_F, 1); mpz_mul_2exp(_F, _F, n); mpz_add_ui(_F, _F, 1); }
	static unsigned int get_n() { return _n; }

	Mod() { mpz_init(_z); }
	Mod(const Mod & rhs) { mpz_init_set(_z, rhs._z); }
	Mod(unsigned int n) { mpz_init_set_ui(_z, n); }
	Mod(const mpz_t & z) { mpz_init_set(_z, z); }
	virtual ~Mod() { mpz_clear(_z); }

	Mod & operator=(const Mod & rhs) { mpz_set(_z, rhs._z); return *this; }

	const mpz_t & get() const { return _z; }

 	Mod operator-() const
	{
		Mod r; if (mpz_sgn(_z) == 0) mpz_set(r._z, _z); else mpz_sub(r._z, _F, _z);
		return r;
	}

	Mod operator+(const Mod & rhs) const
	{
		Mod r; mpz_add(r._z, _z, rhs._z);
		if (mpz_cmp(r._z, _F) >= 0) mpz_sub(r._z, r._z, _F);
		return r;
	}

	Mod operator-(const Mod & rhs) const
	{
		Mod r; mpz_sub(r._z, _z, rhs._z);
		if (mpz_sgn(r._z) < 0) mpz_add(r._z, r._z, _F);
		return r;
	}

	Mod operator<<(const size_t s) const
	{
		Mod r; mpz_mul_2exp(r._z, _z, s);
		r._mod();
		return r;
	}

	Mod & operator*=(const Mod & rhs)
	{
		mpz_mul(_z, _z, rhs._z);
		_mod();
		return *this;
	}
};

unsigned int Mod::_n = 0;
mpz_t Mod::_F;

static void mul(const size_t m, size_t e, Mod * const x, Mod * const y)
{
	if (m == 0) { x[0] *= y[0]; return; }

	const size_t e_2 = e / 2;	// previous root is r^2 = 2^e, new root r = 2^{e/2}

	for (size_t k = 0; k < m; ++k)
	{
		const Mod u0 = x[k + 0 * m], u1 = x[k + 1 * m] << e_2;
		x[k + 0 * m] = u0 + u1; x[k + 1 * m] = u0 - u1;
	}

	for (size_t k = 0; k < m; ++k)
	{
		const Mod u0 = y[k + 0 * m], u1 = y[k + 1 * m] << e_2;
		y[k + 0 * m] = u0 + u1; y[k + 1 * m] = u0 - u1;
	}

	mul(m / 2, e_2, &x[0 * m], &y[0 * m]);					//  r = 2^{e/2}
	mul(m / 2, e_2 + Mod::get_n(), &x[1 * m], &y[1 * m]);	// -r = 2^{e/2 + n}

	const size_t me_2 = Mod::get_n() - e_2;		// 2^n = -1 then r^-1 = 2^-{e/2} = -2^{n - e/2}
	for (size_t k = 0; k < m; ++k)
	{
		const Mod u0 = x[k + 0 * m], u1 = x[k + 1 * m];
		x[k + 0 * m] = u0 + u1; x[k + 1 * m] = (u1 - u0) << me_2;
	}
}

static void mul_Mersenne(const size_t m, Mod * const x, Mod * const y)
{
	if (m == 0) { x[0] *= y[0]; return; }

	// We have e = 0: r = 1

	for (size_t k = 0; k < m; ++k)
	{
		const Mod u0 = x[k + 0 * m], u1 = x[k + 1 * m];
		x[k + 0 * m] = u0 + u1; x[k + 1 * m] = u0 - u1;
	}

	for (size_t k = 0; k < m; ++k)
	{
		const Mod u0 = y[k + 0 * m], u1 = y[k + 1 * m];
		y[k + 0 * m] = u0 + u1; y[k + 1 * m] = u0 - u1;
	}

	mul_Mersenne(m / 2, &x[0 * m], &y[0 * m]);		// root of 1 is still 1
	mul(m / 2, Mod::get_n(), &x[1 * m], &y[1 * m]);	// root is -1 = 2^n

	for (size_t k = 0; k < m; ++k)
	{
		const Mod u0 = x[k + 0 * m], u1 = x[k + 1 * m];
		x[k + 0 * m] = u0 + u1; x[k + 1 * m] = u0 - u1;
	}
}

static double get_param(const size_t N, const unsigned int k, size_t & l, size_t & M, size_t & n)
{
	// See Pierrick Gaudry, Alexander Kruppa, Paul Zimmermann.
	// A GMP-based implementation of Schönhage-Strassen's large integer multiplication algorithm.
	// ISSAC 2007, Jul 2007, Waterloo, Ontario, Canada. pp.167-174, ⟨10.1145/1277548.1277572⟩. ⟨inria-00126462v2⟩
	l = size_t(1) << k;
	M = (N % l == 0) ? N / l : N / l + 1;
	const size_t t = 2 * M + k;
	n = t; if (n % l != 0) n = (n / l + 1) * l;
	return	double(t) / n;	// efficiency
}

static void get_best_param(const size_t N, unsigned int & k, size_t & l, size_t & M, size_t & n)
{
	k = 0;
	for (unsigned int i = 1; true; ++i)
	{
		size_t l_i, M_i, n_i; const double efficiency = get_param(N, i, l_i, M_i, n_i);
		if (l_i > 2 * std::sqrt(M_i * l_i)) break;
		if (efficiency > 0.8) k = i;
	}

	const double efficiency = get_param(N, k, l, M, n);
 	std::cout << "N = " << N << ", sqrt(N) = " << int(std::sqrt(N)) << ", N' = " << M * l << ", M = " << M
		<< ", l = " << l << ", n = " << n << ", efficiency = " << efficiency << ", ";
}

// a is a vector of l M-bit part of x
static void fill_vector(Mod * const a, const size_t l, const mpz_t & x, const size_t M)
{
	mpz_t t, r; mpz_inits(t, r, nullptr);
	mpz_set(t, x); for (size_t i = 0; i < l; ++i) { mpz_fdiv_r_2exp(r, t, M); a[i] = Mod(r); mpz_div_2exp(t, t, M); }
	mpz_clears(t, r, nullptr);
}

// Recursive Schönhage-Strassen-Gallot algorithm, z = x * y (mod 2^N + 1)
static void SSG_mul_Fermat(mpz_t & z, const mpz_t & x, const mpz_t & y, const unsigned int k, const size_t l, const size_t M, const size_t n)
{
	Mod::set_n(n);	// modulo 2^n + 1

	Mod a[l], b[l]; fill_vector(a, l, x, M); fill_vector(b, l, y, M);

	mul(l / 2, n, a, b);	// top-most recursion level, the initial root is -1 = 2^n

	// Components are to halved during the reverse transform then multiply outputs by 1/l = -2^n / l = -2^{n - k}
	for (size_t i = 0; i < l; ++i) a[i] = -(a[i] << (n - k));

	// Compute sum of the l M-bit part of z
	mpz_t F, F_half, c; mpz_inits(F, F_half, c, nullptr);
	mpz_set_ui(F, 1); mpz_mul_2exp(F, F, n); mpz_add_ui(F, F, 1);
	mpz_div_2exp(F_half, F, 1);
	mpz_set_ui(z, 0);
	for (size_t i = 0; i < l; ++i)
	{
		mpz_mul_2exp(z, z, M);
		// negacyclic convolution can generate negative outputs
		mpz_set(c, a[l - i - 1].get());
		if (mpz_cmp(c, F_half) > 0) mpz_sub(c, c, F);
		mpz_add(z, z, c);
	}
	mpz_clears(F, F_half, c, nullptr);
}

// Recursive Schönhage-Strassen-Gallot algorithm, z = x * y (mod 2^N - 1)
static void SSG_mul_Mersenne(mpz_t & z, const mpz_t & x, const mpz_t & y, unsigned int k, const size_t l, const size_t M, const size_t n)
{
	Mod::set_n(n);	// modulo 2^n + 1

	Mod a[l], b[l]; fill_vector(a, l, x, M); fill_vector(b, l, y, M);

	mul_Mersenne(l / 2, a, b);	// top-most recursion level

	// Components are to halved during the reverse transform then multiply outputs by 1/l = -2^n / l = -2^{n - k}
	for (size_t i = 0; i < l; ++i) a[i] = -(a[i] << (n - k));

	// Compute sum of the l M-bit part of z
	mpz_set_ui(z, 0); for (size_t i = 0; i < l; ++i) { mpz_mul_2exp(z, z, M); mpz_add(z, z, a[l - i - 1].get()); }
}

// Recursive Schönhage-Strassen-Gallot algorithm, z < 2^N - 1
static void SSG_mul(mpz_t & z, const mpz_t & x, const mpz_t & y, const size_t N)
{
	unsigned int k = 0;	size_t l, M, n; get_best_param(N, k, l, M, n);
	SSG_mul_Mersenne(z, x, y, k, l, M, n);
}

int main()
{
	Mod::init();

	gmp_randstate_t randstate;
	gmp_randinit_default(randstate);
	gmp_randseed_ui(randstate, (unsigned long int)(time(nullptr)));
	mpz_t x, y, z, zp, t; mpz_inits(x, y, z, zp, t, nullptr);

	const size_t n_min = 13761, n_max = 50000000;

	std::cout << "Check SSG algorithm modulo 2^{M*l} + 1:" << std::endl;
	for (size_t N = n_min; N < n_max; N *= 3)
	{
		// Generate two random numbers of size N
		mpz_urandomb(x, randstate, N); mpz_urandomb(y, randstate, N);

		// Get SSA parameters
		unsigned int k = 0;	size_t l, M, n; get_best_param(N, k, l, M, n);

		// x * y (mod 2^{M*l} + 1)
		mpz_set_ui(t, 1); mpz_mul_2exp(t, t, M * l); mpz_add_ui(t, t, 1);
		mpz_mul(z, x, y); mpz_mod(z, z, t);

		SSG_mul_Fermat(zp, x, y, k, l, M, n);
		mpz_mod(zp, zp, t);

		std::cout << ((mpz_cmp(z, zp) == 0) ? "OK" : "Error") << std::endl;
	}

	std::cout << std::endl << "Check SSG algorithm modulo 2^{M*l} - 1:" << std::endl;
	for (size_t N = n_min; N < n_max; N *= 3)
	{
		// Generate two random numbers of size N
		mpz_urandomb(x, randstate, N); mpz_urandomb(y, randstate, N);

		// Get SSA parameters
		unsigned int k = 0;	size_t l, M, n; get_best_param(N, k, l, M, n);

		// x * y (mod 2^{M*l} - 1)
		mpz_set_ui(t, 1); mpz_mul_2exp(t, t, M * l); mpz_sub_ui(t, t, 1);
		mpz_mul(z, x, y); mpz_mod(z, z, t);

		SSG_mul_Mersenne(zp, x, y, k, l, M, n);
		mpz_mod(zp, zp, t);

		std::cout << ((mpz_cmp(z, zp) == 0) ? "OK" : "Error") << std::endl;
	}

	std::cout << std::endl << "Check SSG algorithm (using 2^N - 1 and such that x * y < 2^N - 1):" << std::endl;
	for (size_t N = n_min; N < n_max; N *= 3)
	{
		// Generate two random numbers such that x * y < 2^N - 1
		const size_t x_size = 4 * N / 5, y_size = N / 5;
		mpz_urandomb(x, randstate, x_size); mpz_urandomb(y, randstate, y_size);

		// z = x * y
		mpz_mul(z, x, y);
		SSG_mul(zp, x, y, N);

		std::cout << ((mpz_cmp(z, zp) == 0) ? "OK" : "Error") << std::endl;
	}

	mpz_clears(x, y, z, zp, t, nullptr);
	gmp_randclear(randstate);

	Mod::clear();

	return EXIT_SUCCESS;
}
