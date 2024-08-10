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
public:
	static unsigned int n;

private:
	mpz_t _z;

	static void set_F(mpz_t & F) { mpz_set_ui(F, 1); mpz_mul_2exp(F, F, n); mpz_add_ui(F, F, 1); }

public:
	Mod() { mpz_init(_z); }
	Mod(const Mod & rhs) { mpz_init_set(_z, rhs._z); }
	Mod(unsigned int n) { mpz_init_set_ui(_z, n); }
	Mod(const mpz_t & z) { mpz_init_set(_z, z); }
	virtual ~Mod() { mpz_clear(_z); }

	Mod & operator=(const Mod & rhs) { mpz_set(_z, rhs._z); return *this; }

	const mpz_t & get() const { return _z; }

	Mod operator+(const Mod & rhs) const
	{
		mpz_t F; mpz_init(F); set_F(F);
		Mod r; mpz_add(r._z, _z, rhs._z);
		mpz_mod(r._z, r._z, F);
		mpz_clear(F);
		return r;
	}

	Mod operator-(const Mod & rhs) const
	{
		mpz_t F; mpz_init(F); set_F(F);
		Mod r; mpz_sub(r._z, _z, rhs._z); mpz_add(r._z, r._z, F);
		mpz_mod(r._z, r._z, F);
		mpz_clear(F);
		return r;

	}

	Mod operator*(const Mod & rhs) const
	{
		mpz_t F; mpz_init(F); set_F(F);
		Mod r; mpz_mul(r._z, _z, rhs._z);
		mpz_mod(r._z, r._z, F);
		mpz_clear(F);
		return r;
	}

	Mod & operator*=(const Mod & rhs)
	{
		mpz_t F; mpz_init(F); set_F(F);
		mpz_mul(_z, _z, rhs._z);
		mpz_mod(_z, _z, F);
		mpz_clear(F);
		return *this;
	}

	Mod half() const
	{
		mpz_t F; mpz_init(F); set_F(F);
		Mod r = *this;
		if (mpz_odd_p(_z)) mpz_add(r._z, r._z, F);
		mpz_div_2exp(r._z, r._z, 1);
		return r;
	}

	Mod pow(const size_t e) const
	{
		if (e == 0) return Mod(1);
		Mod r = Mod(1), y = *this;
		for (size_t i = e; i != 1; i /= 2) { if (i % 2 != 0) r *= y; y *= y; }
		return r * y;
	}
};

unsigned int Mod::n = 0;

static void mul(const size_t m, const size_t s, const size_t e, Mod * const x, Mod * const y)
{
	if (m == 1) { x[0] *= y[0]; return; }

	const size_t m_2 = m / 2;

	const Mod sr = Mod(2).pow(e / 2);	// // root is 2^e, a square root is 2 ^{e/2}

	for (size_t k = 0; k < m_2; ++k)
	{
		const Mod u0 = x[k + 0 * m_2], u1 = x[k + 1 * m_2] * sr;
		x[k + 0 * m_2] = u0 + u1; x[k + 1 * m_2] = u0 - u1;
	}

	for (size_t k = 0; k < m_2; ++k)
	{
		const Mod u0 = y[k + 0 * m_2], u1 = y[k + 1 * m_2] * sr;
		y[k + 0 * m_2] = u0 + u1; y[k + 1 * m_2] = u0 - u1;
	}

	mul(m_2, 2 * s, e / 2 + 0 * Mod::n, &x[0 * m_2], &y[0 * m_2]);	// w = 2^e
	mul(m_2, 2 * s, e / 2 + 1 * Mod::n, &x[1 * m_2], &y[1 * m_2]);	// root of -w

	const Mod msri = Mod(2).pow(Mod::n - e / 2);	// 2^n = -1 then sr^-1 = 2^-{e/2} = -2^{n - e/2}
	for (size_t k = 0; k < m_2; ++k)
	{
		const Mod u0 = x[k + 0 * m_2], u1 = x[k + 1 * m_2];
		x[k + 0 * m_2] = Mod(u0 + u1).half();
		x[k + 1 * m_2] = Mod((u1 - u0) * msri).half();
	}
}

static void mul_Mersenne(const size_t m, const size_t s, Mod * const x, Mod * const y)
{
	if (m == 1) { x[0] *= y[0]; return; }

	const size_t m_2 = m / 2;

	// We have e = 0, root = 2^e = 1

	for (size_t k = 0; k < m_2; ++k)
	{
		const Mod u0 = x[k + 0 * m_2], u1 = x[k + 1 * m_2];
		x[k + 0 * m_2] = u0 + u1; x[k + 1 * m_2] = u0 - u1;
	}

	for (size_t k = 0; k < m_2; ++k)
	{
		const Mod u0 = y[k + 0 * m_2], u1 = y[k + 1 * m_2];
		y[k + 0 * m_2] = u0 + u1; y[k + 1 * m_2] = u0 - u1;
	}

	mul_Mersenne(m_2, 2 * s, &x[0 * m_2], &y[0 * m_2]);	// root of 1 is still 1
	mul(m_2, 2 * s, Mod::n, &x[1 * m_2], &y[1 * m_2]);	// root is -1 = 2^n

	for (size_t k = 0; k < m_2; ++k)
	{
		const Mod u0 = x[k + 0 * m_2], u1 = x[k + 1 * m_2];
		x[k + 0 * m_2] = Mod(u0 + u1).half();
		x[k + 1 * m_2] = Mod(u0 - u1).half();
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

static double get_best_param(const size_t N, unsigned int & k, size_t & l, size_t & M, size_t & n)
{
	k = 0;
	for (unsigned int i = 1; true; ++i)
	{
		size_t l_i, M_i, n_i; const double efficiency = get_param(N, i, l_i, M_i, n_i);
		if (l_i > 2 * std::sqrt(M_i * l_i)) break;
		if (efficiency > 0.95) k = i;
	}

	return get_param(N, k, l, M, n);
}

// a is a vector of l M-bit part of x
static void fill_vector(Mod * const a, const size_t l, const mpz_t & x, const size_t M)
{
	mpz_t t, r; mpz_inits(t, r, nullptr);
	mpz_set(t, x); for (size_t i = 0; i < l; ++i) { mpz_fdiv_r_2exp(r, t, M); a[i] = Mod(r); mpz_div_2exp(t, t, M); }
	mpz_clears(t, r, nullptr);
}

// Recursive Schönhage-Strassen-Gallot algorithm, z = x * y (mod 2^N + 1)
static void SSG_mul_Fermat(mpz_t & z, const mpz_t & x, const mpz_t & y, const size_t l, const size_t M, const size_t n)
{
	Mod::n = n;	// modulo 2^n + 1

	Mod a[l], b[l]; fill_vector(a, l, x, M);  fill_vector(b, l, y, M);

	mul(l, 1, n, a, b);	// top-most recursion level, the initial root is -1 = 2^n

	// Compute sum of the l M-bit part of z
	mpz_t F, F_half, c; mpz_inits(F, F_half, c, nullptr);
	mpz_set_ui(F, 1); mpz_mul_2exp(F, F, n); mpz_add_ui(F, F, 1);
	mpz_div_2exp(F_half, F, 1);
	mpz_set_ui(z, 0);
	for (size_t i = 0; i < l; ++i)
	{
		mpz_mul_2exp(z, z, M);
		// negacyclic convolution can generate negative outputs
		mpz_t c; mpz_init(c);
		mpz_set(c, a[l - i - 1].get());
		if (mpz_cmp(c, F_half) > 0) mpz_sub(c, c, F);
		mpz_add(z, z, c);
	}
	mpz_clears(F, F_half, c, nullptr);
}

// Recursive Schönhage-Strassen-Gallot algorithm, z = x * y (mod 2^N - 1)
static void SSG_mul_Mersenne(mpz_t & z, const mpz_t & x, const mpz_t & y, const size_t l, const size_t M, const size_t n)
{
	Mod::n = n;	// modulo 2^n + 1

	Mod a[l], b[l]; fill_vector(a, l, x, M);  fill_vector(b, l, y, M);

	mul_Mersenne(l, 1, a, b);	// top-most recursion level

	// Compute sum of the l M-bit part of z
	mpz_set_ui(z, 0); for (size_t i = 0; i < l; ++i) { mpz_mul_2exp(z, z, M); mpz_add(z, z, a[l - i - 1].get()); }
}

// Recursive Schönhage-Strassen-Gallot algorithm
static void SSG_mul(mpz_t & z, const mpz_t & x, const mpz_t & y, const size_t N)
{
	unsigned int k = 0;	size_t l, M, n; const double efficiency = get_best_param(N, k, l, M, n);
	std::cout << "N = " << N << ", N' = " << M * l <<", M = " << M << ", l = " << l << ", n = " << n << ", efficiency = " << efficiency << ", ";

	Mod::n = n;	// modulo 2^n + 1

	Mod a[l], b[l]; fill_vector(a, l, x, M);  fill_vector(b, l, y, M);

	mul_Mersenne(l, 1, a, b);	// top-most recursion level

	// 2^n = -1 then 1/l = -2^n / l = -2^{n - k}
	// for (size_t i = 0; i < l; ++i)

	// Compute sum of the l M-bit part of z
	mpz_set_ui(z, 0); for (size_t i = 0; i < l; ++i) { mpz_mul_2exp(z, z, M); mpz_add(z, z, a[l - i - 1].get()); }
}

int main()
{
	gmp_randstate_t randstate;
	gmp_randinit_default(randstate);
	gmp_randseed_ui(randstate, (unsigned long int)(time(nullptr)));
	mpz_t x, y, z, zp, t; mpz_inits(x, y, z, zp, t, nullptr);

	std::cout << "Check SSG algorithm modulo 2^{M*l} + 1:" << std::endl;
	for (size_t N = 13761; N < 10000000; N *= 3)
	{
		// Generate two random numbers
		mpz_urandomb(x, randstate, N); mpz_urandomb(y, randstate, N);

		// Get SSA parameters
		unsigned int k = 0;	size_t l, M, n; const double efficiency = get_best_param(N, k, l, M, n);
		std::cout << "N = " << N << ", N' = " << M * l <<", M = " << M << ", l = " << l << ", n = " << n << ", efficiency = " << efficiency << ", ";

		// x * y (mod 2^{M*l} + 1)
		mpz_set_ui(t, 1); mpz_mul_2exp(t, t, M * l); mpz_add_ui(t, t, 1);
		mpz_mul(z, x, y); mpz_mod(z, z, t);

		SSG_mul_Fermat(zp, x, y, l, M, n);
		mpz_mod(zp, zp, t);

		std::cout << ((mpz_cmp(z, zp) == 0) ? "OK" : "Error") << std::endl;
	}

	std::cout << std::endl << "Check SSG algorithm modulo 2^{M*l} - 1:" << std::endl;
	for (size_t N = 13761; N < 10000000; N *= 3)
	{
		// Generate two random numbers
		mpz_urandomb(x, randstate, N); mpz_urandomb(y, randstate, N);

		// Get SSA parameters
		unsigned int k = 0;	size_t l, M, n; const double efficiency = get_best_param(N, k, l, M, n);
		std::cout << "N = " << N << ", N' = " << M * l <<", M = " << M << ", l = " << l << ", n = " << n << ", efficiency = " << efficiency << ", ";

		// x * y (mod 2^{M*l} - 1)
		mpz_set_ui(t, 1); mpz_mul_2exp(t, t, M * l); mpz_sub_ui(t, t, 1);
		mpz_mul(z, x, y); mpz_mod(z, z, t);

		SSG_mul_Mersenne(zp, x, y, l, M, n);
		mpz_mod(zp, zp, t);

		std::cout << ((mpz_cmp(z, zp) == 0) ? "OK" : "Error") << std::endl;
	}

	std::cout << std::endl << "Check SSG algorithm (using 2^N - 1 and such that x * y < 2^N - 1):" << std::endl;
	for (size_t N = 13761; N < 10000000; N *= 3)
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

	return EXIT_SUCCESS;
}
