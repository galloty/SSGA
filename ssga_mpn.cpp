/*
Copyright 2024, Yves Gallot

SSGA is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <cstdint>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

#include <gmp.h>

// modulo 2^n + 1
class Mod
{
private:
	uint64_t * const _d;
	static size_t _size, _n;
	static uint64_t * _buf;

	static uint64_t _addc(const uint64_t x, const uint64_t y, uint64_t & carry)
	{
		const __uint128_t t = x + __uint128_t(y) + carry;
		carry = uint64_t(t >> 64);
		return uint64_t(t);
	}

	static uint64_t _subb(const uint64_t x, const uint64_t y, uint64_t & borrow)
	{
		const __uint128_t t = x - __uint128_t(y) - borrow;
		borrow = uint64_t(t >> 64) & 1;
		return uint64_t(t);
	}

	static void _copy(uint64_t * const y, const uint64_t * const x, const size_t size)
	{
		for (size_t i = 0; i < size; ++i) y[i] = x[i];
	}

	// *this ?>= 2^n + 1
	bool ge_F() const
	{
		const size_t size = _size;
		uint64_t * const d = _d;

		if (d[size - 1] != 1) return (d[size - 1] > 1);
		if (d[0] >= 1) return true;
		for (size_t i = 1; i < size - 1; ++i) if (d[i] > 0) return true;
		return false;
	}

	// *this += 2^n + 1
	void add_F()
	{
		const size_t size = _size;
		uint64_t * const d = _d;

		uint64_t carry = 0;
		d[0] = _addc(d[0], 1, carry);
		if (carry != 0) for (size_t i = 0; i < size - 1; ++i) { d[i] = _addc(d[i], 0, carry); if (carry == 0) break; }
		d[size - 1] = _addc(d[size - 1], 1, carry);
	}

	// *this -= 2^n + 1
	void sub_F()
	{
		const size_t size = _size;
		uint64_t * const d = _d;

		uint64_t borrow = 0;
		d[0] = _subb(d[0], 1, borrow);
		if (borrow != 0) for (size_t i = 1; i < size - 1; ++i) { d[i] = _subb(d[i], 0, borrow); if (borrow == 0) break; }
		d[size - 1] = _subb(d[size - 1], 1, borrow);
	}

	// *this = 2^n + 1 - *this
	void F_sub()
	{
		const size_t size = _size;
		uint64_t * const d = _d;

		uint64_t borrow = 0;
		d[0] = _subb(1, d[0], borrow);
		for (size_t i = 1; i < size - 1; ++i) d[i] = _subb(0, d[i], borrow);
		d[size - 1] = _subb(1, d[size - 1], borrow);
	}

	// *this = buf (mod 2^n + 1)
	void mod_F()
	{
		const size_t size = _size;
		uint64_t * const d = _d;
		const uint64_t * const buf = _buf;

		// *this = buf mod 2^n
		for (size_t i = 0; i < size - 1; ++i) d[i] = buf[i];
		d[size - 1] = 0;

		// buf[size - 1 + i] is buf / 2^n
	 	uint64_t borrow = 0;
	 	for (size_t i = 0; i < size; ++i) d[i] = _subb(d[i], buf[size - 1 + i], borrow);
	 	if (borrow != 0) add_F();
	}

	void to_mpz(mpz_t & t) const
	{
		const size_t size = _size;
		const uint64_t * const d = _d;

		mpz_set_ui(t, 0);
		for (size_t i = 0; i < size; ++i)
		{
			const uint32_t hi = uint32_t(d[size - i - 1] >> 32), lo = uint32_t(d[size - i - 1]);
			mpz_mul_2exp(t, t, 32); mpz_add_ui(t, t, hi); mpz_mul_2exp(t, t, 32); mpz_add_ui(t, t, lo);
		}
	}

	void from_mpz(const mpz_t & t)
	{
		const size_t size = _size;
		uint64_t * const d = _d;

		mpz_t r; mpz_init_set(r, t);
		for (size_t i = 0; i < size; ++i)
		{
			d[i] = (mpz_size(r) != 0) ? r->_mp_d[0] : 0;
			mpz_div_2exp(r, r, 64);
		}
		if (mpz_size(r) != 0) throw std::runtime_error("from_mpz failed.");
		mpz_clear(r);
	}

public:
	static void set_n(const size_t n)
	{
		if (n % 64 != 0) throw std::runtime_error("set_n failed.");
		_n = n; _size = n / 64 + 1;
		if (_buf != nullptr) delete[] _buf;
		_buf = new uint64_t[2 * _size];
	}

	static size_t get_n() { return _n; }

	Mod() : _d(new uint64_t[_size]) {}
	// Mod(const Mod & rhs) : _d(new uint64_t[_size]) {  _copy(_d, rhs._d, _size); }
	// Mod(const unsigned int u) : _d(new uint64_t[_size]) { _d[0] = u; for (size_t i = 0; i < _size; ++i) _d[i] = 0; }
	Mod(const mpz_t & z) : _d(new uint64_t[_size]) { from_mpz(z); }
	virtual ~Mod() { delete[] _d; }

	Mod & operator=(const Mod & rhs) { _copy(_d, rhs._d, _size); return *this; }

	void get(mpz_t & t) const { to_mpz(t); }

	bool iszero() const
	{
		const size_t size = _size;
		uint64_t * const d = _d;
		for (size_t i = 0; i < size; ++i) if (d[i] != 0) return false;
		return true;
	}

 	Mod & neg() { if (!iszero()) F_sub(); return *this; }

	Mod & add_sub(Mod & rhs)
	{
		const size_t size = _size;
		uint64_t * const x = _d;
		uint64_t * const y = rhs._d;

		uint64_t carry = 0, borrow = 0;
		for (size_t i = 0; i < size; ++i)
		{
			const uint64_t x_i = x[i], y_i = y[i];
			x[i] = _addc(x_i, y_i, carry);
			y[i] = _subb(x_i, y_i, borrow);
		}

		if (ge_F()) sub_F();
		if (borrow != 0) rhs.add_F();
		return *this;
	}

	Mod & add_rsub(Mod & rhs)
	{
		const size_t size = _size;
		uint64_t * const x = _d;
		uint64_t * const y = rhs._d;

		uint64_t carry = 0, borrow = 0;
		for (size_t i = 0; i < size; ++i)
		{
			const uint64_t x_i = x[i], y_i = y[i];
			x[i] = _addc(x_i, y_i, carry);
			y[i] = _subb(y_i, x_i, borrow);
		}

		if (ge_F()) sub_F();
		if (borrow != 0) rhs.add_F();
		return *this;
	}

	Mod & operator<<=(const size_t s)
	{
		const size_t size = _size;
		uint64_t * const buf = _buf;

		// buf = *this << s
		const size_t s_size = s / 64, s_64 = s % 64;	// s_size < size - 1
		for (size_t i = 0; i < s_size; ++i) buf[i] = 0;
		buf[s_size] = _d[0] << s_64;
		for (size_t i = s_size + 1; i < s_size + size; ++i) buf[i] = (_d[i - s_size] << s_64) | (_d[i - s_size - 1] >> (64 - s_64));
		buf[size + s_size] = _d[size - 1] >> (64 - s_64);
		for (size_t i = size + s_size + 1; i < 2 * size; ++i) buf[i] = 0;

		mod_F();

		return *this;
	}

	Mod & operator*=(const Mod & rhs)
	{
 		mpn_mul_n(_buf, _d, rhs._d, _size);
		mod_F();
		return *this;
	}
};

size_t Mod::_size = 0, Mod::_n = 0;
uint64_t * Mod::_buf = nullptr;

static void mul(const size_t m, size_t e, Mod * const x, Mod * const y)
{
	if (m == 0) { x[0] *= y[0]; return; }

	const size_t e_2 = e / 2;	// previous root is r^2 = 2^e, new root r = 2^{e/2}

	for (size_t k = 0; k < m; ++k) { x[k + 1 * m] <<= e_2; x[k + 0 * m].add_sub(x[k + 1 * m]); }
	for (size_t k = 0; k < m; ++k) { y[k + 1 * m] <<= e_2; y[k + 0 * m].add_sub(y[k + 1 * m]); }

	mul(m / 2, e_2, &x[0 * m], &y[0 * m]);					//  r = 2^{e/2}
	mul(m / 2, e_2 + Mod::get_n(), &x[1 * m], &y[1 * m]);	// -r = 2^{e/2 + n}

	const size_t me_2 = Mod::get_n() - e_2;		// 2^n = -1 then r^-1 = 2^-{e/2} = -2^{n - e/2}

	for (size_t k = 0; k < m; ++k) { x[k + 0 * m].add_rsub(x[k + 1 * m]); x[k + 1 * m] <<= me_2; }
}

static void mul_Mersenne(const size_t m, Mod * const x, Mod * const y)
{
	if (m == 0) { x[0] *= y[0]; return; }

	// We have e = 0: r = 1

	for (size_t k = 0; k < m; ++k) x[k + 0 * m].add_sub(x[k + 1 * m]);
	for (size_t k = 0; k < m; ++k) y[k + 0 * m].add_sub(y[k + 1 * m]);

	mul_Mersenne(m / 2, &x[0 * m], &y[0 * m]);		// root of 1 is still 1
	mul(m / 2, Mod::get_n(), &x[1 * m], &y[1 * m]);	// root is -1 = 2^n

	for (size_t k = 0; k < m; ++k) x[k + 0 * m].add_sub(x[k + 1 * m]);
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
		if (n_i % 64 != 0) continue;
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

// Recursive Schönhage-Strassen-Gallot algorithm, z = x * y (mod 2^N - 1)
static void SSG_mul_Mersenne(mpz_t & z, const mpz_t & x, const mpz_t & y, unsigned int k, const size_t l, const size_t M, const size_t n)
{
	Mod::set_n(n);	// modulo 2^n + 1

	Mod a[l], b[l]; fill_vector(a, l, x, M); fill_vector(b, l, y, M);

	mul_Mersenne(l / 2, a, b);	// top-most recursion level

	// Components are to halved during the reverse transform then multiply outputs by 1/l = -2^n / l = -2^{n - k}
	for (size_t i = 0; i < l; ++i) { a[i] <<= (n - k); a[i].neg(); }

	// Compute sum of the l M-bit part of z
	mpz_t c; mpz_init(c);
	mpz_set_ui(z, 0); for (size_t i = 0; i < l; ++i) { mpz_mul_2exp(z, z, M); a[l - i - 1].get(c); mpz_add(z, z, c); }
	mpz_clear(c);
}

// Recursive Schönhage-Strassen-Gallot algorithm, z < 2^N - 1
static void SSG_mul(mpz_t & z, const mpz_t & x, const mpz_t & y, const size_t N)
{
	unsigned int k = 0;	size_t l, M, n; get_best_param(N, k, l, M, n);
	SSG_mul_Mersenne(z, x, y, k, l, M, n);
}

int main()
{
	gmp_randstate_t randstate;
	gmp_randinit_default(randstate);
	gmp_randseed_ui(randstate, (unsigned long int)(time(nullptr)));
	mpz_t x, y, z, zp, t; mpz_inits(x, y, z, zp, t, nullptr);

	const size_t n_min = 13761, n_max = 50000000;

	std::cout << std::endl << "Check SSG algorithm:" << std::endl;
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

	return EXIT_SUCCESS;
}
