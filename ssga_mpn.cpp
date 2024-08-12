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

	static uint64_t _subb(const uint64_t x, const uint64_t y, int64_t & borrow)
	{
		const __int128_t t = x - __int128_t(y) + borrow;
		borrow = int64_t(t >> 64);
		return uint64_t(t);
	}

	bool _add_sub(Mod & rhs)
	{
		const size_t size = _size;
		uint64_t * const x = _d;
		uint64_t * const y = rhs._d;

		uint64_t carry = 0; int64_t borrow = 0;
		for (size_t i = 0; i < size; ++i)
		{
			const uint64_t x_i = x[i], y_i = y[i];
			x[i] = _addc(x_i, y_i, carry);
			y[i] = _subb(x_i, y_i, borrow);
		}

		return (borrow != 0);
	}

	bool _add_subr(Mod & rhs)
	{
		const size_t size = _size;
		uint64_t * const x = _d;
		uint64_t * const y = rhs._d;

		uint64_t carry = 0; int64_t borrow = 0;
		for (size_t i = 0; i < size; ++i)
		{
			const uint64_t x_i = x[i], y_i = y[i];
			x[i] = _addc(x_i, y_i, carry);
			y[i] = _subb(y_i, x_i, borrow);
		}

		return (borrow != 0);
	}

	bool is_zero() const
	{
		const size_t size = _size;
		const uint64_t * const d = _d;

		for (size_t i = 0; i < size; ++i) if (d[i] != 0) return false;
		return true;
	}

	// *this ?>= 2^n + 1
	bool ge_F() const
	{
		const size_t size = _size;
		const uint64_t * const d = _d;

		if (d[size - 1] != 1) return (d[size - 1] > 1);
		for (size_t i = 0; i < size - 1; ++i) if (d[i] != 0) return true;
		return false;
	}

	// *this += 2^n + 1
	void add_F()
	{
		const size_t size = _size;
		uint64_t * const d = _d;

		uint64_t carry = 0;
		d[0] = _addc(d[0], 1, carry);
		if (carry != 0) for (size_t i = 1; i < size - 1; ++i) { d[i] = _addc(d[i], 0, carry); if (carry == 0) break; }
		d[size - 1] = _addc(d[size - 1], 1, carry);
	}

	// *this -= 2^n + 1
	void sub_F()
	{
		const size_t size = _size;
		uint64_t * const d = _d;

		int64_t borrow = 0;
		d[0] = _subb(d[0], 1, borrow);
		if (borrow != 0) for (size_t i = 1; i < size - 1; ++i) { d[i] = _subb(d[i], 0, borrow); if (borrow == 0) break; }
		d[size - 1] = _subb(d[size - 1], 1, borrow);
	}

	// *this = 2^n + 1 - *this
	void F_sub()
	{
		const size_t size = _size;
		uint64_t * const d = _d;

		int64_t borrow = 0;
		d[0] = _subb(1, d[0], borrow);
		for (size_t i = 1; i < size - 1; ++i) d[i] = _subb(0, d[i], borrow);
		d[size - 1] = _subb(1, d[size - 1], borrow);
	}

	// buf = *this << s
	void lshift(const size_t s) const
	{
		const size_t size = _size;
		uint64_t * const buf = _buf;

		const size_t s_64 = s / 64, s_mod64 = s % 64;	// s_64 < size
		for (size_t i = 0; i < s_64; ++i) buf[i] = 0;
		if (s_mod64 != 0) buf[size + s_64] = mpn_lshift(&buf[s_64], _d, size, s_mod64);
		else { mpn_copyi(&buf[s_64], _d, size); buf[size + s_64] = 0; }
		for (size_t i = size + s_64 + 1; i < 2 * size; ++i) buf[i] = 0;
	}

	// *this = buf (mod 2^n + 1)
	void mod_F()
	{
		const size_t size = _size;
		uint64_t * const d = _d;
		const uint64_t * const buf = _buf;

		// buf mod 2^n - buf / 2^n
	 	int64_t borrow = 0;
	 	for (size_t i = 0; i < size - 1; ++i) d[i] = _subb(buf[i], buf[size - 1 + i], borrow);
		d[size - 1] = _subb(0, buf[2 * size - 2], borrow);
	 	if (borrow != 0) add_F();
	}

public:
	static void set_n(const size_t n)
	{
		_n = n; _size = n / 64 + 1;
		if (_buf != nullptr) delete[] _buf;
		_buf = new uint64_t[2 * _size];
	}

	static size_t get_n() { return _n; }
	static size_t get_size() { return _size; }

	Mod() : _d(new uint64_t[_size]) {}
	virtual ~Mod() { delete[] _d; }

	void get(uint64_t * const x) const
	{
		const size_t size = _size;
		const uint64_t * const d = _d;

		for (size_t i = 0; i < size; ++i) x[i] = d[i];
	}

	void set(const uint64_t * const x, const size_t x_size)
	{
		const size_t size = _size;
		uint64_t * const d = _d;

		for (size_t i = 0; i < x_size; ++i) d[i] = x[i];
		for (size_t i = x_size; i < size; ++i) d[i] = 0;
	}

 	Mod & neg() { if (!is_zero()) F_sub(); return *this; }

	Mod & add_sub(Mod & rhs)
	{
		const bool borrow = _add_sub(rhs);
		if (ge_F()) sub_F();
		if (borrow) rhs.add_F();
		return *this;
	}

	Mod & add_subr(Mod & rhs)
	{
		const bool borrow = _add_subr(rhs);
		if (ge_F()) sub_F();
		if (borrow) rhs.add_F();
		return *this;
	}

	Mod & operator<<=(const size_t s)
	{
		lshift(s);
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

	for (size_t k = 0; k < m; ++k) { x[k + 0 * m].add_subr(x[k + 1 * m]); x[k + 1 * m] <<= me_2; }
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

static double get_param(const size_t N, const unsigned int k, size_t & M, size_t & n)
{
	// See Pierrick Gaudry, Alexander Kruppa, Paul Zimmermann.
	// A GMP-based implementation of Schönhage-Strassen's large integer multiplication algorithm.
	// ISSAC 2007, Jul 2007, Waterloo, Ontario, Canada. pp.167-174, ⟨10.1145/1277548.1277572⟩. ⟨inria-00126462v2⟩
	const size_t K = size_t(1) << k;
	M = (N % K == 0) ? N / K : N / K + 1;
	const size_t t = 2 * M + k;
	n = t; if (n % K != 0) n = (n / K + 1) * K;
	return	double(t) / n;	// efficiency
}

static void get_best_param(const size_t N, unsigned int & k, size_t & M, size_t & n)
{
	k = 0;
	for (unsigned int i = 1; true; ++i)
	{
		size_t M_i, n_i; const double efficiency = get_param(N, i, M_i, n_i);
		if (n_i % 64 != 0) continue;
		const size_t K_i = size_t(1) << i;
		if (K_i > 2 * std::sqrt(M_i * K_i)) break;
		if (efficiency > 0.8) k = i;
	}

	const double efficiency = get_param(N, k, M, n);
 	std::cout << "N = " << N << ", sqrt(N) = " << int(std::sqrt(N)) << ", N' = " << (M << k) << ", M = " << M
		<< ", k = " << k << ", n = " << n << ", efficiency = " << efficiency << ", ";
}

// v is a vector of l M-bit part of x
static void fill_vector(Mod * const v, const size_t l, const uint64_t * const x, const size_t size, const size_t M)
{
	const size_t M_64 = M / 64, M_mod64 = M % 64;

	uint64_t * const t = new uint64_t[size];
	for (size_t i = 0; i < size; ++i) t[i] = x[i];

	uint64_t * const r = new uint64_t[M_64 + 1];

	for (size_t i = 0; i < l; ++i)
	{
		for (size_t j = 0; j < M_64; ++j) r[j] = t[j];
		r[M_64] = (t[M_64] & ((size_t(1) << M_mod64) - 1));
		v[i].set(r, M_64 + 1);

		for (size_t j = 0; j < size - M_64; ++j) t[j] = t[j + M_64];
		for (size_t j = size - M_64; j < size; ++j) t[j] = 0;
		if (M_mod64 != 0) mpn_rshift(t, t, size, M_mod64);
	}

	delete[] t;
	delete[] r;
}

// Recursive Schönhage-Strassen-Gallot algorithm
static void SSG_mul(uint64_t * const z, const uint64_t * const x, const size_t x_size, const uint64_t * const y, const size_t y_size)
{
	const size_t z_size = x_size + y_size;
	unsigned int k = 0;	size_t M, n; get_best_param(64 * z_size, k, M, n);
	const size_t l = size_t(1) << k;

	Mod::set_n(n);	// modulo 2^n + 1

	Mod a[l], b[l]; fill_vector(a, l, x, x_size, M); fill_vector(b, l, y, y_size, M);

	mul_Mersenne(l / 2, a, b);	// top-most recursion level

	// Components are not halved during the reverse transform then multiply outputs by 1/l = -2^n / l = -2^{n - k}
	for (size_t i = 0; i < l; ++i) { a[i] <<= (n - k); a[i].neg(); }

	// Compute sum of the l M-bit part of z
	for (size_t i = 0; i < z_size; ++i) z[i] = 0;
	const size_t M_64 = M / 64, M_mod64 = M % 64;
	const size_t c_size = Mod::get_size();
	uint64_t * const c = new uint64_t[c_size];
	for (size_t i = 0; i < l; ++i)
	{
		for (size_t j = z_size - 1; j >= M_64; --j) z[j] = z[j - M_64];
		for (size_t j = 0; j < M_64; ++j) z[j] = 0;
		if (M_mod64 != 0) mpn_lshift(z, z, z_size, M_mod64);
		a[l - i - 1].get(c);
		mpn_add(z, z, z_size, c, c_size);
	}
	delete[] c;
}

int main()
{
	std::cout << std::endl << "Check SSG algorithm:" << std::endl;
	for (size_t N = 13761; N < 50000000ull; N *= 3)
	{
		// Generate two random numbers such that x * y < 2^N - 1
		const size_t x_bitcount = 4 * N / 5, y_bitcount = N / 5;
		const size_t x_size = x_bitcount / 64 + 1, y_size = y_bitcount / 64 + 1, z_size = x_size + y_size;
		uint64_t * const x = new uint64_t[x_size];
		mpn_random2(x, x_size);	// mpn_random2 fails
		uint64_t * const y = new uint64_t[y_size];
		mpn_random2(y, y_size);
		uint64_t * const z = new uint64_t[z_size];
		uint64_t * const zp = new uint64_t[z_size];

		// z = x * y
		mpn_mul(z, x, x_size, y, y_size);
		SSG_mul(zp, x, x_size, y, y_size);

		std::cout << ((mpn_cmp(z, zp, z_size) == 0) ? "OK" : "Error") << std::endl;

		delete[] x;
		delete[] y;
		delete[] z;
		delete[] zp;
	}

	return EXIT_SUCCESS;
}
