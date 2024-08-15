/*
Copyright 2024, Yves Gallot

SSGA is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <cstdint>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <chrono>

#include <gmp.h>

#define finline	__attribute__((always_inline))

// A vector of l elements, operation are modulo 2^n + 1
class ModVector
{
private:
	const size_t _size, _n;
	uint64_t * const _buf;
	uint64_t * const _d;

#ifndef __x86_64
	finline static uint64_t _addc(const uint64_t x, const uint64_t y, uint64_t & carry)
	{
		const __uint128_t t = x + __uint128_t(y) + carry;
		carry = uint64_t(t >> 64);
		return uint64_t(t);
	}

	finline static uint64_t _subb(const uint64_t x, const uint64_t y, int64_t & borrow)
	{
		const __int128_t t = x - __int128_t(y) + borrow;
		borrow = int64_t(t >> 64);
		return uint64_t(t);
	}
#endif

	finline void _get(const uint64_t * const x, uint64_t * const dst, size_t & dst_size) const
	{
		size_t size = _size; while ((size != 0) && (x[size - 1] == 0)) --size;
		for (size_t i = 0; i < size; ++i) dst[i] = x[i];
		dst_size = size;
	}

	finline void _set(uint64_t * const x, const uint64_t * const src, const size_t src_size) const
	{
		for (size_t i = 0; i < src_size; ++i) x[i] = src[i];
		for (size_t i = src_size, size = _size; i < size; ++i) x[i] = 0;
	}

	finline bool _add_sub(uint64_t * const x, uint64_t * const y) const
	{
#ifdef __x86_64
		const size_t size_4 = _size / 4;	// size = 4 * size_4 + 1
		char borrow = 0;
		asm volatile
		(
			"movq	%[size_4], %%rcx\n\t"
			"movq	%[x], %%rsi\n\t"
			"movq	%[y], %%rdi\n\t"
			"xorq	%%rax, %%rax\n\t"		// carry of sbb
			"xorq	%%rdx, %%rdx\n\t"		// carry of adc
			"clc\n\t"

			"loop%=:\n\t"
			"neg	%%al\n\t"				// CF is set to 0 if dl = 0, otherwise it is set to 1

			"movq	(%%rsi), %%rbx\n\t"
			"movq	(%%rdi), %%r9\n\t"
			"movq	%%rbx, %%r8\n\t"		// r8 = x[0], r9 = y[0]
			"sbbq	%%r9, %%rbx\n\t"
			"movq	%%rbx, (%%rdi)\n\t"		// y[i] = rbx = x[i] - y[i]

			"movq	8(%%rsi), %%rbx\n\t"
			"movq	8(%%rdi), %%r11\n\t"
			"movq	%%rbx, %%r10\n\t"		// r10 = x[1], r11 = y[1]
			"sbbq	%%r11, %%rbx\n\t"
			"movq	%%rbx, 8(%%rdi)\n\t"

			"movq	16(%%rsi), %%rbx\n\t"
			"movq	16(%%rdi), %%r13\n\t"
			"movq	%%rbx, %%r12\n\t"		// r12 = x[2], r13 = y[2]
			"sbbq	%%r13, %%rbx\n\t"
			"movq	%%rbx, 16(%%rdi)\n\t"

			"movq	24(%%rsi), %%rbx\n\t"
			"movq	24(%%rdi), %%r15\n\t"
			"movq	%%rbx, %%r14\n\t"		// r14 = x[3], r15 = y[3]
			"sbbq	%%r15, %%rbx\n\t"
			"movq	%%rbx, 24(%%rdi)\n\t"

			"setc	%%al\n\t"
			"neg	%%dl\n\t"

			"adcq	%%r9, %%r8\n\t"
			"movq	%%r8, (%%rsi)\n\t"		// x[i] = x[i] + y[i]
			"adcq	%%r11, %%r10\n\t"
			"movq	%%r10, 8(%%rsi)\n\t"
			"adcq	%%r13, %%r12\n\t"
			"movq	%%r12, 16(%%rsi)\n\t"
			"adcq	%%r15, %%r14\n\t"
			"movq	%%r14, 24(%%rsi)\n\t"

			"setc	%%dl\n\t"

			"decq	%%rcx\n\t"
			"addq	$32, %%rsi\n\t"
			"addq	$32, %%rdi\n\t"
			"testq	%%rcx, %%rcx\n\t"
			"jne	loop%=\n\t"

			"neg	%%al\n\t"

			"movq	(%%rsi), %%rbx\n\t"
			"movq	(%%rdi), %%r9\n\t"
			"movq	%%rbx, %%r8\n\t"
			"sbbq	%%r9, %%rbx\n\t"
			"movq	%%rbx, (%%rdi)\n\t"

			"setc	%[borrow]\n\t"
			"neg	%%dl\n\t"

			"adcq	%%r9, %%r8\n\t"
			"movq	%%r8, (%%rsi)\n\t"

			: [borrow] "=rm" (borrow)
			: [x] "rm" (x), [y] "rm" (y), [size_4] "rm" (size_4)
			: "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "cc", "memory"
		);
#else
	uint64_t carry = 0; int64_t borrow = 0;
	for (size_t i = 0, size = _size; i < size; ++i)
	{
		const uint64_t x_i = x[i], y_i = y[i];
		x[i] = _addc(x_i, y_i, carry);
		y[i] = _subb(x_i, y_i, borrow);
	}
#endif
		return (borrow != 0);
	}

	finline bool _add_subr(uint64_t * const x, uint64_t * const y) const
	{
#ifdef __x86_64
		const size_t size_4 = _size / 4;	// size = 4 * size_4 + 1
		char borrow = 0;
		asm volatile
		(
			"movq	%[size_4], %%rcx\n\t"
			"movq	%[x], %%rsi\n\t"
			"movq	%[y], %%rdi\n\t"
			"xorq	%%rax, %%rax\n\t"		// carry of sbb
			"xorq	%%rdx, %%rdx\n\t"		// carry of adc
			"clc\n\t"

			"loop%=:\n\t"
			"neg	%%al\n\t"				// CF is set to 0 if dl = 0, otherwise it is set to 1

			"movq	(%%rsi), %%r8\n\t"
			"movq	(%%rdi), %%rbx\n\t"
			"movq	%%rbx, %%r9\n\t"		// r8 = x[0], r9 = y[0]
			"sbbq	%%r8, %%rbx\n\t"
			"movq	%%rbx, (%%rdi)\n\t"		// y[i] = rbx = y[i] - x[i]

			"movq	8(%%rsi), %%r10\n\t"
			"movq	8(%%rdi), %%rbx\n\t"
			"movq	%%rbx, %%r11\n\t"		// r10 = x[1], r11 = y[1]
			"sbbq	%%r10, %%rbx\n\t"
			"movq	%%rbx, 8(%%rdi)\n\t"

			"movq	16(%%rsi), %%r12\n\t"
			"movq	16(%%rdi), %%rbx\n\t"
			"movq	%%rbx, %%r13\n\t"		// r12 = x[2], r13 = y[2]
			"sbbq	%%r12, %%rbx\n\t"
			"movq	%%rbx, 16(%%rdi)\n\t"

			"movq	24(%%rsi), %%r14\n\t"
			"movq	24(%%rdi), %%rbx\n\t"
			"movq	%%rbx, %%r15\n\t"		// r14 = x[3], r15 = y[3]
			"sbbq	%%r14, %%rbx\n\t"
			"movq	%%rbx, 24(%%rdi)\n\t"

			"setc	%%al\n\t"
			"neg	%%dl\n\t"

			"adcq	%%r9, %%r8\n\t"
			"movq	%%r8, (%%rsi)\n\t"		// x[i] = x[i] + y[i]
			"adcq	%%r11, %%r10\n\t"
			"movq	%%r10, 8(%%rsi)\n\t"
			"adcq	%%r13, %%r12\n\t"
			"movq	%%r12, 16(%%rsi)\n\t"
			"adcq	%%r15, %%r14\n\t"
			"movq	%%r14, 24(%%rsi)\n\t"

			"setc	%%dl\n\t"

			"decq	%%rcx\n\t"
			"addq	$32, %%rsi\n\t"
			"addq	$32, %%rdi\n\t"
			"testq	%%rcx, %%rcx\n\t"
			"jne	loop%=\n\t"

			"neg	%%al\n\t"

			"movq	(%%rsi), %%r8\n\t"
			"movq	(%%rdi), %%rbx\n\t"
			"movq	%%rbx, %%r9\n\t"
			"sbbq	%%r8, %%rbx\n\t"
			"movq	%%rbx, (%%rdi)\n\t"

			"setc	%[borrow]\n\t"
			"neg	%%dl\n\t"

			"adcq	%%r9, %%r8\n\t"
			"movq	%%r8, (%%rsi)\n\t"

			: [borrow] "=rm" (borrow)
			: [x] "rm" (x), [y] "rm" (y), [size_4] "rm" (size_4)
			: "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "cc", "memory"
		);
#else
		uint64_t carry = 0; int64_t borrow = 0;
		for (size_t i = 0, size = _size; i < size; ++i)
		{
			const uint64_t x_i = x[i], y_i = y[i];
			x[i] = _addc(x_i, y_i, carry);
			y[i] = _subb(y_i, x_i, borrow);
		}
#endif
		return (borrow != 0);
	}

	finline bool _is_zero(const uint64_t * const x) const
	{
		for (size_t i = 0, size = _size; i < size; ++i) if (x[i] != 0) return false;
		return true;
	}

	// x ?>= 2^n + 1
	finline bool _ge_F(const uint64_t * const x) const
	{
		const size_t size = _size;
		if (x[size - 1] != 1) return (x[size - 1] > 1);
		for (size_t i = 0; i < size - 1; ++i) if (x[i] != 0) return true;
		return false;
	}

	// x += 2^n + 1
	finline void _add_F(uint64_t * const x) const
	{
		const size_t size = _size;
		x[size - 1] += 1;
		const uint64_t x_0 = x[0] + 1; x[0] = x_0;
		if (x_0 == 0)	// carry
		{
			for (size_t i = 1; i < size; ++i)
			{
				const uint64_t x_i = x[i] + 1; x[i] = x_i;
				if (x_i != 0) break;
			}
		}
	}

	// x -= 2^n + 1
	finline void _sub_F(uint64_t * const x) const
	{
		const size_t size = _size;
		x[size - 1] -= 1;
		const uint64_t x_0 = x[0]; x[0] = x_0 - 1;
		if (x_0 == 0)	// borrow
		{
			for (size_t i = 1; i < size; ++i)
			{
				const uint64_t x_i = x[i]; x[i] = x_i - 1;
				if (x_i != 0) break;
			}
		}
	}

	// x = 2^n + 1 - x
	finline void _F_sub(uint64_t * const x) const
	{
#ifdef __x86_64
		const size_t size_4 = _size / 4;	// size = 4 * size_4 + 1
		asm volatile
		(
			"movq	%[size_4], %%rcx\n\t"
			"movq	%[x], %%rsi\n\t"
			"xorq	%%rdx, %%rdx\n\t"
			"clc\n\t"

			"movq	$1, %%rax\n\t"

			"loop%=:\n\t"
			"sbbq	(%%rsi), %%rax\n\t"
			"movq	%%rax, (%%rsi)\n\t"
			"movq	%%rdx, %%rax\n\t"
			"sbbq	8(%%rsi), %%rax\n\t"
			"movq	%%rax, 8(%%rsi)\n\t"
			"movq	%%rdx, %%rax\n\t"
			"sbbq	16(%%rsi), %%rax\n\t"
			"movq	%%rax, 16(%%rsi)\n\t"
			"movq	%%rdx, %%rax\n\t"
			"sbbq	24(%%rsi), %%rax\n\t"
			"movq	%%rax, 24(%%rsi)\n\t"
			"movq	%%rdx, %%rax\n\t"

			"leaq	32(%%rsi), %%rsi\n\t"
			"decq	%%rcx\n\t"
			"jnz	loop%=\n\t"

			"movq	$1, %%rax\n\t"
			"sbbq	(%%rsi), %%rax\n\t"
			"movq	%%rax, (%%rsi)\n\t"

			:
			: [x] "rm" (x), [size_4] "rm" (size_4)
			: "rax", "rcx", "rdx", "rsi", "cc", "memory"
		);
#else
		const size_t size = _size;
		int64_t borrow = 0;
		x[0] = _subb(1, x[0], borrow);
		for (size_t i = 1; i < size - 1; ++i) x[i] = _subb(0, x[i], borrow);
		x[size - 1] = _subb(1, x[size - 1], borrow);
#endif
	}

	// buf = x << s
	finline void _lshift(const uint64_t * const x, const size_t s) const
	{
		const size_t size = _size;
		uint64_t * const buf = _buf;
		const size_t s_64 = s / 64, s_mod64 = s % 64;	// s_64 < size
		for (size_t i = 0; i < s_64; ++i) buf[i] = 0;
		if (s_mod64 != 0) buf[size + s_64] = mpn_lshift(&buf[s_64], x, size, s_mod64);
		else { mpn_copyi(&buf[s_64], x, size); buf[size + s_64] = 0; }
		for (size_t i = size + s_64 + 1; i < 2 * size; ++i) buf[i] = 0;
	}

	// x = buf (mod 2^n + 1)
	finline void _mod_F(uint64_t * const x) const
	{
#ifdef __x86_64
		const size_t size = _size;
		char borrow = 0;
		asm volatile
		(
			"movq	%[size], %%rcx\n\t"
			"movq	%[buf], %%rsi\n\t"
			"movq	%[x], %%rdi\n\t"
			"leaq	-8(%%rsi,%%rcx,8), %%rbx\n\t"	// buf[size - 1]
			"shrq	$2,	%%rcx\n\t"					// size = 4 * size_4 + 1
			"clc\n\t"

			"loop%=:\n\t"
			"movq	(%%rsi), %%rax\n\t"
			"sbbq	(%%rbx), %%rax\n\t"
			"movq	%%rax, (%%rdi)\n\t"
			"movq	8(%%rsi), %%rax\n\t"
			"sbbq	8(%%rbx), %%rax\n\t"
			"movq	%%rax, 8(%%rdi)\n\t"
			"movq	16(%%rsi), %%rax\n\t"
			"sbbq	16(%%rbx), %%rax\n\t"
			"movq	%%rax, 16(%%rdi)\n\t"
			"movq	24(%%rsi), %%rax\n\t"
			"sbbq	24(%%rbx), %%rax\n\t"
			"movq	%%rax, 24(%%rdi)\n\t"

			"leaq	32(%%rsi), %%rsi\n\t"
			"leaq	32(%%rbx), %%rbx\n\t"
			"leaq	32(%%rdi), %%rdi\n\t"
			"decq	%%rcx\n\t"
			"jnz	loop%=\n\t"

			"movq	$0, %%rax\n\t"
			"movq	(%%rbx), %%rdx\n\t"
			"sbbq	%%rdx, %%rax\n\t"
			"movq	%%rax, (%%rdi)\n\t"

			"setc	%[borrow]\n\t"

			: [borrow] "=rm" (borrow)
			: [x] "rm" (x), [buf] "rm" (_buf), [size] "rm" (size)
			: "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "cc", "memory"
		);
#else
		const size_t size = _size;
		const uint64_t * const buf = _buf;
		// buf mod 2^n - buf / 2^n
	 	int64_t borrow = 0;
	 	for (size_t i = 0; i < size - 1; ++i) x[i] = _subb(buf[i], buf[size - 1 + i], borrow);
		x[size - 1] = _subb(0, buf[2 * size - 2], borrow);
#endif
	 	if (borrow != 0) _add_F(x);
	}

	static const size_t _gap = 7;	// Cache line size is 64 bytes

public:
	ModVector(const size_t n, const size_t l) : _size(n / 64 + 1), _n(n), _buf(new uint64_t[2 * _size]), _d(new uint64_t[l * _size]) {}
	virtual ~ModVector() { delete[] _buf; delete[] _d; }

	size_t get_size() const { return _size; }

	void get(const size_t i, uint64_t * const x, size_t & x_size) const { _get(&_d[i * _size], x, x_size); }
	void set(const size_t i, const uint64_t * const x, const size_t x_size) { _set(&_d[i * _size], x, x_size); }

	void neg(const size_t i)
	{
		uint64_t * const d_i = &_d[i * _size];
		if (!_is_zero(d_i)) _F_sub(d_i);
	}

	void add_sub(const size_t i, const size_t j)
	{
		uint64_t * const d_i = &_d[i * _size];
		uint64_t * const d_j = &_d[j * _size];
		const bool borrow = _add_sub(d_i, d_j);
		if (_ge_F(d_i)) _sub_F(d_i);
		if (borrow) _add_F(d_j);
	}

	void add_subr(const size_t i, const size_t j)
	{
		uint64_t * const d_i = &_d[i * _size];
		uint64_t * const d_j = &_d[j * _size];
		const bool borrow = _add_subr(d_i, d_j);
		if (_ge_F(d_i)) _sub_F(d_i);
		if (borrow) _add_F(d_j);
	}

	void lshift(const size_t i, const size_t s)
	{
		uint64_t * const d_i = &_d[i * _size];
		_lshift(d_i, s);
		_mod_F(d_i);
	}

	void negacyclic(const ModVector & rhs, const size_t i)
	{
		uint64_t * const d_i = &_d[i * _size];
		mpn_mul_n(_buf, d_i, &rhs._d[i * _size], _size);
		_mod_F(d_i);
	}
};

class SSG
{
private:
	const unsigned int _k;
	const size_t _M, _n, _l;
	ModVector _x;
	ModVector _y;

	// v is a vector of l M-bit slices of x
	void set_vector(ModVector & v, const uint64_t * const x, const size_t size)
	{
		const size_t M_64 = _M / 64, M_mask = (size_t(1) << (_M % 64)) - 1;

		uint64_t * const r = new uint64_t[M_64 + 2];

		for (size_t k = 0, l = _l; k < l; ++k)
		{
			const size_t bit_index = k * _M, index = bit_index / 64;

			if (index < size)
			{
				const size_t left = size - index;
				for (size_t i = 0, n = std::min(left, M_64 + 2); i < n; ++i) r[i] = x[index + i];
				for (size_t i = left; i < M_64 + 2; ++i) r[i] = 0;

				const size_t s = bit_index % 64;
				if (s != 0) mpn_rshift(r, r, M_64 + 2, s);
				r[M_64] &= M_mask;

				v.set(k, r, M_64 + 1);
			}
			else v.set(k, r, 0);
		}

		delete[] r;
	}

	// Compute sum of the l slices
	void get_vector(uint64_t * const x, const size_t size, const ModVector & v)
	{
		for (size_t i = 0; i < size; ++i) x[i] = 0;

		uint64_t * const r = new uint64_t[v.get_size() + 1];

		for (size_t k = 0, l = _l; k < l; ++k)
		{
			const size_t bit_index = k * _M, index = bit_index / 64, s = bit_index % 64;

			size_t r_size; v.get(k, r, r_size);
			if (r_size != 0)
			{
				if (s != 0) r[r_size] = mpn_lshift(r, r, r_size, s); else r[r_size] = 0;
				mpn_add_n(&x[index], &x[index], r, r_size + 1);
			}
		}

		delete[] r;
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
		while (n % (64 * 4) != 0) n += K;
		return	double(t) / n;	// efficiency
	}

	void mul(const size_t m, const size_t k, const size_t e)
	{
		if (m == 0) { _x.negacyclic(_y, k); return; }

		const size_t e_2 = e / 2;	// previous root is r^2 = 2^e, new root r = 2^{e/2}

		for (size_t i = 0; i < m; ++i) { _x.lshift(k + i + 1 * m, e_2); _x.add_sub(k + i + 0 * m, k + i + 1 * m); }
		for (size_t i = 0; i < m; ++i) { _y.lshift(k + i + 1 * m, e_2); _y.add_sub(k + i + 0 * m, k + i + 1 * m); }

		mul(m / 2, k + 0 * m, e_2);			//  r = 2^{e/2}
		mul(m / 2, k + 1 * m, e_2 + _n);	// -r = 2^{e/2 + n}

		const size_t me_2 = _n - e_2;		// 2^n = -1 then r^-1 = 2^-{e/2} = -2^{n - e/2}

		for (size_t i = 0; i < m; ++i) { _x.add_subr(k + i + 0 * m, k + i + 1 * m); _x.lshift(k + i + 1 * m, me_2); }
	}

	void mul_Mersenne(const size_t m, const size_t k)
	{
		if (m == 0) { _x.negacyclic(_y, k); return; }

		// We have e = 0: r = 1

		for (size_t i = 0; i < m; ++i) _x.add_sub(k + i + 0 * m, k + i + 1 * m);
		for (size_t i = 0; i < m; ++i) _y.add_sub(k + i + 0 * m, k + i + 1 * m);

		mul_Mersenne(m / 2, k + 0 * m);		// root of 1 is still 1
		mul(m / 2, k + 1 * m, _n);			// root is -1 = 2^n

		for (size_t i = 0; i < m; ++i) _x.add_sub(k + i + 0 * m, k + i + 1 * m);
	}

public:
	SSG(const unsigned int k, const size_t M, const size_t n) : _k(k), _M(M), _n(n), _l(size_t(1) << k), _x(n, _l), _y(n, _l) {}

	virtual ~SSG() {}

	void set_x(const uint64_t * const x, const size_t size) { set_vector(_x, x, size); }
	void set_y(const uint64_t * const y, const size_t size) { set_vector(_y, y, size); }

	void get_x(uint64_t * const x, const size_t size) {get_vector(x, size, _x); }

	void mul()
	{
		const size_t l = _l;

		mul_Mersenne(l / 2, 0);	// top-most recursion level

		// Components are not halved during the reverse transform then multiply outputs by 1/l = -2^n / l = -2^{n - k}
		const size_t s = _n - _k;
		for (size_t k = 0; k < l; ++k) { _x.lshift(k, s); _x.neg(k); }
	}

	static void get_best_param(const size_t N, unsigned int & k, size_t & M, size_t & n, const bool verbose)
	{
		k = 0;
		for (unsigned int i = 1; true; ++i)
		{
			size_t M_i, n_i; const double efficiency = get_param(N, i, M_i, n_i);
			const size_t K_i = size_t(1) << i;
			if (K_i > 2 * std::sqrt(M_i * K_i)) break;
			if (efficiency > 0.95) k = i;
	}

		const double efficiency = get_param(N, k, M, n);
		if (verbose) std::cout << ", N' = " << (M << k) << ", M = " << M << ", k = " << k << ", n = 64 * " << n / 64 << ", efficiency = " << efficiency << ", ";
	}
};

// Recursive Schönhage-Strassen-Gallot algorithm
static void SSG_mul(uint64_t * const z, const uint64_t * const x, const size_t x_size, const uint64_t * const y, const size_t y_size, const bool verbose)
{
	const size_t z_size = x_size + y_size;
	unsigned int k = 0;	size_t M, n; SSG::get_best_param(64 * z_size, k, M, n, verbose);

	SSG ssg(k, M, n);
	ssg.set_x(x, x_size);
	ssg.set_y(y, y_size);
	ssg.mul();
	ssg.get_x(z, z_size);
}

int main()
{
	std::cout << std::fixed << std::setprecision(3) << std::endl << "Check SSG algorithm:" << std::endl;
	bool parity = true;
	for (unsigned int d = 4; d <= 9; ++d)
	{
		const size_t N = size_t(std::pow(10, d) * std::log2(10));

		std::cout << "10^" << d << " digits, N = " << N;

		// Generate two random numbers
		const size_t x_bitcount = 4 * N / 5, y_bitcount = N / 5;
		const size_t x_size = x_bitcount / 64 + 1, y_size = y_bitcount / 64 + 1, z_size = x_size + y_size;
		uint64_t * const x = new uint64_t[x_size];
		uint64_t * const y = new uint64_t[y_size];
		uint64_t * const z = new uint64_t[z_size];
		uint64_t * const zp = new uint64_t[z_size];
		if (parity) { mpn_random2(x, x_size); mpn_random2(y, y_size); }
		else { mpn_random(x, x_size); mpn_random(y, y_size); }
		parity = !parity;

		// z = x * y
		const size_t count = std::max(1000000000 / N, size_t(1));
		const auto start_mpn = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i < count; ++i) mpn_mul(z, x, x_size, y, y_size);
		const double elapsed_time_mpn = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_mpn).count() / count;
		const auto start_ssg = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i < count; ++i) SSG_mul(zp, x, x_size, y, y_size, i == 0);
		const double elapsed_time_ssg = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_ssg).count() / count;

		std::cout << ((mpn_cmp(z, zp, z_size) == 0) ? "OK" : "Error")
			<< ", mpn: " << elapsed_time_mpn << " sec, SSG: " << elapsed_time_ssg << " sec (" << int(100 * elapsed_time_ssg / elapsed_time_mpn) << "%)"
			<< "." << std::endl;

		delete[] x;
		delete[] y;
		delete[] z;
		delete[] zp;
	}

	return EXIT_SUCCESS;
}
