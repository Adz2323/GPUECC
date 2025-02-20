#pragma once
#ifndef GEC_BIGINT_MIXIN_QUADRATIC_RESIDUE_HPP
#define GEC_BIGINT_MIXIN_QUADRATIC_RESIDUE_HPP

#include <bigint/mixin/random.hpp>
#include <utils/basic.hpp>
#include <utils/crtp.hpp>

namespace gec {

namespace bigint {

namespace _legendre_ {

template <typename Int>
GEC_HD int legendre(Int &GEC_RSTRCT a, Int &GEC_RSTRCT p) {
    Int r;

    int k = 1;
    while (!p.is_one()) {
        if (a.is_zero()) {
            return 0;
        }
        size_t z = a.trailing_zeros();
        a.shift_right(z);
        auto v_mod_8 = p.array()[0] & 0x7;
        if ((z & 1) && (v_mod_8 == 3 || v_mod_8 == 5)) {
            k = -k;
        }
        if ((a.array()[0] & 0x3) == 3 && (p.array()[0] & 0x3) == 3) {
            k = -k;
        }
        r = a;
        Int::rem(a, p, r);
        p = r;
    }
    return k;
}

} // namespace _legendre_

template <class Core>
class GEC_EMPTY_BASES Legendre : protected CRTP<Core, Legendre<Core>> {
    friend CRTP<Core, Legendre<Core>>;

  public:
    GEC_HD GEC_INLINE int legendre() {
        Core la(this->core()), lp(this->core().mod());
        return _legendre_::legendre(la, lp);
    }
};

template <class Core>
class GEC_EMPTY_BASES MonLegendre : protected CRTP<Core, MonLegendre<Core>> {
    friend CRTP<Core, MonLegendre<Core>>;

  public:
    GEC_HD GEC_INLINE int legendre() {
        Core la, lp(this->core().mod());
        Core::from_montgomery(la, this->core());
        return _legendre_::legendre(la, lp);
    }
};

// TODO: specify sqrt calculation method in compile time
// TODO: add more specialized sqrt calculation method

template <class Core>
class GEC_EMPTY_BASES ModSqrt : protected CRTP<Core, ModSqrt<Core>> {
    friend CRTP<Core, ModSqrt<Core>>;

    GEC_INLINE GEC_HD static bool mod_sqrt_inner(Core &GEC_RSTRCT x,
                                                 const Core &GEC_RSTRCT a,
                                                 Core &GEC_RSTRCT b) {
        using T = typename Core::LimbT;

        Core y, r, t;
        Core::sub(r, a.mod(), T(1));   // p - 1 = 2^s r
        size_t s = r.trailing_zeros(); //
        r.shift_right(s);              //
        Core::pow(y, b, r);            // y = b^r
        Core::sub(r, T(1));            // r = (r - 1) / 2
        r.shift_right(1);              //
        Core::pow(x, a, r);            // x = a^r
        Core::mul(t, a, x);            // b = a x^2
        Core::mul(b, t, x);            //
        x = t;                         // x = a x

        while (!b.is_mul_id()) {
            size_t m = 1;
            Core::mul(r, b, b);
            while (!r.is_mul_id()) {
                Core::mul(t, r, r);
                r = t;
                ++m;
            }
            if (m == s) {
                return false;
            }
            t.set_pow2(s - m - 1); // r = y^(2^(s - m - 1))
            Core::pow(r, y, t);    //
            Core::mul(y, r, r);    // y = r^2
            s = m;                 // s = m
            Core::mul(t, r, x);    // x = r x
            x = t;                 //
            Core::mul(t, y, b);    // b = y b
            b = t;                 //
        }

        return true;
    }

  public:
    GEC_HD static bool mod_sqrt(Core &GEC_RSTRCT x, const Core &GEC_RSTRCT a,
                                Core &GEC_RSTRCT b) {
        return mod_sqrt_inner(x, a, b);
    }

    template <typename Rng>
    GEC_HD static bool mod_sqrt(Core &GEC_RSTRCT x, const Core &GEC_RSTRCT a,
                                GecRng<Rng> &rng) {
        Core b;
        do {
            Core::sample(b, rng);
        } while (b.legendre() != -1);

        return mod_sqrt_inner(x, a, b);
    }

    GEC_HD static bool mod_sqrt(Core &GEC_RSTRCT x, const Core &GEC_RSTRCT a) {
        Core b;
        do {
            Core::add(b, 1);
        } while (b.legendre() != -1);

        return mod_sqrt_inner(x, a, b);
    }
};

/**
 * @brief mixin that enables quadratic residue related methods
 */
template <class Core>
class GEC_EMPTY_BASES QuadraticResidue : public Legendre<Core>,
                                         public ModSqrt<Core> {};

/**
 * @brief mixin that enables quadratic residue related methods, for montgomery
 * representation
 */
template <class Core>
class GEC_EMPTY_BASES MonQuadraticResidue : public MonLegendre<Core>,
                                            public ModSqrt<Core> {};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_QUADRATIC_RESIDUE_HPP
