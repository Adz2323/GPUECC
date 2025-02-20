#pragma once
#ifndef GEC_BIGINT_MIXIN_MONTGOMERY_HPP
#define GEC_BIGINT_MIXIN_MONTGOMERY_HPP

#ifdef GEC_ENABLE_AVX2
#include <immintrin.h>
#endif // GEC_ENABLE_AVX2

#include <utils/arithmetic.hpp>
#include <utils/crtp.hpp>
#include <utils/sequence.hpp>

namespace gec {

namespace bigint {

/** @brief mixin that enables Montgomery multiplication
 *
 * require `Core::set_zero`, `Core::set_one`, `Core::set_pow2` methods
 */
template <class Core, typename LIMB_T, size_t LIMB_N>
class GEC_EMPTY_BASES MontgomeryOps
    : protected CRTP<Core, MontgomeryOps<Core, LIMB_T, LIMB_N>> {
    friend CRTP<Core, MontgomeryOps<Core, LIMB_T, LIMB_N>>;

  public:
    GEC_HD GEC_INLINE static void to_montgomery(Core &GEC_RSTRCT a,
                                                const Core &GEC_RSTRCT b) {
        mul(a, b, a.r_sqr());
    }
    GEC_HD GEC_INLINE static void from_montgomery(Core &GEC_RSTRCT a,
                                                  const Core &GEC_RSTRCT b) {
        using namespace utils;

        LIMB_T *a_arr = a.array();
        const LIMB_T *b_arr = b.array();

        fill_seq<LIMB_N>(a_arr, b_arr);

        for (size_t i = 0; i < LIMB_N; ++i) {
            LIMB_T m = a_arr[0] * a.mod_p();
            LIMB_T last = seq_add_mul_limb<LIMB_N>(a_arr, a.mod().array(), m);

            seq_shift_right<LIMB_N, utils::type_bits<LIMB_T>::value>(a_arr);
            a_arr[LIMB_N - 1] = last;
        }

        if (VtSeqCmp<LIMB_N, LIMB_T>::call(a_arr, a.mod().array()) !=
            CmpEnum::Lt) {
            seq_sub<LIMB_N>(a_arr, a.mod().array());
        }
    }

    GEC_HD static void mul(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
                           const Core &GEC_RSTRCT c) {
        using namespace utils;
        a.set_zero();
        LIMB_T *a_arr = a.array();
        const LIMB_T *b_arr = b.array();
        const LIMB_T *c_arr = c.array();

        bool carry = false;
        for (size_t i = 0; i < LIMB_N; ++i) {
            LIMB_T m = (a_arr[0] + b_arr[i] * c_arr[0]) * a.mod_p();
            LIMB_T last0 = seq_add_mul_limb<LIMB_N>(a_arr, c_arr, b_arr[i]);
            LIMB_T last1 = seq_add_mul_limb<LIMB_N>(a_arr, a.mod().array(), m);
            carry = uint_add_with_carry(last0, last1, carry);

            seq_shift_right<LIMB_N, utils::type_bits<LIMB_T>::value>(a_arr);
            a_arr[LIMB_N - 1] = last0;
        }

        if (carry || VtSeqCmp<LIMB_N, LIMB_T>::call(a_arr, a.mod().array()) !=
                         CmpEnum::Lt) {
            seq_sub<LIMB_N>(a_arr, a.mod().array());
        }
    }

    GEC_HD GEC_INLINE static void inv(Core &GEC_RSTRCT a,
                                      const Core &GEC_RSTRCT b) {
        a = b;
        inv(a);
    }

    GEC_HD static void inv(Core &GEC_RSTRCT a) {
        using utils::CmpEnum;
        constexpr size_t LimbBit = utils::type_bits<LIMB_T>::value;
        constexpr size_t Bits = LimbBit * LIMB_N;
        constexpr LIMB_T mask = LIMB_T(1) << (LimbBit - 1);

        Core r, s, t;
        LIMB_T *a_arr = a.array();
        LIMB_T *r_arr = r.array();
        LIMB_T *s_arr = s.array();
        LIMB_T *t_arr = t.array();

        r = a;
        s.set_one();
        utils::fill_seq<LIMB_N>(t_arr, a.mod().array());
        a.set_zero();
        size_t k = 0;
        bool a_carry = false, s_carry = false;
        while (!utils::SeqEqLimb<LIMB_N, LIMB_T>::call(r_arr, 0)) {
            if (!(t_arr[0] & 0x1)) {
                utils::seq_shift_right<LIMB_N, 1>(t_arr);
                utils::seq_shift_left<LIMB_N, 1>(s_arr);
            } else if (!(r_arr[0] & 0x1)) {
                utils::seq_shift_right<LIMB_N, 1>(r_arr);
                utils::seq_shift_left<LIMB_N, 1>(a_arr);
            } else if (utils::VtSeqCmp<LIMB_N, LIMB_T>::call(t_arr, r_arr) ==
                       CmpEnum::Gt) {
                utils::seq_sub<LIMB_N>(t_arr, r_arr);
                utils::seq_shift_right<LIMB_N, 1>(t_arr);
                bool carry = utils::seq_add<LIMB_N>(a_arr, s_arr);
                a_carry = a_carry || s_carry || carry;
                s_carry = s_carry || bool(mask & s_arr[LIMB_N - 1]);
                utils::seq_shift_left<LIMB_N, 1>(s_arr);
            } else {
                utils::seq_sub<LIMB_N>(r_arr, t_arr);
                utils::seq_shift_right<LIMB_N, 1>(r_arr);
                bool carry = utils::seq_add<LIMB_N>(s_arr, a_arr);
                s_carry = a_carry || s_carry || carry;
                a_carry = a_carry || bool(mask & a_arr[LIMB_N - 1]);
                utils::seq_shift_left<LIMB_N, 1>(a_arr);
            }
            ++k;
        }
        if (a_carry || utils::VtSeqCmp<LIMB_N, LIMB_T>::call(
                           a_arr, a.mod().array()) != CmpEnum::Lt) {
            utils::seq_sub<LIMB_N>(a_arr, a.mod().array());
        }
        utils::seq_sub<LIMB_N>(s_arr, a.mod().array(), a_arr);
        if (k < Bits) {
            mul(t, s, a.r_sqr());
            k += Bits;

            mul(s, t, a.r_sqr());

            if (k == Bits) {
                a = s;
            } else {
                r.set_pow2(2 * Bits - k);
                mul(a, s, r);
            }
        } else {
            mul(t, s, a.r_sqr());

            if (k == Bits) {
                a = t;
            } else {
                r.set_pow2(2 * Bits - k);
                mul(a, t, r);
            }
        }
    }
};

/** @brief mixin that enables Montgomery multiplication without checking carry
 * bit
 *
 * Note this mixin does not check overflow during calculation.
 *
 * If `Core` can hold twice as `MOD`, than replacing `ModAddSubMixin` with this
 * mixin might have a performance boost. Otherwise, the mixin could lead to
 * incorrect result.
 *
 * require `Core::set_zero`, `Core::set_one`, `Core::set_pow2` methods
 */
template <class Core, typename LIMB_T, size_t LIMB_N>
class GEC_EMPTY_BASES CarryFreeMontgomeryOps
    : protected CRTP<Core, CarryFreeMontgomeryOps<Core, LIMB_T, LIMB_N>> {
    friend CRTP<Core, CarryFreeMontgomeryOps<Core, LIMB_T, LIMB_N>>;

  public:
    GEC_HD GEC_INLINE static void to_montgomery(Core &GEC_RSTRCT a,
                                                const Core &GEC_RSTRCT b) {
        mul(a, b, a.r_sqr().array());
    }
    GEC_HD GEC_INLINE static void from_montgomery(Core &GEC_RSTRCT a,
                                                  const Core &GEC_RSTRCT b) {
        using namespace utils;

        LIMB_T *a_arr = a.array();
        const LIMB_T *b_arr = b.array();

        fill_seq<LIMB_N>(a_arr, b_arr);

        for (int i = 0; i < LIMB_N; ++i) {
            LIMB_T m = a_arr[0] * a.mod_p();
            LIMB_T last = seq_add_mul_limb<LIMB_N>(a_arr, a.mod().array(), m);

            seq_shift_right<LIMB_N, utils::type_bits<LIMB_T>::value>(a_arr);
            a_arr[LIMB_N - 1] = last;
        }

        if (VtSeqCmp<LIMB_N, LIMB_T>::call(a_arr, a.mod().array()) !=
            CmpEnum::Lt) {
            seq_sub<LIMB_N>(a_arr, a.mod().array());
        }
    }

    GEC_HD static void mul(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
                           const Core &GEC_RSTRCT c) {
        using namespace utils;
        a.set_zero();
        LIMB_T *a_arr = a.array();
        const LIMB_T *b_arr = b.array();
        const LIMB_T *c_arr = c.array();

        for (int i = 0; i < LIMB_N; ++i) {
            LIMB_T m = (a_arr[0] + b_arr[i] * c_arr[0]) * a.mod_p();
            LIMB_T last(0);
            last += seq_add_mul_limb<LIMB_N>(a_arr, c_arr, b_arr[i]);
            last += seq_add_mul_limb<LIMB_N>(a_arr, a.mod().array(), m);

            seq_shift_right<LIMB_N, utils::type_bits<LIMB_T>::value>(a_arr);
            a_arr[LIMB_N - 1] = last;
        }

        if (VtSeqCmp<LIMB_N, LIMB_T>::call(a_arr, a.mod().array()) !=
            CmpEnum::Lt) {
            seq_sub<LIMB_N>(a_arr, a.mod().array());
        }
    }

    GEC_HD GEC_INLINE static void inv(Core &GEC_RSTRCT a,
                                      const Core &GEC_RSTRCT b) {
        a = b;
        inv(a);
    }

    GEC_HD static void inv(Core &GEC_RSTRCT a) {
        using utils::CmpEnum;
        constexpr size_t LimbBit = utils::type_bits<LIMB_T>::value;
        constexpr size_t Bits = LimbBit * LIMB_N;
        constexpr LIMB_T mask = LIMB_T(1) << (LimbBit - 1);

        Core r, s, t;
        LIMB_T *a_arr = a.array();
        LIMB_T *r_arr = r.array();
        LIMB_T *s_arr = s.array();
        LIMB_T *t_arr = t.array();

        r = a;
        s.set_one();
        utils::fill_seq<LIMB_N>(t_arr, a.mod().array());
        a.set_zero();
        int k = 0;
        while (!utils::SeqEqLimb<LIMB_N, LIMB_T>::call(r_arr, 0)) {
            if (!(t_arr[0] & 0x1)) {
                utils::seq_shift_right<LIMB_N, 1>(t_arr);
                utils::seq_shift_left<LIMB_N, 1>(s_arr);
            } else if (!(r_arr[0] & 0x1)) {
                utils::seq_shift_right<LIMB_N, 1>(r_arr);
                utils::seq_shift_left<LIMB_N, 1>(a_arr);
            } else if (utils::VtSeqCmp<LIMB_N, LIMB_T>::call(t_arr, r_arr) ==
                       CmpEnum::Gt) {
                utils::seq_sub<LIMB_N>(t_arr, r_arr);
                utils::seq_shift_right<LIMB_N, 1>(t_arr);
                utils::seq_add<LIMB_N>(a_arr, s_arr);
                utils::seq_shift_left<LIMB_N, 1>(s_arr);
            } else {
                utils::seq_sub<LIMB_N>(r_arr, t_arr);
                utils::seq_shift_right<LIMB_N, 1>(r_arr);
                utils::seq_add<LIMB_N>(s_arr, a_arr);
                utils::seq_shift_left<LIMB_N, 1>(a_arr);
            }
            ++k;
        }
        if (utils::VtSeqCmp<LIMB_N, LIMB_T>::call(a_arr, a.mod().array()) !=
            CmpEnum::Lt) {
            utils::seq_sub<LIMB_N>(a_arr, a.mod().array());
        }
        utils::seq_sub<LIMB_N>(s_arr, a.mod().array(), a_arr);
        if (k < Bits) {
            mul(t, s, a.r_sqr());
            k += Bits;

            mul(s, t, a.r_sqr());

            r.set_pow2(2 * Bits - k);
            mul(a, s, r);
        } else {
            mul(t, s, a.r_sqr());

            r.set_pow2(2 * Bits - k);
            mul(a, t, r);
        }
    }
};

#ifdef GEC_ENABLE_AVX2

/** @brief mixin that enables Montgomery Multiplication with AVX2
 */
template <class Core, typename LIMB_T, size_t LIMB_N>
class GEC_EMPTY_BASES AVX2MontgomeryOps
    : protected CRTP<Core, AVX2MontgomeryOps<Core, LIMB_T, LIMB_N>> {};

/** @brief mixin that enables Montgomery Multiplication with AVX2
 */
template <class Core>
class GEC_EMPTY_BASES AVX2MontgomeryOps<Core, uint32_t, 8>
    : protected CRTP<Core, AVX2MontgomeryOps<Core, uint32_t, 8>> {
    using LIMB_T = uint32_t;
    constexpr static size_t LIMB_N = 8;
    friend CRTP<Core, AVX2MontgomeryOps<Core, LIMB_T, LIMB_N>>;

    GEC_H GEC_INLINE static __m256i add_limbs(__m256i &a, const __m256i &b,
                                              const __m256i &c,
                                              const __m256i &least_mask) {
        __m256i m = _mm256_max_epu32(b, c);
        a = _mm256_add_epi32(b, c);
        return _mm256_andnot_si256(
            _mm256_cmpeq_epi32(_mm256_max_epu32(a, m), a), least_mask);
    }

    GEC_H GEC_INLINE static void mul_limbs(__m256i &l, __m256i &h,
                                           const __m256i &a, const __m256i &b) {
        __m256i a_odd = _mm256_shuffle_epi32(a, 0xf5);
        __m256i b_odd = _mm256_shuffle_epi32(b, 0xf5);
        __m256i p_even = _mm256_mul_epu32(a, b);
        __m256i p_odd = _mm256_mul_epu32(a_odd, b_odd);
        __m256i lo = _mm256_unpacklo_epi32(p_even, p_odd);
        __m256i hi = _mm256_unpackhi_epi32(p_even, p_odd);
        l = _mm256_unpacklo_epi64(lo, hi);
        h = _mm256_unpackhi_epi64(lo, hi);
    }

  public:
    GEC_H GEC_INLINE static void to_montgomery(Core &GEC_RSTRCT a,
                                               const Core &GEC_RSTRCT b) {
        mul(a, b, a.r_sqr());
    }
    GEC_H GEC_INLINE static void from_montgomery(Core &GEC_RSTRCT a,
                                                 const Core &GEC_RSTRCT b) {
        using namespace utils;
        using V = __m256i *;
        using CV = const __m256i *;

        constexpr static uint32_t cir_right[8] = {1, 2, 3, 4, 5, 6, 7, 0};
        constexpr static uint32_t least_mask[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        uint32_t carries[8];

        LIMB_T *a_arr = a.array();
        const LIMB_T *b_arr = b.array();

        __m256i lm = _mm256_loadu_si256(reinterpret_cast<CV>(least_mask));
        __m256i cr = _mm256_loadu_si256(reinterpret_cast<CV>(cir_right));
        __m256i vm = _mm256_loadu_si256(reinterpret_cast<CV>(a.mod().array()));
        __m256i va = _mm256_loadu_si256(reinterpret_cast<CV>(b_arr));
        __m256i mp = _mm256_set1_epi32(static_cast<int>(a.mod_p()));
        __m256i carry = _mm256_setzero_si256();

        for (size_t i = 0; i < LIMB_N; ++i) {
            __m256i vl, vh, new_carry;
            __m256i m = _mm256_mullo_epi32(
                _mm256_broadcastd_epi32(_mm256_castsi256_si128(va)), mp);

            mul_limbs(vl, vh, vm, m);
            new_carry = add_limbs(va, va, vl, lm);
            carry = _mm256_add_epi32(carry, new_carry);

            va = _mm256_permutevar8x32_epi32(va, cr);

            carry = add_limbs(va, va, carry, lm);
            new_carry = add_limbs(va, va, vh, lm);
            carry = _mm256_add_epi32(carry, new_carry);
        }

        _mm256_storeu_si256(reinterpret_cast<V>(carries), carry);
        _mm256_storeu_si256(reinterpret_cast<V>(a_arr), va);

        seq_add<LIMB_N - 1>(a_arr + 1, carries);

        if (carries[LIMB_N - 1] || VtSeqCmp<LIMB_N, LIMB_T>::call(
                                       a_arr, a.mod().array()) != CmpEnum::Lt) {
            seq_sub<LIMB_N>(a_arr, a.mod().array());
        }
    }

    GEC_H static void mul(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
                          const Core &GEC_RSTRCT c) {
        using namespace utils;
        using V = __m256i *;
        using CV = const __m256i *;

        constexpr static uint32_t cir_right[8] = {1, 2, 3, 4, 5, 6, 7, 0};
        constexpr static uint32_t least_mask[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        uint32_t carries[8];

        LIMB_T *a_arr = a.array();
        const LIMB_T *b_arr = b.array();
        const LIMB_T *c_arr = c.array();

        __m256i lm = _mm256_loadu_si256(reinterpret_cast<CV>(least_mask));
        __m256i cr = _mm256_loadu_si256(reinterpret_cast<CV>(cir_right));
        __m256i vm = _mm256_loadu_si256(reinterpret_cast<CV>(a.mod().array()));
        __m256i va = _mm256_setzero_si256();
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<CV>(b_arr));
        __m256i vc = _mm256_loadu_si256(reinterpret_cast<CV>(c_arr));
        __m256i c0 = _mm256_broadcastd_epi32(_mm256_castsi256_si128(vc));
        __m256i mp = _mm256_set1_epi32(static_cast<int>(a.mod_p()));
        __m256i carry = _mm256_setzero_si256();

        for (size_t i = 0; i < LIMB_N; ++i) {
            __m256i vl, vh1, vh2, new_carry;
            __m256i bi = _mm256_broadcastd_epi32(_mm256_castsi256_si128(vb));
            __m256i m = _mm256_mullo_epi32(
                _mm256_add_epi32(
                    _mm256_broadcastd_epi32(_mm256_castsi256_si128(va)),
                    _mm256_mullo_epi32(bi, c0)),
                mp);

            mul_limbs(vl, vh1, bi, vc);
            new_carry = add_limbs(va, va, vl, lm);
            carry = _mm256_add_epi32(carry, new_carry);

            mul_limbs(vl, vh2, m, vm);
            new_carry = add_limbs(va, va, vl, lm);
            carry = _mm256_add_epi32(carry, new_carry);

            va = _mm256_permutevar8x32_epi32(va, cr);
            carry = add_limbs(va, va, carry, lm);

            new_carry = add_limbs(va, va, vh1, lm);
            carry = _mm256_add_epi32(carry, new_carry);
            new_carry = add_limbs(va, va, vh2, lm);
            carry = _mm256_add_epi32(carry, new_carry);

            vb = _mm256_permutevar8x32_epi32(vb, cr);
        }

        _mm256_storeu_si256(reinterpret_cast<V>(carries), carry);
        _mm256_storeu_si256(reinterpret_cast<V>(a_arr), va);

        seq_add<LIMB_N - 1>(a_arr + 1, carries);

        if (carries[LIMB_N - 1] || VtSeqCmp<LIMB_N, LIMB_T>::call(
                                       a_arr, a.mod().array()) != CmpEnum::Lt) {
            seq_sub<LIMB_N>(a_arr, a.mod().array());
        }
    }

    GEC_H GEC_INLINE static void inv(Core &GEC_RSTRCT a,
                                     const Core &GEC_RSTRCT b) {
        a = b;
        inv(a);
    }

    GEC_H static void inv(Core &GEC_RSTRCT a) {
        using utils::CmpEnum;
        constexpr size_t LimbBit = utils::type_bits<LIMB_T>::value;
        constexpr size_t Bits = LimbBit * LIMB_N;
        constexpr LIMB_T mask = LIMB_T(1) << (LimbBit - 1);

        Core r, s, t;
        LIMB_T *a_arr = a.array();
        LIMB_T *r_arr = r.array();
        LIMB_T *s_arr = s.array();
        LIMB_T *t_arr = t.array();

        r = a;
        s.set_one();
        utils::fill_seq<LIMB_N>(t_arr, a.mod().array());
        a.set_zero();
        size_t k = 0;
        bool a_carry = false, s_carry = false;
        while (!utils::SeqEqLimb<LIMB_N, LIMB_T>::call(r_arr, 0)) {
            if (!(t_arr[0] & 0x1)) {
                utils::seq_shift_right<LIMB_N, 1>(t_arr);
                utils::seq_shift_left<LIMB_N, 1>(s_arr);
            } else if (!(r_arr[0] & 0x1)) {
                utils::seq_shift_right<LIMB_N, 1>(r_arr);
                utils::seq_shift_left<LIMB_N, 1>(a_arr);
            } else if (utils::VtSeqCmp<LIMB_N, LIMB_T>::call(t_arr, r_arr) ==
                       CmpEnum::Gt) {
                utils::seq_sub<LIMB_N>(t_arr, r_arr);
                utils::seq_shift_right<LIMB_N, 1>(t_arr);
                bool carry = utils::seq_add<LIMB_N>(a_arr, s_arr);
                a_carry = a_carry || s_carry || carry;
                s_carry = s_carry || bool(mask & s_arr[LIMB_N - 1]);
                utils::seq_shift_left<LIMB_N, 1>(s_arr);
            } else {
                utils::seq_sub<LIMB_N>(r_arr, t_arr);
                utils::seq_shift_right<LIMB_N, 1>(r_arr);
                bool carry = utils::seq_add<LIMB_N>(s_arr, a_arr);
                s_carry = a_carry || s_carry || carry;
                a_carry = a_carry || bool(mask & a_arr[LIMB_N - 1]);
                utils::seq_shift_left<LIMB_N, 1>(a_arr);
            }
            ++k;
        }
        if (a_carry || utils::VtSeqCmp<LIMB_N, LIMB_T>::call(
                           a_arr, a.mod().array()) != CmpEnum::Lt) {
            utils::seq_sub<LIMB_N>(a_arr, a.mod().array());
        }
        utils::seq_sub<LIMB_N>(s_arr, a.mod().array(), a_arr);
        if (k < Bits) {
            mul(t, s, a.r_sqr());
            k += Bits;

            mul(s, t, a.r_sqr());

            r.set_pow2(2 * Bits - k);
            mul(a, s, r);
        } else {
            mul(t, s, a.r_sqr());

            r.set_pow2(2 * Bits - k);
            mul(a, t, r);
        }
    }
};

#endif // GEC_ENABLE_AVX2

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_MONTGOMERY_HPP
