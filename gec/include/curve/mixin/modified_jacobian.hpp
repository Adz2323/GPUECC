#pragma once
#ifndef GEC_CURVE_MIXIN_MODIFIED_JACOBAIN_HPP
#define GEC_CURVE_MIXIN_MODIFIED_JACOBAIN_HPP

#include <utils/crtp.hpp>

namespace gec {

namespace curve {

/** @brief mixin that enables ...
 */
template <typename Core, typename FIELD_T, const FIELD_T &A, const FIELD_T &B>
class GEC_EMPTY_BASES JacobianM
    : protected CRTP<Core, JacobianM<Core, FIELD_T, A, B>> {
    friend CRTP<Core, JacobianM<Core, FIELD_T, A, B>>;

  public:
    GEC_HD GEC_INLINE bool is_inf(Core &GEC_RSTRCT a) {
        // TODO
    }
    GEC_HD GEC_INLINE void set_inf(Core &GEC_RSTRCT a) {
        // TODO
    }

    GEC_HD GEC_INLINE static bool eq(const Core &GEC_RSTRCT a,
                                     const Core &GEC_RSTRCT b) {
        // TODO
    }

    GEC_HD GEC_INLINE static bool on_curve(const Core &GEC_RSTRCT a) {
        // TODO
    }

    GEC_HD static void add_distinct(Core &GEC_RSTRCT a,
                                    const Core &GEC_RSTRCT b,
                                    const Core &GEC_RSTRCT c) {
        //  TODO
    }

    GEC_HD static void add_self(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b) {
        //  TODO
    }

    GEC_HD static void add(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
                           const Core &GEC_RSTRCT c) {
        //  TODO
    }

    GEC_HD GEC_INLINE static void neg(Core &GEC_RSTRCT a,
                                      const Core &GEC_RSTRCT b) {
        // TODO
    }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_MIXIN_MODIFIED_JACOBAIN_HPP
