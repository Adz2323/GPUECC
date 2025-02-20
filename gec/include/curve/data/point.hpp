#pragma once
#ifndef GEC_CURVE_DATA_POINT_HPP
#define GEC_CURVE_DATA_POINT_HPP

#include "../mixin/arr_get_comp.hpp"
#include "../mixin/named_comp.hpp"
#include <bigint/data/array.hpp>

namespace gec {

namespace curve {

/** @brief N-dimensional point
 */
template <typename COMP_T, size_t N>
class GEC_EMPTY_BASES Point : public bigint::ArrayLE<COMP_T, N>,
                              public ArrGetCompLE<Point<COMP_T, N>>,
                              public NamedComp<Point<COMP_T, N>> {
    // using Base = bigint::ArrayLE<COMP_T, N>;

  public:
    using CompT = COMP_T;
    constexpr static size_t CompN = N;
    using bigint::ArrayLE<COMP_T, N>::ArrayLE;
    constexpr GEC_HD GEC_INLINE Point() : bigint::ArrayLE<COMP_T, N>() {
        // for some mysterious reason, without defining the custom default
        // constructor, MSVC will complain the default constructor are ambiguous
        // due to multiple candidates from different base classes.
    }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_DATA_POINT_HPP
