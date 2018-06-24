#pragma once

#include "xtensor/xarray.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xstrided_view.hpp"


// overload ostream operator for xindex
inline std::ostream & operator << (std::ostream & os, const xt::xindex & coord) {
    os << "xindex(";
    for(const auto & cc: coord) {
        os << " " << cc;
    }
    os << " )";
    return os;
}


namespace xt_util {


    // fill a xt::slice_vector for a roi from
    // the offset and shape of the roi
    template<class COORD>
    inline void slice_from_roi(xt::slice_vector & slice,
                               const COORD & offset,
                               const COORD & shape) {
        for(int d = 0; d < offset.size(); ++d) {
            slice.push_back(xt::range(offset[d], offset[d] + shape[d]));
        }
    }


    // copy buffer to view into xcontainer
    template<typename T, typename VIEW, typename COORD>
    inline void copy_buffer_to_view_impl(const std::vector<T> & buffer,
                                         xt::xexpression<VIEW> & view_expression,
                                         const COORD & strides) {
        // get the view into the out array and the number of dimension
        auto & view = view_expression.derived_cast();
        const std::size_t dim = view.dimension();
        // buffer size and view shape
        const std::size_t buffer_size = buffer.size();
        const auto & view_shape = view.shape();
        // initialize the (1d) offset into the buffer and view
        std::size_t buffer_offset = 0;
        std::size_t view_offset = 0;
        // vector to keep track along the position along each dimension
        std::vector<size_t> positions(dim);
        // THIS ASSUMES C-ORDER
        // -> memory is consecutive along the last axis
        const std::size_t mem_len = view_shape[dim - 1];

        // we copy data to consecutive pieces of memory in the view
        // until we have exhausted the buffer

        // we start the outer loop at the second from last dimension
        // (last dimension is the fastest moving and consecutive in memory)
        for(int d = dim - 2; d >= 0;) {
            // copy the piece of buffer that is consectuve to our view
            std::copy(buffer.begin() + buffer_offset,
                      buffer.begin() + buffer_offset + mem_len,
                      &view(0) + view_offset);

            // increase the buffer offset by what we have just written to the view
            buffer_offset += mem_len;
            // increase the view offsets by the strides along the second from last dimension
            view_offset += strides[dim - 2];

            // check if we need to decrease the dimension
            for(d = dim - 2; d >= 0; --d) {
                // increase the position in the current dimension
                positions[d] += 1;

                // if we are smaller than the shape in this dimension, stay in this dimension
                // (i.e. break and go back to the copy loop)
                if(positions[d] < view_shape[d]) {
                    break;
                // otherwise, decrease the dimension
                } else {

                    // reset the position in this dimension
                    positions[d] = 0;

                    // we don't need to increase the view offset if we are at
                    // the end of the next lower dim !
                    if(d > 0) {
                        if(positions[d - 1] + 1 == view_shape[d - 1]) {
                            continue;
                        }
                    }

                    // increase the view_offset to jump to the next point in memory
                    // for this, we increase by the stride of the next lower dimension
                    // but need to correct to jump back to the front of the view
                    // in that dimension
                    if(d > 0) {

                        // the correction to jump back to the front of the view
                        // in the next dim
                        std::size_t correction = 0;
                        for(int dd = dim - 2; dd >= d; --dd) {
                            correction += strides[dd] * (view_shape[dd] - 1);
                        }
                        // further correction because we incremented
                        // one time to much
                        correction += strides[dim - 2];

                        // increase the view offset
                        view_offset += (strides[d - 1] - correction);
                    }
                }
            }
        }
    }


    // TODO implement for F order
    template<typename T, typename VIEW, typename COORD>
    inline void copy_buffer_to_view(const std::vector<T> & buffer,
                                    xt::xexpression<VIEW> & view_expression,
                                    const COORD & strides) {
        auto & view = view_expression.derived_cast();
        // ND impl doesn't work for 1D
        if(view.dimension() == 1) {
            // std::copy(buffer.begin(), buffer.end(), view.begin());
            const auto buffer_view = xt::adapt(buffer, view.shape());
            view = buffer_view;
        } else {
            copy_buffer_to_view_impl(buffer, view_expression, strides);
        }
    }


    template<typename T, typename VIEW, typename COORD>
    inline void copy_view_to_buffer_impl(const xt::xexpression<VIEW> & view_expression,
                                         std::vector<T> & buffer,
                                         const COORD & strides) {
        // get the view into the out array and the number of dimension
        const auto & view = view_expression.derived_cast();
        const std::size_t dim = view.dimension();
        // buffer size and view shape
        const std::size_t buffer_size = buffer.size();
        const auto & view_shape = view.shape();
        // initialize the (1d) offset into the buffer and view
        std::size_t buffer_offset = 0;
        std::size_t view_offset = 0;
        // vector to keep track along the position along each dimension
        std::vector<size_t> positions(dim);
        // THIS ASSUMES C-ORDER
        // -> memory is consecutive along the last axis
        const std::size_t mem_len = view_shape[dim - 1];

        // we copy data that is consecutive in the view to the buffer
        // until we have exhausted the iew

        // we start the outer loop at the second from last dimension
        // (last dimension is the fastest moving and consecutive in memory)
        for(int d = dim - 2; d >= 0;) {
            // copy the piece of buffer that is consectuve to our view
            std::copy(&view(0) + view_offset,
                      &view(0) + view_offset + mem_len,
                      buffer.begin() + buffer_offset);

            // increase the buffer offset by what we have just written to the view
            buffer_offset += mem_len;
            // increase the view offsets by the strides along the second from last dimension
            view_offset += strides[dim - 2];

            // check if we need to decrease the dimension
            for(d = dim - 2; d >= 0; --d) {
                // increase the position in the current dimension
                positions[d] += 1;

                // if we are smaller than the shape in this dimension, stay in this dimension
                // (i.e. break and go back to the copy loop)
                if(positions[d] < view_shape[d]) {
                    break;
                // otherwise, decrease the dimension
                } else {

                    // reset the position in this dimension
                    positions[d] = 0;

                    // we don't need to increase the view offset if we are at
                    // the end of the next lower dim !
                    if(d > 0) {
                        if(positions[d - 1] + 1 == view_shape[d - 1]) {
                            continue;
                        }
                    }

                    // increase the view_offset to jump to the next point in memory
                    // for this, we increase by the stride of the next lower dimension
                    // but need to correct to jump back to the front of the view
                    // in that dimension
                    if(d > 0) {

                        // the correction to jump back to the front of the view
                        // in the next dim
                        std::size_t correction = 0;
                        for(int dd = dim - 2; dd >= d; --dd) {
                            correction += strides[dd] * (view_shape[dd] - 1);
                        }
                        // further correction because we incremented
                        // one time to much
                        correction += strides[dim - 2];

                        // increase the view offset
                        view_offset += (strides[d - 1] - correction);
                    }
                }
            }
        }
    }


    // TODO implement for F order
    template<typename T, typename VIEW, typename COORD>
    inline void copy_view_to_buffer(const xt::xexpression<VIEW> & view_expression,
                                    std::vector<T> & buffer,
                                    const COORD & strides) {
        const auto & view = view_expression.derived_cast();
        // can't use the ND implementation in 1d, hence we resort to xtensor adapt
        // which should be fine in 1D
        if(view.dimension() == 1) {
            // std::copy(view.begin(), view.end(), buffer.begin());
            auto buffer_view = xt::adapt(buffer, view.shape());
            buffer_view = view;
        } else {
            copy_view_to_buffer_impl(view_expression, buffer, strides);
        }
    }


    //
    // for each coordinate:
    // iterate over single shape
    //

    template<typename COORD, typename F>
    inline void for_each_coordinate_c(const COORD & shape, F && f) {
        const int dim = shape.size();
        xt::xindex coord(dim);
        std::fill(coord.begin(), coord.end(), 0);

        // C-Order: last dimension is the fastest moving one
        for(int d = dim - 1; d >= 0;) {
            f(coord);
            for(d = dim - 1; d >= 0; --d) {
                ++coord[d];
                if(coord[d] < shape[d]) {
                    break;
                } else {
                    coord[d] = 0;
                }
            }
        }
    }


    template<typename COORD, typename F>
    inline void for_each_coordinate_f(const COORD & shape, F && f) {
        const int dim = shape.size();
        xt::xindex coord(dim);
        std::fill(coord.begin(), coord.end(), 0);

        // F-Order: last dimension is the fastest moving one
        for(int d = 0; d < dim;) {
            f(coord);
            for(d = 0; d < dim; ++d) {
                ++coord[d];
                if(coord[d] < shape[d]) {
                    break;
                } else {
                    coord[d] = 0;
                }
            }
        }
    }


    template<typename COORD, typename F>
    inline void for_each_coordinate(const COORD & shape, F && f, const bool c_order=true) {
        if(c_order) {
            for_each_coordinate_c(shape, f);
        } else {
            for_each_coordinate_f(shape, f);
        }
    }


    //
    // for each coordinate:
    // iterate between begin and end
    //


    template<typename COORD, typename F>
    inline void for_each_coordinate_c(const COORD & begin, const COORD & end, F && f) {
        const int dim = begin.size();
        xt::xindex coord(dim);
        std::copy(begin.begin(), begin.end(), coord.begin());

        // C-Order: last dimension is the fastest moving one
        for(int d = dim - 1; d >= 0;) {
            f(coord);
            for(d = dim - 1; d >= 0; --d) {
                ++coord[d];
                if(coord[d] < end[d]) {
                    break;
                } else {
                    coord[d] = 0;
                }
            }
        }
    }


    template<typename COORD, typename F>
    inline void for_each_coordinate_f(const COORD & begin, const COORD & end, F && f) {
        const int dim = begin.size();
        xt::xindex coord(dim);
        std::copy(begin.begin(), begin.end(), coord.begin());

        // F-Order: last dimension is the fastest moving one
        for(int d = 0; d < dim;) {
            f(coord);
            for(d = 0; d < dim; ++d) {
                ++coord[d];
                if(coord[d] < end[d]) {
                    break;
                } else {
                    coord[d] = 0;
                }
            }
        }
    }


    template<typename COORD, typename F>
    inline void for_each_coordinate(const COORD & begin, const COORD & end, F && f, const bool c_order=true) {
        if(c_order) {
            for_each_coordinate_c(begin, end, f);
        } else {
            for_each_coordinate_f(begin, end, f);
        }
    }

}
