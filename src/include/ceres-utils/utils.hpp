//
// Created by csl on 1/27/23.
//

#ifndef CERES_UTILS_UTILS_HPP
#define CERES_UTILS_UTILS_HPP

#include "Eigen/Dense"
#include "deque"
#include "map"

namespace ns_ceres_utils {
    // eigen

    template<typename T>
    using Vector1 = Eigen::Matrix<T, 1, 1, 0>;

    template<typename T>
    using Vector2 = Eigen::Matrix<T, 2, 1>;

    template<typename T>
    using Vector3 = Eigen::Matrix<T, 3, 1>;

    template<typename T>
    using Vector6 = Eigen::Matrix<T, 6, 1>;

    template<typename T>
    using Vector9 = Eigen::Matrix<T, 9, 1>;

    template<typename T>
    using Matrix1 = Eigen::Matrix<T, 1, 1>;

    template<typename T>
    using Matrix2 = Eigen::Matrix<T, 2, 2>;

    template<typename T>
    using Matrix3 = Eigen::Matrix<T, 3, 3>;

    template<typename T>
    using aligned_vector = std::vector<T, Eigen::aligned_allocator<T>>;

    template<typename T>
    using aligned_deque = std::deque<T, Eigen::aligned_allocator<T>>;

    template<typename K, typename V>
    using aligned_map = std::map<K, V, std::less<K>,
            Eigen::aligned_allocator<std::pair<K const, V>>>;

    template<typename K, typename V>
    using aligned_unordered_map = std::unordered_map<K, V, std::hash<K>, std::equal_to<K>,
            Eigen::aligned_allocator<std::pair<K const, V>>>;

    /**
     * @tparam EigenVectorType
     * @param eigenVec the eigen vector object
     * @return the stl vector object
     */
    template<typename EigenVectorType>
    inline auto EigenVecToVector(const EigenVectorType &eigenVec) {
        std::vector<typename EigenVectorType::Scalar> vec(eigenVec.rows());
        for (int i = 0; i < vec.size(); ++i) {
            vec.at(i) = eigenVec(i);
        }
        return vec;
    }

    /**
     * @tparam EigenMatrixType
     * @param mat the eigen matrix
     * @return the stl 2d vector object
     */
    template<typename EigenMatrixType>
    inline auto EigenMatToVector(const EigenMatrixType &mat) {
        std::vector<std::vector<typename EigenMatrixType::Scalar>> vec(
                mat.rows(), std::vector<typename EigenMatrixType::Scalar>(mat.cols())
        );
        for (int i = 0; i < mat.rows(); ++i) {
            for (int j = 0; j < mat.cols(); ++j) {
                vec.at(i).at(j) = mat(i, j);
            }
        }
        return vec;
    }

    /**
     * @tparam ScaleType
     * @tparam M the rows
     * @tparam N the cols
     * @param mat the eigen matrix
     * @return the reduced row echelon form
     */
    template<typename ScaleType, int M, int N>
    Eigen::Matrix<ScaleType, M, N> ReducedRowEchelonForm(const Eigen::Matrix<ScaleType, M, N> &mat) {
        Eigen::Matrix<ScaleType, M, N> rMat = mat;

        std::vector<std::pair<int, int>> indexVec;
        int r = 0, c = 0;
        for (; r < rMat.rows() && c < rMat.cols(); ++c) {
            if (std::abs(rMat(r, c)) < 1E-8) {
                int i = r + 1;
                for (; i < rMat.rows(); ++i) {
                    if (std::abs(rMat(i, c)) > 1E-8) {
                        auto row = rMat.row(r);
                        rMat.row(r) = rMat.row(i);
                        rMat.row(i) = row;
                        break;
                    }
                }
                if (i == rMat.rows()) {
                    continue;
                }
            }
            indexVec.emplace_back(r, c);
            for (int i = r + 1; i < rMat.rows(); ++i) {
                rMat.row(i) -= rMat(i, c) * rMat.row(r) / rMat(r, c);
            }
            r += 1;
        }
        for (auto iter = indexVec.rbegin(); iter != indexVec.rend(); ++iter) {
            auto [row, col] = *iter;
            for (int i = row - 1; i >= 0; --i) {
                rMat.row(i) -= rMat(i, col) / rMat(row, col) * rMat.row(row);
            }
        }
        return rMat;
    }

}
#endif //CERES_UTILS_UTILS_HPP
