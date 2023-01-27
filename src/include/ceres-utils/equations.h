//
// Created by csl on 12/14/22.
//

#ifndef LIC_CALIB_SYSTEM_OBSERVABILITY_H
#define LIC_CALIB_SYSTEM_OBSERVABILITY_H

#include "ceres-utils/utils.hpp"
#include "ceres/ceres.h"
#include "cereal/types/vector.hpp"
#include "cereal/archives/json.hpp"
#include "cereal/cereal.hpp"
#include "fstream"

namespace ns_ceres_utils {
    struct CeresFactor {
    protected:
        // the ceres cost function
        ceres::CostFunction *_costFunction;
        // the relative param blocks
        std::vector<double *> _paramBlocks;
        // the 'CostFunctor' hash code
        std::size_t _costFunctorHashCode;

        // param address, jacobian matrix
        std::map<const double *, Eigen::MatrixXd> jacobians;
        // the residuals vector
        Eigen::VectorXd residuals;

    public:
        // constructor
        CeresFactor(ceres::CostFunction *costFunction, const std::vector<double *> &paramBlocks,
                    std::size_t costFunctorHashCode);

        // evaluate function
        void Evaluate(const std::map<const double *, std::string> &targetParams);

        // getters
        [[nodiscard]] const std::map<const double *, Eigen::MatrixXd> &GetJacobians() const;

        [[nodiscard]] const Eigen::VectorXd &GetResiduals() const;

        [[nodiscard]] size_t GetCostFunctorHashCode() const;
    };

    struct Equation {
    protected:
        Eigen::MatrixXd _hMat;
        Eigen::VectorXd _bVec;

        // param name, param, dime
        std::vector<std::pair<std::string, std::size_t>> _paramDesc;

        Eigen::MatrixXd _hMatEchelonForm;
        Eigen::VectorXd _bVecEchelonForm;

        // factor type id, residuals
        std::map<std::size_t, aligned_vector<Eigen::VectorXd>> _residualsMap;

    public:
        Equation(Eigen::MatrixXd hMat, Eigen::VectorXd bVec,
                 const std::vector<std::pair<std::string, std::size_t>> &paramDesc,
                 const std::map<std::size_t, aligned_vector<Eigen::VectorXd>> &residualsMap);

        [[nodiscard]] const Equation &
        SaveEquationToDisk(const std::string &filepath, int precision = 3, bool echelonForm = false) const;

        [[nodiscard]] Eigen::VectorXd ZeroSpace() const;

        template<class CostFunctor>
        [[nodiscard]]const Equation &SaveResiduals(const std::string &filepath, int precision = 3) const {
            if (auto errors = GetResiduals<CostFunctor>();errors) {
                aligned_vector<Eigen::VectorXd> residuals = *errors;
                std::ofstream file(filepath, std::ios::out);
                cereal::JSONOutputArchive ar(file);
                ar(cereal::make_size_tag(residuals.size()));
                for (auto &&v: residuals) { ar(EigenVecToVector(v)); }
            }
            return *this;
        }

        // factor type id, residuals
        [[nodiscard]] const std::map<std::size_t, aligned_vector<Eigen::VectorXd>> &GetResidualsMap() const;

    protected:
        [[nodiscard]] const Equation &
        SaveEquationToDisk(const std::string &filepath, int precision,
                           const Eigen::MatrixXd &hMat, const Eigen::VectorXd &bVec) const;

        template<class CostFunctor>
        [[nodiscard]] std::optional<aligned_vector<Eigen::VectorXd>> GetResiduals() const {
            auto hashCode = typeid(CostFunctor).hash_code();
            if (auto iter = _residualsMap.find(hashCode);iter == _residualsMap.cend()) {
                return {};
            } else {
                return iter->second;
            }
        }
    };

    class Evaluator {
    protected:
        std::vector<CeresFactor> _factors;

    public:
        explicit Evaluator(const std::vector<CeresFactor> &factors = {});

        template<typename CostFunctor, int Stride = 4>
        Evaluator &AddCostFunction(ceres::DynamicAutoDiffCostFunction<CostFunctor, Stride> *costFunc,
                                   const std::vector<double *> &paramBlocks) {
            _factors.emplace_back(costFunc, paramBlocks, typeid(CostFunctor).hash_code());
            return *this;
        }

        Equation Evaluate(const std::map<const double *, std::string> &targetParamsInfoMap);

        Equation Evaluate(const std::initializer_list<std::map<const double *, std::string>> &targetParamsInfoMaps);

    };
}


#endif //LIC_CALIB_SYSTEM_OBSERVABILITY_H
