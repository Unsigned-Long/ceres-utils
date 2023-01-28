//
// Created by csl on 12/14/22.
//

#include "ceres-utils/equations.h"
#include "ceres/ceres.h"
#include "iomanip"

namespace ns_ceres_utils {

    //-------------
    // CeresFactor
    //-------------
    CeresFactor::CeresFactor(ceres::CostFunction *costFunction, const std::vector<double *> &paramBlocks,
                             std::size_t costFunctorHashCode)
            : _costFunction(costFunction), _paramBlocks(paramBlocks), _costFunctorHashCode(costFunctorHashCode) {}

    void CeresFactor::Evaluate(const std::map<const double *, std::string> &targetParams) {
        const auto &parameterBlockSizes = _costFunction->parameter_block_sizes();
        const int numParameterBlocks = static_cast<int>(parameterBlockSizes.size());
        const int numResiduals = _costFunction->num_residuals();

        std::vector<double> rawResiduals(numResiduals);
        std::vector<double *> rawJacobians(numParameterBlocks);

        // allocate
        for (int i = 0; i < numParameterBlocks; ++i) {
            rawJacobians.at(i) = new double[numResiduals * parameterBlockSizes.at(i)];
        }

        // evaluate: get the raw residuals and jacobians
        _costFunction->Evaluate(_paramBlocks.data(), rawResiduals.data(), rawJacobians.data());

        // save rawResiduals
        residuals.resize(numResiduals);
        for (int i = 0; i < numResiduals; ++i) {
            residuals(i) = rawResiduals.at(i);
        }

        // save rawJacobians
        jacobians.clear();
        for (int i = 0; i < numParameterBlocks; ++i) {
            // the param block that we don't interest
            if (targetParams.find(_paramBlocks.at(i)) == targetParams.end()) { continue; }

            // get the jacobian matrix [using all cost functions to evaluate each param block]
            Eigen::MatrixXd jMat(numResiduals, parameterBlockSizes.at(i));
            for (int r = 0; r < numResiduals; ++r) {
                for (int c = 0; c < parameterBlockSizes.at(i); ++c) {
                    jMat(r, c) = rawJacobians.at(i)[r * parameterBlockSizes.at(i) + c];
                }
            }
            // save jacobian matrix with param block address
            jacobians.insert(std::make_pair(_paramBlocks.at(i), jMat));
        }

        // deallocate
        for (int i = 0; i < numParameterBlocks; ++i) {
            delete[] rawJacobians.at(i);
        }
    }

    const std::map<const double *, Eigen::MatrixXd> &CeresFactor::GetJacobians() const {
        return jacobians;
    }

    const Eigen::VectorXd &CeresFactor::GetResiduals() const {
        return residuals;
    }

    size_t CeresFactor::GetCostFunctorHashCode() const {
        return _costFunctorHashCode;
    }

    //----------
    // Equation
    //----------
    Equation::Equation(Eigen::MatrixXd hMat, Eigen::VectorXd bVec,
                       const std::vector<std::pair<std::string, std::size_t>> &paramDesc,
                       const std::map<std::size_t, aligned_vector<Eigen::VectorXd>> &residualsMap)
            : _hMat(std::move(hMat)), _bVec(std::move(bVec)), _paramDesc(paramDesc), _residualsMap(residualsMap) {
        Eigen::MatrixXd hMatBar(_hMat.rows(), _hMat.cols() + 1);
        hMatBar.topLeftCorner(_hMat.rows(), _hMat.cols()) = _hMat;
        hMatBar.topRightCorner(_hMat.rows(), 1) = _bVec;
        hMatBar = ReducedRowEchelonForm(hMatBar);
        _hMatEchelonForm = hMatBar.topLeftCorner(_hMat.rows(), _hMat.cols());
        _bVecEchelonForm = hMatBar.topRightCorner(_hMat.rows(), 1);
    }

    const Equation &Equation::SaveEquationToDisk(const std::string &filepath, bool echelonForm) const {
        if (echelonForm) {
            return SaveEquationToDisk(filepath, _hMatEchelonForm, _bVecEchelonForm);
        } else {
            return SaveEquationToDisk(filepath, _hMat, _bVec);
        }
    }

    const Equation &Equation::SaveEquationToDisk(const std::string &filepath,
                                                 const Eigen::MatrixXd &hMat, const Eigen::VectorXd &bVec) const {
        std::ofstream file(filepath, std::ios::out);
        cereal::JSONOutputArchive ar(file);
        ar.setNextName("param_blocks");
        ar.startNode();
        int count = 0;
        for (const auto &item: _paramDesc) {
            ar.setNextName(("blocks_" + std::to_string(count++)).c_str());
            ar.startNode();
            ar(cereal::make_nvp("name", item.first), cereal::make_nvp("dime", item.second));
            ar.finishNode();
        }
        ar.finishNode();
        ar(
                cereal::make_nvp("h_matrix", EigenMatToVector(hMat)),
                cereal::make_nvp("b_vector", EigenVecToVector(bVec))
        );
        return *this;
    }

    Eigen::VectorXd Equation::ZeroSpace() const {
        Eigen::FullPivLU<Eigen::MatrixXd> lu(_hMat);
        return lu.kernel();
    }

    const std::map<std::size_t, aligned_vector<Eigen::VectorXd>> &Equation::GetResidualsMap() const {
        return _residualsMap;
    }

    //-----------
    // Evaluator
    //-----------
    Evaluator::Evaluator(const std::vector<CeresFactor> &factors) : _factors(factors) {}

    Equation Evaluator::Evaluate(const std::map<const double *, std::string> &targetParamsInfoMap) {
        std::size_t rows = 0, cols = 0;
        // param address, param dime, param start column
        std::map<const double *, std::pair<std::size_t, std::size_t>> paramDimeInfo;
        for (auto &factor: _factors) {
            factor.Evaluate(targetParamsInfoMap);
            // record size info
            rows += factor.GetResiduals().size();
            for (const auto &[pAddress, pJacobian]: factor.GetJacobians()) {
                // using map assign operator, if the 'pAddress' not exists, a new element will be created,
                // otherwise, the value of the key will be updated
                paramDimeInfo[pAddress] = {pJacobian.cols(), 0};
            }
        }
        // param string description, the param block size[dime]
        std::vector<std::pair<std::string, std::size_t>> paramDesc;
        for (auto &[pAddress, column]: paramDimeInfo) {
            column.second = cols;
            cols += column.first;
            paramDesc.emplace_back(targetParamsInfoMap.at(pAddress), column.first);
        }

        Eigen::MatrixXd jMat(rows, cols), hMat(cols, cols);
        Eigen::VectorXd rVec(rows), bVec(cols);
        jMat.setZero(), rVec.setZero();

        // factor type id, residuals
        std::map<std::size_t, aligned_vector<Eigen::VectorXd>> residualsMap;

        // organize the jMat and rVec
        int cr = 0;
        for (const auto &factor: _factors) {
            const auto &jacobians = factor.GetJacobians();
            const auto &residuals = factor.GetResiduals();

            // residuals vector
            rVec.block(cr, 0, residuals.rows(), 1) = residuals;

            // jacobians matrix
            for (const auto &[pAddress, pJacobian]: jacobians) {
                jMat.block(
                        cr, static_cast<int>(paramDimeInfo.find(pAddress)->second.second),
                        pJacobian.rows(), pJacobian.cols()
                ) = pJacobian;
            }

            // draw error
            residualsMap[factor.GetCostFunctorHashCode()].push_back(residuals);

            cr += static_cast<int>(residuals.rows());
        }
        hMat = jMat.transpose() * jMat;
        bVec = -jMat.transpose() * rVec;

        return {hMat, bVec, paramDesc, residualsMap};
    }

    Equation
    Evaluator::Evaluate(
            const std::initializer_list<std::map<const double *, std::string>> &targetParamsInfoMaps) {

        std::map<const double *, std::string> targetParamsInfoMap;
        for (const auto &item: targetParamsInfoMaps) {
            for (const auto &p: item) {
                targetParamsInfoMap.insert(p);
            }
        }

        return Evaluate(targetParamsInfoMap);
    }
}