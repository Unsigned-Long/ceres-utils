//
// Created by csl on 1/27/23.
//

#include <utility>
#include "artwork/logger/logger.h"
#include "ceres-utils/equations.h"
#include "random"
#include "chrono"
#include "fstream"


struct Utils {
    template<typename T>
    static ns_ceres_utils::Vector3<T> RecoveryConstNormVec(Eigen::Map<ns_ceres_utils::Vector2<T> const> &dir, T norm) {
        T cr = ceres::cos(dir[0]), sr = ceres::sin(dir[0]);
        T cp = ceres::cos(dir[1]), sp = ceres::sin(dir[1]);
        return Eigen::Matrix<T, 3, 1>(-sp * cr * T(norm), sr * T(norm), -cr * cp * T(norm));
    }

    template<typename T>
    static ns_ceres_utils::Vector3<T> RecoveryConstNormVec(const ns_ceres_utils::Vector2<T> &dir, T norm) {
        T cr = ceres::cos(dir[0]), sr = ceres::sin(dir[0]);
        T cp = ceres::cos(dir[1]), sp = ceres::sin(dir[1]);
        return Eigen::Matrix<T, 3, 1>(-sp * cr * T(norm), sr * T(norm), -cr * cp * T(norm));
    }

    template<typename T>
    static ns_ceres_utils::Vector3<T> ComputeDirAndNorm(const ns_ceres_utils::Vector3<T> &dir) {
        ns_ceres_utils::Vector3<T> unitDir = (dir / dir.norm()).eval();
        double cr = std::sqrt(unitDir[0] * unitDir[0] + unitDir[2] * unitDir[2]);
        double v1 = std::acos(cr);
        double v2 = std::acos(-unitDir[2] / cr);
        return Eigen::Matrix<T, 3, 1>(v1, v2, dir.norm());
    }

    static Eigen::MatrixXd TangentBasis(Eigen::Vector3d &vec) {
        Eigen::Vector3d b, c;
        Eigen::Vector3d a = vec.normalized();
        Eigen::Vector3d tmp(0, 0, 1);
        if (a == tmp)
            tmp << 1, 0, 0;
        b = (tmp - a * (a.transpose() * tmp)).normalized();
        c = a.cross(b);
        Eigen::MatrixXd bc(3, 2);
        bc.block<3, 1>(0, 0) = b;
        bc.block<3, 1>(0, 1) = c;
        return bc;
    }

    static ns_ceres_utils::aligned_vector<Eigen::Vector2d>
    GeneratePoints(const double a, const double b, const double c, const double sigma) {
        // y = ax^2 + bx + c
        ns_ceres_utils::aligned_vector<Eigen::Vector2d> points;
        double mid = -b / (2.0 * a);
        std::default_random_engine engine(std::chrono::steady_clock::now().time_since_epoch().count());
        std::uniform_real_distribution u(mid - 10.0, mid + 10.0);
        std::normal_distribution n(0.0, sigma);

        for (int i = 0; i < 100; ++i) {
            double x = u(engine), y = a * x * x + b * x + c + n(engine);
            points.emplace_back(x, y);
        }
        return points;
    }

    static void SavePoints(const std::string &filename, const ns_ceres_utils::aligned_vector<Eigen::Vector2d> &points) {
        std::ofstream file(filename);
        for (const auto &item: points) {
            file << item(0) << ',' << item(1) << std::endl;
        }
        file.close();
    }
};

namespace ns_factors {
    struct FittingCostFunctor {
    protected:
        double x, y;

    public:
        explicit FittingCostFunctor(Eigen::Vector2d pts) : x(pts(0)), y(pts(1)) {}

        static auto
        Create(const Eigen::Vector2d &pts) {
            return new ceres::DynamicAutoDiffCostFunction<FittingCostFunctor>(new FittingCostFunctor(pts));
        }

        /**
         * param blocks:
         * [[a, b], c]
         */
        template<class T>
        bool operator()(T const *const *sKnots, T *sResiduals) const {
            T a = sKnots[0][0], b = sKnots[0][1], c = sKnots[1][0];
            sResiduals[0] = a * x * x + b * x + c - y;
            return true;
        }
    };

    struct S2ClosestPoint {
    protected:
        const Eigen::Vector3d _target;

    public:
        explicit S2ClosestPoint(Eigen::Vector3d target) : _target(std::move(target)) {}

        static auto
        Create(const Eigen::Vector3d &target) {
            return new ceres::DynamicAutoDiffCostFunction<S2ClosestPoint>(new S2ClosestPoint(target));
        }

        /**
         * param blocks:
         * [ p ]
         */
        template<class T>
        bool operator()(T const *const *sKnots, T *sResiduals) const {
            Eigen::Map<const ns_ceres_utils::Vector3<T>> source(sKnots[0]);
            Eigen::Map<ns_ceres_utils::Vector3<T>> residuals(sResiduals);
            residuals = (source - _target);
            return true;
        }
    };

    struct ConstNormFactor {
    protected:
        const Eigen::Vector3d _source;

    public:
        explicit ConstNormFactor(Eigen::Vector3d source) : _source(std::move(source)) {}

        static auto
        Create(const Eigen::Vector3d &source) {
            return new ceres::DynamicAutoDiffCostFunction<ConstNormFactor>(new ConstNormFactor(source));
        }

        /**
         * param blocks:
         * [ p ]
         */
        template<class T>
        bool operator()(T const *const *sKnots, T *sResiduals) const {
            Eigen::Map<const ns_ceres_utils::Vector3<T>> source(sKnots[0]);
            Eigen::Map<ns_ceres_utils::Vector1<T>> residuals(sResiduals);
            residuals = 1E1 * ns_ceres_utils::Vector1<T>(source.norm() - _source.norm());
            return true;
        }
    };

    struct DirFactor {
    protected:
        const Eigen::Vector3d _target;
        const double _norm;

    public:
        explicit DirFactor(Eigen::Vector3d target, double norm) : _target(std::move(target)), _norm(norm) {}

        static auto
        Create(const Eigen::Vector3d &target, double norm) {
            return new ceres::DynamicAutoDiffCostFunction<DirFactor>(new DirFactor(target, norm));
        }

        /**
         * param blocks:
         * [ p ]
         */
        template<class T>
        bool operator()(T const *const *sKnots, T *sResiduals) const {
            Eigen::Map<ns_ceres_utils::Vector2<T> const> source(sKnots[0]);
            Eigen::Map<ns_ceres_utils::Vector3<T>> residuals(sResiduals);
            residuals = (Utils::RecoveryConstNormVec(source, T(_norm)) - _target);
            return true;
        }
    };

    struct CeresDebugCallBack : public ceres::IterationCallback {

        const Eigen::Vector3d &_source;
        std::ofstream _file;

        explicit CeresDebugCallBack(const Eigen::Vector3d &source, const std::string &filename)
                : _source(source), _file(filename, std::ios::out) {}

        static auto Create(const Eigen::Vector3d &source, const std::string &filename) {
            return new CeresDebugCallBack(source, filename);
        }

        ~CeresDebugCallBack() override { _file.close(); }

        ceres::CallbackReturnType operator()(const ceres::IterationSummary &summary) override {
            // for drawing
            _file << _source(0) << ',' << _source(1) << ',' << _source(2) << std::endl;
            return ceres::SOLVER_CONTINUE;
        }

    };

    struct CeresDebugCallBack2 : public ceres::IterationCallback {

        const Eigen::Vector2d &_dir;
        const double _norm;
        std::ofstream _file;

        explicit CeresDebugCallBack2(const Eigen::Vector2d &dir, double norm, const std::string &filename)
                : _dir(dir), _norm(norm), _file(filename, std::ios::out) {}

        static auto Create(const Eigen::Vector2d &dir, double norm, const std::string &filename) {
            return new CeresDebugCallBack2(dir, norm, filename);
        }

        ~CeresDebugCallBack2() override { _file.close(); }

        ceres::CallbackReturnType operator()(const ceres::IterationSummary &summary) override {
            // for drawing
            Eigen::Vector3d p = Utils::RecoveryConstNormVec(_dir, _norm);
            _file << p(0) << ',' << p(1) << ',' << p(2) << std::endl;
            return ceres::SOLVER_CONTINUE;
        }

    };
}

void TestEquation() {
    // fitting
    double ab[2] = {0.5, 10}, c = -5;
    auto points = Utils::GeneratePoints(ab[0], ab[1], c, 3.0);
    Utils::SavePoints("/home/csl/CppWorks/artwork/ceres-utils/src/output/points.txt", points);

    //-------------
    // use library
    ns_ceres_utils::Evaluator evaluator;
    //-------------

    ceres::Problem problem;
    for (const auto &pt: points) {
        auto costFunc = ns_factors::FittingCostFunctor::Create(pt);
        // a, b
        costFunc->AddParameterBlock(2);
        // c
        costFunc->AddParameterBlock(1);
        costFunc->SetNumResiduals(1);

        problem.AddResidualBlock(costFunc, nullptr, {ab, &c});
        //-------------
        // use library
        evaluator.AddCostFunction<ns_factors::FittingCostFunctor>(costFunc, {ab, &c});
        //-------------
    }
    //-------------
    // use library
    auto eqBefore = evaluator.Evaluate(
            {{ab, "ab"},
             {&c, "c"}}
    ).SaveEquationToDisk(
            "/home/csl/CppWorks/artwork/ceres-utils/src/output/equation_before.json"
    ).SaveResiduals<ns_factors::FittingCostFunctor>(
            "/home/csl/CppWorks/artwork/ceres-utils/src/output/residuals_before.json"
    );
    //-------------

    ab[0] = ab[1] = c = 1;
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << "a: " << ab[0] << ", b: " << ab[1] << ", c: " << c << std::endl;

    //-------------
    // use library
    auto eqAfter = evaluator.Evaluate(
            {{ab, "ab"},
             {&c, "c"}}
    ).SaveEquationToDisk(
            "/home/csl/CppWorks/artwork/ceres-utils/src/output/equation_after.json"
    ).SaveResiduals<ns_factors::FittingCostFunctor>(
            "/home/csl/CppWorks/artwork/ceres-utils/src/output/residuals_after.json"
    );
    //-------------
}

void TestS2Manifold() {
    Eigen::Vector3d target(1, 1, 1), source(-2, -1, -3);
    Eigen::Vector2d dir(0, 0);
    ceres::Problem prob;


    // solution one
    {
        auto costFun = ns_factors::S2ClosestPoint::Create(target);
        costFun->AddParameterBlock(3);
        costFun->SetNumResiduals(3);
        prob.AddResidualBlock(costFun, nullptr, source.data());
        prob.SetManifold(source.data(), new ceres::SphereManifold<3>());
    }
    // solution two
    {
        // auto costFun = ns_factors::S2ClosestPoint::Create(target);
        // costFun->AddParameterBlock(3);
        // costFun->SetNumResiduals(3);
        // prob.AddResidualBlock(costFun, nullptr, source.data());
        // auto costNorm = ns_factors::ConstNormFactor::Create(source);
        // costNorm->AddParameterBlock(3);
        // costNorm->SetNumResiduals(1);
        // prob.AddResidualBlock(costNorm, nullptr, source.data());
    }
    // solution three
    {
        // Eigen::Vector3d dirAndNorm = Utils::ComputeDirAndNorm(source);
        // dir = dirAndNorm.head(2);
        // auto costFun = ns_factors::DirFactor::Create(target, dirAndNorm(2));
        // costFun->AddParameterBlock(2);
        // costFun->SetNumResiduals(3);
        // prob.AddResidualBlock(costFun, nullptr, dir.data());
    }


    ceres::Solver::Summary sum;
    ceres::Solver::Options opt;
    opt.minimizer_progress_to_stdout = true;
    opt.update_state_every_iteration = true;
    opt.callbacks.push_back(
            ns_factors::CeresDebugCallBack::Create(
                    source, "/home/csl/CppWorks/artwork/ceres-utils/output/sphere_opt.txt"
            )
    );

    ceres::Solve(opt, &prob, &sum);

    LOG_VAR(source.transpose())
    LOG_VAR(sum.FullReport())
}

void TestVINSMono() {
    Eigen::Vector3d target(1, 1, 1), source(-2, -1, -3);
    std::ofstream file("/home/csl/CppWorks/artwork/ceres-utils/output/sphere_opt.txt", std::ios::out);
    for (int i = 0; i < 20; ++i) {
        file << source(0) << ',' << source(1) << ',' << source(2) << std::endl;

        Eigen::Matrix<double, 3, 2> jacobian = Utils::TangentBasis(source);
        Eigen::Vector3d error = source - target;
        Eigen::Matrix2d HMat = jacobian.transpose() * jacobian;
        Eigen::Vector2d gVec = jacobian.transpose() * error;
        Eigen::Vector2d delta = HMat.ldlt().solve(gVec);

        source = (source + jacobian * delta).normalized() * source.norm();
        LOG_VAR(source.transpose(), source.norm())
    }
    file.close();
}

int main() {
    // TestEquation();
    TestS2Manifold();
    // TestVINSMono();
    return 0;
}