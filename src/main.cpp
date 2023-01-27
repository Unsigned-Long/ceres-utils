//
// Created by csl on 1/27/23.
//

#include <utility>

#include "ceres-utils/equations.h"
#include "random"
#include "chrono"
#include "fstream"

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

struct Utils {
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


int main(int argc, char **argv) {
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
        auto costFunc = FittingCostFunctor::Create(pt);
        // a, b
        costFunc->AddParameterBlock(2);
        // c
        costFunc->AddParameterBlock(1);
        costFunc->SetNumResiduals(1);

        problem.AddResidualBlock(costFunc, nullptr, {ab, &c});
        //-------------
        // use library
        evaluator.AddCostFunction(costFunc, {ab, &c});
        //-------------
    }
    //-------------
    // use library
    auto eqBefore = evaluator.Evaluate(
            {{ab, "ab"},
             {&c, "c"}}
    ).SaveEquationToDisk(
            "/home/csl/CppWorks/artwork/ceres-utils/src/output/equation_before.json"
    ).SaveResiduals<FittingCostFunctor>(
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
    ).SaveResiduals<FittingCostFunctor>(
            "/home/csl/CppWorks/artwork/ceres-utils/src/output/residuals_after.json"
    );;
    //-------------
    return 0;
}