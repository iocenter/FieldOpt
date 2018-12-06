//
// Created by bellout on 11/27/18.
//

#include <gtest/gtest.h>

// Define settings for tr-dfo
#include "Optimization/tests/test_resource_optimizer.h"
#include "Optimization/tests/test_resource_cases.h"

#include "Runner/tests/test_resource_runner.hpp"
#include "Reservoir/tests/test_resource_grids.h"
#include "Optimization/tests/test_resource_test_functions.h"

#include "Optimization/optimizers/trust_region/TrustRegionOptimization.h"

#include <Eigen/Core>
#include "Utilities/math.hpp"
#include "Utilities/colors.hpp"
#include "Utilities/stringhelpers.hpp"

#include "test_tr-model-data.hpp"

using namespace TestResources::TestFunctions;
using namespace Optimization::Optimizers;
using namespace std;

namespace {

    class TrustRegionTest : public ::testing::Test,
                            public TestResources::TestResourceOptimizer,
                            public TestResources::TestResourceGrids {

    protected:
        TrustRegionTest() {}

        virtual ~TrustRegionTest() {}
        virtual void SetUp() {}

        TrustRegionOptimization *tr_dfo_;
        Optimization::Case *test_case_tr_dfo_probs_;
        VariablePropertyContainer *varcont_tr_dfo_probs_;
        TestResources::TrustRegionModelData tr_mdata;

        void SetUpOptimizer(const MatrixXd &x0, double (*tr_dfo_prob)(VectorXd xs)){

            // Dummy var container based on initial point
            varcont_tr_dfo_probs_ = new VariablePropertyContainer();
            QString base_varname = "BHP#PRODUCER#"; // dummy var name
            for (int i = 0; i < x0.rows(); ++i) {
                // Use initial point values to construct container
                auto *prop = new ContinousProperty(x0(i,0));
                prop->setName(base_varname + QString::number(i));
                varcont_tr_dfo_probs_->AddVariable(prop);
            }

            // Set up base case using dummy var containter
            test_case_tr_dfo_probs_ = new Optimization::Case(
                    QHash<QUuid, bool>(), QHash<QUuid, int>(),
                    varcont_tr_dfo_probs_->GetContinousVariableValues());

            test_case_tr_dfo_probs_->set_objective_function_value(tr_dfo_prob(x0.col(0)));

            // cout << "SetUpOptimizer" << endl;
            tr_dfo_ = new TrustRegionOptimization(
                    settings_tr_opt_max_,
                    test_case_tr_dfo_probs_,
                    varcont_tr_dfo_probs_,
                    grid_5spot_,
                    logger_);
        }

        void RunnerSubs(double (*tr_dfo_prob)(VectorXd xs)){
            // cout << "RunnerSubs" << endl;
            stringstream ss; ss << "[          ] " << FMAGENTA;

            while (!tr_dfo_->IsFinished()) {

                // RUNNER CALL (START)
                auto next_case = tr_dfo_->GetCaseForEvaluation();

                next_case->set_objective_function_value(
                        tr_dfo_prob(next_case->GetRealVarVector()));

                cout << ss.str() << "----------------------" << END << endl;
                cout << ss.str() << "Case id:" << next_case->GetId().toString().toStdString() << END << endl;
                cout << ss.str() << "x:" << next_case->GetRealVarVector().transpose() << END << endl;
                cout << ss.str() << "f:" << next_case->objective_function_value() << END << endl;

                // RUNNER CALL (FINISH)
                tr_dfo_->SubmitEvaluatedCase(next_case);

            }

            cout << ss.str() << "----------------------" << END << endl;
            auto best_case = tr_dfo_->GetTentativeBestCase();
            cout << best_case->objective_function_value() << endl;
        }

    };

    //  auto values = variables->GetContinousVariableValues();
    //  VectorXd test_vec = VectorXd::Zero(values.size());
    //  Case * init_case = new Case(base_case);
    //  init_case->SetRealVarValues(test_vec);
    //  case_handler_->AddNewCase(init_case);

    TEST_F(TrustRegionTest, tr_dfo_prob1) {
        cout << FMAGENTA
             << "[CG.prob1  ] f = @(x) (1 - x(1))^2; x0=[-1.2 2.0]"
             << END << endl;
        SetUpOptimizer(tr_mdata.prob1.xm, tr_dfo_prob1);
        RunnerSubs(tr_dfo_prob1);
    }

    TEST_F(TrustRegionTest, tr_dfo_prob2) {
        cout << FMAGENTA
             << "[CG.prob2  ] f = @(x) log1p(x(1)^2) + x(2)^2; x0=[2.0 2.0]"
             << END << endl;
        SetUpOptimizer(tr_mdata.prob1.xm, tr_dfo_prob2);
        RunnerSubs(tr_dfo_prob2);
    }

    TEST_F(TrustRegionTest, tr_dfo_prob3) {
        cout << FMAGENTA
             << "[CG.prob3  ] f = @(x) sin(pi*x(1)/12) * cos(pi*x(2)/16); x0=[0.0 0.0]"
             << END << endl;
        SetUpOptimizer(tr_mdata.prob1.xm, tr_dfo_prob3);
        RunnerSubs(tr_dfo_prob3);
    }

    TEST_F(TrustRegionTest, tr_dfo_initial_model_created) {
        bool is_model_present = false;
        SetUpOptimizer(tr_mdata.prob1.xm, tr_dfo_prob1);

        // RUNNER CALL (START)
        auto next_case = tr_dfo_->GetCaseForEvaluation();

        next_case->set_objective_function_value(
                tr_dfo_prob1(next_case->GetRealVarVector()));

        cout << "----------------------" << END << endl;
        cout << "Case id:" << next_case << END << endl;
        cout << "x:" << next_case->GetRealVarVector().transpose() << END << endl;
        cout << "f:" << next_case->objective_function_value() << END << endl;

        // RUNNER CALL (FINISH)
        tr_dfo_->SubmitEvaluatedCase(next_case);

        auto tr_model = tr_dfo_->getTrustRegionModel();

        if (tr_model) {
            if (tr_model->getDimension() >=2) { //<!at least 2 points are needed in the
                is_model_present = true;
            } else {
                cout << "Not enough points in the model." << endl;
            }
        } else {
            cout << "model not created" << endl;
        }
        EXPECT_TRUE(is_model_present);
    }
}
