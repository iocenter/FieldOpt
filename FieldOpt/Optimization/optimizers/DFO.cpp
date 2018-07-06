//
// Created by pcg1 on 12.01.18.
//

#include "DFO.h"
#include "GradientEnhancedModel.h"
#include "VirtualSimulator.h"
#include <iostream>
#include <iomanip>
#include <casadi/casadi.hpp>

#define FIND_POINTS1 0
#define FIND_POINTS2 1
#define INITIALIZE_MODEL 2
#define CRITICALITY_STEP_START 3
#define CRITICALITY_STEP_CHECK_CONVERGENCE 4
#define FIND_TRIAL_POINT 5
#define ACCEPTANCE_OF_TRIAL_POINT 6
#define MODEL_IMPROVEMENT_STEP_START 7
#define MODEL_IMPROVEMENT_STEP_END 8
#define TRUST_REGION_RADIUS_UPDATE_STEP 9
#define CRITICALITY_STEP_ADD_POINTS 10

VirtualSimulator vs;

/// Test functions - start
double matyasFunction(Eigen::VectorXd x) {
  double val = 0.26 * (x(0) * x(0) + x(1) * x(1)) - 0.46 * x(0) * x(1);
  return val;
}

Eigen::VectorXd matyasFunctionWithGradients(Eigen::VectorXd x) {
  double val = 0.26 * (x(0) * x(0) + x(1) * x(1)) - 0.46 * x(0) * x(1);
  Eigen::VectorXd ret(1 + 2);
  ret[0] = val;
  ret[1] = 0.52 * x(0) - 0.46 * x(1);
  ret[2] = 0.52 * x(1) - 0.46 * x(0);
  return ret;
}

Eigen::VectorXd matyasFunctionWithGradients1(Eigen::VectorXd x) {
  double val = 0.26 * (x(0) * x(0) + 0.5 * x(1) * x(1)) - 0.46 * x(0) * x(1) + 2 * x(1) + 5.5 - 3.2 * x(0);
  Eigen::VectorXd ret(1 + 1);
  ret[0] = val;
  ret[1] = 0.52 * x(1) - 0.46 * x(0) + 2;
  return ret;
}

Eigen::VectorXd test(Eigen::VectorXd x) {
  Eigen::VectorXd ret(1 + 4);
  ret.setOnes();
  ret[0] = x.norm();
  return ret;
}

Eigen::VectorXd evaluateFunctionVS(Eigen::VectorXd x, int ng) {
  Eigen::VectorXd result(1 + ng);
  result(0) = vs.evaluateFunction(x);
  std::cout << "input is:\n" << x << "\noutput is: " << result(0) << "\n";

  Eigen::VectorXd gradients1 = vs.evaluateFunctionGradients(x);
  Eigen::VectorXd grads = gradients1.tail(ng);
  for (int i = 0; i < grads.rows(); ++i) {
    //result(i+1) = grads(grads.rows()-1-i);
    result(i + 1) = grads(i);
  }
  //result(1) = -1;
  //result(2) = -1;

  return result;
}

Eigen::VectorXd sphere(Eigen::VectorXd x, int ng) {
  Eigen::VectorXd result(1 + ng);
  x(0) = x(0) / 100;
  Eigen::VectorXd scaledX = x * 10;
  double val = 0;
  for (int i = 0; i < x.rows(); i++) {
    val += (x(i)) * (x(i));
//    val += (x(i)  - 23 ) * (x(i) - 4);
  }
  double val2 = 0.26 * (x(0) * x(0) + x(1) * x(1)) - 0.46 * x(0) * x(1);
  double valros = (x(0) - 1) * (x(0) - 1) + 100 * (x(1) - x(0) * x(0)) * (x(1) - x(0) * x(0));
  double valSixHump = (4 - 2.1 * x(0) * x(0) + 1.0 / 3.0 * x(0) * x(0) * x(0) * x(0)) * x(0) * x(0) + x(0) * x(1)
      + 4 * (x(1) * x(1) - 1) * x(1) * x(1);
  double valXinSheYangs = (std::abs(x(0)) + std::abs(x(1))) * std::exp(-std::sin(x(0) * x(0)) - std::sin(x(1) * x(1)));
  //double valAbs = std::abs(x(0)) + std::abs(x(1));
  double valAbs = std::abs(x(0)) + std::abs(x(1));
  double funky = std::abs(x(0)) + x(1) * x(1) * 10;
  result(0) = val2;
  if (ng > 0) {
    //for (int i = ng; i >= 1; ++i){
    //  result(i) = 2*x(i-1);
    //}
    /*
    if (ng >= 1)
      result(1) = 2*x(4);
    if (ng >= 2)
      result(2) = 2*x(3);
    if (ng >= 3)
      result(3) = 2*x(2);
    if (ng >= 4)
      result(4) = 2*x(1);
    if (ng >= 5)
      result(5) = 2*x(0);
    */

    result(1) = 0.26 * 2 * x(1) - 0.46 * x(0);
    if (ng >= 2) {
      result(2) = 0.26 * 2 * x(0) - 0.46 * x(1);
    }
    //if (ng == 2)
    //result(1) = 2*x(1)*10 ;

    //if (ng == 2)
    //  result(2) = 2*x(0);
  }
  return result;


  /// OBS: result(1) <- derivert med hensyn på siste variabel.
  /// OBS: result(2) <- derivert med hensyn på nest siste variabel.
}

double sphereorg(Eigen::VectorXd x) {
  Eigen::VectorXd scaledX = x * 10;
  double val = 0;
  for (int i = 0; i < x.rows(); i++) {
    val += (x(i) - 10) * (x(i) - 10);
//    val += (x(i)  - 23 ) * (x(i) - 4);
  }
  return val;
}
Eigen::VectorXd sphereGradients(Eigen::VectorXd x, int ng) {
  Eigen::VectorXd ret(1 + ng);
  double val = 0;
  for (int i = 0; i < x.rows(); i++) {
    val += x(i) * x(i);
  }
  ret[0] = val;
  int n = x.rows();
  int j = 1;
  for (int i = (n - ng); i < n; ++i) {
    ret[j] = 2 * x(i);
    j++;
  }
  return ret;
}

/// Test functions - end



namespace Optimization {
namespace Optimizers {
double matyasFunction2D(Eigen::VectorXd x);
double rosenbrockFunction2D(Eigen::VectorXd x);
double polynomial3(Eigen::VectorXd x);
double polynomial32(Eigen::VectorXd x);
double polynomial5(Eigen::VectorXd x);
double polynomial6(Eigen::VectorXd x);
double evaluateFunction(Eigen::VectorXd x, Eigen::VectorXd y0, int select);
double evaluateFunction(Eigen::VectorXd x, Eigen::VectorXd y0, int select) {
  Eigen::VectorXd y = x + y0;
  switch (select) {
    case 0:return matyasFunction2D(y);
      break;
    case 1:return rosenbrockFunction2D(y);
      break;
    case 2:return polynomial6(y);
      break;
    case 3:return polynomial5(y);
      break;
    case 4:return polynomial3(y);
      break;
    case 5:return polynomial32(y);
      break;
    default:std::cout << "Invalid function selection" << std::endl;
      break;
  }

}

DFO::DFO(Settings::Optimizer *settings,
         Optimization::Case *base_case,
         Model::Properties::VariablePropertyContainer *variables,
         Reservoir::Grid::Grid *grid,
         Logger *logger)
    : Optimizer(settings, base_case, variables, grid, logger),
      DFO_model_(settings->parameters().number_of_interpolation_points,
                 settings->parameters().number_of_variables,
                 settings->parameters().number_of_variables_with_gradients,
                 base_case->GetRealVarVector(),
                 settings->parameters().initial_trust_region_radius,
                 settings->parameters().required_poisedness,
                 settings->parameters().weight_model_determination_minimum_change_hessian,
                 settings->parameters().weights_distance_from_optimum_lsq,
                 settings) {
  // Set parameters and stuff here
  if (settings->parameters().initial_trust_region_radius > 0.0)
    initial_trust_region_radius_ = settings->parameters().initial_trust_region_radius;
  else
    initial_trust_region_radius_ = 600;

  if (settings->parameters().number_of_interpolation_points > 0)
    number_of_interpolation_points_ = settings->parameters().number_of_interpolation_points;
  else
    number_of_interpolation_points_ = 21;

  if (settings->parameters().number_of_variables > 0)
    number_of_variables_ = settings->parameters().number_of_variables;
  else
    number_of_variables_ = 10;

  if (settings->parameters().required_poisedness > 0)
    required_poisedness_ = settings->parameters().required_poisedness;
  else
    required_poisedness_ = 5;

  settings_ = settings;
  varcont_ = variables;
  iterations_ = 0;
  previous_iterate_type_ = 0;
  base_case_ = new Case(base_case);
  last_action_ = -1;
  CreateFiles();
  weights_distance_from_optimum_lsq_ =
      Eigen::VectorXd::Zero(settings->parameters().weights_distance_from_optimum_lsq.size());
  int j = 0;
  for (auto i = settings->parameters().weights_distance_from_optimum_lsq.begin();
       i != settings->parameters().weights_distance_from_optimum_lsq.end(); ++i) {
    weights_distance_from_optimum_lsq_[j] = *i;
    j++;
    if (j >= 3) {
      break;
    }
  }

  // Set up the DFO model.
  Eigen::VectorXd initialStartPoint = base_case->GetRealVarVector();
  auto as = base_case->GetRealVarIdVector(); //QList<QUuid>
  auto aa = base_case->real_variables(); // QHash<QUuid, double>



  for (int i = 0; i < 6; i++) {
    std::cout << aa[as.at(i)] << "\n";
    as.at(i);
  }
  std::cout << "\n";



  /// Making example of model improvement algorithm. for report.

  //DFO_model_.makeExample();
  //std::cout << "too far" << std::endl;
  /// Testing snopt solver
/*
  std::cout << std::scientific;
  std::cout << std::setprecision(15);
  vector<double> xsolmin;
  vector<double> xsolmax;
  vector<double> fsolmin;
  vector<double> fsolmax;
  Subproblem mySub(settings);
  double trr = 0.000000084360314/2.0;
  double c = 0.000001268519113;
  Eigen::MatrixXd h(number_of_variables_, number_of_variables_);
  h << 0.000000000000748, -0.000000000000000,
  -0.000000000000000,  0.000000000000748;
  Eigen::VectorXd g(number_of_variables_);
  g << 5.495226660131417, -5.495226829650895;
  Eigen::VectorXd ybr(number_of_variables_);
  ybr << -0.000000175677456, 0.000000175677456;
  Eigen::VectorXd sp(number_of_variables_);
  sp = ybr;
  mySub.setConstant(c); mySub.setGradient(g); mySub.setHessian(h);
  mySub.SetTrustRegionRadius(trr);
  Eigen::VectorXd cp(number_of_variables_);
  cp << 1000.000000160129503, 999.999999839866064;
  Eigen::VectorXd ttt(2);
  ttt.setZero();
  mySub.Solve(xsolmin, fsolmin, (char*)"Minimize",cp,ybr,ybr);
  mySub.Solve(xsolmax, fsolmax, (char*)"Maximize",cp,ybr,ybr);

  Eigen::VectorXd xmin(number_of_variables_);
  Eigen::VectorXd xmax(number_of_variables_);
  for (int i = 0; i < xsolmax.size(); ++i) {
    xmin[i] = xsolmin[i];
    xmax[i] = xsolmax[i];
  }
  std::cout << "\n\n\n\n ----------------------------------------------------------------------------\n\n\n\n";
  std::cout << "ybr: \n" << ybr <<"\n";
  std::cout << "xsolmin: \n" << xmin << "\n" << "xsolMax: \n" << xmax <<"\n";
  std::cout << "minval = " << fsolmin[0] << "\t maxval = " << fsolmax[0] <<"\n";

  trr = 20000;
  c = 2;
  h << 255, -853,
      -853,  55555;
  g << 4000, -3600;
  ybr << 1000, 2000;
  sp = ybr;
  mySub.setConstant(c); mySub.setGradient(g); mySub.setHessian(h);
  mySub.SetTrustRegionRadius(trr);
  cp << 1000.000000160129503, 999.999999839866064;
  mySub.Solve(xsolmin, fsolmin, (char*)"Minimize",cp,ybr,ybr);
  mySub.Solve(xsolmax, fsolmax, (char*)"Maximize",cp,ybr,ybr);

  for (int i = 0; i < xsolmax.size(); ++i) {
    xmin[i] = xsolmin[i];
    xmax[i] = xsolmax[i];
  }
  std::cout << "\n\n\n\n ----------------------------------------------------------------------------\n\n\n\n";
  std::cout << "ybr: \n" << ybr <<"\n";
  std::cout << "xsolmin: \n" << xmin << "\n" << "xsolMax: \n" << xmax <<"\n";
  std::cout << "minval = " << fsolmin[0] << "\t maxval = " << fsolmax[0] <<"\n";


  trr = 9.38361523181344736778e-08;
  c = 1.73332985074814979819e-16;
  h <<  3.16185306180997827141e-01, -3.16185306180997827141e-01,
      -3.16185306180997827141e-01,  3.16185306180997827141e-01;
  g << -3.03179261538150991457e-06,
  3.03363980306762893570e-06;
  ybr << 0.00000000000000000000e+00, 0.00000000000000000000e+00;
  sp = ybr;
  mySub.setConstant(c); mySub.setGradient(g); mySub.setHessian(h);
  mySub.SetTrustRegionRadius(trr);
  cp << 9.99999997200520738261e+02, 9.99999997206447460485e+02;
  mySub.Solve(xsolmin, fsolmin, (char*)"Minimize",cp,ybr,ybr);
  mySub.Solve(xsolmax, fsolmax, (char*)"Maximize",cp,ybr,ybr);

  for (int i = 0; i < xsolmax.size(); ++i) {
    xmin[i] = xsolmin[i];
    xmax[i] = xsolmax[i];
  }
  std::cout << "\n\n\n\n ----------------------------------------------------------------------------\n\n\n\n";
  std::cout << "ybr: \n" << ybr <<"\n";
  std::cout << "xsolmin: \n" << xmin << "\n" << "xsolMax: \n" << xmax <<"\n";
  std::cout << "minval = " << fsolmin[0] << "\t maxval = " << fsolmax[0] <<"\n";
*/


  //DFO_model_ = DFO_Model(number_of_interpolation_points_, number_of_variables_, initialStartPoint, initial_trust_region_radius_, required_poisedness_,settings_);
  //Eigen::VectorXd tt(number_of_variables_);
  //tt = DFO_model_.FindLocalOptimum();
/*
  DFO_model_.findFirstSetOfInterpolationPoints();

  Eigen::VectorXd* refFvals = DFO_model_.getFvalsReference(); // a temporarily (and bad coding practice) solution
  Eigen::MatrixXd* Yref = DFO_model_.getYReference(); // a temporarily (and bad coding practice) solution
  int v = 0;
  std::cout << (*refFvals).rows() << " " << (*refFvals).cols() << "\n ";
  for (int i = 0; i < 2*number_of_variables_+1; i++) {
    ++v;
    (*refFvals)[i] = evaluateFunction((*Yref).col(i), DFO_model_.getCenterPoint(), 0); // Fill up the function evaluations.
  }

  if (number_of_interpolation_points_ >= 2 * number_of_variables_ + 1) {
    DFO_model_.findLastSetOfInterpolationPoints();
    for (int i = 2 * number_of_variables_ + 1; i < number_of_interpolation_points_; i++) {
      ++v;
      (*refFvals)[i] = evaluateFunction((*Yref).col(i), DFO_model_.getCenterPoint(), 0);
    }
  }
  DFO_model_.initializeModel();
*/
// Shift the center point of the model if the best point is not the center point.
  //int bestPointIndex = DFO_model_.getBestPointIndex();
  //Eigen::VectorXd bestPoint(number_of_variables_);
  //bestPoint = (*Yref).col(bestPointIndex - 1);

  //if (bestPointIndex != 1) {
  //  DFO_model_.shiftCenterPointOfQuadraticModel(bestPoint);
  //}

  // Model Improvement algorithm
  /*
   double upperBound = 0;
   double lowerBound = 0;
   int k = 0;
   while (1) {
     int yk = 0;
     Eigen::VectorXd d(number_of_variables_);
     d.setZero();
     DFO_model_.findWorstPointInInterpolationSet(d, yk);

     if (yk == -1) {
       std::cout << "breaking. The required poisedness is acheived. k = " << k << std::endl;
       break;
     }
     else {
       ++v;
       double funcVal = evaluateFunction(d, DFO_model_.getCenterPoint(), 0);
       DFO_model_.update(d, funcVal, yk, DFO_Model::IMPROVE_POISEDNESS);

       //myModel.printParametersMatlabFriendly();
       k++;

       if (k == 2000) {
         std::cout << "breaking because of too many iterations., k = " << k << std::endl;
         break;
       }
     }
   }

   std::cout << *(Yref) << std::endl << std::endl;
   DFO_model_.printParametersMatlabFriendly();
   std::cout << "%Number of function evaluation is: v = " << v << std::endl;

   //GradientEnhancedModel  enhancedModel(n,m,ng,weights_derivatives,weight_objective_minimum_change);
   GradientEnhancedModel  enhancedModel(number_of_variables_,number_of_interpolation_points_,settings->parameters().number_of_variables_with_gradients,settings->parameters().weights_distance_from_optimum_lsq,settings->parameters().weight_model_determination_minimum_change_hessian);

   std::cin.get(); // To avoid exiting terminal.
 */

  //Subproblem mySub(settings);
/*
  vector<double> xsol;
  vector<double> fsol;
  //mySub.Solve(xsol, fsol, (char*)"Maximize");
  mySub.Solve(xsol, fsol, (char*)"Minimize");
  int numVars = 10;
  cout << "xsol: " << endl;
  for (int j = 0; j < numVars; j++) {
    cout << xsol[j] << endl;
  }
  cout << endl << "Objective values:" << endl;
  for (int j = 0; j < 2; j++) {
    cout << fsol[j] << endl;
  }
  cout << "stop " << endl;
*/
  /*
   for (int p = 0; p < 10; p++){
   static std::random_device rd;
   static std::mt19937 gen(p);
   static std::uniform_real_distribution<> dis(-30, 30);
   // Testing with 20 (wells) * 6 (variables pr wells) = 120 variables.
   int numVars = 10;
   Eigen::MatrixXd H(numVars,numVars);
   int k = 0;
   for (int j = 0; j < numVars; ++j) {
     for (int i = 0; i < j; ++i){
       H(i,j) = dis(gen);
       if (i != j){
         H(j,i) = H(i,j);
       }
     }
   }


   Eigen::VectorXd g(numVars);
   g.setRandom();
   mySub.setConstant(1.2);
   mySub.setGradient(g);
   mySub.setHessian(H);
   vector<double> xsol;
   vector<double> fsol;
   //mySub.Solve(xsol, fsol, (char*)"Maximize");
   mySub.Solve(xsol, fsol, (char*)"Minimize");

   cout << "xsol: " << endl;
   for (int j = 0; j < numVars; j++) {
     cout << xsol[j] << endl;
   }
   cout << endl << "Objective values:" << endl;
   for (int j = 0; j < 2; j++) {
     cout << fsol[j] << endl;
   }
   }
   cout << "stop " << endl;
 */

  /*
  cout << "[DFO constructor] ..." << endl;


  //cout << settings_->verb_vector()[0] << endl;
  realvar_uuid_ = GetTentativeBestCase()->GetRealVarIdVector();


  Eigen::VectorXd vars = base_case->GetRealVarVector();

  cout<< "realvar \n" << vars << endl;


  vars[0] += 1;
  vars[1] += 1;
  vars[2] += 1;
  vars[3] += 1;
  vars[4] += 1;
  vars[5] += 1;
  QString old = base_case->GetId().toString();
  Optimization::Case *newCase = new Case(base_case);
  QString neww = newCase->GetId().toString();

  cout << "The ID of the base case is: " << old.toStdString() << endl;
  cout << "The ID of the copied case is: " << neww.toStdString() << endl;

  if (old == neww){
    cout << "WTF \n";
  }
  if (old == old)
    cout << "GOOD " << endl;
  if (neww == neww)
    cout << "GOOD 2" << endl;
  if (old.toStdString() == old.toStdString())
    cout << "GOOD 3" << endl;

  if (base_case->GetId() == base_case->GetId())
    cout << "GOOD 4\n";

  if (base_case->GetId() == newCase->GetId())
    cout << "THIS IS BAD\n";

  if (base_case != newCase)
    cout << "INteresting. comparing cases!" << endl;

  auto fval0 = base_case->objective_function_value();

  newCase->SetRealVarValues(vars);
  newCase->set_objective_function_value(std::numeric_limits<double>::max());
  //auto fval1 = newCase->objective_function_value();

  cout << fval0 << "\t \t" << "oooouups" << endl;

  case_handler_->AddNewCase(newCase);
*/




  //std::cout << "Reached the end of DFO constructor\n";
  //std::cin.get();

}
/*
Optimization::Optimizer::TerminationCondition DFO::IsFinished() {
    return TerminationCondition::NOT_FINISHED;
}

 */
void DFO::handleEvaluatedCase(Optimization::Case *c) {

}

void DFO::iterate() {

  //std::string problem =  "/home/joakim/git/casadi/docs/examples/nl_files/hs107.nl";
  std::string problem = "/home/joakim/git/casadi/docs/examples/nl_files/nlTestSet/qptest1.nl";

  // Parse an NL-file
  casadi::NlpBuilder nl;
  nl.import_nl(problem);

  //std::vector<casadi::MX> xcasted = nl.x;

  std::vector<casadi::MX> f1 = {nl.f};
  std::vector<casadi::MX> f2 = {casadi::MX::vertcat(nl.x)};
  casadi::Function f = casadi::Function("obj", f2, f1);
  //std::pair<int,int> mypair = {1,int(nl.x.size())};
  //auto in = casadi::DM::ones(mypair); //
  //std::vector<casadi::DM> a = {1,1,1,1,1,1,1,1,1};
  std::vector<casadi::DM> a = {casadi::DM(nl.x_init)};
  //Convert eigen::vector into DM:
  Eigen::VectorXd d;// = Eigen::VectorXd(a);
  auto out = f(a);
  std::cout << "obj = " << (out[0]) << std::endl;

  auto rows = a[0].size1();

  /*
  d.resize(rows);
  d.setZero();
  d << -1,-1;
  std::cout<< "\n" << d << std::endl;
  auto dense_a = casadi::DM::densify(a[0]);

  std::memcpy(d.data(), dense_a.ptr(), sizeof(double)*rows);
  std::cout<< "\n"  << d << std::endl;
*/
  casadi::Function fj = f.factory("jacf", {f.name_in()}, {"jac:o0:i0", "o0"});
  auto out_fj = fj(a);
  std::cout << "out_fj =\n " << (out_fj[0]) << std::endl;


  //constraints
  auto gcat = {casadi::MX::vertcat(nl.g)};
  casadi::Function g = casadi::Function("obj", f2, gcat);
  auto out_g = g(a);
  std::cout << "constraints\n" << out_g[0] << "\n";

  casadi::Function gj = g.factory("jacg", {f.name_in()}, {"jac:o0:i0", "o0"});
  auto out_gj = gj(a);
  auto dd = out_gj[0];
  auto dd_dens = casadi::DM::densify(dd);
  //std::cout << "size1 (rows) =  " << dd_dens.size1() << ",\t size2 (cols) = " << dd_dens.size2() <<"\n";
  std::cout << "dd_dens \n" << dd_dens << "\n"; //<<" " << out_fj[1] << std::endl;
  //std::cout << "d_dens(0) \n" << casadi::DM::densify(dd_dens(0)) << "\n"; //<<" " << out_fj[1] << std::endl;
  //std::cout << "d_dens(5) \n" << casadi::DM::densify(dd_dens(5)) << "\n"; //<<" " << out_fj[1] << std::endl;Schøyens gate 10 C, 7030 Trondheim
  //std::cout << "d_dens(2,1) \n" << casadi::DM::densify(dd_dens(2,1)) << "\n"; //<<" " << out_fj[1] << std::endl;,,


  /// Testing VirtualSimulator

  vs = VirtualSimulator(settings_->parameters().test_problem_file);
  vs.Solve();
/*
  /////////////////////////////////////////////////////////////////////////////////////////////////
  int n = number_of_variables_;
  Eigen::VectorXd yTry(n); //Displacement from current center point
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-100, 100);
  int k = 0;

  for (int i = 0; i < n; ++i) {
    yTry(i) = dis(gen);
  }
  Eigen::VectorXd yBest = yTry;

  double minValue = vs.evaluateFunction(yBest);
  double value = 0;

  while (k < 5000000 && false) {
    for (int i = 0; i < n; ++i) {
      yTry(i) = dis(gen);
    }
    value = vs.evaluateFunction(yTry);
    if (value < minValue) {
      minValue = value;
      yBest = yTry;
    }
    ++k;
  }
  std::cout << "The smallest value ffound is: "  << minValue <<"\nWith the x= \n" << yBest << "\n\n";
  */
  /////////////////////////////////////////////////////////////////////////////////////////////////

  //Eigen::VectorXd asda(2);
  //asda  << 2,5;
  //std::cout <<"constraints \n" << vs.evaluateConstraints(asda) << "\n";
  DFO_model_.initLagrangeMultipliers(vs.GetNumberOfConstraints());


  // d_dens (ci,vi) <- constraint_i derivert mhp variabel_i

  //std::cout << "out_g[0] \n" << (out_g[0]) << "\n"; //<<" " << out_fj[1] << std::endl;
  //std::cout << "out_g[1]  \n" << (out_g[1]) << "\n"; //<<" " << out_fj[1] << std::endl;
  //std::cout << "(out_g[1,1])  " << (out_g[1,1]) << "\n"; //<<" " << out_fj[1] << std::endl;
  //std::cout << "(out_g[1,2])  " << (out_g[1,2]) << "\n"; //<<" " << out_fj[1] << std::endl;


/*
  size_t rows = casadi_matrix.size1();
  size_t cols = casadi_matrix.size2();

  eigen_matrix.resize(rows,cols);
  eigen_matrix.setZero(rows,cols);

  std::memcpy(eigen_matrix.data(), casadi_matrix.ptr(), sizeof(double)*rows*cols);
*/



  // Need:
  // function evaluation. gradients of obj.fun.
  //       constraints,         gradients of contraints(?)



  /*
  casadi::Dict opts;
  opts["expand"] = true;
  casadi::Function solver = nlpsol("nlpsol", "ipopt", nl, opts);
  std::map<std::string, casadi::DM> arg, res;

  // Solve NLP
  arg["lbx"] = nl.x_lb;
  arg["ubx"] = nl.x_ub;
  arg["lbg"] = nl.g_lb;
  arg["ubg"] = nl.g_ub;
  arg["x0"] = nl.x_init;
  res = solver(arg);
  res = solver(arg);
  for (auto&& s : res) {
    std::cout << std::setw(10) << s.first << ": " << std::vector<double>(s.second) << std::endl;
  }
  */
  /*
  Eigen::VectorXd input = Eigen::VectorXd::Zero(2);
  std::cout << "from virtual simulator: \n";
  std::cout << "obj = " << vs.evaluateFunction(input) << "\n";
  std::cout << "obj gradients \n" << vs.evaluateFunctionGradients(input) << "\n";
  std::cout << "constraints\n" <<     vs.evaluateConstraints(input) << "\n";
  std::cout << "constraints gradients \n" << vs.evaluateConstraintGradients(input) << "\n";
*/

  std::cout << std::setprecision(20);
  std::cout << std::scientific;
  //GradientEnhancedModel  enhancedModel(number_of_variables_,number_of_interpolation_points_,settings_->parameters().number_of_variables_with_gradients,settings_->parameters().weights_distance_from_optimum_lsq,settings_->parameters().weight_model_determination_minimum_change_hessian);
  int number_of_crit_step_finished_with_bad_poisedness = 0;
  int number_of_crit_step_finished_without_checking_poisedness = 0;
  MatrixXd aa(2, 2);
  aa << 1.1, 2.1, 1.2, 2.2;
  std::cout << "aa\n" << aa << "\n";
  aa.col(0).swap(aa.col(1));
  std::cout << "aa\n" << aa << "\n";

  int ng = settings_->parameters().number_of_variables_with_gradients;
  double alpha = settings_->parameters().weight_model_determination_minimum_change_hessian;

  std::string color_from = "31";
  std::string color_to = "33";

  /// Extract information from cases
  QList<Case *> newCases = case_handler_->RecentlyEvaluatedCases();
  /*
  Need the fval... the rest is already known.
  */
  int number_of_tiny_improvements = 0;
  // This loop is used for testing. Instead of running simulations, a simple analytical function is used.
  // Most of the content in this loop should be in "iterate()", when the testing is done.
  double r =
      settings_->parameters().r; // expansion factor of the trust-region radius to accept more points. (even though they are outside the "true" trust region)
  double w = settings_->parameters().w; // decreasing factor of the trust-region radius during the criticality step.
  double u = settings_->parameters().u;
  double beta = settings_->parameters().beta;
  double epsilon_c = settings_->parameters().epsilon_c;
  double tau = settings_->parameters().tau;
  double eta1 = settings_->parameters().eta1;
  double gamma = settings_->parameters().gamma;
  double gamma_inc = settings_->parameters().gamma_inc;
  double trust_region_radius_inc = settings_->parameters().initial_trust_region_radius;
  double trust_region_radius_max = settings_->parameters().max_trust_region_radius;
  double trust_region_radius_end = settings_->parameters().end_trust_region_radius;

  bool notConverged = true;

  int crit_steps = 0;
  int accept_steps = 0;
  int model_impr_steps = 0;

  // These are the ones that are used the most. Used in all other places than "Step 4 - Model Improvement".

  // Only used in the initialization process.

  Eigen::VectorXd *refFuncVals = DFO_model_.getFvalsReference();
  Eigen::MatrixXd *refY = DFO_model_.getYReference();
  Eigen::MatrixXd *refDerivatives = DFO_model_.getDerivativeReference();

  for (int i = 0; i < number_of_interpolation_points_; ++i) {
    (*refFuncVals)(i) = -1;
  }

  int number_of_new_points = 0;
  int number_of_function_calls = 0;
  int number_of_parallell_function_calls = 0;
  Eigen::MatrixXd new_gradients;
  Eigen::MatrixXd new_points;
  Eigen::VectorXd new_point;
  Eigen::VectorXd last_trial_point(number_of_variables_);
  Eigen::VectorXd new_gradient;
  Eigen::VectorXi new_points_indicies;
  Eigen::VectorXd function_evaluations(number_of_interpolation_points_);
  function_evaluations.setZero();
  double function_evaluation;
  int new_point_index = -1;
  int criticality_step_iteration = 0;

  bool force_criticality_step = false;
  {
    Eigen::VectorXd tmp = DFO_model_.getCenterPoint();
    for (int i = 0; i < number_of_variables_; ++i) {
      last_trial_point(i) = tmp(i) + DFO_model_.GetTrustRegionRadius() * (2 * r) * 10;
    }
  }
  Eigen::VectorXd tmp_eval(1 + ng);

  next_step_ = FIND_POINTS1;
  bool is_model_CFL = false; //false means we don't know.

  double trust_region_radius_tilde = settings_->parameters().initial_trust_region_radius;;
  double trust_region_radius_icb = settings_->parameters().initial_trust_region_radius;;
  double rho = 0;
  double multiple_new_points = false;
  int number_of_points_first_set = 0;

  while (notConverged) {
    print:
    std::cout << "\033[1;34;m " << " ---------- New iterate " << iterations_ << " ---------- " << "\033[0m"
              << std::endl;
    std::cout << "\033[1;34;m " << "Fvals = \n" << "\033[0m" << *refFuncVals << "\n";
    std::cout << "\033[1;34;m " << "Y = \n" << "\033[0m" << *refY << "\n";
    if (iterations_ != 0 && iterations_ != 1 && iterations_ != 2) {
      std::cout << "\033[1;34;m " << "Best point index = \n" << "\033[0m" << DFO_model_.getBestPointIndex() << "\n";

    }
    std::cout << "\033[1;34;m " << "y0 = \n" << "\033[0m" << DFO_model_.getCenterPoint() << "\n";
    std::cout << "\033[1;34;m " << "Trust region radius is: " << "\033[0m" << DFO_model_.GetTrustRegionRadius()
              << std::endl;
    std::cout << "\033[1;34;m " << "Trust region radius Tilde is: " << "\033[0m" << trust_region_radius_tilde
              << std::endl;


    /// --------------------------------------------------------------------------------------------------------------------------------

    top:
    if (next_step_ == FIND_POINTS1) {
      new_points = DFO_model_.findFirstSetOfInterpolationPoints();
      std::cout << "Y new points:\n" << new_points <<"\n";
      std::cout << "Y:\n" << *refY << "\n";
      new_points_indicies.resize(new_points.cols());
      number_of_points_first_set = new_points.cols();

      multiple_new_points = true;
      set_next_step(FIND_POINTS2);
      goto evaluate;
    } else if (next_step_ == FIND_POINTS2) {
      // add the points.
      for (int i = 0; i < number_of_new_points; ++i) {
        //DFO_model_.SetFunctionValue(i + 1, function_evaluations[i]);
        DFO_model_.SetFunctionValueAndDerivatives(i + 1, function_evaluations(i), new_gradients.col(i));
      }

      set_next_step(INITIALIZE_MODEL);
      // find the remaining points.
      if (number_of_interpolation_points_ == number_of_function_calls) {
        // All points are found.
        number_of_new_points = 0;
        goto print;
      } else {
        number_of_new_points = number_of_interpolation_points_ - number_of_function_calls;
        new_points_indicies.resize(number_of_new_points);
        new_points.resize(number_of_variables_, number_of_new_points);
        new_points = DFO_model_.findLastSetOfInterpolationPoints();
        multiple_new_points = true;
        goto evaluate;
      }
    } else if (next_step_ == INITIALIZE_MODEL) {
      if (number_of_new_points != 0) {
        // Add the points
        for (int i = 0; i < number_of_new_points; ++i) {
          //DFO_model_.SetFunctionValue(i + 1, function_evaluations[i - number_of_new_points]);
          DFO_model_.SetFunctionValueAndDerivatives(number_of_points_first_set + i + 1,
                                                    function_evaluations(i),
                                                    new_gradients.col(i));
        }
      }

      DFO_model_.initializeModel();
      set_next_step(CRITICALITY_STEP_START);
      goto print;
    } else if (next_step_ == CRITICALITY_STEP_START) {
      //Eigen::VectorXd gradient = DFO_model_.GetGradientAtPoint(DFO_model_.GetBestPoint());
      Eigen::VectorXd gradient = DFO_model_.GetLagrangianGradient(DFO_model_.GetBestPoint());
      if ((gradient.norm() > epsilon_c && force_criticality_step == false
          && number_of_crit_step_finished_without_checking_poisedness <= 4)) {
        DFO_model_.SetTrustRegionRadius(trust_region_radius_icb);
        set_next_step(FIND_TRIAL_POINT);
        number_of_crit_step_finished_without_checking_poisedness++;
        goto top;
      } else {
        number_of_crit_step_finished_without_checking_poisedness = 0;
        bool is_poised =
            DFO_model_.ModelImprovementAlgorithm(r * trust_region_radius_icb, new_points, new_points_indicies);
        if ((!is_poised) || (trust_region_radius_icb > u * gradient.norm())) {
          criticality_step_iteration = 0;
          if (is_poised) {
            force_criticality_step = false;
            set_next_step(CRITICALITY_STEP_CHECK_CONVERGENCE);
            goto top;
          } else {
            force_criticality_step = false;
            set_next_step(CRITICALITY_STEP_ADD_POINTS);
            multiple_new_points = true;
            goto evaluate;
          }
        } else {
          if (force_criticality_step) {
            trust_region_radius_icb = trust_region_radius_icb * gamma;
            force_criticality_step = false;
            set_next_step(CRITICALITY_STEP_START);
          } else {
            DFO_model_.SetTrustRegionRadius(trust_region_radius_icb);
            set_next_step(FIND_TRIAL_POINT);
          }
          goto top;
        }
      }
    } else if (next_step_ == CRITICALITY_STEP_ADD_POINTS) {
      // add the points
      Eigen::VectorXi dummyI;
      Eigen::MatrixXd dummyP;
      bool is_poised = DFO_model_.ModelImprovementAlgorithm(r * trust_region_radius_tilde, dummyP, dummyI);
      std::cout << "is_poised1 = " << is_poised << "\n";
      DFO_model_.update(new_points,
                        function_evaluations,
                        new_gradients,
                        new_points_indicies,
                        number_of_new_points,
                        DFO_Model::IMPROVE_POISEDNESS);
      is_poised = DFO_model_.ModelImprovementAlgorithm(r * trust_region_radius_tilde, new_points, new_points_indicies);
      std::cout << "is_poised2 = " << is_poised << "\n";
      set_next_step(CRITICALITY_STEP_CHECK_CONVERGENCE);
      crit_steps++;
      goto print;
    } else if (next_step_ == CRITICALITY_STEP_CHECK_CONVERGENCE) {
      //Eigen::VectorXd gradient = DFO_model_.GetGradientAtPoint(DFO_model_.GetBestPoint());
      Eigen::VectorXd gradient = DFO_model_.GetLagrangianGradient(DFO_model_.GetBestPoint());

      bool is_poised =
          DFO_model_.ModelImprovementAlgorithm(r * trust_region_radius_tilde, new_points, new_points_indicies);

      if (trust_region_radius_tilde <= u * gradient.norm() && is_poised) {
        // The radius have been reduced, and the gradient is now sufficient large. Proceed to find trial point.
        is_model_CFL = true;
        if (iterations_ >= 62) {
          std::cout << "iteration 62++\n";
        }
        double temp = max(trust_region_radius_tilde, beta * gradient.norm());
        double new_trust_region_radius = min(temp, trust_region_radius_icb);
        DFO_model_.SetTrustRegionRadius(new_trust_region_radius);
        set_next_step(FIND_TRIAL_POINT);
        std::cout << "Criticality step iterations = " << criticality_step_iteration << "\n";
        goto print;
      } else {
        do {
          //decrease radius. check poisedness.
          criticality_step_iteration++;
          trust_region_radius_tilde = pow(w, (criticality_step_iteration - 1)) * trust_region_radius_icb;
          if (trust_region_radius_tilde <= trust_region_radius_end) {
            //std::cout << "crit_steps: " << crit_steps << "\nmodel_impr: " << model_impr_steps << "\nacceptance: " << accept_steps <<"\n";
            //DFO_model_.Converged(iterations_, 0, number_of_function_calls,number_of_parallell_function_calls);
          }
          is_poised =
              DFO_model_.ModelImprovementAlgorithm(r * trust_region_radius_tilde, new_points, new_points_indicies);
        } while (is_poised);

        if (trust_region_radius_tilde <= trust_region_radius_end*0.9){
          std::cout << "crit_steps: " << crit_steps << "\nmodel_impr: " << model_impr_steps << "\nacceptance: "
                    << accept_steps << "\n";
          DFO_model_.Converged(iterations_, 0, number_of_function_calls, number_of_parallell_function_calls);
        }

        set_next_step(CRITICALITY_STEP_ADD_POINTS);
        number_of_new_points = new_points_indicies.rows();
        multiple_new_points = true;
        goto evaluate;
      }
    } else if (next_step_ == FIND_TRIAL_POINT) {

      Eigen::VectorXd gradient = DFO_model_.GetLagrangianGradient(DFO_model_.GetBestPoint());


      /// Can add a test to see if currently best found point is "bestpoint"
      //DFO_model_.UpdateOptimum();

      if (DFO_model_.GetTrustRegionRadius() <= trust_region_radius_end) {
        std::cout << "crit_steps: " << crit_steps << "\nmodel_impr: " << model_impr_steps << "\nacceptance: "
                  << accept_steps << "\n";
        DFO_model_.Converged(iterations_, 0, number_of_function_calls, number_of_parallell_function_calls);
      }
      new_point = DFO_model_.FindLocalOptimum();
      if ((new_point - last_trial_point).norm() <= 0.0001*DFO_model_.GetTrustRegionRadius()) {
        force_criticality_step = true;
        trust_region_radius_icb = gamma * DFO_model_.GetTrustRegionRadius();
        set_next_step(CRITICALITY_STEP_START);
        goto top;
      } else {
        last_trial_point = new_point;
        double maxDistance = DFO_model_.findLargestDistanceBetweenPointsAndOptimum();
        if (DFO_model_.norm((new_point - DFO_model_.GetBestPoint())) < tau * maxDistance) {
          // Too close.
          if (!is_model_CFL) {
            is_model_CFL = DFO_model_.isPoised(new_point, new_point_index, DFO_model_.GetTrustRegionRadius());
          }
          if (is_model_CFL) {
            set_next_step(TRUST_REGION_RADIUS_UPDATE_STEP);
            goto top;
          } else {
            set_next_step(MODEL_IMPROVEMENT_STEP_START);
            goto top;
          }
        } else {
          set_next_step(ACCEPTANCE_OF_TRIAL_POINT);
          number_of_new_points = 0;
          multiple_new_points = false;
          goto evaluate;
        }
      }
    } else if (next_step_ == ACCEPTANCE_OF_TRIAL_POINT) {
      accept_steps++;
      DFO_model_.SetTrustRegionRadiusForSubproblem(DFO_model_.GetTrustRegionRadius());
      int t = DFO_model_.findPointToReplaceWithNewOptimum(new_point);
      rho = (DFO_model_.GetFunctionValue(DFO_model_.getBestPointIndex()) - function_evaluation)
          / (DFO_model_.evaluateQuadraticModel(DFO_model_.GetBestPoint())
              - DFO_model_.evaluateQuadraticModel(new_point));
      std::cout << "eta1 = " << eta1 << "\trho = " << rho << "\n";

      if (!is_model_CFL) {
        Eigen::VectorXd dummyVec(number_of_variables_);
        dummyVec.setZero();
        int dummyInt = 0;
        is_model_CFL = DFO_model_.isPoised(dummyVec, dummyInt, DFO_model_.GetTrustRegionRadius());
      }

      if ((rho >= eta1) || (is_model_CFL && rho > 0)) {
        DFO_model_.update(new_point, function_evaluation, new_gradient, t, DFO_Model::INCLUDE_NEW_OPTIMUM);
        std::cout << "The gradient of the lagrangian is:\n"
                  << DFO_model_.GetLagrangianGradient(new_point + DFO_model_.getCenterPoint()) << "\n";
        set_next_step(TRUST_REGION_RADIUS_UPDATE_STEP);
        goto print;
      } else {
        int index = DFO_model_.isPointAcceptable(new_point);
        if (index != -1) {
          DFO_model_.update(new_point, function_evaluation, new_gradient, index, DFO_Model::INCLUDE_NEW_POINT);
        }
        set_next_step(MODEL_IMPROVEMENT_STEP_START);
        goto top;
      }
    } else if (next_step_ == MODEL_IMPROVEMENT_STEP_START) {
      DFO_model_.modelImprovementStep(new_point, new_point_index);
      if (new_point_index == -1) {
        rho = 0;
        std::cout << "The model improvement step was performed, BUT the model is CFL. No change done.\n";
        force_criticality_step = true;
        is_model_CFL = true;
        set_next_step(TRUST_REGION_RADIUS_UPDATE_STEP);
        goto top;
      } else {
        set_next_step(MODEL_IMPROVEMENT_STEP_END);
      }
    } else if (next_step_ == MODEL_IMPROVEMENT_STEP_END) {
      DFO_model_.update(new_point, function_evaluation, new_gradient, new_point_index, DFO_Model::INCLUDE_NEW_POINT);
      model_impr_steps++;
      set_next_step(TRUST_REGION_RADIUS_UPDATE_STEP);
      goto print;
    } else if (next_step_ == TRUST_REGION_RADIUS_UPDATE_STEP) {
      if (rho >= eta1) {
        /// if point is not better, but rho is high. lower radius!!!
        if (function_evaluation > DFO_model_.GetFunctionValue(DFO_model_.getBestPointIndex())) {
          trust_region_radius_icb = gamma * DFO_model_.GetTrustRegionRadius();
        } else {
          double tmp = std::min(gamma_inc * DFO_model_.GetTrustRegionRadius(), trust_region_radius_max);
          double weight = 0;
          trust_region_radius_icb = weight * DFO_model_.GetTrustRegionRadius() + (1 - weight) * tmp;
          rho = 0;
        }
      } else {
        if (!is_model_CFL) {
          Eigen::VectorXd dummyVec(number_of_variables_);
          dummyVec.setZero();
          int dummyInt = 0;
          is_model_CFL = DFO_model_.isPoised(dummyVec, dummyInt, DFO_model_.GetTrustRegionRadius());
        }
        if (is_model_CFL) {
          trust_region_radius_icb = gamma * DFO_model_.GetTrustRegionRadius();
        } else {
          trust_region_radius_icb = DFO_model_.GetTrustRegionRadius();
        }
      }
      set_next_step(CRITICALITY_STEP_START);
      goto top;

    } else {
      cout << "This should never be ran. \n";
      cin.get();
    }




    /// --------------------------------------------------------------------------------------------------------------------------------






    if (DFO_model_.isModelInitialized()) {
      std::cout
          << "___________________________________________________Interpolating?______________________________________________\n";
      DFO_model_.isInterpolating();
      std::cout << "__________________________________\n";
      DFO_model_.isInterpolatingLagrange();
      std::cout << "__________________________________\n";
      DFO_model_.isInterpolatingEnhanced();
      std::cout
          << "___________________________________________________Interpolating?______________________________________________\n";
      DFO_model_.printParametersMatlabFriendly();
      //DFO_model_.printParametersMatlabFriendlyFromLagrangePolynomials();

      //DFO_model_.printParametersMatlabFriendlyGradientEnhanced();

      std::cout
          << "___________________________________________________Lagrange?______________________________________________\n";
      DFO_model_.isLagrangePoly();
      std::cout
          << "___________________________________________________Lagrange?______________________________________________\n";
    }

    /*
              Get the function evaluation(s)
    */

    evaluate:

    if (multiple_new_points) {
      number_of_new_points = new_points_indicies.rows();
      new_gradients.resize(ng, number_of_new_points);
      function_evaluations.resize(number_of_new_points);
      for (int i = 0; i < number_of_new_points; ++i) {
        number_of_function_calls++;
        Eigen::VectorXd tmppp = new_points.col(i) + DFO_model_.getCenterPoint();
        Eigen::VectorXd tmpp = ScaleVariablesFromAlgorithmToApplication(tmppp);
        //tmp_eval = sphere(tmpp, ng);
        tmp_eval = evaluateFunctionVS(tmpp, ng);
        eigen_col(new_gradients, tmp_eval.tail(ng), i);
        function_evaluations(i) = tmp_eval(0);

        WritePointToFile(tmpp, 1);

      }
      number_of_parallell_function_calls++;
    } else {
      Eigen::VectorXd tmpp = ScaleVariablesFromAlgorithmToApplication(new_point + DFO_model_.getCenterPoint());

      //tmp_eval = sphere(tmpp, ng);
      tmp_eval = evaluateFunctionVS(tmpp, ng);
      function_evaluation = tmp_eval(0);
      new_gradient = tmp_eval.tail(ng);
      number_of_function_calls++;
      number_of_parallell_function_calls++;


      if (next_step_ == ACCEPTANCE_OF_TRIAL_POINT){
        WritePointToFile(tmpp, 0);
      }
      else{
        WritePointToFile(tmpp, 1);
      }
    }
    iterations_++;

  }/// End of while loop



  // QList<Case *> new_cases = ConvertPointsToCases(new_points);
  // case_handler_->AddNewCases(new_cases);

}

std::string DFO::get_action_name(int a) {
  switch (a) {
    case 0: return "FIND_POINTS1 							";
    case 1: return "FIND_POINTS2 							";
    case 2: return "INITIALIZE_MODEL 						";
    case 3: return "CRITICALITY_STEP_START 				    ";
    case 4: return "CRITICALITY_STEP_CHECK_CONVERGENCE    	";
    case 5: return "FIND_TRIAL_POINT 						";
    case 6: return "ACCEPTANCE_OF_TRIAL_POINT 				";
    case 7: return "MODEL_IMPROVEMENT_STEP_START 			";
    case 8: return "MODEL_IMPROVEMENT_STEP_END 		    	";
    case 9: return "TRUST_REGION_RADIUS_UPDATE_STEP 		";
    case 10: return "CRITICALITY_STEP_ADD_POINTS 			";
    default:return "Not a valid state                       ";
  }
}

void DFO::set_next_step(int a) {
  std::cout << "From " << "\033[1;" + color_from + "m " << get_action_name(next_step_) << "\033[0m";
  next_step_ = a;
  std::cout << "\tTo " << "\033[1;" + color_to + ";m " << get_action_name(next_step_) << "\033[0m" << std::endl;
}

Eigen::VectorXd DFO::ScaleVariablesFromAlgorithmToApplication(Eigen::VectorXd point) {
  Eigen::VectorXd ret(point.rows());
  //ret(0) = point(0);
  //ret(1) = point(1);
  //ret(2) = point(2)*100;
  return point;
}
Eigen::VectorXd DFO::ScaleVariablesFromApplicationToAlgorithm(Eigen::VectorXd point) {
  Eigen::VectorXd ret(point.rows());
  //ret(0) = point(0);
  //ret(1) = point(1);
  return point;
  //ret(2) = point(2)*10;
}

void DFO::CreateFiles(){
  std::string filename = settings_->parameters().test_problem_file;
  std::size_t botDirPos = filename.find_last_of("/");
  std::string name2 = filename.substr(botDirPos, filename.length());
  std::size_t pos2 = name2.find_last_of(".");
  std::string name = name2.substr(0, pos2);


  filenamePoint = "/home/joakim/dat_and_mod/testcases/results" + name  + "points" + "m"+ std::to_string(settings_->parameters().number_of_interpolation_points);
  filenameType = "/home/joakim/dat_and_mod/testcases/results" + name + "types"+ "m" + std::to_string(settings_->parameters().number_of_interpolation_points);
  filenameTrr = "/home/joakim/dat_and_mod/testcases/results" + name + "trr"+ "m" + std::to_string(settings_->parameters().number_of_interpolation_points);
  std::ofstream o(filenameType);
  std::ofstream o2(filenamePoint);
  std::ofstream o3(filenameTrr);

}

void DFO::WritePointToFile(Eigen::VectorXd point, int t) {
  std::fstream fs;
  fs.open(filenamePoint,std::fstream::in |std::fstream::out | std::fstream::app );

  for (int i = 0; i < settings_->parameters().number_of_variables; ++i){
    fs << point[i] << " ";
  }
  fs << std::endl;
  fs.close();

  std::fstream fs2;
  fs2.open(filenameType,std::fstream::in |std::fstream::out | std::fstream::app );

  fs2  << t << std::endl;
  fs2.close();

  std::fstream fs3;
  fs3.open(filenameTrr,std::fstream::in |std::fstream::out | std::fstream::app );

  fs3  << DFO_model_.GetTrustRegionRadius()*settings_->parameters().r << std::endl;
  fs3.close();

}
void DFO::WritePointTypeToFile(int t) {
  std::fstream fs;
  fs.open("aa.txt",std::fstream::in |std::fstream::out | std::fstream::app );

  fs  << t << "\n";
  fs.close();
}

double matyasFunction2D(Eigen::VectorXd x) {
  double squaredx1 = std::pow(x(0), 2);
  double squaredx2 = std::pow(x(1), 2);
  return 0.26 * (squaredx1 + squaredx2) - 0.48 * x(0) * x(1);
}
double rosenbrockFunction2D(Eigen::VectorXd x) {
  double a = std::pow(x(0), 2);
  double b = std::pow(x(1) - a, 2);
  double c = std::pow(1 - x(0), 2);
  return 100 * b + c;
}
double polynomial6(Eigen::VectorXd y) {
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(y.rows());
  Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(y.rows(), y.rows());
  double c = 140;

  grad << 1, 2, 3, 4, 5, 6;
  hess <<
       3, 0, 0, 5, 0, 0,
      0, 4, 0, 0, 4, 88,
      0, 0, 1, 77, 0, 0,
      5, 0, 77, 2, 0, 6,
      0, 4, 0, 0, 7, 0,
      0, 88, 0, 6, 0, 9;

  return c + grad.transpose() * y + 0.5 * y.transpose() * hess * y;

}
double polynomial5(Eigen::VectorXd y) {
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(y.rows());
  Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(y.rows(), y.rows());
  double c = 140;

  grad << 1, 2, 3, 4, 5;
  hess <<
       3, 0, 0, 5, 0,
      0, 4, 0, 0, 4,
      0, 0, 1, 77, 0,
      5, 0, 77, 2, 0,
      0, 4, 0, 0, 7;

  return c + grad.transpose() * y + 0.5 * y.transpose() * hess * y;

}
double polynomial3(Eigen::VectorXd y) {
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(y.rows());
  Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(y.rows(), y.rows());
  double c = 0;

  grad << 5, 1, 92;
  hess <<
       3, -1, 10,
      -1, 4, 7,
      10, 7, 1;

  return c + grad.transpose() * y + 0.5 * y.transpose() * hess * y;

}
double polynomial32(Eigen::VectorXd y) {
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(y.rows());
  Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(y.rows(), y.rows());
  double c = 0;

  grad << 5, 1, 92;
  hess <<
       3, -1, 10,
      -1, 4, 7,
      10, 7, 1;

  return c + grad.transpose() * y + 0.5 * y.transpose() * hess * y + y(1) * y(2) * y(0) + y(0) * y(0) * y(1) * y(1)
      + y(2) * y(2) * y(2) * y(2) * y(2) + y(1) * y(1) * y(1) * y(1);

}
}
}

