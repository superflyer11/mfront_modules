
#include <MoFEM.hpp>

static char help[] = "mfront --obuild --interface=generic BEHAVIOUR.mfront \n";
//  mfront -query-- material-properties BEHAVIOUR.mfront

using namespace MoFEM;
using namespace FTensor;

using EntData = DataForcesAndSourcesCore::EntData;
using DomainEle = VolumeElementForcesAndSourcesCore;
using DomainEleOp = DomainEle::UserDataOperator;

#include <BasicFiniteElements.hpp>

// #ifdef WITH_MFRONT
#include <MGIS/Behaviour/Behaviour.hxx>
#include <MGIS/Behaviour/BehaviourData.hxx>
#include "MGIS/Behaviour/MaterialDataManager.h"
#include "MGIS/Behaviour/Integrate.hxx"

using namespace mgis;
using namespace mgis::behaviour;

#include <Operators.hpp>

using namespace MFrontInterface;

int main(int argc, char *argv[]) {
  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);

    moab::Core mb_instance;              ///< mesh database
    moab::Interface &moab = mb_instance; ///< mesh database interface

    MoFEM::Core core(moab);           ///< finite element database
    MoFEM::Interface &m_field = core; ///< finite element database insterface

    Simple *simple = m_field.getInterface<Simple>();
    CHKERR simple->getOptions();
    CHKERR simple->loadFile("");
    int order = 2;

    auto get_options_from_command_line = [&]() {
      MoFEMFunctionBeginHot;
      CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "", "none");
      CHKERR PetscOptionsInt("-order", "approximation order", "", order, &order,
                             PETSC_NULL);
      ierr = PetscOptionsEnd();
      CHKERRQ(ierr);
      MoFEMFunctionReturnHot(0);
    };

    CHKERR get_options_from_command_line();

    // Add field
    CHKERR simple->addDomainField("U", H1, AINSWORTH_LEGENDRE_BASE, 3);
    // CHKERR simple->addDomainField("U", H1, AINSWORTH_BERNSTEIN_BEZIER_BASE,
    // 3);
    CHKERR simple->addBoundaryField("U", H1, AINSWORTH_LEGENDRE_BASE, 3);
    CHKERR simple->setFieldOrder("U", order);
    PetscBool is_partitioned = PETSC_TRUE;

    CHKERR simple->defineFiniteElements();

    // Add Neumann forces elements
    CHKERR MetaNeumannForces::addNeumannBCElements(m_field, "U");
    CHKERR MetaNodalForces::addElement(m_field, "U");
    CHKERR MetaEdgeForces::addElement(m_field, "U");

    simple->getOtherFiniteElements().push_back("FORCE_FE");
    simple->getOtherFiniteElements().push_back("PRESSURE_FE");

    CHKERR simple->defineProblem(is_partitioned);
    CHKERR simple->buildFields();
    CHKERR simple->buildFiniteElements();
    CHKERR simple->buildProblem();

    boost::shared_ptr<CommonData> commonDataPtr;
    boost::shared_ptr<PostProcVolumeOnRefinedMesh> postProcFe;
    // mofem boundary conditions
    boost::ptr_map<std::string, NeumannForcesSurface> neumann_forces;
    boost::ptr_map<std::string, NodalForce> nodal_forces;
    boost::ptr_map<std::string, EdgeForce> edge_forces;
    boost::shared_ptr<DirichletDisplacementBc> dirichlet_bc_ptr;
    boost::shared_ptr<DomainEle> update_int_variables;

    commonDataPtr = boost::make_shared<CommonData>(m_field);
    commonDataPtr->setBlocks();
    commonDataPtr->createTags();

    commonDataPtr->mGradPtr = boost::make_shared<MatrixDouble>();
    commonDataPtr->mStressPtr = boost::make_shared<MatrixDouble>();
    commonDataPtr->materialTangent = boost::make_shared<MatrixDouble>();
    commonDataPtr->internalVariablePtr = boost::make_shared<MatrixDouble>();

    commonDataPtr->mGisBehaviour = boost::make_shared<Behaviour>(
        load("src/libBehaviour.so", "IsotropicLinearHardeningPlasticity",
             Hypothesis::TRIDIMENSIONAL));

    cout << "Parameters for this behaviour are: \n";
    for (const auto &mp : commonDataPtr->mGisBehaviour->mps) {
      cerr << mp.name << endl;
    }

    cout << "Parameters for this behaviour are: \n";
    for (auto &p : commonDataPtr->mGisBehaviour->params)
      cout << p << endl;
    // FIXME: TODO: HARD CODE PARAMETERS

    commonDataPtr->setOfBlocksData.begin()->second.params[0] = 10;
    commonDataPtr->setOfBlocksData.begin()->second.params[1] = 0.3;
    commonDataPtr->setOfBlocksData.begin()->second.params[2] = 0.1;
    commonDataPtr->setOfBlocksData.begin()->second.params[3] = 10000;
    commonDataPtr->setOfBlocksData.begin()->second.params[4] = 100;

    update_int_variables =
        boost::make_shared<DomainEle>(m_field);
    auto integration_rule = [&](int, int, int approx_order) {
      return 2 * order + 1;
    };
    update_int_variables->getRuleHook = integration_rule;
    update_int_variables->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<3, 3>("U", commonDataPtr->mGradPtr));
    for (auto &sit : commonDataPtr->setOfBlocksData)
      update_int_variables->getOpPtrVector().push_back(
          new OpStress<true>("U", commonDataPtr, sit.second));

    // forces and pressures on surface
    CHKERR MetaNeumannForces::setMomentumFluxOperators(m_field, neumann_forces,
                                                       PETSC_NULL, "U");
    // nodal forces
    CHKERR MetaNodalForces::setOperators(m_field, nodal_forces, PETSC_NULL,
                                         "U");
    // edge forces
    CHKERR MetaEdgeForces::setOperators(m_field, edge_forces, PETSC_NULL, "U");

    for (auto mit = neumann_forces.begin(); mit != neumann_forces.end();
         mit++) {
      mit->second->methodsOp.push_back(
          new TimeForceScale("-load_history", true));
    }
    for (auto mit = nodal_forces.begin(); mit != nodal_forces.end(); mit++) {
      mit->second->methodsOp.push_back(
          new TimeForceScale("-load_history", true));
    }
    for (auto mit = edge_forces.begin(); mit != edge_forces.end(); mit++) {
      mit->second->methodsOp.push_back(
          new TimeForceScale("-load_history", true));
    }

    dirichlet_bc_ptr =
        boost::make_shared<DirichletDisplacementBc>(m_field, "U");
    // dirichlet_bc_ptr->dIag = 1;
    dirichlet_bc_ptr->methodsOp.push_back(
        new TimeForceScale("-load_history", true));

    PipelineManager *pipeline_mng = m_field.getInterface<PipelineManager>();
    auto add_domain_base_ops = [&](auto &pipeline) {
      pipeline.push_back(new OpCalculateVectorFieldGradient<3, 3>(
          "U", commonDataPtr->mGradPtr));
    };

    auto add_domain_ops_lhs = [&](auto &pipeline) {
      for (auto &sit : commonDataPtr->setOfBlocksData)
        pipeline.push_back(
            new OpAssembleLhs("U", "U", commonDataPtr, sit.second));
    };

    auto add_domain_ops_rhs = [&](auto &pipeline) {
      for (auto &sit : commonDataPtr->setOfBlocksData)
        pipeline.push_back(new OpStress<false>("U", commonDataPtr, sit.second));
      pipeline.push_back(new OpAssembleRhs("U", commonDataPtr));
    };

    add_domain_base_ops(pipeline_mng->getOpDomainLhsPipeline());
    add_domain_base_ops(pipeline_mng->getOpDomainRhsPipeline());
    add_domain_ops_lhs(pipeline_mng->getOpDomainLhsPipeline());
    add_domain_ops_rhs(pipeline_mng->getOpDomainRhsPipeline());

    CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
    CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);

    ISManager *is_manager = m_field.getInterface<ISManager>();

    // auto solver = pipeline_mng->createTS();
    auto create_custom_ts = [&]() {
      auto set_dm_section = [&](auto dm) {
        MoFEMFunctionBegin;
        PetscSection section;
        CHKERR m_field.getInterface<ISManager>()->sectionCreate(
            simple->getProblemName(), &section);
        CHKERR DMSetDefaultSection(dm, section);
        CHKERR DMSetDefaultGlobalSection(dm, section);
        CHKERR PetscSectionDestroy(&section);
        MoFEMFunctionReturn(0);
      };

      auto dm = simple->getDM();
      CHKERR set_dm_section(dm);

      boost::shared_ptr<FEMethod> null;
      auto preProc = boost::make_shared<FePrePostProcess>();
      preProc->methodsOp.push_back(new TimeForceScale("-load_history", true));

      // Add element to calculate lhs of stiff part
      if (pipeline_mng->getDomainLhsFE()) {

        CHKERR DMMoFEMTSSetIJacobian(dm, simple->getDomainFEName(), null,
                                     preProc, null);
        CHKERR DMMoFEMTSSetIJacobian(dm, simple->getDomainFEName(), null,
                                     dirichlet_bc_ptr, null);

        CHKERR DMMoFEMTSSetIJacobian(dm, simple->getDomainFEName(),
                                     pipeline_mng->getDomainLhsFE(), null,
                                     null);
        CHKERR DMMoFEMTSSetIJacobian(dm, simple->getDomainFEName(), null, null,
                                     dirichlet_bc_ptr);
        CHKERR DMMoFEMTSSetIJacobian(dm, simple->getDomainFEName(), null, null,
                                     preProc);
      }
      if (pipeline_mng->getBoundaryLhsFE())
        CHKERR DMMoFEMTSSetIJacobian(dm, simple->getBoundaryFEName(),
                                     pipeline_mng->getBoundaryLhsFE(), null,
                                     null);
      if (pipeline_mng->getSkeletonLhsFE())
        CHKERR DMMoFEMTSSetIJacobian(dm, simple->getSkeletonFEName(),
                                     pipeline_mng->getSkeletonLhsFE(), null,
                                     null);

      // Add element to calculate rhs of stiff part
      if (pipeline_mng->getDomainRhsFE()) {

        // add dirichlet boundary conditions
        CHKERR DMMoFEMTSSetIFunction(dm, simple->getDomainFEName(), null,
                                     preProc, null);
        CHKERR DMMoFEMTSSetIFunction(dm, simple->getDomainFEName(), null,
                                     dirichlet_bc_ptr, null);

        CHKERR DMMoFEMTSSetIFunction(dm, simple->getDomainFEName(),
                                     pipeline_mng->getDomainRhsFE(), null,
                                     null);

        // Add surface forces

        for (auto fit = neumann_forces.begin(); fit != neumann_forces.end();
             fit++) {
          CHKERR DMMoFEMTSSetIFunction(dm, fit->first.c_str(),
                                       &fit->second->getLoopFe(), NULL, NULL);
        }

        // Add edge forces
        for (auto fit = edge_forces.begin(); fit != edge_forces.end(); fit++) {
          cerr << fit->first.c_str() << endl;
          CHKERR DMMoFEMTSSetIFunction(dm, fit->first.c_str(),
                                       &fit->second->getLoopFe(), NULL, NULL);
        }

        // Add nodal forces
        for (auto fit = nodal_forces.begin(); fit != nodal_forces.end();
             fit++) {
          CHKERR DMMoFEMTSSetIFunction(dm, fit->first.c_str(),
                                       &fit->second->getLoopFe(), NULL, NULL);
        }

        CHKERR DMMoFEMTSSetIFunction(dm, simple->getDomainFEName(), null, null,
                                     dirichlet_bc_ptr);
        CHKERR DMMoFEMTSSetIFunction(dm, simple->getDomainFEName(), null, null,
                                     preProc);
      }
      if (pipeline_mng->getBoundaryRhsFE())
        CHKERR DMMoFEMTSSetIFunction(dm, simple->getBoundaryFEName(),
                                     pipeline_mng->getBoundaryRhsFE(), null,
                                     null);
      if (pipeline_mng->getSkeletonRhsFE())
        CHKERR DMMoFEMTSSetIFunction(dm, simple->getSkeletonFEName(),
                                     pipeline_mng->getSkeletonRhsFE(), null,
                                     null);

      auto ts = MoFEM::createTS(m_field.get_comm());
      CHKERR TSSetDM(ts, dm);
      return ts;
    };

    auto solver = create_custom_ts();

    CHKERR TSSetExactFinalTime(solver, TS_EXACTFINALTIME_STEPOVER);
    auto dm = simple->getDM();
    auto D = smartCreateDMVector(dm);

    CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_FORWARD);

    CHKERR TSSetSolution(solver, D);
    CHKERR TSSetFromOptions(solver);
    CHKERR TSSetUp(solver);

    auto set_section_monitor = [&]() {
      MoFEMFunctionBegin;
      SNES snes;
      CHKERR TSGetSNES(solver, &snes);
      PetscViewerAndFormat *vf;
      CHKERR PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD,
                                        PETSC_VIEWER_DEFAULT, &vf);
      CHKERR SNESMonitorSet(
          snes,
          (MoFEMErrorCode(*)(SNES, PetscInt, PetscReal,
                             void *))SNESMonitorFields,
          vf, (MoFEMErrorCode(*)(void **))PetscViewerAndFormatDestroy);
      MoFEMFunctionReturn(0);
    };

    auto create_post_process_element = [&]() {
      MoFEMFunctionBegin;
      // postProcFe =
      // boost::make_shared<PostProcFaceOnRefinedMeshFor2D>(m_field);
      postProcFe = boost::make_shared<PostProcVolumeOnRefinedMesh>(m_field);
      postProcFe->generateReferenceElementMesh();

      postProcFe->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>("U",
                                                   commonDataPtr->mGradPtr));
      for (auto &sit : commonDataPtr->setOfBlocksData)
        postProcFe->getOpPtrVector().push_back(new OpPostProcElastic(
            "U", postProcFe->postProcMesh, postProcFe->mapGaussPts,
            commonDataPtr, sit.second));

      postProcFe->addFieldValuesPostProc("U", "DISPLACEMENT");
      MoFEMFunctionReturn(0);
    };

    auto set_time_monitor = [&]() {
      MoFEMFunctionBegin;
      boost::shared_ptr<Monitor> monitor_ptr(
          new Monitor(dm, postProcFe, update_int_variables));
      boost::shared_ptr<ForcesAndSourcesCore> null;
      CHKERR DMMoFEMTSSetMonitor(dm, solver, simple->getDomainFEName(),
                                 monitor_ptr, null, null);
      MoFEMFunctionReturn(0);
    };

    // CHKERR set_section_monitor();
    CHKERR create_post_process_element();
    CHKERR set_time_monitor();

    CHKERR TSSolve(solver, D);

    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

    // auto odd = FiniteStrainBehaviourOptions{};
    // odd.stress_measure = FiniteStrainBehaviourOptions::PK1;
    // odd.tangent_operator = FiniteStrainBehaviourOptions::DPK1_DF;

    // const auto mgis_bv =
    //     // load("src/libBehaviour.so", "Plasticity",
    //     load("src/libBehaviour.so", "Elasticity",
    // //          Hypothesis::TRIDIMENSIONAL);
    // auto &mgis_bv = *commonDataPtr->mGisBehaviour;
    // auto beh_data = BehaviourData{mgis_bv};

    // auto check_this = mgis_bv.gradients.size();
    // vector<Variable> test_params;
    // // mgis_bv.mps = &*test_params.begin();
    // mgis_bv.mps;
    // for (const auto &mp : mgis_bv.mps) {
    //   cerr << mp.name << endl;
    // }
    // beh_data.K[0] = 4; // consistent tangent
    // beh_data.K[1] = 2; // first Piola stress
    // beh_data.K[2] = 2; // dP / dF derivative
    // cerr << beh_data.K << endl;
    // // const auto offset1 = getVariableOffset(mgis_bv.isvs,
    // // "EquivalentPlasticStrain",
    // //                                  mgis_bv.hypothesis);
    // // const auto offset2 = getVariableOffset(mgis_bv.isvs, "ElasticStrain",
    // //                                  mgis_bv.hypothesis);

    // // MaterialDataManager mat{mgis_bv, 50};

    // // std::cout << "member s0 " << m.s0 << '\n';
    // // const auto nb = getArraySize(mgis_bv.isvs, mgis_bv.hypothesis);
    // // std::cout << "offset of EquivalentPlasticStrain: " << offset1 << '\n';
    // // std::cout << "offset of ElasticStrain: " << offset2 << '\n';
    // // // get these sizes to initialize the size of internal and external
    // // variables std::cout << "array size " << nb << '\n';
    // auto b_view = make_view(beh_data);
    // vector<double> my_material_parameters(5);
    // my_material_parameters[0] = 100;
    // my_material_parameters[1] = 0.2;
    // my_material_parameters[2] = .2;
    // // mgis_bv.params["YoungModulus"] = 200;
    // // auto check_params = beh_data.getParameters();
    // // beh_data.getMaterialProperties();
    // vector<double> my_test_gradients(14, 0.);
    // // for (auto &it : my_test_gradients)
    // //   it = (double)rand() * 0.2 / (double)RAND_MAX;

    // b_view.s0.gradients = &*my_test_gradients.begin();
    // b_view.s0.material_properties = my_material_parameters.data();
    // b_view.s1.material_properties = my_material_parameters.data();

    // cout << "Parameters for this behaviour are: \n";
    // for (auto &p : mgis_bv.params)
    //   cout << p << endl;
    // int check = integrate(b_view, mgis_bv);

    // cerr << beh_data.K << endl;
    // std::cout << "check " << check << '\n';
    // auto set_mat_props = [&](auto &s) {
    //   setMaterialProperty(s, "YoungModulus", 100);
    //   setMaterialProperty(s, "PoissonRatio", .3);
    //   setMaterialProperty(s, "HardeningSlope", 50);
    //   setMaterialProperty(s, "YieldStrength", 10);
    //   setExternalStateVariable(s, "Temperature", 293.15);
    // };

    // set_mat_props(b_view.s0);
    // set_mat_props(mat.s0);
    // set_mat_props(mat.s1);
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();
}
// #endif