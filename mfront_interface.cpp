/**
 * @file mfront_interface.cpp
 * @brief
 * @date 2023-01-25
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <MoFEM.hpp>

static char help[] = "mfront --obuild --interface=generic BEHAVIOUR.mfront \n";
//  mfront -query-- material-properties BEHAVIOUR.mfront

using namespace MoFEM;
using namespace FTensor;

#include <BasicFiniteElements.hpp>
#include <quad.h>

#include <MGIS/Behaviour/Behaviour.hxx>
#include <MGIS/Behaviour/BehaviourData.hxx>
#include "MGIS/Behaviour/MaterialDataManager.h"
#include "MGIS/Behaviour/Integrate.hxx"
#include "MGIS/LibrariesManager.hxx"

using namespace mgis;
using namespace mgis::behaviour;

#include <Operators.hpp>

using namespace MFrontInterface;

#include <ElasticMaterials.hpp>
#include <NonlinearElasticElementInterface.hpp>

#include <BasicBoundaryConditionsInterface.hpp>
#include <SurfacePressureComplexForLazy.hpp>

// #ifdef WITH_MODULE_MFRONT_INTERFACE
#include <MFrontMoFEMInterface.hpp>
// #endif

using Ele = ForcesAndSourcesCore;
using EntData = EntitiesFieldData::EntData;
using DomainEle = VolumeElementForcesAndSourcesCore;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEle = FaceElementForcesAndSourcesCore;
using BoundaryEleOp = BoundaryEle::UserDataOperator;
using PostProcEle = PostProcVolumeOnRefinedMesh;
using PostProcSkinEle = PostProcFaceOnRefinedMesh;
using SetPtsData = FieldEvaluatorInterface::SetPtsData;

int main(int argc, char *argv[]) {

  const string default_options = "-ksp_type fgmres \n"
                                 "-pc_type lu \n"
                                 "-pc_factor_mat_solver_type mumps \n"
                                 "-ksp_atol 1e-10 \n"
                                 "-ksp_rtol 1e-10 \n"
                                 "-snes_monitor \n"
                                 "-snes_max_it 100 \n"
                                 "-snes_linesearch_type bt \n"
                                 "-snes_linesearch_max_it 3 \n"
                                 "-snes_atol 1e-8 \n"
                                 "-snes_rtol 1e-8 \n"
                                 "-ts_monitor \n"
                                 "-ts_alpha_radius 1 \n"
                                 "-mat_mumps_icntl_20 0 \n"
                                 "-mat_mumps_icntl_14 1200 \n"
                                 "-mat_mumps_icntl_24 1 \n"
                                 "-mat_mumps_icntl_13 1 \n";

  string param_file = "param_file.petsc";
  if (!static_cast<bool>(ifstream(param_file))) {
    std::ofstream file(param_file.c_str(), std::ios::ate);
    if (file.is_open()) {
      file << default_options;
      file.close();
    }
  }

  MoFEM::Core::Initialize(&argc, &argv, param_file.c_str(), help);

  // Add logging channel for example
  auto core_log = logging::core::get();
  core_log->add_sink(LogManager::createSink(LogManager::getStrmWorld(),
                                            "MoFEM_MFront_Interface"));
  LogManager::setLog("MoFEM_MFront_Interface");
  MOFEM_LOG_TAG("MoFEM_MFront_Interface", "module_manager");

  core_log->add_sink(
      LogManager::createSink(LogManager::getStrmSync(), "ATOM_TEST"));
  LogManager::setLog("ATOM_TEST");
  MOFEM_LOG_TAG("ATOM_TEST", "atom_test");

  try {

    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    int order;
    int save_every_nth_step = 1;
    PetscBool is_quasi_static = PETSC_TRUE;
    PetscBool is_partitioned = PETSC_TRUE;

    int atom_test = -1;
    std::vector<std::pair<std::pair<double, std::vector<double>>, bool>>
        atom_test_data;
    double atom_test_threshold = 1;

    PetscBool field_eval_flag = PETSC_FALSE;
    std::array<double, 3> field_eval_coords;
    boost::shared_ptr<SetPtsData> field_eval_data;

    SmartPetscObj<TS> tSolver;
    SmartPetscObj<DM> dM;

    boost::shared_ptr<std::vector<unsigned char>> boundaryMarker;
    boost::shared_ptr<FEMethod> monitor_ptr;

    CHKERR PetscOptionsGetBool(PETSC_NULL, "-is_quasi_static", &is_quasi_static,
                               PETSC_NULL);
    CHKERR PetscOptionsGetInt(PETSC_NULL, "-order", &order, PETSC_NULL);
    CHKERR PetscOptionsGetInt(PETSC_NULL, "-output_every", &save_every_nth_step,
                              PETSC_NULL);
    CHKERR PetscOptionsGetInt(PETSC_NULL, "-atom_test", &atom_test, PETSC_NULL);
    int dim = 3;
    CHKERR PetscOptionsGetRealArray(NULL, NULL, "-field_eval_coords",
                                    field_eval_coords.data(), &dim,
                                    &field_eval_flag);

    CHKERR PetscOptionsGetBool(PETSC_NULL, "-is_partitioned", &is_partitioned,
                               PETSC_NULL);
    MOFEM_LOG("WORLD", Sev::inform)
        << "Mesh Partition Flag Status: " << is_partitioned;

    switch (atom_test) {
    case 1:
      atom_test_data = {{{0.1, {-0.1101}}, false}, {{0.2, {-0.2246}}, false},
                        {{0.3, {-0.3380}}, false}, {{0.4, {-0.4528}}, false},
                        {{0.5, {-0.5769}}, false}, {{0.6, {-0.7539}}, false},
                        {{0.7, {-1.1205}}, false}, {{0.8, {-1.5959}}, false},
                        {{0.9, {-2.1240}}, false}, {{1.0, {-2.6948}}, false}};
      atom_test_threshold = 3e-3;
      break;
    case 2:
      atom_test_data = {{{0.14, {0.0855}}, false}, {{0.28, {0.1706}}, false},
                        {{0.42, {0.2612}}, false}, {{0.56, {0.3847}}, false},
                        {{0.70, {0.6871}}, false}, {{0.84, {1.1362}}, false},
                        {{0.98, {1.6878}}, false}};
      // FIXME: these times cannot be reached with MGIS 2.0
      // {{1.12, {2.3067}}, false},
      // {{1.26, {2.8729}}, false},
      // {{1.40, {3.2957}}, false}};
      atom_test_threshold = 6e-2;
      break;
    case 3:
      atom_test_data = {
          {{0.01, {0.001157, 100}}, false}, {{0.02, {0.001196, 100}}, false},
          {{0.04, {0.001258, 100}}, false}, {{0.06, {0.001304, 100}}, false},
          {{0.08, {0.001337, 100}}, false}, {{0.10, {0.001362, 100}}, false},
          {{0.14, {0.001393, 100}}, false}, {{0.16, {0.001402, 100}}, false},
          {{0.20, {0.001414, 100}}, false}, {{0.50, {0.001428, 100}}, false}};
      atom_test_threshold = 1e-2;
      break;
    case 4:
      break;
    case 5:
      break;
    default:
      if (atom_test > -1)
        SETERRQ1(PETSC_COMM_WORLD, MOFEM_NOT_IMPLEMENTED,
                 "Atom test number %d is not yet implemented", atom_test);
      break;
    }

    Simple *simple = m_field.getInterface<Simple>();
    CHKERR simple->getOptions();

    if (is_partitioned)
      CHKERR simple->loadFile("");
    else
      CHKERR simple->loadFile("", "");

    simple->getProblemName() = "MoFEM MFront Interface module";
    simple->getDomainFEName() = "MFRONT_EL";

    FieldApproximationBase base = AINSWORTH_LEGENDRE_BASE;
    // Add displacement field
    CHKERR m_field.add_field("U", H1, base, 3);

    // Add field representing ho-geometry
    CHKERR m_field.add_field("MESH_NODE_POSITIONS", H1, base, 3);

    // Add entities to field
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "U");
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "MESH_NODE_POSITIONS");
    CHKERR m_field.set_field_order(0, MBVERTEX, "U", 1);
    CHKERR m_field.set_field_order(0, MBEDGE, "U", order);
    CHKERR m_field.set_field_order(0, MBTRI, "U", order);
    CHKERR m_field.set_field_order(0, MBTET, "U", order);

    CHKERR m_field.set_field_order(0, MBVERTEX, "MESH_NODE_POSITIONS", 1);
    // CHKERR m_field.set_field_order(0, MBEDGE, "MESH_NODE_POSITIONS", 2);
    CHKERR m_field.set_field_order(0, MBEDGE, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBTRI, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBTET, "MESH_NODE_POSITIONS", 1);

    // setup the modules
    boost::ptr_vector<GenericElementInterface> m_modules;

    // Basic Boundary Conditions module should always be first (dirichlet)
    m_modules.push_back(new BasicBoundaryConditionsInterface(
        m_field, "U", "MESH_NODE_POSITIONS", simple->getProblemName(),
        simple->getDomainFEName(), true, is_quasi_static, nullptr,
        is_partitioned));

    m_modules.push_back(new NonlinearElasticElementInterface(
        m_field, "U", "MESH_NODE_POSITIONS", true, is_quasi_static));

    m_modules.push_back(new MFrontMoFEMInterface(
        m_field, "U", "MESH_NODE_POSITIONS", true, is_quasi_static));

    for (auto &&mod : m_modules) {
      CHKERR mod.getCommandLineParameters();
      CHKERR mod.addElementFields();
      CHKERR mod.createElements();
    }

    CHKERR m_field.build_fields();

    DMType dm_name = "DMMOFEM";
    // Register DM problem
    CHKERR DMRegister_MoFEM(dm_name);
    dM = createSmartDM(m_field.get_comm(), dm_name);
    CHKERR DMSetType(dM, dm_name);
    CHKERR DMMoFEMSetIsPartitioned(dM, is_partitioned);
    // Create DM instance
    CHKERR DMMoFEMCreateMoFEM(dM, &m_field, simple->getProblemName().c_str(),
                              simple->getBitRefLevel());
    CHKERR DMSetFromOptions(dM);

    for (auto &&mod : m_modules)
      CHKERR mod.addElementsToDM(dM);

    Projection10NodeCoordsOnField ent_method(m_field, "MESH_NODE_POSITIONS");
    CHKERR m_field.loop_dofs("MESH_NODE_POSITIONS", ent_method);
    CHKERR m_field.build_finite_elements();
    CHKERR m_field.build_adjacencies(simple->getBitRefLevel());

    CHKERR DMSetUp(dM);

    boost::shared_ptr<MatrixDouble> field_ptr;
    if (field_eval_flag) {
      field_eval_data =
          m_field.getInterface<FieldEvaluatorInterface>()->getData<DomainEle>();
      CHKERR m_field.getInterface<FieldEvaluatorInterface>()->buildTree3D(
          field_eval_data, simple->getDomainFEName());
      field_eval_data->setEvalPoints(field_eval_coords.data(), 1);

      auto no_rule = [](int, int, int) { return -1; };

      auto fe_ptr = field_eval_data->feMethodPtr.lock();
      fe_ptr->getRuleHook = no_rule;

      field_ptr = boost::make_shared<MatrixDouble>();
      fe_ptr->getOpPtrVector().push_back(
          new OpCalculateVectorFieldValues<3>("U", field_ptr));
    }

    monitor_ptr = boost::make_shared<FEMethod>();
    monitor_ptr->preProcessHook = [&]() { 
      MoFEMFunctionBegin;
      t_dt = monitor_ptr->ts_dt;
      MoFEMFunctionReturn(0);
      };
    monitor_ptr->operatorHook = []() { return 0; };
    monitor_ptr->postProcessHook = [&]() {
      MoFEMFunctionBegin;
      
      auto ts_time = monitor_ptr->ts_t;
      auto ts_step = monitor_ptr->ts_step;

      for (auto &&mod : m_modules) {
        CHKERR mod.updateElementVariables();
        if (ts_step % save_every_nth_step == 0)
          CHKERR mod.postProcessElement(ts_step);
      }

      for (auto &it : atom_test_data) {
        if (fabs(ts_time - it.first.first) < 1e-3) {
          it.second = true;

          if (field_eval_flag) {
            CHKERR m_field.getInterface<FieldEvaluatorInterface>()
                ->evalFEAtThePoint3D(
                    field_eval_coords.data(), 1e-12, simple->getProblemName(),
                    simple->getDomainFEName(), field_eval_data,
                    m_field.get_comm_rank(), m_field.get_comm_rank(), nullptr,
                    MF_EXIST, QUIET);
          }

          double rel_dif = 0;
          switch (atom_test) {
          case 1:
            if (field_ptr->size1()) {
              auto t_p = getFTensor1FromMat<3>(*field_ptr);
              rel_dif =
                  fabs(it.first.second[0] - t_p(0)) / fabs(it.first.second[0]);
              MOFEM_LOG("ATOM_TEST", Sev::verbose)
                  << "Relative difference: " << rel_dif;
            }
            break;
          case 2:
            if (field_ptr->size1()) {
              auto t_p = getFTensor1FromMat<3>(*field_ptr);
              rel_dif =
                  fabs(it.first.second[0] - t_p(1)) / fabs(it.first.second[0]);
              MOFEM_LOG("ATOM_TEST", Sev::verbose)
                  << "Relative difference: " << rel_dif;
            }
            break;
          case 3: {
            double eps_rel_dif =
                fabs(it.first.second[0] -
                     *getGradient(
                         commonDataPtr->setOfBlocksData[1].behDataPtr->s1, 1)) /
                fabs(it.first.second[0]);
            double sig_rel_dif =
                fabs(it.first.second[1] -
                     *getThermodynamicForce(
                         commonDataPtr->setOfBlocksData[1].behDataPtr->s1, 1)) /
                fabs(it.first.second[1]);
            rel_dif = (eps_rel_dif > sig_rel_dif) ? eps_rel_dif : sig_rel_dif;
            MOFEM_LOG("WORLD", Sev::verbose)
                << "Relative difference eps: " << eps_rel_dif
                << " sig: " << sig_rel_dif;
          } break;
          default:
            break;
          }

          MOFEM_LOG_SYNCHRONISE(m_field.get_comm());

          if (rel_dif > atom_test_threshold) {
            SETERRQ2(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID,
                     "Atom test failed for time %1.2f: relative difference is "
                     "greater than %0.3f",
                     it.first.first, atom_test_threshold);
          }

          break;
        }
      }
      MoFEMFunctionReturn(0);
    };

    auto t_type = GenericElementInterface::IM2;
    if (is_quasi_static)
      t_type = GenericElementInterface::IM;
    for (auto &&mod : m_modules) {
      CHKERR mod.setOperators();
      CHKERR mod.setupSolverFunctionTS(t_type);
      CHKERR mod.setupSolverJacobianTS(t_type);
    }

    auto set_time_monitor = [&](auto solver) {
      MoFEMFunctionBegin;
      boost::shared_ptr<ForcesAndSourcesCore> null;
      CHKERR DMMoFEMTSSetMonitor(dM, solver, simple->getDomainFEName(),
                                 monitor_ptr, null, null);
      MoFEMFunctionReturn(0);
    };

    auto set_dm_section = [&](auto dm) {
      MoFEMFunctionBeginHot;
      auto section = m_field.getInterface<ISManager>()->sectionCreate(
          simple->getProblemName());
      CHKERR DMSetSection(dm, section);
      MoFEMFunctionReturnHot(0);
    };

    CHKERR set_dm_section(dM);

    auto D = smartCreateDMVector(dM);
    tSolver = MoFEM::createTS(m_field.get_comm());

    if (is_quasi_static) {
      CHKERR TSSetSolution(tSolver, D);
    } else {
      CHKERR TSSetType(tSolver, TSALPHA2);
      auto DD = smartVectorDuplicate(D);
      CHKERR TS2SetSolution(tSolver, D, DD);
    }

    CHKERR TSSetDM(tSolver, dM);

    // default max time is 1
    CHKERR TSSetMaxTime(tSolver, 1.0);
    CHKERR TSSetExactFinalTime(tSolver, TS_EXACTFINALTIME_MATCHSTEP);
    CHKERR TSSetFromOptions(tSolver);

    CHKERR set_time_monitor(tSolver);

    CHKERR TSSetUp(tSolver);
    CHKERR TSSolve(tSolver, NULL);

    switch (atom_test) {
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
      for (auto it : atom_test_data) {
        if (!it.second) {
          SETERRQ1(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID,
                   "Atom test failed: output for time %1.2f was not observed",
                   it.first.first);
        }
      }
      break;
    default:
      break;
    }
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();
  return 0;
}