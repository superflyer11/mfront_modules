
namespace MFrontInterface {

constexpr double sqr2 = boost::math::constants::root_two<double>();
constexpr double inv_sqr2 = boost::math::constants::half_root_two<double>();

struct BlockData {
  int iD;
  int locID;
  int oRder;
  string behaviourPath;
  string behaviourName;
  array<double, 15> params;
  Range tEts;
  BlockData()
      : oRder(-1), behaviourPath("src/libBehaviour.so"),
        behaviourName("IsotropicLinearHardeningPlasticity") {}
};

template <typename T>
inline size_t get_paraview_size(T &vsize) {
  return vsize > 1 ? (vsize > 3 ? 9 : 3) : 1;
};

template <typename T>
inline auto integrateMGIS(BehaviourDataView &b_view,
                          mgis::behaviour::Behaviour &mgis_bv, T &t_grad,
                          MatrixDouble &mat_int, int gg) {
  MoFEMFunctionBeginHot;

  auto vec_sym = get_voigt_vec_symm(t_grad);

  VectorDouble internal_var = row(mat_int, gg); // FIXME: copyß
  ublas::matrix_row<MatrixDouble> int_var_at_gauss(mat_int, gg);

  b_view.s0.internal_state_variables = internal_var.data().data();
  b_view.s0.gradients = vec_sym.data();

  int check = integrate(b_view, mgis_bv);

  MoFEMFunctionReturnHot(0);
};

template <typename T> inline auto get_voigt_vec_symm(T &t_grad) {
  Tensor2_symmetric<double, 3> t_strain;
  Index<'i', 3> i;
  Index<'j', 3> j;
  t_strain(i, j) = (t_grad(i, j) || t_grad(j, i)) / 2;

  vector<double> vec_sym{t_strain(0, 0),        t_strain(1, 1),
                         t_strain(2, 2),        sqr2 * t_strain(0, 1),
                         sqr2 * t_strain(0, 2), sqr2 * t_strain(1, 2)};
  return vec_sym;
};

template <typename T>
inline auto to_non_symm(const Tensor2_symmetric<T, 3> &symm) {
  Tensor2<double, 3, 3> non_symm;
  Number<0> N0;
  Number<1> N1;
  Number<2> N2;
  non_symm(N0, N0) = symm(N0, N0);
  non_symm(N1, N1) = symm(N1, N1);
  non_symm(N2, N2) = symm(N2, N2);
  non_symm(N0, N1) = non_symm(N1, N0) = symm(N0, N1);
  non_symm(N0, N2) = non_symm(N2, N0) = symm(N0, N2);
  non_symm(N1, N2) = non_symm(N2, N1) = symm(N1, N2);
  return non_symm;
};


template <typename T1, typename T2>
inline MoFEMErrorCode get_ddg_from_voigt(const T1 &K, Ddg<T2, 3, 3> &D) {
  // Ddg<double, 3, 3> D;
  MoFEMFunctionBeginHot;

  D(0, 0, 0, 0) = K[0];
  D(0, 0, 1, 1) = K[1];
  D(0, 0, 2, 2) = K[2];

  D(0, 0, 0, 1) = inv_sqr2 * K[3];
  D(0, 0, 0, 2) = inv_sqr2 * K[4];
  D(0, 0, 1, 2) = inv_sqr2 * K[5];

  D(1, 1, 0, 0) = K[6];
  D(1, 1, 1, 1) = K[7];
  D(1, 1, 2, 2) = K[8];

  D(1, 1, 0, 1) = inv_sqr2 * K[9];
  D(1, 1, 0, 2) = inv_sqr2 * K[10];
  D(1, 1, 1, 2) = inv_sqr2 * K[11];

  D(2, 2, 0, 0) = K[12];
  D(2, 2, 1, 1) = K[13];
  D(2, 2, 2, 2) = K[14];

  D(2, 2, 0, 1) = inv_sqr2 * K[15];
  D(2, 2, 0, 2) = inv_sqr2 * K[16];
  D(2, 2, 1, 2) = inv_sqr2 * K[17];

  D(0, 1, 0, 0) = inv_sqr2 * K[18];
  D(0, 1, 1, 1) = inv_sqr2 * K[19];
  D(0, 1, 2, 2) = inv_sqr2 * K[20];

  D(0, 1, 0, 1) = 0.5 * K[21];
  D(0, 1, 0, 2) = 0.5 * K[22];
  D(0, 1, 1, 2) = 0.5 * K[23];

  D(0, 2, 0, 0) = inv_sqr2 * K[24];
  D(0, 2, 1, 1) = inv_sqr2 * K[25];
  D(0, 2, 2, 2) = inv_sqr2 * K[26];

  D(0, 2, 0, 1) = 0.5 * K[27];
  D(0, 2, 0, 2) = 0.5 * K[28];
  D(0, 2, 1, 2) = 0.5 * K[29];

  D(1, 2, 0, 0) = inv_sqr2 * K[30];
  D(1, 2, 1, 1) = inv_sqr2 * K[31];
  D(1, 2, 2, 2) = inv_sqr2 * K[32];

  D(1, 2, 0, 1) = 0.5 * K[33];
  D(1, 2, 0, 2) = 0.5 * K[34];
  D(1, 2, 1, 2) = 0.5 * K[35];
  MoFEMFunctionReturnHot(0);
  // return D;
};

struct CommonData {
  // Ddg<double, 3, 3> tD;

  MoFEM::Interface &mField;
  vector<boost::shared_ptr<Behaviour>> mGisBehavioursVec;
  Behaviour *mGisBehaviour;
  boost::shared_ptr<MatrixDouble> mGradPtr;
  boost::shared_ptr<MatrixDouble> mStressPtr;
  boost::shared_ptr<MatrixDouble> mDispPtr;
  boost::shared_ptr<MatrixDouble> materialTangentPtr;
  boost::shared_ptr<MatrixDouble> internalVariablePtr;
  std::map<int, BlockData> setOfBlocksData;
  Tag internalVariableTag;

  CommonData(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode setBlocks() {
    MoFEMFunctionBegin;
    string block_name = "MAT";
    int loc_id = 0;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      // FIXME: set up a proper name
      if (it->getName().compare(0, 3, block_name) == 0) {
        std::vector<double> block_data;
        CHKERR it->getAttributes(block_data);
        int id = it->getMeshsetId();
        EntityHandle meshset = it->getMeshset();
        CHKERR mField.get_moab().get_entities_by_type(
            meshset, MBTET, setOfBlocksData[id].tEts, true);
        setOfBlocksData[id].iD = id;
        setOfBlocksData[id].locID = loc_id++;
        for (int n = 0; n != block_data.size(); n++)
          setOfBlocksData[id].params[n] = block_data[n];
      }
    }

    MoFEMFunctionReturn(0);
  }

  // FIXME: this should properly assign the behaviour and internal variables
  // instead
  MoFEMErrorCode getBlockData(BlockData &data) {
    MoFEMFunctionBegin;
    mGisBehaviour = mGisBehavioursVec[data.locID].get();
    // auto yOung = data.yOung;
    // auto pOisson = data.pOisson;
    // auto lAmbda = (yOung * pOisson) / ((1. + pOisson) * (1. - 2. * pOisson));
    // auto mU = yOung / (2. * (1. + pOisson));

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode getInternalVar(const EntityHandle fe_ent,
                                const int nb_gauss_pts, const int var_size) {
    MoFEMFunctionBegin;
    double *tag_data;
    int tag_size;
    rval = mField.get_moab().tag_get_by_ptr(
        internalVariableTag, &fe_ent, 1, (const void **)&tag_data, &tag_size);

    if (rval != MB_SUCCESS || tag_size != var_size * nb_gauss_pts) {
      internalVariablePtr->resize(nb_gauss_pts, var_size, false);
      internalVariablePtr->clear();
      void const *tag_data[] = {&*internalVariablePtr->data().begin()};
      const int tag_size = internalVariablePtr->data().size();
      CHKERR mField.get_moab().tag_set_by_ptr(internalVariableTag, &fe_ent, 1,
                                              tag_data, &tag_size);

    } else {
      MatrixAdaptor tag_vec = MatrixAdaptor(
          nb_gauss_pts, var_size, 
          ublas::shallow_array_adaptor<double>(tag_size, tag_data));

      *internalVariablePtr = tag_vec;
    }

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setInternalVar(const EntityHandle fe_ent) {
    MoFEMFunctionBegin;
    void const *tag_data[] = {&*internalVariablePtr->data().begin()};
    const int tag_size = internalVariablePtr->data().size();
    CHKERR mField.get_moab().tag_set_by_ptr(internalVariableTag, &fe_ent, 1,
                                            tag_data, &tag_size);
    // TODO: FIXME: this is just for debug
    // MatrixDouble test_mat;
    // {
    //   double *tag_data;
    //   int tag_size;
    //   rval = mField.get_moab().tag_get_by_ptr(
    //       internalVariableTag, &fe_ent, 1, (const void **)&tag_data, &tag_size);
    //   MatrixAdaptor tag_vec = MatrixAdaptor(
    //       nb_gauss_pts, var_size, 
    //       ublas::shallow_array_adaptor<double>(tag_size, tag_data));
    //   test_mat = tag_vec;
    //   double test500 = test_mat(0, 0);
    //   double test501 = test_mat(1, 0);
    //   double test502 = test_mat(2, 0);
    // }
    // cout << test_mat << endl;
    MoFEMFunctionReturn(0);
  }

  inline double getInternalValAvg(size_t gg) {
    double avg = 0.;
    auto &mat = *internalVariablePtr;
    VectorDouble internal_var = row(mat, gg);
    for (double val : internal_var)
      avg += val;
    return avg;
  }

  MoFEMErrorCode createTags() {
    MoFEMFunctionBegin;
    double def_val = 0.0;
    const int default_length = 0;
    CHKERR mField.get_moab().tag_get_handle(
        "_INTERNAL_VAR", default_length, MB_TYPE_DOUBLE, internalVariableTag,
        MB_TAG_CREAT | MB_TAG_VARLEN | MB_TAG_SPARSE, PETSC_NULL);

    MoFEMFunctionReturn(0);
  }
};

struct Monitor : public FEMethod {

  Monitor(SmartPetscObj<DM> &dm,
          boost::shared_ptr<PostProcVolumeOnRefinedMesh> post_proc_fe,
          boost::shared_ptr<DomainEle> update_history,
          moab::Interface &moab_mesh, bool print_gauss)
      : dM(dm), postProcFe(post_proc_fe), updateHist(update_history),
        internalVarMesh(moab_mesh), printGauss(print_gauss){};

  MoFEMErrorCode preProcess() { return 0; }
  MoFEMErrorCode operator()() { return 0; }

  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;

    auto make_vtks = [&]() {
      MoFEMFunctionBegin;
      CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProcFe);
      CHKERR postProcFe->writeFile(
          "out_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");

      if (printGauss) {
        string file_name =
            "out_gauss_" + boost::lexical_cast<std::string>(ts_step) + ".h5m";

        CHKERR internalVarMesh.write_file(file_name.c_str(), "MOAB",
                                          "PARALLEL=WRITE_PART");
        CHKERR internalVarMesh.delete_mesh();
      }

      MoFEMFunctionReturn(0);
    };

    CHKERR DMoFEMLoopFiniteElements(dM, "dFE", updateHist);

    CHKERR make_vtks();

    // switch (atom_test_nb) {
    // case 1: {
    //   if (ts_step == 12)
    //     if (fabs(min_disp + 9.3536) > 1e-4)
    //       SETERRQ(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
    //               "atom test diverged!");
    //   break;
    // }
    // default:
    //   break;
    // }

    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<DM> dM;
  boost::shared_ptr<PostProcVolumeOnRefinedMesh> postProcFe;
  boost::shared_ptr<DomainEle> updateHist;
  moab::Interface &internalVarMesh;
  bool printGauss;
};

template <bool UPDATE>
struct OpStress : public DomainEleOp {
  OpStress(const std::string field_name,
           boost::shared_ptr<CommonData> common_data_ptr,
           BlockData &block_data);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  BlockData &dAta;
};

struct OpAssembleRhs : public DomainEleOp {
  OpAssembleRhs(const std::string field_name,
                boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpTangent : public DomainEleOp {
  OpTangent(const std::string field_name,
           boost::shared_ptr<CommonData> common_data_ptr,
           BlockData &block_data);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  BlockData &dAta;
};

struct OpAssembleLhs : public DomainEleOp {
  OpAssembleLhs(const std::string row_field_name,
                const std::string col_field_name,
                boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data);

private:
  MatrixDouble locK;
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpPostProcElastic : public DomainEleOp {
  OpPostProcElastic(const std::string field_name,
                    moab::Interface &post_proc_mesh,
                    std::vector<EntityHandle> &map_gauss_pts,
                    boost::shared_ptr<CommonData> common_data_ptr,
                    BlockData &block_data);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  moab::Interface &postProcMesh;
  std::vector<EntityHandle> &mapGaussPts;
  boost::shared_ptr<CommonData> commonDataPtr;
  BlockData &dAta;
};

struct OpSaveGaussPts : public DomainEleOp {
  OpSaveGaussPts(const std::string field_name, moab::Interface &moab_mesh,
           boost::shared_ptr<CommonData> common_data_ptr, 
           BlockData &block_data);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  moab::Interface &internalVarMesh;
  BlockData &dAta;
};

struct FePrePostProcess : public FEMethod {

  boost::ptr_vector<MethodForForceScaling> methodsOp;

  // FePrePostProcess(){}

  MoFEMErrorCode preProcess() {
    //
    MoFEMFunctionBegin;
    switch (ts_ctx) {
    case CTX_TSSETIJACOBIAN: {
      snes_ctx = CTX_SNESSETJACOBIAN;
      snes_B = ts_B;
      break;
    }
    case CTX_TSSETIFUNCTION: {
      snes_ctx = CTX_SNESSETFUNCTION;
      snes_f = ts_F;
      break;
    }
    default:
      break;
    }

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode postProcess() { return 0; }
};

} // namespace MFrontInterface