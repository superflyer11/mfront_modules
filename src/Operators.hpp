#define MAX_INTERNAL_VAR 9

struct BlockData {
  int iD;
  int oRder;
  double &yOung;
  double &pOisson;
  array<double, 10> params;
  Range tEts;
  BlockData() : oRder(-1), yOung(params[0]), pOisson(params[1]) {}
};

// Index<'i', 2> i;
// Index<'j', 2> j;
// Index<'k', 2> k;
// Index<'l', 2> l;
// Index<'m', 2> m;
// Index<'n', 2> n;

struct CommonData {
  Ddg<double, 3, 3> tD;

  MoFEM::Interface &mField;

  boost::shared_ptr<MatrixDouble> mGradPtr;
  // boost::shared_ptr<MatrixDouble> mStrainPtr;
  // boost::shared_ptr<MatrixDouble> mStressPtr;
  boost::shared_ptr<MatrixDouble> materialTangent;
  std::map<int, BlockData> setOfBlocksData;
  boost::shared_ptr<MatrixDouble> internalVariablePtr;
  Tag internalVariableTag;

  CommonData(MoFEM::Interface &m_field) : mField(m_field) {

    // ierr = setBlocks();
    // CHKERRABORT(PETSC_COMM_WORLD, ierr);
    // ierr = createTags();
    // CHKERRABORT(PETSC_COMM_WORLD, ierr);
  }

  MoFEMErrorCode setBlocks() {
    MoFEMFunctionBegin;
    string block_name = "MAT_ELASTIC";
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      if (it->getName().compare(0, block_name.size(), block_name) == 0) {

        std::vector<double> block_data;
        CHKERR it->getAttributes(block_data);
        int id = it->getMeshsetId();
        EntityHandle meshset = it->getMeshset();
        CHKERR mField.get_moab().get_entities_by_type(
            meshset, MBTET, setOfBlocksData[id].tEts, true);
        setOfBlocksData[id].iD = id;
        setOfBlocksData[id].yOung = block_data[0];
        setOfBlocksData[id].pOisson = block_data[1];
        for (int n = 2; n < block_data.size(); n++)
          setOfBlocksData[id].params[n] = block_data[n];
      }
    }

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode getBlockData(BlockData &data) {
    MoFEMFunctionBegin;

    auto yOung = data.yOung;
    auto pOisson = data.pOisson;
    auto lAmbda = (yOung * pOisson) / ((1. + pOisson) * (1. - 2. * pOisson));
    auto mU = yOung / (2. * (1. + pOisson));

    Index<'i', 3> i;
    Index<'j', 3> j;
    Index<'k', 3> k;
    Index<'l', 3> l;

    // FIXME: DON'T CALCULATE IT EVERY SINGLE TIME!!!!
    auto &tD = this->tD;
    tD(i, j, k, l) = 0.;

    tD(0, 0, 0, 0) = tD(1, 1, 1, 1) = tD(2, 2, 2, 2) = lAmbda + 2 * mU;

    tD(0, 0, 1, 1) = tD(0, 0, 2, 2) = tD(1, 1, 2, 2) = tD(2, 2, 1, 1) =
        tD(1, 1, 2, 2) = tD(2, 2, 0, 0) = tD(1, 1, 0, 0) = lAmbda;

    tD(0, 1, 0, 1) = tD(0, 2, 0, 2) = tD(1, 2, 1, 2) = tD(0, 1, 1, 0) =
        tD(0, 2, 2, 0) = tD(1, 2, 2, 1) = tD(2, 0, 0, 2) = tD(1, 0, 0, 1) =
            tD(2, 1, 1, 2) = tD(2, 0, 2, 0) = tD(1, 0, 1, 0) = tD(2, 1, 2, 1) =
                mU;

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode getInternalVar(const EntityHandle fe_ent,
                                const int nb_gauss_pts) {
    MoFEMFunctionBegin;
    double *tag_data;
    int tag_size;
    rval = mField.get_moab().tag_get_by_ptr(
        internalVariableTag, &fe_ent, 1, (const void **)&tag_data, &tag_size);

    if (rval != MB_SUCCESS || tag_size != MAX_INTERNAL_VAR * nb_gauss_pts) {
      internalVariablePtr->resize(MAX_INTERNAL_VAR, nb_gauss_pts);
      internalVariablePtr->clear();
      void const *tag_data[] = {&*internalVariablePtr->data().begin()};
      const int tag_size = internalVariablePtr->data().size();
      CHKERR mField.get_moab().tag_set_by_ptr(internalVariableTag, &fe_ent, 1,
                                              tag_data, &tag_size);

    } else {
      MatrixAdaptor tag_vec = MatrixAdaptor(
          MAX_INTERNAL_VAR, nb_gauss_pts,
          ublas::shallow_array_adaptor<double>(tag_size, tag_data));

      *internalVariablePtr = tag_vec;
    }

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setInternalVar(const EntityHandle fe_ent,
                                const int nb_gauss_pts) {
    MoFEMFunctionBegin;
    void const *tag_data[] = {&*internalVariablePtr->data().begin()};
    const int tag_size = internalVariablePtr->data().size();
    CHKERR mField.get_moab().tag_set_by_ptr(internalVariableTag, &fe_ent, 1,
                                            tag_data, &tag_size);

    MoFEMFunctionReturn(0);
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
          boost::shared_ptr<VolumeElementForcesAndSourcesCore> update_history)
      : dM(dm), postProcFe(post_proc_fe), updateHist(update_history){};

  MoFEMErrorCode preProcess() { return 0; }
  MoFEMErrorCode operator()() { return 0; }

  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;

    auto make_vtk = [&]() {
      MoFEMFunctionBegin;
      CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProcFe);
      CHKERR postProcFe->writeFile(
          "out_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");
      MoFEMFunctionReturn(0);
    };

    CHKERR DMoFEMLoopFiniteElements(dM, "dFE", updateHist);

    CHKERR make_vtk();

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
  boost::shared_ptr<VolumeElementForcesAndSourcesCore> updateHist;
};

struct OpAssembleRhs : public DomainEleOp {
  OpAssembleRhs(const std::string field_name,
                boost::shared_ptr<CommonData> common_data_ptr,
                BlockData &block_data);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  BlockData &dAta;
};

struct OpUpdateInternalVar : public DomainEleOp {
  OpUpdateInternalVar(const std::string field_name,
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
                boost::shared_ptr<CommonData> common_data_ptr,
                BlockData &block_data);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data);

private:
  MatrixDouble locK;
  boost::shared_ptr<CommonData> commonDataPtr;
  BlockData &dAta;
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
