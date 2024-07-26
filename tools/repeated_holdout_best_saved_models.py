"""Store configurations of each task's best AE/MLP in each holdout split."""

best_AE = {}
best_MLP = {}
best_LGBM = {}
"""Best AEs for gene_essentiality."""
AE_ge = {}
AE_ge['repr_dim'] = (81, 211, 177, 148, 239, 98, 217, 248, 201, 101)
AE_ge['hidden_n_layers'] = (2, 0, 1, 1, 1, 2, 2, 1, 1, 1)
AE_ge['hidden_n_units_first'] = (975, 402, 798, 451, 648, 304, 350, 776, 1021, 575)
AE_ge['hidden_decrease_rate'] = (1, 0.5, 1, 0.5, 1, 1, 0.5, 1, 0.5, 1)
AE_ge['dropout'] = (
    0.026707604682255158,
    0.006902129865380862,
    0.1288816716035408,
    0.015310644009979673,
    0.11467619407900875,
    0.0005409401379203177,
    0.003861001743877747,
    0.06778188542021994,
    0.043302363492544814,
    0.08946182302065186,
)
AE_ge['batch_size'] = (1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)
AE_ge['learning_rate'] = (
    0.0003622043580251975,
    0.0004599818401673607,
    0.0004452779896494631,
    0.00016732332211089287,
    0.0003176542073429854,
    0.00048262479195945707,
    0.0004992411734961119,
    0.000250958780536394,
    0.0001537354785487756,
    0.00018322438008989495,
)
best_AE['gene_essentiality'] = AE_ge
"""Best LGBMs for gene_essentiality."""
LGBM_ge = {}
LGBM_ge['learning_rate'] = (
    0.010930447521036828,
    0.01866199010084843,
    0.022076822039315387,
    0.01578194524504481,
    0.048433593813957454,
    0.046279288494692636,
    0.015797333919066348,
    0.029223502588794213,
    0.013832505887926273,
    0.01210842520690128,
)
LGBM_ge['reg_alpha'] = (
    0.24454378785389297,
    0.9402733630916281,
    0.4810618841592823,
    0.274193262018251,
    16.10394374358799,
    15.68688775331011,
    4.705368755560066,
    3.5739340672319235,
    0.4375338541407743,
    0.7854045009772121,
)
best_LGBM['gene_essentiality'] = LGBM_ge
"""Best MLPs for gene_essentiality."""
MLP_GE = {}
MLP_GE['learning_rate'] = (
    0.000005,
    0.000005,
    0.000005,
    0.000005,
    0.000005,
    0.000005,
    0.000005,
    0.000005,
    0.000005,
    0.000005,
)

MLP_GE['dropout'] = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
MLP_GE['batch_size'] = (1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)
mlp_hidden = (
    [512],
    [512],
    [512],
    [512],
    [512],
    [512],
    [512],
    [512],
    [512],
    [512],
)
MLP_GE['mlp_hidden'] = mlp_hidden
mlp_plus_ae = (
    [975, 975, 975, 81, 512],
    [402, 211, 512],
    [798, 798, 177, 512],
    [451, 225, 148, 512],
    [648, 648, 239, 512],
    [304, 304, 304, 98, 512],
    [350, 175, 87, 217, 512],
    [776, 776, 248, 512],
    [1021, 510, 201, 512],
    [575, 575, 101, 512],
)
MLP_GE['mlp_plus_ae'] = mlp_plus_ae
best_MLP['gene_essentiality'] = MLP_GE
"""Best AEs for survival_prediction_tcga_task ALL_TCGA_COHORTS."""
AE_pan = {}
AE_pan['repr_dim'] = (223, 254, 236, 143, 141, 254, 233, 190, 249, 15)
AE_pan['hidden_n_layers'] = (0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
AE_pan['hidden_n_units_first'] = (810, 481, 648, 294, 719, 685, 479, 545, 535, 703)
AE_pan['hidden_decrease_rate'] = (0.5, 0.5, 1, 0.5, 0.5, 0.5, 0.5, 1, 0.5, 0.5)
AE_pan['dropout'] = (
    0.166847183416907,
    0.1816053849826712,
    0.12215642045197754,
    0.19473812091640064,
    0.05486087687330104,
    0.19259182798466432,
    0.11814846270068519,
    0.1582788455392451,
    0.10838499896062394,
    0.015404250947797845,
)
AE_pan['batch_size'] = (256, 256, 256, 256, 256, 256, 256, 256, 256, 256)
AE_pan['learning_rate'] = (
    0.002219492147073428,
    0.00394895998089286,
    0.0014555707986638792,
    0.0026953898012066794,
    0.0026743592991526487,
    0.0012761863363679902,
    0.0022462085393022985,
    0.001941162576376331,
    0.0015848859324959052,
    0.0022993371681613996,
)
best_AE['survival_prediction_tcga_task_ALL_TCGA_COHORTS'] = AE_pan
"""Best MLPs for survival_prediction_tcga_task ALL_TCGA_COHORTS."""
MLP_pan = {}
MLP_pan['learning_rate'] = (
    0.002126839545790741,
    0.0010272923402118374,
    0.0004841547163922125,
    0.002287566561920128,
    0.0039222709193499775,
    0.00011774072090418213,
    0.003643576602728036,
    0.00010645749209107666,
    0.0009177380641653688,
    0.0006686807778475124,
)
MLP_pan['dropout'] = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
MLP_pan['batch_size'] = (256, 256, 256, 256, 256, 256, 256, 256, 256, 256)
mlp_hidden = (
    [128, 64],
    [64],
    [128, 64],
    [64],
    [64],
    [256, 128],
    [64],
    [256, 128],
    [256, 128],
    [64],
)
MLP_pan['mlp_hidden'] = mlp_hidden
best_MLP['survival_prediction_tcga_task_ALL_TCGA_COHORTS'] = MLP_pan
"""Best AEs for survival_prediction_tcga_task BRCA."""
AE_BRCA = {}
AE_BRCA['repr_dim'] = (209, 219, 40, 55, 220, 122, 242, 145, 156, 239)
AE_BRCA['hidden_n_layers'] = (1, 0, 0, 2, 0, 1, 0, 1, 1, 0)
AE_BRCA['hidden_n_units_first'] = (836, 412, 778, 709, 845, 718, 799, 648, 766, 903)
AE_BRCA['hidden_decrease_rate'] = (0.5, 1, 1, 0.5, 0.5, 0.5, 0.5, 1, 1, 0.5)
AE_BRCA['dropout'] = (
    0.006345897497758678,
    0.07501433586280853,
    0.11379893095183834,
    0.16485946170638321,
    0.1514403349468112,
    0.17423660004133526,
    0.11465064517415172,
    0.14758006109592028,
    0.10415893691780852,
    0.08354326403724739,
)
AE_BRCA['batch_size'] = (256, 256, 256, 256, 256, 256, 256, 256, 256, 256)
AE_BRCA['learning_rate'] = (
    0.0017120078921071889,
    0.0007752500036053875,
    0.0014999612219985582,
    0.00317530126441056,
    0.0017322972493147818,
    5.849093960893527e-05,
    0.002545312146155517,
    0.002180010883636929,
    0.0028480871807677337,
    0.0006858873866691916,
)
best_AE['survival_prediction_tcga_task_[BRCA]'] = AE_BRCA
"""Best MLPs for survival_prediction_tcga_task BRCA."""
MLP_BRCA = {}
MLP_BRCA['learning_rate'] = (
    0.002675335028962231,
    0.004012638238729195,
    0.003755574084989659,
    0.002951706500378655,
    0.004517228254951432,
    0.0019690873765534607,
    0.001373233694231096,
    0.0021153760257700367,
    0.0026409309043380247,
    0.004009643095628705,
)
MLP_BRCA['dropout'] = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
MLP_BRCA['batch_size'] = (256, 256, 256, 256, 256, 256, 256, 256, 256, 256)
mlp_hidden = (
    [256, 128],
    [128, 64],
    [64],
    [64],
    [128, 64],
    [256, 128],
    [64],
    [256, 128],
    [64],
    [128, 64],
)
MLP_BRCA['mlp_hidden'] = mlp_hidden
mlp_plus_ae = (
    [836, 418, 209, 256, 128],
    [412, 219, 128, 64],
    [778, 40, 64],
    [709, 354, 177, 55, 64],
    [845, 220, 128, 64],
    [718, 359, 122, 256, 128],
    [799, 242, 64],
    [648, 648, 145, 256, 128],
    [766, 766, 156, 64],
    [903, 239, 128, 64],
)
MLP_BRCA['mlp_plus_ae'] = mlp_plus_ae
best_MLP['survival_prediction_tcga_task_[BRCA]'] = MLP_BRCA
"""Best AEs for survival_prediction_tcga_task UCEC."""
AE_UCEC = {}
AE_UCEC['repr_dim'] = (93, 214, 148, 39, 160, 134, 105, 6, 53, 256)
AE_UCEC['hidden_n_layers'] = (2, 2, 2, 2, 0, 0, 0, 2, 0, 0)
AE_UCEC['hidden_n_units_first'] = (969, 833, 986, 702, 1015, 775, 718, 967, 557, 271)
AE_UCEC['hidden_decrease_rate'] = (1, 1, 1, 1, 0.5, 1, 1, 1, 1, 1)
AE_UCEC['dropout'] = (
    0.13768088039523318,
    0.09975276779929079,
    0.12695022592494273,
    0.08021552744879526,
    0.0425090105762429,
    0.05747906327371688,
    0.17007205573090756,
    0.05315778399457369,
    0.08171313260971205,
    0.025751737861254495,
)
AE_UCEC['batch_size'] = (256, 256, 256, 256, 256, 256, 256, 256, 256, 256)
AE_UCEC['learning_rate'] = (
    0.0031232531831626035,
    6.675135651489485e-05,
    0.0010220670279335734,
    0.002686904535429112,
    0.00022083910678006284,
    0.003528998586393882,
    0.004163521475731023,
    0.002114615161820498,
    0.004972939962142053,
    0.002740832592392298,
)
best_AE['survival_prediction_tcga_task_[UCEC]'] = AE_UCEC
"""Best MLPs for survival_prediction_tcga_task UCEC."""
MLP_UCEC = {}
MLP_UCEC['learning_rate'] = (
    0.001672523801823238,
    0.004150036715600435,
    0.0036379820276453826,
    0.0025027507132469153,
    0.004672454684199134,
    0.003994046659941026,
    0.0026331143093385094,
    0.00039403335414287454,
    0.002558725584568731,
    0.002427946455962707,
)
MLP_UCEC['dropout'] = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
MLP_UCEC['batch_size'] = (256, 256, 256, 256, 256, 256, 256, 256, 256, 256)
mlp_hidden = (
    [128, 64],
    [128, 64],
    [256, 128],
    [128, 64],
    [128, 64],
    [64],
    [64],
    [128, 64],
    [128, 64],
    [64],
)
MLP_UCEC['mlp_hidden'] = mlp_hidden
mlp_plus_ae = (
    [969, 969, 969, 93, 128, 64],
    [833, 833, 833, 214, 128, 64],
    [986, 986, 986, 148, 256, 128],
    [702, 702, 702, 39, 128, 64],
    [1015, 160, 128, 64],
    [775, 134, 64],
    [718, 105, 64],
    [967, 967, 967, 6, 128, 64],
    [557, 53, 128, 64],
    [271, 256, 64],
)
MLP_UCEC['mlp_plus_ae'] = mlp_plus_ae
best_MLP['survival_prediction_tcga_task_[UCEC]'] = MLP_UCEC
"""Best AEs for survival_prediction_tcga_task KIRC."""
AE_KIRC = {}
AE_KIRC['repr_dim'] = (83, 196, 41, 64, 209, 252, 103, 256, 202, 172)
AE_KIRC['hidden_n_layers'] = (0, 1, 1, 2, 1, 0, 1, 2, 1, 2)
AE_KIRC['hidden_n_units_first'] = (892, 508, 507, 612, 574, 699, 777, 625, 961, 1023)
AE_KIRC['hidden_decrease_rate'] = (1, 1, 1, 1, 1, 1, 0.5, 1, 1, 0.5)
AE_KIRC['dropout'] = (
    0.08378235719238811,
    0.1827778941899262,
    0.010070254728308827,
    0.021604416197119315,
    0.13913317096363584,
    0.19311942372376154,
    0.012271924129967747,
    0.122819780676321,
    0.09864860436599487,
    0.19642676026447367,
)
AE_KIRC['batch_size'] = (256, 256, 256, 256, 256, 256, 256, 256, 256, 256)
AE_KIRC['learning_rate'] = (
    0.0017981999511232807,
    0.0005938770153049394,
    0.0016125394978166452,
    0.000978069647451689,
    0.0006163282665248341,
    0.0029773360475498704,
    0.00043494411088890617,
    0.0009254337982316555,
    7.544898039225716e-05,
    0.0003084621567585065,
)
best_AE['survival_prediction_tcga_task_[KIRC]'] = AE_KIRC
"""Best MLPs for survival_prediction_tcga_task KIRC."""
MLP_KIRC = {}
MLP_KIRC['learning_rate'] = (
    0.00014320014250770582,
    0.0006456788777265709,
    0.003925183561288188,
    0.0008005671338012883,
    0.00018339534356986309,
    0.0032578949094681916,
    0.003422021351547887,
    0.0029483526359428804,
    0.00357707712061373,
    0.0021481705744211626,
)
MLP_KIRC['dropout'] = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
MLP_KIRC['batch_size'] = (256, 256, 256, 256, 256, 256, 256, 256, 256, 256)
mlp_hidden = (
    [64],
    [128, 64],
    [128, 64],
    [64],
    [64],
    [64],
    [256, 128],
    [64],
    [64],
    [128, 64],
)
MLP_KIRC['mlp_hidden'] = mlp_hidden
mlp_plus_ae = (
    [892, 83, 64],
    [508, 508, 196, 128, 64],
    [507, 507, 41, 128, 64],
    [612, 612, 612, 64, 64],
    [574, 574, 209, 64],
    [699, 252, 64],
    [777, 388, 103, 256, 128],
    [625, 625, 625, 256, 64],
    [961, 961, 202, 64],
    [1023, 511, 255, 172, 128, 64],
)
MLP_KIRC['mlp_plus_ae'] = mlp_plus_ae
best_MLP['survival_prediction_tcga_task_[KIRC]'] = MLP_KIRC
"""Best AEs for survival_prediction_tcga_task HNSC."""
AE_HNSC = {}
AE_HNSC['repr_dim'] = (45, 100, 30, 18, 51, 182, 4, 87, 51, 141)
AE_HNSC['hidden_n_layers'] = (1, 2, 2, 1, 2, 0, 2, 2, 2, 2)
AE_HNSC['hidden_n_units_first'] = (807, 946, 795, 495, 767, 354, 1009, 787, 670, 733)
AE_HNSC['hidden_decrease_rate'] = (0.5, 1, 0.5, 1, 0.5, 1, 0.5, 0.5, 0.5, 1)
AE_HNSC['dropout'] = (
    0.079189911527789,
    0.16706543088792733,
    0.1381173765387244,
    0.0026422149006933987,
    0.14188865686000407,
    0.11360216541757884,
    0.026908346087115412,
    0.07891137099137321,
    0.02738008508703467,
    0.051440152422144725,
)
AE_HNSC['batch_size'] = (256, 256, 256, 256, 256, 256, 256, 256, 256, 256)
AE_HNSC['learning_rate'] = (
    0.0014742258516843728,
    0.0027544344787215295,
    0.002815567257345179,
    0.0030926376007197384,
    0.001013584657635809,
    0.004206639086874691,
    0.0027608788883038693,
    0.0010277027140718202,
    0.0035694614824692368,
    0.00039269860344545823,
)
best_AE['survival_prediction_tcga_task_[HNSC]'] = AE_HNSC
"""Best MLPs for survival_prediction_tcga_task HNSC."""
MLP_HNSC = {}
MLP_HNSC['learning_rate'] = (
    0.0008326788516084991,
    0.001995995220289004,
    0.001121574190778335,
    0.0018152288070861452,
    0.001263928936528991,
    0.0026665722416313563,
    0.004453724838545947,
    0.0010615890639878265,
    0.002448250805828439,
    0.000773699774455072,
)
MLP_HNSC['dropout'] = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
MLP_HNSC['batch_size'] = (256, 256, 256, 256, 256, 256, 256, 256, 256, 256)
mlp_hidden = (
    [256, 128],
    [256, 128],
    [256, 128],
    [64],
    [64],
    [128, 64],
    [128, 64],
    [64],
    [128, 64],
    [128, 64],
)
MLP_HNSC['mlp_hidden'] = mlp_hidden
mlp_plus_ae = (
    [807, 403, 45, 256, 128],
    [946, 946, 946, 100, 256, 128],
    [795, 397, 198, 30, 256, 128],
    [495, 495, 18, 64],
    [767, 383, 191, 51, 64],
    [354, 182, 128, 64],
    [1009, 504, 252, 4, 128, 64],
    [787, 393, 196, 87, 64],
    [670, 335, 167, 51, 128, 64],
    [733, 733, 733, 141, 128, 64],
)
MLP_HNSC['mlp_plus_ae'] = mlp_plus_ae
best_MLP['survival_prediction_tcga_task_[HNSC]'] = MLP_HNSC
"""Best AEs for survival_prediction_tcga_task LUAD."""
AE_LUAD = {}
AE_LUAD['repr_dim'] = (81, 11, 149, 90, 171, 45, 25, 56, 6, 37)
AE_LUAD['hidden_n_layers'] = (1, 2, 0, 2, 1, 0, 2, 1, 1, 0)
AE_LUAD['hidden_n_units_first'] = (344, 976, 685, 291, 679, 716, 968, 310, 394, 298)
AE_LUAD['hidden_decrease_rate'] = (1, 0.5, 0.5, 1, 1, 0.5, 1, 0.5, 1, 0.5)
AE_LUAD['dropout'] = (
    0.017461706294194412,
    0.1832645364692991,
    0.15020436549048694,
    0.1582214551414863,
    0.07500772735613856,
    0.0005018381902404251,
    0.17396534580944412,
    0.10879547719505307,
    0.04954401476385065,
    0.03791559908154203,
)
AE_LUAD['batch_size'] = (256, 256, 256, 256, 256, 256, 256, 256, 256, 256)
AE_LUAD['learning_rate'] = (
    0.0005351957513923084,
    0.0031232977103864976,
    0.0036585027773551053,
    0.0026890560105694373,
    0.001200404736633506,
    0.002709389776140135,
    0.00465879964178072,
    0.004857398104412686,
    0.001032740128320648,
    0.0006254648156454786,
)
best_AE['survival_prediction_tcga_task_[LUAD]'] = AE_LUAD
"""Best MLPs for survival_prediction_tcga_task LUAD."""
MLP_LUAD = {}
MLP_LUAD['learning_rate'] = (
    0.004473040120865103,
    0.0013342465328915125,
    0.0011428699142570086,
    0.0003758443451147423,
    0.002309790551259891,
    0.0013059110992572464,
    0.0022866611030165256,
    0.0008690436635523166,
    0.004228661673339739,
    0.004025872209031554,
)
MLP_LUAD['dropout'] = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
MLP_LUAD['batch_size'] = (256, 256, 256, 256, 256, 256, 256, 256, 256, 256)
mlp_hidden = (
    [64],
    [256, 128],
    [64],
    [128, 64],
    [64],
    [128, 64],
    [256, 128],
    [128, 64],
    [64],
    [256, 128],
)
MLP_LUAD['mlp_hidden'] = mlp_hidden
mlp_plus_ae = (
    [344, 344, 81, 64],
    [976, 488, 244, 11, 256, 128],
    [685, 149, 64],
    [291, 291, 291, 90, 128, 64],
    [679, 679, 171, 64],
    [716, 45, 128, 64],
    [968, 968, 968, 25, 256, 128],
    [310, 155, 56, 128, 64],
    [394, 394, 6, 64],
    [298, 37, 256, 128],
)
MLP_LUAD['mlp_plus_ae'] = mlp_plus_ae
best_MLP['survival_prediction_tcga_task_[LUAD]'] = MLP_LUAD
"""Best AEs for survival_prediction_tcga_task LGG."""
AE_LGG = {}
AE_LGG['repr_dim'] = (186, 256, 236, 115, 161, 131, 231, 158, 179, 95)
AE_LGG['hidden_n_layers'] = (1, 0, 1, 1, 0, 1, 0, 1, 0, 1)
AE_LGG['hidden_n_units_first'] = (625, 767, 379, 805, 420, 952, 432, 533, 624, 407)
AE_LGG['hidden_decrease_rate'] = (0.5, 1, 1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5)
AE_LGG['dropout'] = (
    0.04684963884052379,
    0.033342458127675875,
    0.036735841509191625,
    0.0532435744507544,
    00.061903638083909335,
    0.17512892941201114,
    0.02568666644721443,
    0.17317618864263945,
    0.07637498808902526,
    0.07696847673531118,
)
AE_LGG['batch_size'] = (256, 256, 256, 256, 256, 256, 256, 256, 256, 256)
AE_LGG['learning_rate'] = (
    0.000903804738740849,
    0.0038000970222221327,
    0.0012794504270336117,
    0.0004891196111445032,
    0.004326879344469546,
    0.0003709524640530261,
    0.0015265023190024647,
    0.0006424158996657662,
    0.0032747394795494774,
    0.0016546078179348791,
)
best_AE['survival_prediction_tcga_task_[LGG]'] = AE_LGG
"""Best MLPs for survival_prediction_tcga_task LGG."""
MLP_LGG = {}
MLP_LGG['learning_rate'] = (
    0.002939003623843495,
    0.0019996368995095926,
    0.0013895869978842043,
    0.004281689450888263,
    0.0008269235361913831,
    0.0034168227261542456,
    0.0033531833749941608,
    0.001056686481934718,
    0.0010385465911935095,
    0.002725107808144191,
)
MLP_LGG['dropout'] = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
MLP_LGG['batch_size'] = (256, 256, 256, 256, 256, 256, 256, 256, 256, 256)
mlp_hidden = (
    [64],
    [256, 128],
    [128, 64],
    [256, 128],
    [64],
    [128, 64],
    [64],
    [256, 128],
    [64],
    [256, 128],
)
MLP_LGG['mlp_hidden'] = mlp_hidden
mlp_plus_ae = (
    [625, 312, 186, 64],
    [767, 256, 256, 128],
    [379, 379, 236, 128, 64],
    [805, 402, 115, 256, 128],
    [420, 161, 64],
    [952, 476, 131, 128, 64],
    [432, 231, 64],
    [533, 266, 158, 256, 128],
    [624, 179, 64],
    [407, 203, 95, 256, 128],
)
MLP_LGG['mlp_plus_ae'] = mlp_plus_ae
best_MLP['survival_prediction_tcga_task_[LGG]'] = MLP_LGG
"""Best AEs for survival_prediction_tcga_task LUSC."""
AE_LUSC = {}
AE_LUSC['repr_dim'] = (187, 194, 159, 84, 241, 134, 180, 214, 187, 16)
AE_LUSC['hidden_n_layers'] = (1, 2, 2, 2, 0, 1, 1, 1, 0, 0)
AE_LUSC['hidden_n_units_first'] = (821, 537, 528, 679, 910, 938, 299, 676, 535, 298)
AE_LUSC['hidden_decrease_rate'] = (0.5, 0.5, 0.5, 1, 0.5, 1, 0.5, 1, 1, 1)
AE_LUSC['dropout'] = (
    0.058468905600382565,
    0.08866176067111545,
    0.07495277118676508,
    0.07748713732361664,
    0.13388954909877696,
    0.046120262638010855,
    0.027006952830442958,
    0.05977155166139417,
    0.1669008883583541,
    0.08497723478936911,
)
AE_LUSC['batch_size'] = (256, 256, 256, 256, 256, 256, 256, 256, 256, 256)
AE_LUSC['learning_rate'] = (
    0.00491939364508083,
    0.0018536332529283319,
    0.002900371599541399,
    0.0046270745483694,
    0.003556565378389492,
    0.0017702939204277991,
    0.004837887764484782,
    0.0013629195153448813,
    7.867234683109074e-05,
    0.004188211305786745,
)
best_AE['survival_prediction_tcga_task_[LUSC]'] = AE_LUSC
"""Best MLPs for survival_prediction_tcga_task LUSC."""
MLP_LUSC = {}
MLP_LUSC['learning_rate'] = (
    0.0003396619690425702,
    0.0028449314293035217,
    0.0010752686368204816,
    0.0012117179171438363,
    0.00425710681865857,
    0.0015962358123492375,
    0.0007714673556323704,
    0.0007107899810563359,
    0.002064378084789681,
    0.00025105082394946655,
)
MLP_LUSC['dropout'] = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
MLP_LUSC['batch_size'] = (256, 256, 256, 256, 256, 256, 256, 256, 256, 256)
mlp_hidden = (
    [128, 64],
    [256, 128],
    [128, 64],
    [256, 128],
    [128, 64],
    [64],
    [64],
    [128, 64],
    [256, 128],
    [64],
)
MLP_LUSC['mlp_hidden'] = mlp_hidden
mlp_plus_ae = (
    [821, 410, 187, 128, 64],
    [537, 268, 134, 194, 256, 128],
    [528, 264, 132, 159, 128, 64],
    [679, 679, 679, 84, 256, 128],
    [910, 241, 128, 64],
    [938, 938, 134, 64],
    [299, 149, 180, 64],
    [676, 676, 214, 128, 64],
    [535, 187, 256, 128],
    [298, 16, 64],
)
MLP_LUSC['mlp_plus_ae'] = mlp_plus_ae
best_MLP['survival_prediction_tcga_task_[LUSC]'] = MLP_LUSC
"""Best AEs for survival_prediction_tcga_task SKCM."""
AE_SKCM = {}
AE_SKCM['repr_dim'] = (102, 23, 31, 109, 80, 129, 221, 91, 155, 141)
AE_SKCM['hidden_n_layers'] = (2, 0, 0, 2, 1, 1, 1, 2, 0, 2)
AE_SKCM['hidden_n_units_first'] = (975, 707, 601, 556, 640, 579, 792, 1024, 511, 974)
AE_SKCM['hidden_decrease_rate'] = (0.5, 1, 1, 0.5, 1, 0.5, 0.5, 0.5, 1, 0.5)
AE_SKCM['dropout'] = (
    0.09073379510874723,
    0.14397336840578523,
    0.12061628328371986,
    0.0637007477850440,
    0.0177191605949297,
    0.19643211543379757,
    0.14420954729071359,
    0.08450630696134293,
    0.15510888678112614,
    0.03636163316061158,
)
AE_SKCM['batch_size'] = (256, 256, 256, 256, 256, 256, 256, 256, 256, 256)
AE_SKCM['learning_rate'] = (
    0.0009375195192943451,
    0.0022769736840834273,
    0.0031252949903242135,
    0.0003779274152019617,
    0.0033853234177361616,
    0.0010301001458706037,
    0.0003218532562814526,
    0.00016549985263652876,
    0.002693553862763989,
    0.0003621282336518892,
)
best_AE['survival_prediction_tcga_task_[SKCM]'] = AE_SKCM
"""Best MLPs for survival_prediction_tcga_task SKCM."""
MLP_SKCM = {}
MLP_SKCM['learning_rate'] = (
    0.0027322344030426687,
    0.004324575400837234,
    0.001828231382519155,
    0.0035115087312052574,
    0.0005385651315800869,
    0.0001239174326161357,
    0.004792603412180449,
    0.001223816887685214,
    0.0047622584384693495,
    0.003024360384438305,
)
MLP_SKCM['dropout'] = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
MLP_SKCM['batch_size'] = (256, 256, 256, 256, 256, 256, 256, 256, 256, 256)
mlp_hidden = (
    [256, 128],
    [64],
    [128, 64],
    [128, 64],
    [256, 128],
    [64],
    [256, 128],
    [128, 64],
    [256, 128],
    [64],
)
MLP_SKCM['mlp_hidden'] = mlp_hidden
mlp_plus_ae = (
    [975, 487, 243, 102, 256, 128],
    [707, 23, 64],
    [601, 31, 128, 64],
    [556, 278, 139, 109, 128, 64],
    [640, 640, 80, 256, 128],
    [579, 289, 129, 64],
    [792, 396, 221, 256, 128],
    [1024, 512, 256, 91, 128, 64],
    [511, 155, 256, 128],
    [974, 487, 243, 141, 64],
)
MLP_SKCM['mlp_plus_ae'] = mlp_plus_ae
best_MLP['survival_prediction_tcga_task_[SKCM]'] = MLP_SKCM
"""Best AEs for survival_prediction_tcga_task COAD."""
AE_COAD = {}
AE_COAD['repr_dim'] = (157, 165, 52, 109, 240, 254, 57, 256, 149, 80)
AE_COAD['hidden_n_layers'] = (1, 2, 1, 2, 1, 2, 1, 0, 1, 2)
AE_COAD['hidden_n_units_first'] = (360, 415, 537, 793, 465, 763, 402, 641, 742, 877)
AE_COAD['hidden_decrease_rate'] = (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 1)
AE_COAD['dropout'] = (
    0.1404818465826912,
    0.039925422148883116,
    0.04562480506855678,
    0.061639521496515186,
    0.13663445929554008,
    0.07518786975385101,
    0.15842552577812824,
    0.15236579085209695,
    0.18736584045881732,
    0.058067675017155960,
)
AE_COAD['batch_size'] = (256, 256, 256, 256, 256, 256, 256, 256, 256, 256)
AE_COAD['learning_rate'] = (
    0.0029605298292938583,
    0.0027011788583795235,
    0.001500190574303032,
    0.001707599082756292,
    0.0012486078764542437,
    0.00022218630350692584,
    0.002107844978119903,
    0.000658374107750872,
    0.00040028529232761937,
    0.002122424804772422,
)
best_AE['survival_prediction_tcga_task_[COAD]'] = AE_COAD
"""Best MLPs for survival_prediction_tcga_task COAD."""
MLP_COAD = {}
MLP_COAD['learning_rate'] = (
    0.002096903400819045,
    0.004989135525695439,
    0.0021260295816240603,
    0.0017523374377467964,
    0.004100290010072185,
    0.0012488104091117173,
    0.004421245430388815,
    0.004051155455383567,
    0.003011526931735486,
    0.0003314438107486095,
)
MLP_COAD['dropout'] = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
MLP_COAD['batch_size'] = (256, 256, 256, 256, 256, 256, 256, 256, 256, 256)
mlp_hidden = (
    [64],
    [128, 64],
    [256, 128],
    [64],
    [256, 128],
    [256, 128],
    [256, 128],
    [128, 64],
    [128, 64],
    [128, 64],
)
MLP_COAD['mlp_hidden'] = mlp_hidden
mlp_plus_ae = (
    [360, 360, 157, 64],
    [415, 415, 415, 165, 128, 64],
    [537, 268, 52, 256, 128],
    [793, 396, 198, 109, 64],
    [465, 232, 240, 256, 128],
    [763, 381, 190, 254, 256, 128],
    [402, 201, 57, 256, 128],
    [641, 256, 128, 64],
    [742, 742, 149, 128, 64],
    [877, 877, 877, 80, 128, 64],
)
MLP_COAD['mlp_plus_ae'] = mlp_plus_ae
best_MLP['survival_prediction_tcga_task_[COAD]'] = MLP_COAD
"""Best AEs for survival_prediction_tcga_task STAD."""
AE_STAD = {}
AE_STAD['repr_dim'] = (98, 85, 175, 244, 215, 62, 223, 195, 58, 63)
AE_STAD['hidden_n_layers'] = (1, 1, 1, 2, 1, 1, 2, 0, 0, 0)
AE_STAD['hidden_n_units_first'] = (560, 967, 428, 261, 348, 663, 826, 407, 608, 547)
AE_STAD['hidden_decrease_rate'] = (1, 1, 0.5, 1, 1, 1, 0.5, 1, 1, 0.5)
AE_STAD['dropout'] = (
    0.15135562761810614,
    0.017057553109487394,
    0.10424251247636766,
    0.14710037645220614,
    0.18013101326757458,
    0.08016265607435058,
    0.01982955584925785,
    0.06271452636785756,
    0.09652191361034904,
    0.09008461187667567,
)
AE_STAD['batch_size'] = (256, 256, 256, 256, 256, 256, 256, 256, 256, 256)
AE_STAD['learning_rate'] = (
    0.002453997051326398,
    0.00010948546187948348,
    0.0030224510162863596,
    0.0021608784901432565,
    0.001880372983327088,
    0.0038059115507322,
    0.0004443676710539101,
    0.003453855439724642,
    0.0009386243115098045,
    0.0026259781636248044,
)
best_AE['survival_prediction_tcga_task_[STAD]'] = AE_STAD
"""Best MLPs for survival_prediction_tcga_task STAD."""
MLP_STAD = {}
MLP_STAD['learning_rate'] = (
    0.0033362847431869184,
    6.26947538108942e-05,
    0.002577159144266021,
    0.0005643625686417506,
    0.0030427983947219164,
    0.0003572741361112533,
    0.00047660898951703244,
    0.002492760024946575,
    0.0023045213410701005,
    0.0020467070722105193,
)
MLP_STAD['dropout'] = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
MLP_STAD['batch_size'] = (256, 256, 256, 256, 256, 256, 256, 256, 256, 256)
mlp_hidden = (
    [256, 128],
    [256, 128],
    [256, 128],
    [64],
    [128, 64],
    [256, 128],
    [256, 128],
    [256, 128],
    [64],
    [256, 128],
)
MLP_STAD['mlp_hidden'] = mlp_hidden
mlp_plus_ae = (
    [560, 560, 98, 256, 128],
    [967, 967, 85, 256, 128],
    [428, 214, 175, 256, 128],
    [261, 261, 261, 244, 64],
    [348, 348, 215, 128, 64],
    [663, 663, 62, 256, 128],
    [826, 413, 206, 223, 256, 128],
    [407, 195, 256, 128],
    [608, 58, 64],
    [547, 63, 256, 128],
)
MLP_STAD['mlp_plus_ae'] = mlp_plus_ae
best_MLP['survival_prediction_tcga_task_[STAD]'] = MLP_STAD
"""Best AEs for survival_prediction_tcga_task BLCA."""
AE_BLCA = {}
AE_BLCA['repr_dim'] = (35, 244, 99, 11, 141, 173, 172, 209, 126, 166)
AE_BLCA['hidden_n_layers'] = (0, 0, 0, 1, 0, 0, 0, 1, 0, 1)
AE_BLCA['hidden_n_units_first'] = (974, 669, 586, 611, 567, 976, 803, 359, 567, 541)
AE_BLCA['hidden_decrease_rate'] = (1, 0.5, 1, 0.5, 1, 0.5, 0.5, 1, 0.5, 1)
AE_BLCA['dropout'] = (
    0.15078171018130893,
    0.08059587122543435,
    0.16022861933183308,
    0.13454712123405457,
    0.08725344795998377,
    0.062739462038148580,
    0.17715842535865964,
    0.13540080330220777,
    0.10508180048967158,
    0.06006447307417532,
)
AE_BLCA['batch_size'] = (256, 256, 256, 256, 256, 256, 256, 256, 256, 256)
AE_BLCA['learning_rate'] = (
    0.0016706719341117902,
    0.0006654349945659321,
    0.0009435971233698831,
    0.0014144698461291804,
    0.002902008539726578,
    0.0033696342613834442,
    0.0031669219678676025,
    0.0010494201484264496,
    0.0024317448471322563,
    0.0006511587696378921,
)
best_AE['survival_prediction_tcga_task_[BLCA]'] = AE_BLCA
"""Best MLPs for survival_prediction_tcga_task BLCA."""
MLP_BLCA = {}
MLP_BLCA['learning_rate'] = (
    0.002515059024683099,
    0.0030727122826639535,
    0.002901434303662633,
    0.004839078076659975,
    0.0006690172387671157,
    0.0012620355694987911,
    0.001472357863367714,
    0.0008524506227326244,
    0.0030895414076411295,
    0.0015903227649489858,
)
MLP_BLCA['dropout'] = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
MLP_BLCA['batch_size'] = (256, 256, 256, 256, 256, 256, 256, 256, 256, 256)
mlp_hidden = (
    [64],
    [64],
    [64],
    [128, 64],
    [128, 64],
    [256, 128],
    [64],
    [128, 64],
    [64],
    [256, 128],
)
MLP_BLCA['mlp_hidden'] = mlp_hidden
mlp_plus_ae = (
    [974, 35, 64],
    [669, 244, 64],
    [586, 99, 64],
    [611, 305, 11, 128, 64],
    [567, 141, 128, 64],
    [976, 173, 256, 128],
    [803, 172, 64],
    [359, 359, 209, 128, 64],
    [567, 126, 64],
    [541, 541, 166, 256, 128],
)
MLP_BLCA['mlp_plus_ae'] = mlp_plus_ae
best_MLP['survival_prediction_tcga_task_[BLCA]'] = MLP_BLCA
