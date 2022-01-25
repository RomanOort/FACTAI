import argparse
import utils.parser_utils as parser_utils
    
def arg_parse():
    parser = argparse.ArgumentParser(description="GNN Explainer arguments.")
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument("--dataset", dest="dataset", help="Input dataset.")
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument(
        "--bmname", dest="bmname", help="Name of the benchmark dataset"
    )
    
    io_parser.add_argument("--pkl", dest="pkl_fname", help="Name of the pkl data file")

    parser_utils.parse_optimizer(parser)

    parser.add_argument("--clean-log", action="store_true", help="If true, cleans the specified log directory before running.")
    parser.add_argument("--logdir", dest="logdir", help="Tensorboard log directory")
    parser.add_argument("--ckptdir", dest="ckptdir", help="Model checkpoint directory")
    parser.add_argument("--prefix", dest="prefix", help="prefix for saving files")
    parser.add_argument("--add-self", dest="add_self", help="add self")
    parser.add_argument("--exp-path", dest="exp_path", help="explainer load path")
    parser.add_argument("--exp-path1", dest="exp_path1", help="explainer load path")
    parser.add_argument("--exp-path2", dest="exp_path2", help="explainer load path")
    parser.add_argument("--adversarial-path", dest="adversarial_path", help="path of adversarial examples")
    parser.add_argument("--cuda", dest="cuda", help="CUDA.")
    parser.add_argument(
        "--gpu",
        dest="gpu",
        action="store_true",
        help="whether to use GPU.",
    )
    parser.add_argument(
        "--epochs", dest="num_epochs", type=int, help="Number of epochs to train."
    )
    parser.add_argument(
        "--hidden-dim", dest="hidden_dim", type=int, help="Hidden dimension"
    )
    parser.add_argument(
        "--output-dim", dest="output_dim", type=int, help="Output dimension"
    )
    parser.add_argument(
        "--num-gc-layers",
        dest="num_gc_layers",
        type=int,
        help="Number of graph convolution layers before each pooling",
    )
    parser.add_argument(
        "--start-epoch",
        dest="start_epoch",
        type=int,
        help="start epoch",
    )
    parser.add_argument(
        "--bn",
        dest="bn",
        action="store_const",
        const=True,
        default=False,
        help="Whether batch normalization is used",
    )
    parser.add_argument(
        "--eval",
        dest="eval",
        action="store_const",
        const=True,
        default=False,
        help="only eval",
    )
    parser.add_argument(
        "--apply-filter",
        dest="apply_filter",
        action="store_const",
        const=True,
        default=False,
        help="apply filter",
    )
 
    parser.add_argument(
        "--node-mask",
        dest="node_mask",
        action="store_const",
        const=True,
        default=False,
        help="use node mask",
    )
    parser.add_argument(
        "--shuffle-adj",
        dest="shuffle_adj",
        action="store_const",
        const=True,
        default=False,
        help="shuffle",
    )
    parser.add_argument(
        "--noise",
        dest="noise",
        action="store_const",
        const=True,
        default=False,
        help="add noise",
    )

    parser.add_argument(
        "--noise-percent",
        dest="noise_percent",
        type=float,
        default=0,
    )

    parser.add_argument(
        "--post-processing",
        dest="post_processing",
        action="store_const",
        const=True,
        default=False,
        help="post processing",
    )

    parser.add_argument(
        "--draw-graphs",
        dest="draw_graphs",
        action="store_const",
        const=True,
        default=False,
        help="draw graphs",
    )

    parser.add_argument(
        "--gumbel",
        dest="gumbel",
        action="store_const",
        const=True,
        default=False,
        help="use gumbel",
    )


    parser.add_argument(
        "--inverse-noise",
        dest="inverse_noise",
        action="store_const",
        const=True,
        default=False,
        help="add noise",
    )

    parser.add_argument("--dropout", dest="dropout", type=float, help="Dropout rate.")
    parser.add_argument(
        "--nobias",
        dest="bias",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--no-writer",
        dest="writer",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    # Explainer
    parser.add_argument("--mask-act", dest="mask_act", type=str, help="sigmoid, ReLU.")
    parser.add_argument(
        "--mask-bias",
        dest="mask_bias",
        action="store_const",
        const=True,
        default=False,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--explain-node", dest="explain_node", type=int, help="Node to explain."
    )
    parser.add_argument(
        "--graph-idx", dest="graph_idx", type=int, help="Graph to explain."
    )
    parser.add_argument(
        "--graph-mode",
        dest="graph_mode",
        action="store_const",
        const=True,
        default=False,
        help="whether to run Explainer on Graph Classification task.",
    )
    parser.add_argument(
        "--multigraph-class",
        dest="multigraph_class",
        type=int,
        help="whether to run Explainer on multiple Graphs from the Classification task for examples in the same class.",
    )
    parser.add_argument(
        "--multinode-class",
        dest="multinode_class",
        type=int,
        help="whether to run Explainer on multiple nodes from the Classification task for examples in the same class.",
    )
    parser.add_argument(
        "--align-steps",
        dest="align_steps",
        type=int,
        help="Number of iterations to find P, the alignment matrix.",
    )

    parser.add_argument(
        "--fname", dest="fname", type=str, help="result file"
    )
    parser.add_argument(
        "--lap_c", dest="lap_c", type=float, help="laplacian coeffecient"
    )
    parser.add_argument(
        "--inverse_boundary_c", dest="inverse_boundary_c", type=float, help="boundary coeffecient"
    )
    parser.add_argument(
        "--boundary_c", dest="boundary_c", type=float, help="boundary coeffecient"
    )
    
    parser.add_argument(
        "--sparsity", dest="sparsity", type=float, help="sparsity for eval of fidelity"
    )

    parser.add_argument(
        "--ent_c", dest="ent_c", type=float, help="entropy coeffecient"
    )
    parser.add_argument(
        "--ent_c_2", dest="ent_c_2", type=float, help="entropy coeffecient"
    )
    parser.add_argument(
        "--intersec_c", dest="intersec_c", type=float, help="intersection coeffecient"
    )   
    parser.add_argument(
        "--topk", dest="topk", type=float, help="topk to mask"
    )
    parser.add_argument(
        "--size_c", dest="size_c", type=float, help="size coeffecient"
    )
    parser.add_argument(
        "--size_c_2", dest="size_c_2", type=float, help="size coeffecient"
    )
    parser.add_argument(
        "--method", dest="method", type=str, help="Method. Possible values: base, att."
    )
    parser.add_argument(
        "--name-suffix", dest="name_suffix", help="suffix added to the output filename"
    )
    parser.add_argument(
        "--add_embedding", dest="add_embedding", default=False, help="add embedding layer "
    )
    parser.add_argument(
        "--no-sample", dest="no_sample", action='store_true', help="pgexp sample "
    )

    parser.add_argument(
        "--explainer-suffix",
        dest="explainer_suffix",
        help="suffix added to the explainer log",
    )

    parser.add_argument(
        "--explainer-method",
        dest="explainer_method",
        type=str,
        help="Method to follow (gnnexplainer, boundary)"
    )
    parser.add_argument(
        "--bloss-version",
        dest="bloss_version",
        type=str,
        help="proj | sigmoid loss for boundary"
    )

    parser.add_argument(
        "--train-data-sparsity",
        dest="train_data_sparsity",
        type=float,
        help="(use 0.0 to 1.0 of all training data when training model)"
    )

    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        help="seeding training",
        default=0
    )

    parser.add_argument(
        "--lambda",
        dest="lambda",
        type=int,
        help="lambda hyperparam [0, 1]",
        default=0.5
    )

    parser.add_argument(
        "--pool-size",
        dest="pool_size",
        type=int,
        help="size of pool",
        default=50
    )

    parser.add_argument(
        "--pred-hidden-dim",
        dest="pred_hidden_dim",
        type=int,
        help="hidden dims",
        default=20
    )

    parser.add_argument(
        "--pred-num-layers",
        dest="pred_num_layers",
        type=int,
        help="num layers",
        default=0
    )

    parser.add_argument(
        "--AUC-type",
        dest="AUC_type",
        type=str,
        help="Selector for how AUC is computed:"
             "'original': The default method ignoring false negatives"
             "'FN': Alternative AUC metric, which takes into account edges originally in S but not in S'"
             "'full': Compare full adj matrices, taking into acount true negatives",
        default="original"
    )

    # TODO: Check argument usage
    parser.set_defaults(
        logdir="log",
        ckptdir="ckpt",
        prefix="",
        exp_path="",
        add_self="none",
        dataset="syn1",
        opt="adam",  
        opt_scheduler="none",
        cuda="1",
        lr=0.1,
        clip=2.0,
        batch_size=20,
        num_epochs=100,#100,
        hidden_dim=20,
        output_dim=20,
        num_gc_layers=3,#1,3
        start_epoch=0,
        dropout=0.0,
        method="base",
        name_suffix="",
        explainer_suffix="",
        align_steps=1000,
        explain_node=None,
        graph_idx=-1,
        mask_act="sigmoid",
        multigraph_class=-1,
        multinode_class=-1,
        add_embedding = False,
	    size_c = -1.0,
        size_c_2 = -1.0,
        lap_c = -1.0,
        boundary_c = 0.5,
        inverse_boundary_c = 0.5,
        sparsity = 0.5,
	    ent_c = -1.0,
        ent_c_2 = -1.0,
	    intersec_c = -1.0,
        topk = 8.0,
        noise_percent = 10.0,
        fname = "",
        explainer_method="gnnexplainer",
        bloss_version=""
    )

    return parser.parse_args()