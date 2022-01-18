rsync * tvorden@WS3:/storage2/tvorden/FACT/packages -avzP --exclude "gcn_interpretation/gnn-model-explainer/wandb" &
wait
rsync * robolab@WS9:~/Documents/tvorden/FACTAI/packages -avzP --exclude "gcn_interpretation/gnn-model-explainer/wandb"