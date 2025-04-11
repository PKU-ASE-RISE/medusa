python T5_mutual_mask.py --top_k=80 --mask=soft_magnitude_mutual_mask --out_dir=ckpts/tinymutual/cola_rte/ --device=cuda:0 --datasets cola rte --model=google/t5-v1_1-base --modularized --tiny_sample --smaller_batch=2


python T5merge.py --method=_TIES\
	--out_file=example_logs/cola.txt\
	--models ckpts/tinymutual/cola_rte/cola_best ckpts/tinymutual/cola_rte/rte_best\
	--dataset=cola\
	--base_model=google/t5-v1_1-base\
	--modularized\
	--device=cuda:0


python T5merge.py --method=_TIES\
	--out_file=example_logs/rte.txt\
	--models ckpts/tinymutual/cola_rte/cola_best ckpts/tinymutual/cola_rte/rte_best\
	--dataset=rte\
	--base_model=google/t5-v1_1-base\
	--modularized\
	--device=cuda:0