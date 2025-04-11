python T5_mutual_mask.py --top_k=80 --mask=soft_magnitude_mutual_mask --out_dir=ckpts/mutual/cola_rte/ --device=cuda:0 --datasets cola rte --model=google/t5-v1_1-base --modularized


python T5merge.py --method= _TIES \
	--out_file=logs/merege_cola_rte_test_cola.txt \
	--models ckpts/mutual/cola_rte/cola_best ckpts/mutual/cola_rte/rte_best \ 
	--dataset=cola \ 
	--base_model=google/t5-v1_1-base \
	--modularized \ 
	--device=cuda:0 