seeds="1 2 3 4 5 6 7 8 9 10"
max_iter="500"
dsname="pareto"

for seed in $seeds
do
	# synthetic example
	export SETTING_ARGS="--data_dir=./example/synthetic/data/ --model_dir=./outputs/tmp_model_synthetic_pareto_matched_${seed}/ --output_dir=./outputs/tmp_model_synthetic_pareto_matched_${seed}/ --setting_file=./example/offline_setting/ipw_rank_exp_settings.json --train_data_prefix=train_pareto --valid_data_prefix=val_pareto --test_data_prefix=test_pareto"

	# Run model
	python main.py --dsname=${dsname} --seed=${seed} --max_train_iteration=$max_iter --train_flag=True $SETTING_ARGS

done

