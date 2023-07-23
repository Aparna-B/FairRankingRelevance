seeds="1 2 3 4 5 6 7 8 9 10"
max_iter="500"
dsname="fairtrecnotpretrained"
for seed in $seeds
do
	# Fairtrec example
	export SETTING_ARGS="--data_dir=./example/fairtrecnotpretrained/data/ --train_data_prefix=train_fairtrecnotpretrained --valid_data_prefix=val_fairtrecnotpretrained --test_data_prefix=test_fairtrecnotpretrained --model_dir=./outputs/fairtrecnotpretrained_model_matched_${seed}/ --output_dir=./outputs/fairtrecnotpretrained_output_matched_${seed} --setting_file=./example/offline_setting/ipw_rank_exp_settings.json"
	# Run model
	python main.py --dsname=${dsname} --seed=${seed} --max_train_iteration=$max_iter $SETTING_ARGS
done


