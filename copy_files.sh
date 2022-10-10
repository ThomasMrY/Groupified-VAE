rsync -av --progress --exclude=.ipynb_checkpoints \
                     --exclude=dataset_folder \
                     --exclude=experiments \
                     ~/cloud/2021_4_7/GroupifiedVAEs_old ~/cloud/clean_code_data/GroupifiedVAE