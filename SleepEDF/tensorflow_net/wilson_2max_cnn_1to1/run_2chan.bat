# run 20 folds (1000 conv. filters), 记住传入参数时，用双引号
cd D:\Study\Github\MultitaskSleepNet\SleepEDF\tensorflow_net\wilson_2max_cnn_1to1
python train_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n1/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n1/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n1.txt" --eeg_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/eval_list_n1.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n1.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n1.txt" --eog_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/eval_list_n1.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n1.txt" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n1/" --dropout_keep_prob 0.8 --num_filter 1000
python test_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n1/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n1/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n1.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n1.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n1.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n1.txt" --emg_train_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n1/" --dropout_keep_prob 0.8 --num_filter 1000
python train_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n2/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n2/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n2.txt" --eeg_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/eval_list_n2.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n2.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n2.txt" --eog_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/eval_list_n2.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n2.txt" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n2/" --dropout_keep_prob 0.8 --num_filter 1000
python test_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n2/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n2/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n2.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n2.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n2.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n2.txt" --emg_train_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n2/" --dropout_keep_prob 0.8 --num_filter 1000
python train_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n3/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n3/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n3.txt" --eeg_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/eval_list_n3.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n3.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n3.txt" --eog_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/eval_list_n3.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n3.txt" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n3/" --dropout_keep_prob 0.8 --num_filter 1000
python test_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n3/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n3/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n3.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n3.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n3.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n3.txt" --emg_train_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n3/" --dropout_keep_prob 0.8 --num_filter 1000
python train_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n4/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n4/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n4.txt" --eeg_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/eval_list_n4.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n4.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n4.txt" --eog_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/eval_list_n4.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n4.txt" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n4/" --dropout_keep_prob 0.8 --num_filter 1000
python test_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n4/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n4/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n4.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n4.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n4.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n4.txt" --emg_train_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n4/" --dropout_keep_prob 0.8 --num_filter 1000
python train_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n5/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n5/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n5.txt" --eeg_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/eval_list_n5.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n5.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n5.txt" --eog_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/eval_list_n5.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n5.txt" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n5/" --dropout_keep_prob 0.8 --num_filter 1000
python test_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n5/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n5/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n5.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n5.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n5.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n5.txt" --emg_train_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n5/" --dropout_keep_prob 0.8 --num_filter 1000
python train_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n6/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n6/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n6.txt" --eeg_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/eval_list_n6.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n6.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n6.txt" --eog_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/eval_list_n6.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n6.txt" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n6/" --dropout_keep_prob 0.8 --num_filter 1000
python test_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n6/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n6/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n6.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n6.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n6.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n6.txt" --emg_train_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n6/" --dropout_keep_prob 0.8 --num_filter 1000
python train_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n7/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n7/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n7.txt" --eeg_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/eval_list_n7.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n7.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n7.txt" --eog_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/eval_list_n7.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n7.txt" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n7/" --dropout_keep_prob 0.8 --num_filter 1000
python test_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n7/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n7/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n7.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n7.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n7.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n7.txt" --emg_train_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n7/" --dropout_keep_prob 0.8 --num_filter 1000
python train_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n8/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n8/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n8.txt" --eeg_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/eval_list_n8.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n8.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n8.txt" --eog_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/eval_list_n8.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n8.txt" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n8/" --dropout_keep_prob 0.8 --num_filter 1000
python test_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n8/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n8/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n8.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n8.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n8.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n8.txt" --emg_train_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n8/" --dropout_keep_prob 0.8 --num_filter 1000
python train_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n9/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n9/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n9.txt" --eeg_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/eval_list_n9.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n9.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n9.txt" --eog_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/eval_list_n9.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n9.txt" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n9/" --dropout_keep_prob 0.8 --num_filter 1000
python test_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n9/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n9/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n9.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n9.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n9.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n9.txt" --emg_train_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n9/" --dropout_keep_prob 0.8 --num_filter 1000
python train_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n10/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n10/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n10.txt" --eeg_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/eval_list_n10.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n10.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n10.txt" --eog_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/eval_list_n10.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n10.txt" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n10/" --dropout_keep_prob 0.8 --num_filter 1000
python test_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n10/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n10/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n10.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n10.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n10.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n10.txt" --emg_train_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n10/" --dropout_keep_prob 0.8 --num_filter 1000
python train_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n11/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n11/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n11.txt" --eeg_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/eval_list_n11.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n11.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n11.txt" --eog_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/eval_list_n11.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n11.txt" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n11/" --dropout_keep_prob 0.8 --num_filter 1000
python test_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n11/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n11/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n11.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n11.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n11.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n11.txt" --emg_train_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n11/" --dropout_keep_prob 0.8 --num_filter 1000
python train_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n12/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n12/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n12.txt" --eeg_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/eval_list_n12.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n12.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n12.txt" --eog_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/eval_list_n12.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n12.txt" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n12/" --dropout_keep_prob 0.8 --num_filter 1000
python test_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n12/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n12/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n12.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n12.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n12.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n12.txt" --emg_train_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n12/" --dropout_keep_prob 0.8 --num_filter 1000
python train_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n13/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n13/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n13.txt" --eeg_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/eval_list_n13.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n13.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n13.txt" --eog_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/eval_list_n13.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n13.txt" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n13/" --dropout_keep_prob 0.8 --num_filter 1000
python test_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n13/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n13/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n13.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n13.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n13.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n13.txt" --emg_train_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n13/" --dropout_keep_prob 0.8 --num_filter 1000
python train_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n14/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n14/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n14.txt" --eeg_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/eval_list_n14.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n14.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n14.txt" --eog_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/eval_list_n14.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n14.txt" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n14/" --dropout_keep_prob 0.8 --num_filter 1000
python test_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n14/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n14/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n14.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n14.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n14.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n14.txt" --emg_train_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n14/" --dropout_keep_prob 0.8 --num_filter 1000
python train_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n15/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n15/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n15.txt" --eeg_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/eval_list_n15.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n15.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n15.txt" --eog_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/eval_list_n15.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n15.txt" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n15/" --dropout_keep_prob 0.8 --num_filter 1000
python test_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n15/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n15/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n15.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n15.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n15.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n15.txt" --emg_train_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n15/" --dropout_keep_prob 0.8 --num_filter 1000
python train_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n16/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n16/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n16.txt" --eeg_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/eval_list_n16.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n16.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n16.txt" --eog_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/eval_list_n16.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n16.txt" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n16/" --dropout_keep_prob 0.8 --num_filter 1000
python test_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n16/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n16/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n16.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n16.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n16.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n16.txt" --emg_train_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n16/" --dropout_keep_prob 0.8 --num_filter 1000
python train_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n17/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n17/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n17.txt" --eeg_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/eval_list_n17.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n17.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n17.txt" --eog_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/eval_list_n17.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n17.txt" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n17/" --dropout_keep_prob 0.8 --num_filter 1000
python test_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n17/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n17/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n17.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n17.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n17.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n17.txt" --emg_train_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n17/" --dropout_keep_prob 0.8 --num_filter 1000
python train_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n18/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n18/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n18.txt" --eeg_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/eval_list_n18.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n18.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n18.txt" --eog_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/eval_list_n18.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n18.txt" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n18/" --dropout_keep_prob 0.8 --num_filter 1000
python test_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n18/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n18/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n18.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n18.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n18.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n18.txt" --emg_train_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n18/" --dropout_keep_prob 0.8 --num_filter 1000
python train_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n19/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n19/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n19.txt" --eeg_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/eval_list_n19.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n19.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n19.txt" --eog_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/eval_list_n19.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n19.txt" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n19/" --dropout_keep_prob 0.8 --num_filter 1000
python test_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n19/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n19/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n19.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n19.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n19.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n19.txt" --emg_train_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n19/" --dropout_keep_prob 0.8 --num_filter 1000
python train_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n20/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n20/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n20.txt" --eeg_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/eval_list_n20.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n20.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n20.txt" --eog_eval_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/eval_list_n20.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n20.txt" --emg_train_data "" --emg_eval_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n20/" --dropout_keep_prob 0.8 --num_filter 1000
python test_cnn1d_eval_gpu0.py --eeg_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eeg/n20/filterbank.mat" --eog_pretrainedfb_path "../dnn-filterbank/dnn_filterbank_sleep_20_512_256_512_(08)_eog/n20/filterbank.mat" --emg_pretrainedfb_path "" --eeg_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n20.txt" --eeg_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n20.txt" --eog_train_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/train_list_n20.txt" --eog_test_data "../../data_processing/tf_data/cnn_filterbank_eval_eog/test_list_n20.txt" --emg_train_data "" --emg_test_data "" --out_dir "./cnn1d_sleep_357_1000_(08)_eval_2chan/n20/" --dropout_keep_prob 0.8 --num_filter 1000

