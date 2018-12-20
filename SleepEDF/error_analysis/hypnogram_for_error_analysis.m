clear all;
clc;

model_name = 'wilson_ra_2max_cnn_1to3/';
folder_name = 'cnn1d_sleep_357_300_(08)_eval_2chan_dropout(0.6)_lr(-5)_epoches(400)/';
Nfold = 20;
for fold = 1 : Nfold
    fold
    
    % training set
    train_folder = ['./',model_name, folder_name, '/n', num2str(fold, '%d'), '/train/'];
    train_list = importdata(['../data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n', num2str(fold, '%d'),'.txt']);
    for n = 1 : numel(train_list.textdata)
        disp(['training set: ', num2str(n)])
        load([train_folder, train_list.textdata{n}(27:29), '.mat'], 'yhat');
        % load raw_data info
        load(['../data_processing/raw_data/', train_list.textdata{n}(27:29), '.mat']);
        
        % seperate day1 data
        yt_d1 = raw_label(1 : raw_epoch_num(1));
        yh_d1 = yhat(1 : valid_epoch_num(1));
        if valid_epoch_num(1) ~= raw_epoch_num(1)
            index = find(yt_d1 == 0);
            for ind = 1:length(index)
                yh_d1 = [yh_d1(1 : index(ind)-1); 0; yh_d1(index(ind) : end)];
            end
        end
        save_result_txt(yt_d1, yh_d1, onset(1), train_folder, [train_list.textdata{n}(27:29), '_d1'])
        
        % seperate day2 data
        if length(raw_epoch_num) > 1
            yt_d2 = raw_label(1+raw_epoch_num(1) : end);
            yh_d2 = yhat(1+valid_epoch_num(1) : end);
            if valid_epoch_num(2) ~= raw_epoch_num(2)
                index = find(yt_d2 == 0);
                for ind = 1:length(index)
                    yh_d2 = [yh_d2(1 : index(ind)-1); 0; yh_d2(index(ind) : end)];
                end
            end
            save_result_txt(yt_d2, yh_d2, onset(2), train_folder, [train_list.textdata{n}(27:29), '_d2'])
        end
    end
    
    % testing set
    test_folder = ['./',model_name, folder_name, '/n', num2str(fold, '%d'), '/test/'];
    test_list = importdata(['../data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n', num2str(fold, '%d'),'.txt']);
    for n = 1 : numel(test_list.textdata)
        disp(['testing set: ', num2str(n)])
        load([test_folder, test_list.textdata{n}(27:29), '.mat'], 'yhat');
        % load raw_data info
        load(['../data_processing/raw_data/', test_list.textdata{n}(27:29), '.mat']);
        
        % seperate day1 data
        yt_d1 = raw_label(1 : raw_epoch_num(1));
        yh_d1 = yhat(1 : valid_epoch_num(1));
        if valid_epoch_num(1) ~= raw_epoch_num(1)
            index = find(yt_d1 == 0);
            for ind = 1:length(index)
                yh_d1 = [yh_d1(1 : index(ind)-1); 0; yh_d1(index(ind) : end)];
            end
        end
        save_result_txt(yt_d1, yh_d1, onset(1), test_folder, [test_list.textdata{n}(27:29), '_d1'])
        
        % seperate day2 data
        if length(raw_epoch_num) > 1
            yt_d2 = raw_label(1+raw_epoch_num(1) : end);
            yh_d2 = yhat(1+valid_epoch_num(1) : end);
            if valid_epoch_num(2) ~= raw_epoch_num(2)
                index = find(yt_d2 == 0);
                for ind = 1:length(index)
                    yh_d2 = [yh_d2(1 : index(ind)-1); 0; yh_d2(index(ind) : end)];
                end
            end
            save_result_txt(yt_d2, yh_d2, onset(2), test_folder, [test_list.textdata{n}(27:29), '_d2'])
        end
    end
    
    % evaling set
    eval_folder = ['./',model_name, folder_name, '/n', num2str(fold, '%d'), '/eval/'];
    eval_list = importdata(['../data_processing/tf_data/cnn_filterbank_eval_eeg/eval_list_n', num2str(fold, '%d'),'.txt']);
    for n = 1 : numel(eval_list.textdata)
        disp(['evaling set: ', num2str(n)])
        load([eval_folder, eval_list.textdata{n}(27:29), '.mat'], 'yhat');
        % load raw_data info
        load(['../data_processing/raw_data/', eval_list.textdata{n}(27:29), '.mat']);
        
        % seperate day1 data
        yt_d1 = raw_label(1 : raw_epoch_num(1));
        yh_d1 = yhat(1 : valid_epoch_num(1));
        if valid_epoch_num(1) ~= raw_epoch_num(1)
            index = find(yt_d1 == 0);
            for ind = 1:length(index)
                yh_d1 = [yh_d1(1 : index(ind)-1); 0; yh_d1(index(ind) : end)];
            end
        end
        save_result_txt(yt_d1, yh_d1, onset(1), eval_folder, [eval_list.textdata{n}(27:29), '_d1'])
        
        % seperate day2 data
        if length(raw_epoch_num) > 1
            yt_d2 = raw_label(1+raw_epoch_num(1) : end);
            yh_d2 = yhat(1+valid_epoch_num(1) : end);
            if valid_epoch_num(2) ~= raw_epoch_num(2)
                index = find(yt_d2 == 0);
                for ind = 1:length(index)
                    yh_d2 = [yh_d2(1 : index(ind)-1); 0; yh_d2(index(ind) : end)];
                end
            end
            save_result_txt(yt_d2, yh_d2, onset(2), eval_folder, [eval_list.textdata{n}(27:29), '_d2'])
        end
    end
end