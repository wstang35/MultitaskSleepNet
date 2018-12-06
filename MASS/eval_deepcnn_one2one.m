function [acc, kappa, f1, sens, spec, classwise_sens, classwise_sel, C] = eval_deepcnn_one2one(nchan)

    if(nargin == 0)
        nchan = 3;
    end
    
    Ncat = 5;

    Nfold = 20;
    yh = cell(Nfold,1);
    yt = cell(Nfold,1);
    
    mat_path = './data_processing/mat/';
	listing = dir([mat_path, '*_cnn_filterbank_eeg.mat']);
	load('./data_processing/data_split_eval.mat');
    
    for fold = 1 : Nfold
        fold
        
        % ground truth
        test_s = test_sub{fold};
        sample_size = zeros(numel(test_s), 1);
        for i = 1 : numel(test_s)
            sname = listing(test_s(i)).name;
            load([mat_path,sname], 'label');
            sample_size(i) = numel(label);
            yt{fold} = [yt{fold}; double(label)];
        end
        
        load(['./tensorflow_net/deep_cnn_baseline_1to1/deepcnn_sleep_96_96_1024_1024_(08)_eval_',num2str(nchan),'chan/n', num2str(fold), '/test_ret_model_acc.mat']);
        yh{fold} = double(yhat');
     end
    yh = cell2mat(yh);
    yt = cell2mat(yt);
    
    [acc, kappa, f1, sens, spec] = calculate_overall_metrics(yt, yh);
    [classwise_sens, classwise_sel]  = calculate_classwise_sens_sel(yt, yh);
    C = confusionmat(yt, yh);
end