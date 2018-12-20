% aggregation = 0 % multiplicative aggregation
% aggregation = 1 % additive aggregation
function [test,train,eval] = eval_2maxcnn_one2many(nfilter, nchan, aggregation)

    addpath('./evaluation/');
    if(nargin == 0)
        nfilter = 1000;
        nchan = 2;
        aggregation = 0;
    end
    folder_name = ['wilson_ra_2max_cnn_1to3/cnn1d_sleep_357_',num2str(nfilter),'_(08)_eval_',num2str(nchan),'chan_dropout(0.6)_lr(-5)_epoches(400)'];
    
    Ncat = 5;
    
    output_context_size = 3;
    half = floor(output_context_size/2);

    Nfold = 20;
    yh = cell(Nfold,1);
    yt = cell(Nfold,1);

    %% test set
    for fold = 1 : Nfold
        fold
        
        test_list = importdata(['./data_processing/tf_data/cnn_filterbank_eval_eeg/test_list_n', num2str(fold, '%d'),'.txt']);
        load(['./tensorflow_net/',folder_name,'/n', num2str(fold), '/test_ret_model_acc.mat']);
        score2 = softmax(score2);
        score1 = softmax(score1);
        score3 = softmax(score3);
        
         % wilson: �ֿ�ÿ��individuals
        st_ind = 1;
        for n = 1:length(test_list.data)
            et_ind = st_ind + test_list.data(n) - 1; % update end index
            score1_temp = score1(st_ind:et_ind,:);
            score2_temp = score2(st_ind:et_ind,:);
            score3_temp = score3(st_ind:et_ind,:);
            score1_temp = [score1_temp((1+half):end,:); ones(1,Ncat)];
            score3_temp = [ones(1,Ncat); score3_temp(1:(end-half),:)];
            st_ind = et_ind + 1; % updata start index
            if(aggregation == 0)
                score = (score1_temp.* score2_temp .* score3_temp)/output_context_size;
            else
                score = (score1_temp + score2_temp + score3_temp)/output_context_size;
            end
            yhat = zeros(1,size(score,1));
            for i = 1 : size(score,1)
                [~, yhat(i)] = max(score(i,:));
            end
            yh{fold} = [yh{fold};double(yhat')];
            % save yhat          
            individual = test_list.textdata{n}(27:29);
            yhat = yhat';
            mkdir(['./error_analysis/',folder_name,'/n', num2str(fold, '%d'), '/test'])
            save(['./error_analysis/',folder_name,'/n', num2str(fold, '%d'), '/test','/', individual, '.mat'], 'yhat') 
        end
       
        % load ground-truth labels (test set)
        test_label = [];        
        for i =1:length(test_list.textdata)
            test = load(test_list.textdata{i}(5:end), 'label');
            test_label = [test_label; test.label];
        end
        yt{fold} = double(test_label);     
        
    end
    yh = cell2mat(yh);
    yt = cell2mat(yt);
    
    [test.acc, test.kappa, test.f1, test.sens, test.spec] = calculate_overall_metrics(yt, yh);
    [test.classwise_sens, test.classwise_sel]  = calculate_classwise_sens_sel(yt, yh);
    test.C = confusionmat(yt, yh);
    
   
    %% train set
    yh = cell(Nfold,1);
    yt = cell(Nfold,1);
    
    for fold = 1 : Nfold
        fold
        
        train_list = importdata(['./data_processing/tf_data/cnn_filterbank_eval_eeg/train_list_n', num2str(fold, '%d'),'.txt']);
        load(['./tensorflow_net/',folder_name,'/n', num2str(fold), '/train_ret_model_acc.mat']);
        score2 = softmax(score2);
        score1 = softmax(score1);
        score3 = softmax(score3);
        
        % wilson: �ֿ�ÿ��individuals
        st_ind = 1;
        for n = 1:length(train_list.data)
            et_ind = st_ind + train_list.data(n) - 1; % update end index
            score1_temp = score1(st_ind:et_ind,:);
            score2_temp = score2(st_ind:et_ind,:);
            score3_temp = score3(st_ind:et_ind,:);
            score1_temp = [score1_temp((1+half):end,:); ones(1,Ncat)];
            score3_temp = [ones(1,Ncat); score3_temp(1:(end-half),:)];
            st_ind = et_ind + 1; % updata start index
            if(aggregation == 0)
                score = (score1_temp.* score2_temp .* score3_temp)/output_context_size;
            else
                score = (score1_temp + score2_temp + score3_temp)/output_context_size;
            end
            yhat = zeros(1,size(score,1));
            for i = 1 : size(score,1)
                [~, yhat(i)] = max(score(i,:));
            end
            yh{fold} = [yh{fold};double(yhat')];
            % save yhat
            individual = train_list.textdata{n}(27:29);
            yhat = yhat';
            mkdir(['./error_analysis/',folder_name,'/n', num2str(fold, '%d'), '/train'])
            save(['./error_analysis/',folder_name,'/n', num2str(fold, '%d'), '/train','/', individual, '.mat'], 'yhat') 
        end        
        
        % load ground-truth labels (train set)
        train_label = [];        
        for i =1:length(train_list.textdata)
            train = load(train_list.textdata{i}(5:end), 'label');
            train_label = [train_label; train.label];
        end
        yt{fold} = double(train_label); 
        
    end
    yh = cell2mat(yh);
    yt = cell2mat(yt);
    
    [train.acc, train.kappa, train.f1, train.sens, train.spec] = calculate_overall_metrics(yt, yh);
    [train.classwise_sens, train.classwise_sel]  = calculate_classwise_sens_sel(yt, yh);
    train.C = confusionmat(yt, yh);
    
    %% dev set
    yh = cell(Nfold,1);
    yt = cell(Nfold,1);
    
    for fold = 1 : Nfold
        fold
        
        eval_list = importdata(['./data_processing/tf_data/cnn_filterbank_eval_eeg/eval_list_n', num2str(fold, '%d'),'.txt']);
        load(['./tensorflow_net/',folder_name,'/n', num2str(fold), '/eval_ret_model_acc.mat']);
        score2 = softmax(score2);
        score1 = softmax(score1);
        score3 = softmax(score3);
        
        % wilson: �ֿ�ÿ��individuals
        st_ind = 1;
        for n = 1:length(eval_list.data)
            et_ind = st_ind + eval_list.data(n) - 1; % update end index
            score1_temp = score1(st_ind:et_ind,:);
            score2_temp = score2(st_ind:et_ind,:);
            score3_temp = score3(st_ind:et_ind,:);
            score1_temp = [score1_temp((1+half):end,:); ones(1,Ncat)];
            score3_temp = [ones(1,Ncat); score3_temp(1:(end-half),:)];
            st_ind = et_ind + 1; % updata start index
            if(aggregation == 0)
                score = (score1_temp.* score2_temp .* score3_temp)/output_context_size;
            else
                score = (score1_temp + score2_temp + score3_temp)/output_context_size;
            end
            yhat = zeros(1,size(score,1));
            for i = 1 : size(score,1)
                [~, yhat(i)] = max(score(i,:));
            end
            yh{fold} = [yh{fold};double(yhat')];
            % save yhat
            individual = eval_list.textdata{n}(27:29);
            yhat = yhat';
            mkdir(['./error_analysis/',folder_name,'/n', num2str(fold, '%d'), '/eval'])
            save(['./error_analysis/',folder_name,'/n', num2str(fold, '%d'), '/eval','/', individual, '.mat'], 'yhat') 
        end
      
        % load ground-truth labels (dev set)
        eval_label = [];
        for i =1:length(eval_list.textdata)
            eval = load(eval_list.textdata{i}(5:end), 'label');
            eval_label = [eval_label; eval.label];
        end
        yt{fold} = double(eval_label);
        
    end
    yh = cell2mat(yh);
    yt = cell2mat(yt);
    
    [eval.acc, eval.kappa, eval.f1, eval.sens, eval.spec] = calculate_overall_metrics(yt, yh);
    [eval.classwise_sens, eval.classwise_sel]  = calculate_classwise_sens_sel(yt, yh);
    eval.C = confusionmat(yt, yh);
end