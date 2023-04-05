function train_network(fname, maxEpochs)

fpath = '';
slash_ind = findstr(fname, '\');
if(~isempty(slash_ind))
    fpath = fname(1:slash_ind(end));
    fname = fname(slash_ind(end)+1:end);
end

load([fpath, fname,'_train_data.mat']);
load([fpath, fname,'_val_data.mat']);

%% Train the CNN
numTypes = numel(exerciseTypes);

if(~exist('maxEpochs'))
    maxEpochs = 20;
    disp(['Setting default max epochs to ',num2str(maxEpochs)]);
end

in_format = [size(dataTraining, 1) size(dataTraining,2) 1];
exClassNet = cnn(in_format, numTypes);

analyzeNetwork(exClassNet);

miniBatchSize = 16;
validationFrequency = floor(numel(dataTrainingLabel)/miniBatchSize);

 options = trainingOptions('adam', ...
      'InitialLearnRate',2e-3, ...
      'MaxEpochs',maxEpochs, ...
      'MiniBatchSize',miniBatchSize, ...
      'Shuffle','every-epoch', ...
      'Plots','training-progress', ...
      'Verbose',false, ...
      'ValidationData',{dataValidation,dataValidationLabel}, ...
      'ValidationFrequency',validationFrequency, ...
      'LearnRateSchedule', 'piecewise', ...
      'LearnRateDropPeriod', 9, ...
      'LearnRateDropFactor', 0.1, ...
      'ExecutionEnvironment', 'gpu');
    
%% 
% Either train the network or use the already trained network. By default, this 
% example uses the trained network.
trainedNet = trainNetwork(dataTraining,dataTrainingLabel,exClassNet,options);

save([fpath,fname,'_trained_net.mat'], 'trainedNet', 'exClassNet','options', 'exerciseTypes', '-v7.3');
disp('Saved trained network');

