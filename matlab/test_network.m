function test_network(net_fname, test_fname)

load(net_fname);
load(test_fname);

%% 
% Evaluate the trained network by obtaining the classification accuracy for 
% the test frames. The results show that the network achieves about 90% accuracy 
% for this group of waveforms.

fprintf('Classifying test frames\n')
dataTestPred = classify(trainedNet,dataTest);
testAccuracy = mean(dataTestPred == dataTestLabel);
disp("Test accuracy: " + testAccuracy*100 + "%")

figure
cm = confusionchart(dataTestLabel, dataTestPred);
cm.Title = 'Confusion Matrix for Test Data';
cm.RowSummary = 'row-normalized';
cm.Parent.Position = [cm.Parent.Position(1:2) 740 424];
