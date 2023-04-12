function exClassNet = cnn(in_format, numTypes)

exClassNet = [
  imageInputLayer(in_format, 'Normalization', 'none', 'Name', 'Input Layer')
  
  convolution2dLayer([2 3], 16, 'Padding', 'same', 'Name', 'CNN1')
  batchNormalizationLayer('Name', 'BN1')
  reluLayer('Name', 'ReLU1')
  maxPooling2dLayer([1 2], 'Stride', [1 2], 'Name', 'MaxPool1')
  
  convolution2dLayer([2 3], 16, 'Padding', 'same', 'Name', 'CNN2')
  batchNormalizationLayer('Name', 'BN2')
  reluLayer('Name', 'ReLU2')
  maxPooling2dLayer([1 4], 'Stride', [1 4], 'Name', 'MaxPool2')
  
  convolution2dLayer([2 3], 8, 'Padding', 'same', 'Name', 'CNN3')
  batchNormalizationLayer('Name', 'BN3')
  reluLayer('Name', 'ReLU3')
  maxPooling2dLayer([3 2], 'Stride', [1 2], 'Name', 'MaxPool3')
  %flattenLayer();
  
  fullyConnectedLayer(16, 'Name', 'FC1');
  reluLayer('Name', 'ReLU4')
  
  fullyConnectedLayer(numTypes, 'Name', 'FC2')
  softmaxLayer('Name', 'SoftMax')
  
  classificationLayer('Name', 'Output') ];
