function [train_data,test_data]=LSTM_data_process()
%% 数据加载
train_data_initial= load('shuru_1.txt');
train_data_initial= train_data_initial';

test_data_initial= load('shuchu_1.txt');
test_data_initial= test_data_initial';

data_length=size(train_data_initial,1);            %每个样本的长度
data_num=size(train_data_initial,2);               %样本数目  

for n=1:data_num
      train_data(:,n)=train_data_initial(:,n);
end
for m=1:size(test_data_initial,2)
      test_data(:,m)=test_data_initial(:,m);
end