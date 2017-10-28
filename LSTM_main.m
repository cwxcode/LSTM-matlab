%% LSTM网络
clear all;
clc;
tic  %计时
%% 数据加载
[train_data,test_data]=LSTM_data_process();
data_length=size(train_data,1);
data_num=size(train_data,2);
%% 网络参数初始化
% 结点数设置
input_num=1000;  
cell_num=50;  %隐藏结点数量
output_num=100;  
% 网络中门的偏置
bias_input_gate=rand(1,cell_num);
bias_forget_gate=rand(1,cell_num);
bias_output_gate=rand(1,cell_num);
%网络权重初始化
ab=20;
weight_input_x=rand(input_num,cell_num)/ab;
weight_input_h=rand(output_num,cell_num)/ab;
weight_inputgate_x=rand(input_num,cell_num)/ab;
weight_inputgate_c=rand(cell_num,cell_num)/ab;
weight_forgetgate_x=rand(input_num,cell_num)/ab;
weight_forgetgate_c=rand(cell_num,cell_num)/ab;
weight_outputgate_x=rand(input_num,cell_num)/ab;
weight_outputgate_c=rand(cell_num,cell_num)/ab;

%hidden_output权重
weight_preh_h=rand(cell_num,output_num);

%网络状态初始化
cost_gate=1e-4;
h_state=rand(output_num,data_num);
cell_state=rand(cell_num,data_num);
%% 网络训练学习
for iter=1:10000  %迭代次数
    yita=0.01;   %每次迭代权重调整比例0.01
    for m=1:data_num
        %前馈部分
        if(m==1)
            gate=tanh(train_data(:,m)'*weight_input_x);
            input_gate_input=train_data(:,m)'*weight_inputgate_x+bias_input_gate;
            output_gate_input=train_data(:,m)'*weight_outputgate_x+bias_output_gate;
            for n=1:cell_num
                input_gate(1,n)=1/(1+exp(-input_gate_input(1,n)));
                output_gate(1,n)=1/(1+exp(-output_gate_input(1,n)));
            end
            forget_gate=zeros(1,cell_num);
            forget_gate_input=zeros(1,cell_num);
            cell_state(:,m)=(input_gate.*gate)';
        else
            gate=tanh(train_data(:,m)'*weight_input_x+h_state(:,m-1)'*weight_input_h);
            input_gate_input=train_data(:,m)'*weight_inputgate_x+cell_state(:,m-1)'*weight_inputgate_c+bias_input_gate;
            forget_gate_input=train_data(:,m)'*weight_forgetgate_x+cell_state(:,m-1)'*weight_forgetgate_c+bias_forget_gate;
            output_gate_input=train_data(:,m)'*weight_outputgate_x+cell_state(:,m-1)'*weight_outputgate_c+bias_output_gate;
            for n=1:cell_num
                input_gate(1,n)=1/(1+exp(-input_gate_input(1,n)));
                forget_gate(1,n)=1/(1+exp(-forget_gate_input(1,n)));
                output_gate(1,n)=1/(1+exp(-output_gate_input(1,n)));
            end
            cell_state(:,m)=(input_gate.*gate+cell_state(:,m-1)'.*forget_gate)';   
        end
        pre_h_state=tanh(cell_state(:,m)').*output_gate;
        h_state(:,m)=(pre_h_state*weight_preh_h)';
        %误差计算
        Error=h_state(:,m)-test_data(:,m);
        Error_Cost(1,iter)=sum(Error.^2);
        if(Error_Cost(1,iter)<cost_gate)
            flag=1;
            break;
        else
            [   weight_input_x,...
                weight_input_h,...
                weight_inputgate_x,...
                weight_inputgate_c,...
                weight_forgetgate_x,...
                weight_forgetgate_c,...
                weight_outputgate_x,...
                weight_outputgate_c,...
                weight_preh_h ]=LSTM_updata_weight(m,yita,Error,...
                                                   weight_input_x,...
                                                   weight_input_h,...
                                                   weight_inputgate_x,...
                                                   weight_inputgate_c,...
                                                   weight_forgetgate_x,...
                                                   weight_forgetgate_c,...
                                                   weight_outputgate_x,...
                                                   weight_outputgate_c,...
                                                   weight_preh_h,...
                                                   cell_state,h_state,...
                                                   input_gate,forget_gate,...
                                                   output_gate,gate,...
                                                   train_data,pre_h_state,...
                                                   input_gate_input,...
                                                   output_gate_input,...
                                                   forget_gate_input);

        end
    end
    if(Error_Cost(1,iter)<cost_gate)
        break;
    end
end
%% 绘制Error-Cost曲线图
for n=1:1:iter
    semilogy(n,Error_Cost(1,n),'*');
    hold on;
    title('Error-Cost曲线图');   
end
%% 数据检验
%数据加载
test_final= load('test_1.txt');
test_final= test_final';
test_output=test_data(:,7);  %第7组数据作测试
%前馈
m=7;  %输出向量的个数
gate=tanh(test_final'*weight_input_x+h_state(:,m-1)'*weight_input_h);
input_gate_input=test_final'*weight_inputgate_x+cell_state(:,m-1)'*weight_inputgate_c+bias_input_gate;
forget_gate_input=test_final'*weight_forgetgate_x+cell_state(:,m-1)'*weight_forgetgate_c+bias_forget_gate;
output_gate_input=test_final'*weight_outputgate_x+cell_state(:,m-1)'*weight_outputgate_c+bias_output_gate;
for n=1:cell_num
    input_gate(1,n)=1/(1+exp(-input_gate_input(1,n)));
    forget_gate(1,n)=1/(1+exp(-forget_gate_input(1,n)));
    output_gate(1,n)=1/(1+exp(-output_gate_input(1,n)));
end
cell_state_test=(input_gate.*gate+cell_state(:,m-1)'.*forget_gate)';
pre_h_state=tanh(cell_state_test').*output_gate;
h_state_test=(pre_h_state*weight_preh_h)'  %输出
test_output  %输出

x=1:100;
figure;
plot(x,h_state_test,'r-o',x,test_output,'b--+');
legend('预测值','实际值');
xlabel('时间');
ylabel('某金融标的价格数据');
title('验证集拟合度检验');

%% 预测
%数据加载
test_final= load('predict_1.txt');
test_final= test_final';
%前馈
m=7;  %m=4要改，输出文件的行数
gate=tanh(test_final'*weight_input_x+h_state(:,m-1)'*weight_input_h);
input_gate_input=test_final'*weight_inputgate_x+cell_state(:,m-1)'*weight_inputgate_c+bias_input_gate;
forget_gate_input=test_final'*weight_forgetgate_x+cell_state(:,m-1)'*weight_forgetgate_c+bias_forget_gate;
output_gate_input=test_final'*weight_outputgate_x+cell_state(:,m-1)'*weight_outputgate_c+bias_output_gate;
for n=1:cell_num
    input_gate(1,n)=1/(1+exp(-input_gate_input(1,n)));
    forget_gate(1,n)=1/(1+exp(-forget_gate_input(1,n)));
    output_gate(1,n)=1/(1+exp(-output_gate_input(1,n)));
end
cell_state_test=(input_gate.*gate+cell_state(:,m-1)'.*forget_gate)';
pre_h_state=tanh(cell_state_test').*output_gate;
h_state_test=(pre_h_state*weight_preh_h)'  %输出

x=1:100;
figure;
plot(x,h_state_test,'r-o');
legend('预测值');
xlabel('时间');
ylabel('某金融标的价格数据');
title('预测未来情况');

toc