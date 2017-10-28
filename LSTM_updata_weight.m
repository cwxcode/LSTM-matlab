function [   weight_input_x,weight_input_h,weight_inputgate_x,weight_inputgate_c,weight_forgetgate_x,weight_forgetgate_c,weight_outputgate_x,weight_outputgate_c,weight_preh_h ]=LSTM_updata_weight(n,yita,Error,...
                                                   weight_input_x, weight_input_h, weight_inputgate_x,weight_inputgate_c,weight_forgetgate_x,weight_forgetgate_c,weight_outputgate_x,weight_outputgate_c,weight_preh_h,...
                                                   cell_state,h_state,input_gate,forget_gate,output_gate,gate,train_data,pre_h_state,input_gate_input, output_gate_input,forget_gate_input)
%%% 权重更新函数
input_num=1000;  %这里也要改
cell_num=50;
output_num=100;
data_length=size(train_data,1);
data_num=size(train_data,2);
weight_preh_h_temp=weight_preh_h;

%% 更新weight_preh_h权重
for m=1:output_num
    delta_weight_preh_h_temp(:,m)=2*Error(m,1)*pre_h_state;
end
weight_preh_h_temp=weight_preh_h_temp-yita*delta_weight_preh_h_temp;

%% 更新weight_outputgate_x
for num=1:output_num
    for m=1:data_length
        delta_weight_outputgate_x(m,:)=(2*weight_preh_h(:,num)*Error(num,1).*tanh(cell_state(:,n)))'.*exp(-output_gate_input).*(output_gate.^2)*train_data(m,n);
    end
    weight_outputgate_x=weight_outputgate_x-yita*delta_weight_outputgate_x;
end
%% 更新weight_inputgate_x
for num=1:output_num
for m=1:data_length
    delta_weight_inputgate_x(m,:)=2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*gate.*exp(-input_gate_input).*(input_gate.^2)*train_data(m,n);
end
weight_inputgate_x=weight_inputgate_x-yita*delta_weight_inputgate_x;
end


if(n~=1)
    %% 更新weight_input_x
    temp=train_data(:,n)'*weight_input_x+h_state(:,n-1)'*weight_input_h;
    for num=1:output_num
    for m=1:data_length
        delta_weight_input_x(m,:)=2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*input_gate.*(ones(size(temp))-tanh(temp.^2))*train_data(m,n);
    end
    weight_input_x=weight_input_x-yita*delta_weight_input_x;
    end
    %% 更新weight_forgetgate_x
    for num=1:output_num
    for m=1:data_length
        delta_weight_forgetgate_x(m,:)=2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*cell_state(:,n-1)'.*exp(-forget_gate_input).*(forget_gate.^2)*train_data(m,n);
    end
    weight_forgetgate_x=weight_forgetgate_x-yita*delta_weight_forgetgate_x;
    end
    %% 更新weight_inputgate_c
    for num=1:output_num
    for m=1:cell_num
        delta_weight_inputgate_c(m,:)=2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*gate.*exp(-input_gate_input).*(input_gate.^2)*cell_state(m,n-1);
    end
    weight_inputgate_c=weight_inputgate_c-yita*delta_weight_inputgate_c;
    end
    %% 更新weight_forgetgate_c
    for num=1:output_num
    for m=1:cell_num
        delta_weight_forgetgate_c(m,:)=2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*cell_state(:,n-1)'.*exp(-forget_gate_input).*(forget_gate.^2)*cell_state(m,n-1);
    end
    weight_forgetgate_c=weight_forgetgate_c-yita*delta_weight_forgetgate_c;
    end
    %% 更新weight_outputgate_c
    for num=1:output_num
    for m=1:cell_num
        delta_weight_outputgate_c(m,:)=2*(weight_preh_h(:,num)*Error(num,1))'.*tanh(cell_state(:,n))'.*exp(-output_gate_input).*(output_gate.^2)*cell_state(m,n-1);
    end
    weight_outputgate_c=weight_outputgate_c-yita*delta_weight_outputgate_c;
    end
    %% 更新weight_input_h
    temp=train_data(:,n)'*weight_input_x+h_state(:,n-1)'*weight_input_h;
    for num=1:output_num
    for m=1:output_num
        delta_weight_input_h(m,:)=2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*input_gate.*(ones(size(temp))-tanh(temp.^2))*h_state(m,n-1);
    end
    weight_input_h=weight_input_h-yita*delta_weight_input_h;
    end
else
    %% 更新weight_input_x
    temp=train_data(:,n)'*weight_input_x;
    for num=1:output_num
    for m=1:data_length
        delta_weight_input_x(m,:)=2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*input_gate.*(ones(size(temp))-tanh(temp.^2))*train_data(m,n);
    end
    weight_input_x=weight_input_x-yita*delta_weight_input_x;
    end
end
weight_preh_h=weight_preh_h_temp;

end