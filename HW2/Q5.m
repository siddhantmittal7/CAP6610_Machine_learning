load ionosphere
ionosphere = array2table(X);
ionosphere.Group = Y;
y = ones(351,1);
for i=1:351
    if Y{i,:} == 'g'
        y(i,:) = 1;
    else
        y(i,:) = -1;
    end
end
%disp(y);
train_x = X(1:300,:);
train_y = y(1:300,:);
test_x = X(301:end,:);
test_y = y(301:end,:);
% Adding constant term in our model
train_x = [train_x ones(300,1)];
test_x = [test_x ones(51,1)];

w = mldivide(train_x,train_y);

correct = 0;

for i=1:51
    y_pred = (test_x(i,:)*w);
    %disp("predicted y: " + y_pred);
    %disp("actual y: " + test_y(i,:));
    if y_pred > 0 && test_y(i,:) == 1
        correct = correct + 1;
    end
    if y_pred < 0 && test_y(i,:) == -1
        correct = correct + 1;
    end
end

disp("Prediction accurary for least square loss: " + (correct/51)*100 + "%");

%logistic loss implementation
cvx_begin
    variable b_logistic(35);
    minimize(sum(log( 1 + exp(-train_y.*(train_x*b_logistic) ))));
cvx_end

correct = 0;

for i=1:51
    y_pred = test_x(i,:)*b_logistic;
    %disp("predicted y: " + y_pred);
    %disp("actual y: " + test_y(i,:));
    if y_pred > 0 && test_y(i,:) == 1
        correct = correct + 1;
    end
    if y_pred < 0 && test_y(i,:) == -1
        correct = correct + 1;
    end
end

disp("Prediction accurary for logistic loss: " + (correct/51)*100 + "%");



%hinge_loss implementation
cvx_begin
    variable b_hinge(35,1);
    minimize(sum(hinge_loss(train_y, train_x*b_hinge )));
cvx_end


correct = 0;

for i=1:51
    y_pred = test_x(i,:)*b_hinge;
    if y_pred > 0 && test_y(i,:) == 1
        correct = correct + 1;
    end
    if y_pred < 0 && test_y(i,:) == -1
        correct = correct + 1;
    end
end

disp("Prediction accurary for hinge lost: " + (correct/51)*100 + "%");

function y = hinge_loss( y,t)
    y = max( 1 - y.*t, 0 );
end

