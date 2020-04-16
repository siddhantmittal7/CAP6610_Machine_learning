
% Reading the red wine csv file.
m = dlmread("winequality-red.csv",';',1,0);
% Separating the quality value.
y = m(:,12);
% Extracting 11 feature vectors.
x = m(:,1:end-1);
% Separating testing and training data.
train_x = x(1:1400,:);
train_y = y(1:1400,:);
%disp("Size of training data:");
test_x = x(1401:end,:);
test_y = y(1401:end,:);
train_x = [train_x ones(1400,1)];
test_x = [test_x ones(199,1)];
w = mldivide(train_x,train_y);
MAE = sum(abs(test_y - test_x*w),1)/199;
disp("Mean absolution error for least square lost: " + MAE);

% huber
cvx_begin
    variable b_huber(12,1);
    minimize( sum(huber(train_x*b_huber - train_y,1)) );
cvx_end

MAE = sum(abs(test_y - test_x*b_huber),1)/199;
disp("Mean absolution error for Huber loss: " + MAE);

% hinge_loss 
cvx_begin
    variable b_hinge(12,1);
    minimize(sum(hinge_loss( train_x*b_hinge - train_y )));
cvx_end

MAE = sum(abs(test_y - test_x*b_hinge),1)/199;
disp("Mean absolution error for Hinge loss: " + MAE);


function y = hinge_loss( x )
    y = max( abs( x ) - 0.5, 0 );
end


