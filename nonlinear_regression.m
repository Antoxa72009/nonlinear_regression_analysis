clear; clc; close all;

x = [80 90 100 110 120 130 140 150 160 170]';   % Стабілізована глюкоза
y = [5.2 5.5 5.9 6.4 7.1 7.5 8.1 8.7 9.0 9.3]'; % Гемоглобін

% Генерація нелінійних членів
x2 = x.^2;
x3 = x.^3;
x_inv = 1./x;   % для оберненої функції

% Лінійні моделі за параметрами
% Поліном 2-го ступеня
mdl2 = fitlm([x x2], y);
% Поліном 3-го ступеня
mdl3 = fitlm([x x2 x3], y);
% Зворотна функція
mdl_inv = fitlm(x_inv, y);

% Нелінійні моделі за параметрами
% Степенева модель: y = a * x^b
f_power = @(b,x) b(1)*x.^b(2);
beta0_power = [1,1];
mdl_power = fitnlm(x, y, f_power, beta0_power);
% Експоненціальна модель: y = a * exp(b*x)
f_exp = @(b,x) b(1)*exp(b(2)*x);
beta0_exp = [1,0.01];
mdl_exp = fitnlm(x, y, f_exp, beta0_exp);

% 5. Побудова графіків
x_plot = linspace(min(x), max(x), 100)';
y2 = predict(mdl2, [x_plot x_plot.^2]);
y3 = predict(mdl3, [x_plot x_plot.^2 x_plot.^3]);
y_inv_plot = predict(mdl_inv, 1./x_plot);
y_power_plot = predict(mdl_power, x_plot);
y_exp_plot = predict(mdl_exp, x_plot);

figure('Color','w'); hold on; grid on;
plot(x, y, 'ko', 'MarkerFaceColor','y', 'MarkerSize',8);
plot(x_plot, y2, 'b-', 'LineWidth',1.5);
plot(x_plot, y3, 'r-', 'LineWidth',1.5);
plot(x_plot, y_inv_plot, 'g--', 'LineWidth',1.5);
plot(x_plot, y_power_plot, 'm-.', 'LineWidth',1.5);
plot(x_plot, y_exp_plot, 'c:', 'LineWidth',1.5);
legend('Дані','Поліном 2-го','Поліном 3-го','Зворотна','Степенева','Експоненціальна','Location','northwest');
xlabel('Стабілізована глюкоза (x)');
ylabel('Гемоглобін (y)');
title('Порівняння нелінійних моделей');

% Порівняння моделей за R^2
R2 = [
    mdl2.Rsquared.Ordinary;
    mdl3.Rsquared.Ordinary;
    mdl_inv.Rsquared.Ordinary;
    mdl_power.Rsquared.Ordinary;
    mdl_exp.Rsquared.Ordinary
];

ModelNames = {'Поліном 2','Поліном 3','Зворотна','Степенева','Експоненціальна'};
T = table(ModelNames', R2, 'VariableNames', {'Модель','R2'});
disp('=== Порівняння моделей за R^2 ===');
disp(T);
% Висновок
[~,idx_best] = max(R2);
fprintf('\nНайкраща модель: %s (R^2 = %.3f)\n', ModelNames{idx_best}, R2(idx_best));