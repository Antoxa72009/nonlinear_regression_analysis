clc;
clear;
close all;

% Підготовка даних
X = (0:0.1:10)';                     % Вхід
Y = sin(X) + 0.1*randn(size(X));     % Вихід з шумом
data = [X Y];                         % Об’єднання в таблицю для ANFIS

% Створення початкової нечіткої системи
numMFs = 3;                           % Кількість нечітких множин для входу
mfType = 'gaussmf';                   % Тип нечіткої функції (Gaussian)
fis = genfis1(data, numMFs, mfType);

% Навчання ANFIS
numEpochs = 50;                        % Кількість епох навчання
[fis, trainError] = anfis(data, fis, numEpochs);

% Перевірка моделі
Y_pred = evalfis(fis, X);

% Візуалізація результатів
figure;
plot(X, Y, 'b', 'LineWidth', 1.5); hold on;
plot(X, Y_pred, 'r--', 'LineWidth', 1.5);
legend('Справжні дані', 'Прогноз ANFIS');
title('Прогнозування з використанням нейро-нечіткої мережі');
xlabel('X');
ylabel('Y');
grid on;

% Графік помилки навчання
figure;
plot(trainError, 'LineWidth', 1.5);
title('Помилка навчання ANFIS');
xlabel('Епоха');
ylabel('Помилка');
grid on;