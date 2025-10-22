clear; close all; clc;

% 1) Завантаження даних
load fisheriris % змінні: meas (150x4), species (150x1 cell)
X = meas;                 % предиктори
Y = species;              % ціль (3 класи)

% Імена предикторів
predictorNames = {'SepalLength','SepalWidth','PetalLength','PetalWidth'};

% 2) Попередня перевірка
fprintf('Rows: %d, Predictors: %d\n', size(X,1), size(X,2));
anyMissing = any(isnan(X),'all') || any(cellfun(@isempty,Y));
fprintf('Наявність пропусків: %d\n', anyMissing);

% 3) Розбиття на train/test (70/30)
rng(1,'twister'); % для відтворюваності
clear; close all; clc;

% 1) Завантаження даних
load fisheriris % змінні: meas (150x4), species (150x1 cell)
X = meas;       % предиктори
Y = species;    % ціль (3 класи)

% Імена предикторів
predictorNames = {'SepalLength','SepalWidth','PetalLength','PetalWidth'};

% 2) Попередня перевірка
fprintf('Rows: %d, Predictors: %d\n', size(X,1), size(X,2));
anyMissing = any(isnan(X),'all') || any(cellfun(@isempty,Y));
fprintf('Наявність пропусків: %d\n', anyMissing);

% 3) Розбиття на train/test (70/30)
rng(1,'twister'); % для відтворюваності
cv = cvpartition(Y,'HoldOut',0.3);
Xtrain = X(training(cv),:);
Ytrain = Y(training(cv),:);
Xtest  = X(test(cv),:);
Ytest  = Y(test(cv),:);

% 4) Тренування дерева рішень
tree = fitctree(Xtrain, Ytrain, ...
    'PredictorNames', predictorNames, ...
    'ClassNames', unique(Y), ...
    'CrossVal','off');

% 5) Текстовий опис дерева (правила)
fprintf('\n--- Текстовий опис дерева (правила) ---\n');
view(tree,'Mode','text');

% 6) Відображення повного дерева (без figure()!)
view(tree, 'Mode','graph');

% 7) Прогноз і оцінка на тесті
Ypred = predict(tree, Xtest);

confMat = confusionmat(Ytest, Ypred);
disp('Confusion matrix (rows = actual, cols = predicted):');
disp(confMat);

accuracy = sum(strcmp(Ytest,Ypred)) / numel(Ytest);
fprintf('Test accuracy: %.3f\n', accuracy);

% 8) Крос-валідація (5-fold)
cvTree = crossval(tree, 'KFold', 5);
cvLoss = kfoldLoss(cvTree);
fprintf('5-fold cross-validated loss (classification error): %.3f\n', cvLoss);

% 9) Важливість предикторів
imp = predictorImportance(tree);
tblImp = table(predictorNames', imp', 'VariableNames', {'Predictor','Importance'});
tblImp = sortrows(tblImp,'Importance','descend');
disp('Predictor importance (descending):');
disp(tblImp);

% Графік важливості
figure('Name','Predictor Importance');
bar(tblImp.Importance);
set(gca, 'XTickLabel', tblImp.Predictor);
xlabel('Predictor');
ylabel('Importance');
title('Важливість предикторів');
grid on;

% 10) Проба обрізки дерева
maxLevel = max(tree.PruneList);
errors = zeros(maxLevel+1,1);

for lvl = 0:maxLevel
    tpr = prune(tree,'Level',lvl);
    cvt = crossval(tpr,'KFold',5);
    errors(lvl+1) = kfoldLoss(cvt);
end

[bestErr, bestIdx] = min(errors);
bestLevel = bestIdx - 1;
fprintf('Best pruning level (min CV loss): %d (CV error=%.3f)\n', bestLevel, bestErr);

% 11) Обрізане дерево (без figure()!)
bestPruned = prune(tree,'Level',bestLevel);
view(bestPruned, 'Mode','graph');

% 12) Залежність помилки від рівня обрізки
figure('Name','Cross-validation error vs pruning level');
plot(0:maxLevel, errors, '-o','LineWidth',1.5);
xlabel('Рівень обрізки (Level)');
ylabel('Крос-валідаційна помилка');
title('Оцінка оптимального рівня обрізки дерева рішень');
grid on;
hold on;
plot(bestLevel, bestErr, 'ro', 'MarkerFaceColor','r');
text(bestLevel, bestErr, sprintf('  Optimal level = %d', bestLevel));
hold off;

% 13) Оцінка обрізаного дерева
YpredPruned = predict(bestPruned, Xtest);
accPruned = mean(strcmp(Ytest, YpredPruned));
fprintf('Test accuracy (pruned tree): %.3f\n', accPruned);

% 14) Збереження моделі
save('decisionTreeModel.mat','tree','bestPruned','predictorNames');
fprintf('\nМодель збережена в decisionTreeModel.mat\n');