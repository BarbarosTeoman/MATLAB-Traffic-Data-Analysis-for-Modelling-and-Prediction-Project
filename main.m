clc
clear

function datas = loadOrCacheNGSIMData(matCacheFile)
    if nargin < 1
        matCacheFile = 'ngsim_cached_data.mat';
    end

    if isfile(matCacheFile)
        disp("Loading cached data from: " + matCacheFile);
        load(matCacheFile, 'datas');
        disp(matCacheFile + " is loaded!")
        return;
    end

    varNames = {'Vehicle_ID', ...
                'Frame_ID', ...
                'Total_Frames', ...
                'Global_Time', ...
                'Local_X', ...
                'Local_Y', ...
                'Global_X', ...
                'Global_Y', ...
                'v_Length', ...
                'v_Width', ...
                'v_Class', ...
                'v_Vel', ...
                'v_Acc', ...
                'Lane_ID', ...
                'Preceeding', ...
                'Following', ...
                'Space_Hdwy', ...
                'Time_Hdwy'};

    files = { 'data/trajectories-0750am-0805am.txt', ...
              'data/trajectories-0805am-0820am.txt', ...
              'data/trajectories-0820am-0835am.txt', ...
              'data/trajectories-0515-0530.txt', ...
              'data/trajectories-0500-0515.txt', ...
              'data/trajectories-0400-0415.txt' };

    datas = cell(1, numel(files));
    for i = 1:numel(files)
        disp("Loading: " + files{i});
        raw = load(files{i});
        datas{i} = array2table(raw, 'VariableNames', varNames);
    end

    save(matCacheFile, 'datas');
    disp("Data cached to: " + matCacheFile);
end

% datas = loadOrCacheNGSIMData();

function plot_vehicle_frames(carID, frameIDRange, data, boxWidth, boxHeight)
    arguments
        carID double
        frameIDRange double
        data table
        boxWidth double = 36
        boxHeight double = 195
    end

    figure;
    for idx = 1:length(frameIDRange)
        frameID = frameIDRange(idx);
        clf;
        plotCars(carID, frameID, data, boxWidth, boxHeight);
        drawnow;
        pause(0.1);
    end
end


function plotCars(carID, frameID, data, boxWidth, boxHeight)
    arguments (Input)
        carID double
        frameID double
        data table
        boxWidth double
        boxHeight double
    end

    vhclInBox = vhclBox(carID, frameID, data, boxWidth, boxHeight);
    
    minX = -boxWidth / 2;
    maxX =  boxWidth / 2;
    minY = -boxHeight / 2;
    maxY =  boxHeight / 2;

    axis equal;
    grid off;
    xlabel('Relative X (ft)');
    ylabel('Relative Y (ft)');
    title(['Vehicle Surroundings - Frame ', num2str(frameID)]);
    xlim([minX, maxX]);
    ylim([minY, maxY]);
    for x = -boxWidth/2 : 12 : boxWidth/2
        line([x x], [minY maxY], 'Color', [0.8 0.8 0.8], 'LineStyle', '--');
    end
    
    for y = -boxHeight/2 : 15 : boxHeight/2
        line([minX maxX], [y y], 'Color', [0.8 0.8 0.8], 'LineStyle', '--');
    end

    for i = 1:height(vhclInBox)
        xCenter = vhclInBox.XDiff(i);
        yCenter = vhclInBox.YDiff(i);
        vLength = vhclInBox.Length(i);
        vWidth = vhclInBox.Width(i);
        vSpeed = vhclInBox.Speed(i);
        lane = vhclInBox.Lane(i);
        vehID = vhclInBox.Vehicle_ID(i);

        xLeft = xCenter - vWidth / 2;
        yBottom = yCenter - vLength / 2;

        if vehID == carID
            faceColor = [0.2 0.8 0.2];
        else
            faceColor = [0.5 0.8 0.9];
        end

        rectangle('Position', [xLeft, yBottom, vWidth, vLength], ...
                  'FaceColor', faceColor, 'EdgeColor', 'k');

        text(xCenter + 5, yCenter, string(vehID), ...
             'FontSize', 8, 'Color', 'k');
        
        text(xCenter + 5, yCenter - 5, sprintf('%.2f feet/s', vSpeed), ...
             'FontSize', 8, 'Color', 'k');

        text(xCenter + 5, yCenter - 10, sprintf('line %d', lane), ...
             'FontSize', 8, 'Color', 'k');

        lineLength = vSpeed / 3;
        line([xCenter, xCenter], ...
             [yCenter - vLength/2, yCenter - vLength/2 - lineLength], ...
             'Color', 'r', 'LineWidth', 2);
    end
end

function [vhclInBox, occupancyGrid] = vhclBox(carID, frameID, data, boxWidth, boxHeight)
    vhcl = data(data.Vehicle_ID == carID & data.Frame_ID == frameID, :);
    vhclInFrame = data(data.Frame_ID == frameID, :);

    occupancyGrid = zeros(39,1);

    if isempty(vhcl)
        vhclInBox = table();
        return;
    end

    vhclX = vhcl.Local_X;
    vhclY = vhcl.Local_Y - vhcl.v_Length / 2;

    vhclInFrameX = vhclInFrame.Local_X;
    vhclInFrameY = vhclInFrame.Local_Y - vhclInFrame.v_Length / 2;

    vhclXDiff = vhclInFrameX - vhclX;
    vhclYDiff = vhclInFrameY - vhclY;

    vhclInFrame = table(vhclInFrame.Vehicle_ID, vhclXDiff, vhclYDiff, ...
        vhclInFrame.v_Length, vhclInFrame.v_Width, vhclInFrame.v_Vel, vhclInFrame.Lane_ID);

    vhclInFrame = renamevars(vhclInFrame, ...
        ["Var1", "Var2", "Var3", "Var4", "Var5", "Var6", "Var7"], ...
        ["Vehicle_ID", "XDiff", "YDiff", "Length", "Width", "Speed", "Lane"]);

    withinX = abs(vhclInFrame.XDiff) <= boxWidth / 2;
    withinY = abs(vhclInFrame.YDiff) <= boxHeight / 2;
    vhclInBox = vhclInFrame(withinX & withinY, :);

    cellWidth = 12;
    cellHeight = 15;

    for i = 1:height(vhclInBox)
        x = vhclInBox.XDiff(i);
        y = vhclInBox.YDiff(i);

        col = floor((x + boxWidth/2) / cellWidth) + 1;
        row = floor((y + boxHeight/2) / cellHeight) + 1;

        if row >= 1 && row <= 13 && col >= 1 && col <= 3
            occupancyGrid((13-row)*3+col) = 1;
        end
    end
end

function gridMatrix = get_occupancy_grid_over_frames(carID, frameIDRange, data, boxWidth, boxHeight)
    arguments
        carID double
        frameIDRange double
        data table
        boxWidth double = 36
        boxHeight double = 195
    end

    numFrames = length(frameIDRange);
    gridMatrix = zeros(39, numFrames);

    for idx = 1:numFrames
        frameID = frameIDRange(idx);
        [~, occupancyGrid] = vhclBox(carID, frameID, data, boxWidth, boxHeight);
        gridMatrix(:, idx) = occupancyGrid;
    end
end

function targetData = get_target_data_over_frames(carID, frameIDRange, data)
    arguments
        carID double
        frameIDRange double
        data table
    end

    numFrames = length(frameIDRange);
    targetData = zeros(numFrames, 3);

    for idx = 1:numFrames
        frameID = frameIDRange(idx);
        targetData(idx, :) = table2array(data(data.Vehicle_ID == carID & data.Frame_ID == frameID, [5,6,12]));
    end
end 

function car_chunks = get_occupancy_grid_chunks(carID, data, chunkSize)
    arguments
        carID double
        data table
        chunkSize double = 60
    end

    carRows = data(data.Vehicle_ID == carID, :);

    allFrames = unique(carRows.Frame_ID);
    totalFrames = carRows.Total_Frames(1);

    if length(allFrames) < chunkSize
        car_chunks = {};
        return;
    end

    numChunks = floor(totalFrames / chunkSize);
    car_chunks = [];
    
    for i = 1:numChunks
        frameIdxStart = (i - 1) * chunkSize + 1;
        frameRange = allFrames(frameIdxStart : frameIdxStart + chunkSize - 1);
    
        occMatrix = get_occupancy_grid_over_frames(carID, frameRange, data);
    
        chunkX = occMatrix(:, 1:30);
        reshaped = reshape(chunkX, 3, 13, 30);
        transposedStack = permute(reshaped, [2 1 3]);
    
        chunkY = occMatrix(:, 31:60);
        reshaped2 = reshape(chunkY, 3, 13, 30);
        transposedStack2 = permute(reshaped2, [2 1 3]);
    
        pairChunk = cat(4, transposedStack, transposedStack2);
    
        if isempty(car_chunks)
            car_chunks = pairChunk;
        else
            car_chunks = cat(4, car_chunks, pairChunk);
        end
    end
end

function target_chunks = get_target_data_chunks(carID, data, chunkSize)
    arguments
        carID double
        data table
        chunkSize double = 60
    end

    carRows = data(data.Vehicle_ID == carID, :);

    allFrames = unique(carRows.Frame_ID);
    totalFrames = carRows.Total_Frames(1);

    if length(allFrames) < chunkSize
        target_chunks = {};
        return;
    end

    numChunks = floor(totalFrames / chunkSize);
    target_chunks = [];
    
    for i = 1:numChunks
        frameIdxStart = (i - 1) * chunkSize + 1;
        frameRange = allFrames(frameIdxStart : frameIdxStart + chunkSize - 1);
    
        targetMatrix = get_target_data_over_frames(carID, frameRange, data);
    
        chunkX = targetMatrix(1:30, :);
        chunkY = targetMatrix(31:60, :);
    
        pairChunk = cat(3, chunkX, chunkY);
    
        if isempty(target_chunks)
            target_chunks = pairChunk;
        else
            target_chunks = cat(3, target_chunks, pairChunk);
        end
    end
end

function all_chunks_combined = all_chunks(datas)

    arguments
        datas cell
    end

    all_chunks_cell = {};

    for i = 1:numel(datas)
        data = datas{i};
        vhcls = unique(data.Vehicle_ID);
        n = length(vhcls);
    
        temp_chunks = cell(n, 1);
    
        for j = 1:n
            carID = vhcls(j);
            fprintf('Processing carID %d\n', carID)
            chunks = get_occupancy_grid_chunks(carID, data);
    
            if ~isempty(chunks)
                temp_chunks{j} = chunks; 
            end
        end
    
        for j = 1:n
            if ~isempty(temp_chunks{j})
                all_chunks_cell{end+1} = temp_chunks{j};
            end
        end
    end
    
    all_chunks_combined = cat(4, all_chunks_cell{:});

end

function all_target_chunks_combined = all_target_chunks(datas)

    arguments
        datas cell
    end

    all_chunks_cell = {};

    for i = 1:numel(datas)
        data = datas{i};
        vhcls = unique(data.Vehicle_ID);
        n = length(vhcls);
    
        temp_chunks = cell(n, 1);
    
        parfor j = 1:n
            carID = vhcls(j);
            fprintf('Processing carID %d, data %d, car left %d\n', carID, i, n-j)
            chunks = get_target_data_chunks(carID, data);
    
            if ~isempty(chunks)
                temp_chunks{j} = chunks; 
            end
        end
    
        for j = 1:n
            if ~isempty(temp_chunks{j})
                all_chunks_cell{end+1} = temp_chunks{j};
            end
        end
    end
    
    all_target_chunks_combined = cat(3, all_chunks_cell{:});

end

disp("Train, validation and test datasets are loading...");
load TrainDS.mat
load ValDS.mat
load Test.mat
disp("Train, validation and test datasets are loaded!");

function [trainedNetwork, trainingInfo] = trainNetwork(TrainDS, ValDS)

    if isfile("trainedNetwork.mat") && isfile("trainingInfo.mat")
        disp("Loading pre-trained network and training info...");
        load("trainedNetwork.mat", "trainedNetwork");
        load("trainingInfo.mat", "trainingInfo");
        disp("Pre-trained network and training info loaded!");
        return;
    end

    options = trainingOptions( ...
        "adam", ...
        Plots = "training-progress", ...
        ValidationData = ValDS, ...
        ValidationFrequency = 50, ...
        ValidationPatience = 5, ...
        ExecutionEnvironment = "parallel-auto", ...
        MaxEpochs = 10, ...
        InitialLearnRate = 0.001, ...
        MiniBatchSize = 256, ...
        L2Regularization = 0.0001, ...
        Verbose = false, ...
        LearnRateSchedule = "polynomial");

    numHiddenUnits = 64;
    numNeurons = 128;

    layers1 = [
        sequenceInputLayer([13 3 1], "Name", "seq1", "MinLength", 30)
        convolution2dLayer(3, 32, "Padding", "same", "Name", "conv1")
        batchNormalizationLayer("Name", "batch1")
        reluLayer("Name", "relu1")
        maxPooling2dLayer(2, "Stride", 2, "Name", "maxpool1", "Padding", "same")
        convolution2dLayer(3, 64, "Padding", "same", "Name", "conv2")
        batchNormalizationLayer("Name", "batch2")
        reluLayer("Name", "relu2")
        maxPooling2dLayer(2, "Stride", 2, "Name", "maxpool2", "Padding", "same")
        flattenLayer("Name", "flatten")
        bilstmLayer(numHiddenUnits, "OutputMode", "sequence", "Name", "bilstm1")
        concatenationLayer(1, 2, "Name", "cat")
        fullyConnectedLayer(numNeurons)
        fullyConnectedLayer(3)
    ];

    layers2 = [
        sequenceInputLayer(3, "Name", "seq2", "MinLength", 30)
        bilstmLayer(32, "OutputMode", "sequence", "Name", "bilstm2")
    ];

    net = dlnetwork;
    net = addLayers(net, layers1);
    net = addLayers(net, layers2);
    net = connectLayers(net, "bilstm2", "cat/in2");

    [trainedNetwork, trainingInfo] = trainnet(TrainDS, net, "l2loss", options);

    save("trainedNetwork.mat", "trainedNetwork");
    save("trainingInfo.mat", "trainingInfo");
end

[trainedNetwork, trainingInfo] = trainNetwork(TrainDS, ValDS);

function results = evaluateModel(trainedNetwork, Test)
    predictedTarget = minibatchpredict(trainedNetwork, Test.X1Test(:,:,:,:,:), Test.X2Test(:,:,:), MiniBatchSize=256, UniformOutput=false);

    nSamples = numel(predictedTarget);
    
    YPredMat = zeros(size(Test.YTest));
    
    for i = 1:nSamples
        YPredMat(:,:,i) = predictedTarget{i}; 
    end
    
    squaredError = (YPredMat - Test.YTest).^2;
    mse = mean(squaredError, 'all');

    results.mse = mse;
    results.YPredMat = YPredMat;
end

results = evaluateModel(trainedNetwork, Test);

disp("Mean-squared error is " + results.mse)