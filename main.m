clc
clear

function [TrainDS, ValDS, Test] = loadOrCacheNGSIMData(matCacheFile)
% loadOrCacheNGSIMData loads or processes the NGSIM traffic dataset, caches it,
% prepares it for deep learning training, and returns the Train/Validation/Test sets.
%
% INPUT:
%   - matCacheFile (string, optional): Path to a .mat file used for caching 
%     the processed NGSIM data. Default = 'ngsim_cached_data.mat'
%
% OUTPUT:
%   - TrainDS: Combined datastore of input features and targets for training
%   - ValDS: Combined datastore for validation
%   - Test: Structure containing test inputs and targets (fields: X1Test, X2Test, YTest)
arguments (Input)
    matCacheFile string
end

arguments (Output)
    TrainDS 
    ValDS 
    Test struct
end

    if nargin < 1
        matCacheFile = 'ngsim_cached_data.mat';
    end

    if isfile(matCacheFile)
        disp("Loading cached data from: " + matCacheFile);
        load(matCacheFile, 'datas');
        disp(matCacheFile + " is loaded!");
    else
        % Variable names in raw data files
        varNames = {'Vehicle_ID', 'Frame_ID', 'Total_Frames', 'Global_Time', ...
            'Local_X', 'Local_Y', 'Global_X', 'Global_Y', ...
            'v_Length', 'v_Width', 'v_Class', 'v_Vel', 'v_Acc', ...
            'Lane_ID', 'Preceeding', 'Following', 'Space_Hdwy', 'Time_Hdwy'};

        % List of raw NGSIM data files
        files = {
            'data/trajectories-0750am-0805am.txt', ...
            'data/trajectories-0805am-0820am.txt', ...
            'data/trajectories-0820am-0835am.txt', ...
            'data/trajectories-0515-0530.txt', ...
            'data/trajectories-0500-0515.txt', ...
            'data/trajectories-0400-0415.txt'
        };

        % Load all raw data into tables
        datas = cell(1, numel(files));
        for i = 1:numel(files)
            disp("Loading: " + files{i});
            raw = load(files{i});
            datas{i} = array2table(raw, 'VariableNames', varNames);
        end

        % Save to cache for future use
        save(matCacheFile, 'datas');
        disp("Data cached to: " + matCacheFile);
    end

    % Feature and target extraction
    neigh = all_chunks(datas);               % Feature tensors
    target = all_target_chunks(datas);       % Target tensors

    % Separate odd and even sequences
    input_neigh_odd = neigh{1:2:end};
    input_target_odd = target{1:2:end};
    input_target_even = target{1:2:end};

    % Save raw tensors
    save("neighInputOdd.mat", "input_neigh_odd");
    save("targetInputOdd.mat", "input_target_odd");
    save("targetInputEven.mat", "input_target_even");

    % Load saved tensors
    load("neighInputOdd.mat");
    load("targetInputOdd.mat");
    load("targetInputEven.mat");

    % Center the target sequences relative to the last frame
    input_target_odd_centered = input_target_odd;
    input_target_even_centered = input_target_even;
    for k = 1:size(input_target_odd, 3)
        last_row = input_target_odd(30, 1:2, k); % Position at last frame
        for row = 1:30
            input_target_odd_centered(row, 1:2, k) = input_target_odd(row, 1:2, k) - last_row;
            input_target_even_centered(row, 1:2, k) = input_target_even(row, 1:2, k) - last_row;
        end
    end

    % Save centered target data
    save("targetInputOddCentered.mat", "input_target_odd_centered");
    save("targetInputEvenCentered.mat", "input_target_even_centered");

    % Shuffle and split indices for training, validation, test
    numSamples = size(input_target_odd_centered, 3);
    shuffledIdx = randperm(numSamples);

    nTrain = floor(0.70 * numSamples);
    nVal   = floor(0.15 * numSamples);

    trainIdx = shuffledIdx(1:nTrain);
    valIdx   = shuffledIdx(nTrain+1:nTrain+nVal);
    testIdx  = shuffledIdx(nTrain+nVal+1:end);

    % Prepare arrayDatastores
    X1Train = arrayDatastore(input_neigh_odd(:,:,:,:,trainIdx), "IterationDimension", 5);
    X1Val   = arrayDatastore(input_neigh_odd(:,:,:,:,valIdx), "IterationDimension", 5);
    Test.X1Test  = input_neigh_odd(:,:,:,:,testIdx);

    X2Train = arrayDatastore(input_target_odd_centered(:,:,trainIdx), "IterationDimension", 3);
    X2Val   = arrayDatastore(input_target_odd_centered(:,:,valIdx), "IterationDimension", 3);
    Test.X2Test  = input_target_odd_centered(:,:,testIdx);

    YTrain = arrayDatastore(input_target_even_centered(:,:,trainIdx), "IterationDimension", 3);
    YVal   = arrayDatastore(input_target_even_centered(:,:,valIdx), "IterationDimension", 3);
    Test.YTest  = input_target_even_centered(:,:,testIdx);

    % Combine input-output pairs into datastores
    TrainDS = combine(X1Train, X2Train, YTrain);
    ValDS   = combine(X1Val, X2Val, YVal);

    % Save prepared datasets
    save("TrainValTest/TrainDS.mat", "TrainDS", "-v7.3");
    save("TrainValTest/ValDS.mat", "ValDS", "-v7.3");
    save("TrainValTest/Test.mat", "Test", "-v7.3");
end

function plot_vehicle_frames(carID, frameIDRange, data)
% PLOT_VEHICLE_FRAMES animates the surroundings of a vehicle over a range of frames.
%
% INPUTS:
%   carID         : (double) ID of the vehicle to track
%   frameIDRange  : (array of double) list of frame IDs to visualize
%   data          : (table) full dataset containing vehicle information
%   boxWidth      : (double, optional) width of surrounding box (default: 36 ft)
%   boxHeight     : (double, optional) height of surrounding box (default: 195 ft)
%
% OUTPUTS:
%   None (this function generates a figure/animation)
arguments
    carID double
    frameIDRange double
    data table
end

    figure;
    for idx = 1:length(frameIDRange)
        frameID = frameIDRange(idx);
        clf;  % Clear figure for new frame
        plotCars(carID, frameID, data, boxWidth, boxHeight);  % Plot current frame
        drawnow;  % Force drawing update
        pause(0.1);  % Delay between frames for animation effect
    end
end

function plotCars(carID, frameID, data)
% PLOTCARS plots vehicles surrounding a target car in a given frame.
%
% INPUTS:
%   carID       : (double) ID of the vehicle of interest
%   frameID     : (double) current frame to visualize
%   data        : (table) full dataset
%   boxWidth    : (double) width of the region around the vehicle
%   boxHeight   : (double) height of the region around the vehicle
%
% OUTPUTS:
%   None (draws a static plot for the given frame)

arguments
    carID double
    frameID double
    data table
end

    % Get vehicles within the area-of-interest for the target car
    vhclInBox = vhclBox(carID, frameID, data);

    % Set plot dimensions
    minX = -boxWidth / 2;
    maxX =  boxWidth / 2;
    minY = -boxHeight / 2;
    maxY =  boxHeight / 2;

    % Configure plot layout
    axis equal;
    grid off;
    xlabel('Relative X (ft)');
    ylabel('Relative Y (ft)');
    title(['Vehicle Surroundings - Frame ', num2str(frameID)]);
    xlim([minX, maxX]);
    ylim([minY, maxY]);

    % Draw dashed lane-like vertical and horizontal lines
    for x = -boxWidth/2 : 12 : boxWidth/2
        line([x x], [minY maxY], 'Color', [0.8 0.8 0.8], 'LineStyle', '--');
    end
    for y = -boxHeight/2 : 15 : boxHeight/2
        line([minX maxX], [y y], 'Color', [0.8 0.8 0.8], 'LineStyle', '--');
    end

    % Draw vehicles
    for i = 1:height(vhclInBox)
        xCenter = vhclInBox.XDiff(i);    % Relative X
        yCenter = vhclInBox.YDiff(i);    % Relative Y
        vLength = vhclInBox.Length(i);   % Vehicle length
        vWidth = vhclInBox.Width(i);     % Vehicle width
        vSpeed = vhclInBox.Speed(i);     % Speed in feet/s
        lane = vhclInBox.Lane(i);        % Lane ID
        vehID = vhclInBox.Vehicle_ID(i); % Vehicle ID

        % Position for rectangle (vehicle bounding box)
        xLeft = xCenter - vWidth / 2;
        yBottom = yCenter - vLength / 2;

        % Color the target vehicle differently
        if vehID == carID
            faceColor = [0.2 0.8 0.2];  % Green for target
        else
            faceColor = [0.5 0.8 0.9];  % Blue for others
        end

        % Draw vehicle rectangle
        rectangle('Position', [xLeft, yBottom, vWidth, vLength], ...
                  'FaceColor', faceColor, 'EdgeColor', 'k');

        % Annotate vehicle ID
        text(xCenter + 5, yCenter, string(vehID), ...
             'FontSize', 8, 'Color', 'k');
        
        % Annotate vehicle speed
        text(xCenter + 5, yCenter - 5, sprintf('%.2f feet/s', vSpeed), ...
             'FontSize', 8, 'Color', 'k');

        % Annotate lane number
        text(xCenter + 5, yCenter - 10, sprintf('line %d', lane), ...
             'FontSize', 8, 'Color', 'k');

        % Plot velocity vector (line) behind vehicle
        lineLength = vSpeed / 3;
        line([xCenter, xCenter], ...
             [yCenter - vLength/2, yCenter - vLength/2 - lineLength], ...
             'Color', 'r', 'LineWidth', 2);
    end
end

function [vhclInBox, occupancyGrid] = vhclBox(carID, frameID, data, boxWidth, boxHeight)
% VHCLBOX extracts nearby vehicles and builds an occupancy grid relative to a target vehicle.
%
% INPUTS:
%   carID      : (double) vehicle ID of the ego car
%   frameID    : (double) frame number of interest
%   data       : (table) dataset with vehicle state information
%   boxWidth   : (double) width of observation area (default unit: ft)
%   boxHeight  : (double) height of observation area
%
% OUTPUTS:
%   vhclInBox      : (table) vehicles inside the observation box (relative positions)
%   occupancyGrid  : (39x1 binary vector) occupancy of 3x13 grid cells
arguments (Input)
    carID int8
    frameID double
    data table
    boxWidth double = 36
    boxHeight double = 195
end

arguments (Output)
    vhclInBox (:,7) table
    occupancyGrid (39,1) int8
end

    vhcl = data(data.Vehicle_ID == carID & data.Frame_ID == frameID, :);
    vhclInFrame = data(data.Frame_ID == frameID, :);

    occupancyGrid = zeros(39,1);  % Initialize 3x13 grid (flattened column-major)

    if isempty(vhcl)
        vhclInBox = table();  % Return empty if ego car not found
        return;
    end

    % Get ego car's rear bumper coordinates
    vhclX = vhcl.Local_X;
    vhclY = vhcl.Local_Y - vhcl.v_Length / 2;

    % Compute relative positions for all cars in the same frame
    vhclXDiff = vhclInFrame.Local_X - vhclX;
    vhclYDiff = vhclInFrame.Local_Y - vhclInFrame.v_Length / 2 - vhclY;

    % Create structured table of relevant fields
    vhclInFrame = table(vhclInFrame.Vehicle_ID, vhclXDiff, vhclYDiff, ...
        vhclInFrame.v_Length, vhclInFrame.v_Width, vhclInFrame.v_Vel, vhclInFrame.Lane_ID);

    vhclInFrame = renamevars(vhclInFrame, ...
        ["Var1", "Var2", "Var3", "Var4", "Var5", "Var6", "Var7"], ...
        ["Vehicle_ID", "XDiff", "YDiff", "Length", "Width", "Speed", "Lane"]);

    % Filter to only vehicles inside the ego car's observation box
    withinX = abs(vhclInFrame.XDiff) <= boxWidth / 2;
    withinY = abs(vhclInFrame.YDiff) <= boxHeight / 2;
    vhclInBox = vhclInFrame(withinX & withinY, :);

    % Define grid cell size
    cellWidth = 12;
    cellHeight = 15;

    % Mark occupied cells in a 3x13 occupancy grid
    for i = 1:height(vhclInBox)
        x = vhclInBox.XDiff(i);
        y = vhclInBox.YDiff(i);

        col = floor((x + boxWidth/2) / cellWidth) + 1;
        row = floor((y + boxHeight/2) / cellHeight) + 1;

        if row >= 1 && row <= 13 && col >= 1 && col <= 3
            occupancyGrid((13 - row) * 3 + col) = 1;  % row-major to column-major
        end
    end
end

function gridMatrix = get_occupancy_grid_over_frames(carID, frameIDRange, data, boxWidth, boxHeight)
% GET_OCCUPANCY_GRID_OVER_FRAMES constructs temporal occupancy grid maps.
%
% INPUTS:
%   carID        : (double) ego vehicle ID
%   frameIDRange : (array of double) sequence of frame IDs to analyze
%   data         : (table) dataset of vehicle states
%   boxWidth     : (double) width of the spatial grid (default: 36 ft)
%   boxHeight    : (double) height of the spatial grid (default: 195 ft)
%
% OUTPUT:
%   gridMatrix   : (39 x N) binary occupancy matrix where each column is a frame
arguments (Input)
    carID double
    frameIDRange double
    data table
    boxWidth double = 36
    boxHeight double = 195
end

arguments (Output)
    gridMatrix (39,:) double
end

    numFrames = length(frameIDRange);
    gridMatrix = zeros(39, numFrames);  % Preallocate 39xT occupancy matrix

    for idx = 1:numFrames
        frameID = frameIDRange(idx);
        [~, occupancyGrid] = vhclBox(carID, frameID, data, boxWidth, boxHeight);
        gridMatrix(:, idx) = occupancyGrid;  % Assign column for current frame
    end
end

function targetData = get_target_data_over_frames(carID, frameIDRange, data)
% GET_TARGET_DATA_OVER_FRAMES extracts trajectory and velocity data over time for a specific vehicle.
%
% INPUTS:
%   carID         : (double) ID of the vehicle to track
%   frameIDRange  : (array of double) list of frame IDs to extract data from
%   data          : (table) dataset containing vehicle state information
%
% OUTPUT:
%   targetData    : (N x 3 matrix) where N is the number of frames;
%                   Columns correspond to [Local_X, Local_Y, Velocity] for each frame

arguments (Input)
    carID double
    frameIDRange double
    data table
end

arguments (Output)
    targetData (:,3) double
end

    numFrames = length(frameIDRange);
    targetData = zeros(numFrames, 3);  % Preallocate output

    for idx = 1:numFrames
        frameID = frameIDRange(idx);
        % Extract Local_X (col 5), Local_Y (col 6), v_Vel (col 12)
        targetData(idx, :) = table2array(data(data.Vehicle_ID == carID & ...
                                              data.Frame_ID == frameID, ...
                                              [5, 6, 12]));
    end
end

function car_chunks = get_occupancy_grid_chunks(carID, data, chunkSize)
% GET_OCCUPANCY_GRID_CHUNKS splits occupancy grid data for a vehicle into input/target chunks.
%
% INPUTS:
%   carID     : (double) vehicle ID of the ego car
%   data      : (table) dataset containing all vehicle data
%   chunkSize : (double, optional) number of frames per chunk (default: 60)
%
% OUTPUT:
%   car_chunks : (4D array) [13 x 3 x 30 x N] where:
%                 - 13 x 3 is the spatial occupancy grid (rows x columns)
%                 - 30 is the number of frames (timesteps) per input or output
%                 - N is the number of input/output pairs (chunks)
arguments (Input)
    carID double
    data table
    chunkSize double = 60
end

arguments (Output)
    car_chunks (13,3,30,:) double
end

    % Extract rows for this vehicle
    carRows = data(data.Vehicle_ID == carID, :);

    allFrames = unique(carRows.Frame_ID);
    totalFrames = carRows.Total_Frames(1);

    % Skip if not enough frames for even a single chunk
    if length(allFrames) < chunkSize
        car_chunks = {};
        return;
    end

    numChunks = floor(totalFrames / chunkSize);
    car_chunks = [];

    for i = 1:numChunks
        % Frame range for this chunk
        frameIdxStart = (i - 1) * chunkSize + 1;
        frameRange = allFrames(frameIdxStart : frameIdxStart + chunkSize - 1);

        % Get occupancy grid over 60 frames → size: [39 x 60]
        occMatrix = get_occupancy_grid_over_frames(carID, frameRange, data);

        % First 30 frames are input
        chunkX = occMatrix(:, 1:30);  % [39 x 30]
        reshaped = reshape(chunkX, 3, 13, 30);  % → [3 x 13 x 30]
        transposedStack = permute(reshaped, [2 1 3]);  % → [13 x 3 x 30]

        % Next 30 frames are prediction target
        chunkY = occMatrix(:, 31:60);  % [39 x 30]
        reshaped2 = reshape(chunkY, 3, 13, 30);  % → [3 x 13 x 30]
        transposedStack2 = permute(reshaped2, [2 1 3]);  % → [13 x 3 x 30]

        % Concatenate input/output pair along 4th dimension
        pairChunk = cat(4, transposedStack, transposedStack2);  % [13 x 3 x 30 x 2]

        if isempty(car_chunks)
            car_chunks = pairChunk;
        else
            car_chunks = cat(4, car_chunks, pairChunk);  % Append along 4th dim
        end
    end
end


function target_chunks = get_target_data_chunks(carID, data, chunkSize)
% GET_TARGET_DATA_CHUNKS splits target vehicle trajectory data into chunks.
%
% INPUTS:
%   carID     : (double) target vehicle ID
%   data      : (table) full dataset
%   chunkSize : (double, optional) number of frames per chunk (default: 60)
%
% OUTPUT:
%   target_chunks : (3D array or empty) [30x3xN] where N is number of chunks.
%                   Each chunk is a pair: first 30 frames = input, next 30 = target.
arguments (Input)
    carID double
    data table
    chunkSize double = 60
end

arguments (Output)
    target_chunks (30,3,:) double
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

        % [Local_X, Local_Y, v_Vel]
        targetMatrix = get_target_data_over_frames(carID, frameRange, data);

        % Divide into input and prediction segments
        chunkX = targetMatrix(1:30, :);  % Input segment
        chunkY = targetMatrix(31:60, :); % Prediction segment

        % Concatenate along third dim (60x3 becomes [30x3x2])
        pairChunk = cat(3, chunkX, chunkY);

        if isempty(target_chunks)
            target_chunks = pairChunk;
        else
            target_chunks = cat(3, target_chunks, pairChunk);
        end
    end
end

function all_chunks_combined = all_chunks(datas)
% ALL_CHUNKS collects and combines occupancy grid chunks for all vehicles in all datasets.
%
% INPUT:
%   datas : (cell array) each cell is a table of vehicle data for one time slice
%
% OUTPUT:
%   all_chunks_combined : (4D array) occupancy chunks [39x30x1xN]
arguments (Input)
    datas (1,6) cell
end

arguments (Output)
    all_chunks_combined (39,30,1,:) int8
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

        % Collect valid chunks
        for j = 1:n
            if ~isempty(temp_chunks{j})
                all_chunks_cell{end+1} = temp_chunks{j};
            end
        end
    end

    % Combine into a 4D array: [39 x T x 1 x NumChunks]
    all_chunks_combined = cat(4, all_chunks_cell{:});
end

function all_target_chunks_combined = all_target_chunks(datas)
% ALL_TARGET_CHUNKS collects trajectory chunks for all vehicles across datasets.
%
% INPUT:
%   datas : (cell array) each cell is a table of vehicle data
%
% OUTPUT:
%   all_target_chunks_combined : (3D array) [30x3xN] input/target sequence pairs
arguments (Input)
    datas (1,6) cell
end

arguments (Output)
    all_target_chunks_combined (30,3,:) double
end

    all_chunks_cell = {};

    for i = 1:numel(datas)
        data = datas{i};
        vhcls = unique(data.Vehicle_ID);
        n = length(vhcls);

        temp_chunks = cell(n, 1);

        parfor j = 1:n  % Use parallel loop for performance
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

    % Combine into 3D array: [30 x 3 x NumChunks]
    all_target_chunks_combined = cat(3, all_chunks_cell{:});
end

disp("Train, validation and test datasets are loading...");
load TrainDS.mat
load ValDS.mat
load Test.mat
disp("Train, validation and test datasets are loaded!");

function [trainedNetwork, trainingInfo] = trainNetwork(TrainDS, ValDS)
% TRAINNETWORK trains or loads a deep learning model on traffic sequence data.
%
% INPUTS:
%   TrainDS : Combined datastore containing training input/output (X1, X2, Y)
%   ValDS   : Combined datastore for validation
%
% OUTPUTS:
%   trainedNetwork : Trained dlnetwork object
%   trainingInfo   : Struct with training metrics (loss, validation accuracy, etc.)
arguments (Input)
    TrainDS
    ValDS 
end

arguments (Output)
    trainedNetwork dlnetwork
    trainingInfo
end

    % Load cached model if exists
    if isfile("trainedNetwork.mat") && isfile("trainingInfo.mat")
        disp("Loading pre-trained network and training info...");
        load("trainedNetwork.mat", "trainedNetwork");
        load("trainingInfo.mat", "trainingInfo");
        disp("Pre-trained network and training info loaded!");
        return;
    end

    % Training hyperparameters and options
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

    % Model configuration
    numHiddenUnits = 64;
    numNeurons = 128;

    % --- Branch 1: Spatial (occupancy grid) + temporal (CNN + BiLSTM)
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

        concatenationLayer(1, 2, "Name", "cat")  % to combine with branch 2
        fullyConnectedLayer(numNeurons, "Name", "fc1")
        fullyConnectedLayer(3, "Name", "fc_output")  % Predict X/Y/Speed
    ];

    % --- Branch 2: Ego vehicle input (trajectory features)
    layers2 = [
        sequenceInputLayer(3, "Name", "seq2", "MinLength", 30)
        bilstmLayer(32, "OutputMode", "sequence", "Name", "bilstm2")
    ];

    % Create a layer graph and connect the branches
    net = dlnetwork;
    net = addLayers(net, layers1);
    net = addLayers(net, layers2);
    net = connectLayers(net, "bilstm2", "cat/in2");  % Connect ego path to CNN path

    % Train the network using combined loss
    [trainedNetwork, trainingInfo] = trainnet(TrainDS, net, "l2loss", options);

    % Save trained model and logs
    save("trainedNetwork.mat", "trainedNetwork");
    save("trainingInfo.mat", "trainingInfo");
end

[trainedNetwork, trainingInfo] = trainNetwork(TrainDS, ValDS);

function results = evaluateModel(trainedNetwork, Test)
% EVALUATEMODEL computes predictions and evaluates performance on the test dataset.
%
% INPUTS:
%   trainedNetwork : Trained dlnetwork object
%   Test           : Struct with test inputs and targets:
%                   - X1Test : [13 x 3 x 30 x 1 x N] Occupancy input
%                   - X2Test : [30 x 3 x N] Ego input
%                   - YTest  : [30 x 3 x N] True output/target
%
% OUTPUT:
%   results : Struct with fields:
%               - mse       : Mean squared error across all predictions
%               - YPredMat  : [30 x 3 x N] Predicted outputs (same shape as YTest)
arguments (Input)
    trainedNetwork dlnetwork
    Test struct
end

arguments (Output)
    results struct
end

    % Predict using minibatches for performance (cell output)
    predictedTarget = minibatchpredict( ...
        trainedNetwork, ...
        Test.X1Test(:,:,:,:,:), ...
        Test.X2Test(:,:,:), ...
        MiniBatchSize = 256, ...
        UniformOutput = false ...
    );

    nSamples = numel(predictedTarget);

    % Preallocate prediction matrix to match YTest shape
    YPredMat = zeros(size(Test.YTest));

    % Convert each prediction (cell) into 3D matrix
    for i = 1:nSamples
        YPredMat(:,:,i) = predictedTarget{i};
    end

    % Compute mean squared error over all dimensions
    squaredError = (YPredMat - Test.YTest).^2;
    mse = mean(squaredError, 'all');

    % Package output
    results.mse = mse;
    results.YPredMat = YPredMat;
end

results = evaluateModel(trainedNetwork, Test);

disp("Mean-squared error is " + results.mse)