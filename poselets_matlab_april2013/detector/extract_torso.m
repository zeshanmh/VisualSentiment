global config;
config=init;
time=clock;

% Choose the category here
category = 'person';

data_root = [config.DATA_DIR '/' category];

disp(['Running on ' category]);

faster_detection = true;  % Set this to false to run slower but higher quality
interactive_visualization = true; % Enable browsing the results
enable_bigq = true; % enables context poselets

if faster_detection
    disp('Using parameters optimized for speed over accuracy.');
    config.DETECTION_IMG_MIN_NUM_PIX = 500^2;  % if the number of pixels in a detection image is < DETECTION_IMG_SIDE^2, scales up the image to meet that threshold
    config.DETECTION_IMG_MAX_NUM_PIX = 750^2;
    config.PYRAMID_SCALE_RATIO = 2;
end

% Loads the SVMs for each poselet and the Hough voting params
clear output poselet_patches fg_masks;
load([data_root '/model.mat']); % model
if exist('output','var')
    model=output; clear output;
end
if ~enable_bigq
   model =rmfield(model,'bigq_weights');
   model =rmfield(model,'bigq_logit_coef');
   disp('Context is disabled.');
end
if ~enable_bigq || faster_detection
   disp('*******************************************************');
   disp('* NOTE: The code is running in faster but suboptimal mode.');
   disp('*       Before reporting comparison results, set faster_detection=false; enable_bigq=true;');
   disp('*******************************************************');
end

% read in all images form a folder 
foldername = '../../../data/groupdataset_release/images/all/';
srcFiles = dir(foldername);

% make dir to hold all torso bounds if it doesn't exist
torso_dir = '../../../data/groupdataset_release/all_torsos/';
mkdir(torso_dir);

% for each image
for i = 3 : length(srcFiles)
    i
    filename = strcat(foldername,srcFiles(i).name);
    img = imread(filename);

    [bounds_predictions,poselet_hits,torso_predictions]=...
        detect_objects_in_image(img, model,config);
    
    % get the image filename without the extension to save torso file
    [~, filename_no_ext, ~] = fileparts(filename);
    all_torso_bounds = torso_predictions.bounds';
    all_torso_scores = torso_predictions.score;
    
     % create and store data as csv file
    torso_filename = strcat(torso_dir,filename_no_ext,'_torsos.csv');
    csvwrite(torso_filename, all_torso_bounds);
    
    
%     % for each torso
%     for i = 1:torso_predictions.size
%         % get the bounds
%         torso_bounds = torso_predictions.bounds(i,:);
%         
%        
%     end
end




% display bounding boxes on image