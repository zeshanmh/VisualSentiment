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
foldername = '../../data/TUD/test/';
srcFiles = dir(foldername);
size(srcFiles)

% % make dir to hold all poselet bounds if it doesn't exist
poselets_dir = '../../data/TUD/test/all_poselets/';
mkdir(poselets_dir);

% make dir to hold all torso bounds if it doesn't exist
torso_dir = '../../data/TUD/test/all_torsos/';
mkdir(torso_dir);

% make dir to hold all person bounds
people_dir = '../../data/TUD/test/all_people/';
mkdir(people_dir);


% our wanted poselets
% rel_poselets = [7,16,19,22,24,25,28,30,34,35,45,48,51,53,63,72,74,80,83,100,105,112,115,119,129]; 

% for each image
for i = 3 : length(srcFiles)
    i
    if srcFiles(i).name == '.DS_Store'
        continue
    end
    filename = strcat(foldername,srcFiles(i).name);
    img = imread(filename);

    [bounds_predictions,poselet_hits,torso_predictions]=...
        detect_objects_in_image(img, model,config);
    
    poselet_ids = single(poselet_hits.poselet_id);
    poselet_bounds = poselet_hits.bounds';
    poselet_scores = poselet_hits.score;
    
    poselets = [poselet_bounds poselet_ids poselet_scores];
    
    people_bounds = bounds_predictions.bounds';
    people_scores = bounds_predictions.score;
    people = [people_bounds people_scores];
    
    all_torso_bounds = torso_predictions.bounds';
    all_torso_scores = torso_predictions.score;
    torsos = [all_torso_bounds all_torso_scores];
    
    
%     % get the image filename without the extension to save torso file
    [~, filename_no_ext, ~] = fileparts(filename);
%     all_torso_bounds = torso_predictions.bounds'
%     
%      % create and store data as csv file
    poselet_filename = strcat(poselets_dir,filename_no_ext,'_poselets.csv');
    csvwrite(poselet_filename, poselets);
       
    
    
     % create and store data as csv file
    torso_filename = strcat(torso_dir,filename_no_ext,'_torsos.csv');
    csvwrite(torso_filename, torsos);
    
    people_filename = strcat(people_dir,filename_no_ext,'_people.csv');
    csvwrite(people_filename, people);
    
    
%     % for each torso
%     for i = 1:torso_predictions.size
%         % get the bounds
%         torso_bounds = torso_predictions.bounds(i,:);
%         
%        
%     end
end




% display bounding boxes on image