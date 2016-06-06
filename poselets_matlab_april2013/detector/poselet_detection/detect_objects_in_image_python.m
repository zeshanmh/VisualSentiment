function [torso_bounds, torso_scores] = detect_objects_in_image_python(img)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Given an RGB uint8 image returns the locations and scores of all objects
%%% in the image (bounds_predictions), of all poselet hits (poselet_hits)
%%% and, optionally for mammals, of all torso locations (torso_predictions)
%%%
%%% Copyright (C) 2009, Lubomir Bourdev and Jitendra Malik.
%%% This code is distributed with a non-commercial research license.
%%% Please see the license file license.txt included in the source directory.
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = load('../../poselets_matlab_april2013/data/person/model.mat');
model = model.model;

% get config
config=init;

if config.DEBUG>0
    config.DEBUG_IMG = img;
end

%fprintf('Computing pyramid HOG (hint: You can cache this)... ');
%total_start_time=clock;
% img = [img{:}];
% size(img)
% img = cell2mat(img);
% img = uint8(reshape(img, img_height, img_width, 3));
phog=image2phog(img, config);

%fprintf('Done in %4.2f secs.\n',etime(clock,total_start_time));

%fprintf('Detecting poselets... ');
%start_time=clock;
poselet_hits = nonmax_suppress_hits(detect_poselets(phog,model.svms,config));
poselet_hits.score = -poselet_hits.score.*model.logit_coef(poselet_hits.poselet_id,1)-model.logit_coef(poselet_hits.poselet_id,2);
poselet_hits.score = 1./(1+exp(-poselet_hits.score));
[srt,srtd]=sort(poselet_hits.score,'descend');
poselet_hits = poselet_hits.select(srtd);
%fprintf('Done in %4.2f secs.\n',etime(clock,start_time));

if isfield(model,'bigq_weights')
%   fprintf('Big Q...');
%    start_time=clock;
   hyps=[model.hough_votes.hyp];
   [features,contrib_hits] = get_context_features_in_image(hyps,poselet_hits,config);
   poselet_hits.src_idx = contrib_hits';
   poselet_hits.score = sum(features.*model.bigq_weights(poselet_hits.poselet_id,1:(end-1)),2) + model.bigq_weights(poselet_hits.poselet_id,end);
   poselet_hits.score = -poselet_hits.score.*model.bigq_logit_coef(poselet_hits.poselet_id,1)-model.bigq_logit_coef(poselet_hits.poselet_id,2);
   poselet_hits.score = 1./(1+exp(-poselet_hits.score));
%   fprintf('Done in %4.2f secs.\n',etime(clock,start_time));
end

%fprintf('Clustering poselets... ');
%start_time=clock;
hyps = instantiate_hypotheses([model.hough_votes.hyp],poselet_hits);
cluster_labels = cluster_poselet_hits(poselet_hits,hyps,config);
%fprintf('Done in %4.2f secs.\n',etime(clock,start_time));

%fprintf('Predicting bounding boxes... ');
%start_time=clock;
[H W D] = size(img);
[torso_predictions,bounds_predictions] = cluster2bounds(poselet_hits, hyps,...
    cluster_labels, model, [W H], config);

torso_bounds = torso_predictions.bounds;
torso_scores = torso_predictions.score;
%fprintf('Done in %4.2f secs.\n',etime(clock,start_time));

%disp(sprintf('Total time: %d secs',round(etime(clock,total_start_time))));



