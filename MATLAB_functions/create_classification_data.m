function [nums] = create_classification_data(scene_name, cat_ids_to_use)
% creates images and labels for a classification task:
%     classify an image as one of the BigBIRD objects or background
%
%
% crops the original images around the ground truth bounding boxes
% and with region proposals that are assumed to exist in 'meta_path/region_proposals'
% to generate these proposals:
%        generate_region_proposals(scene_name) in './SelectiveSearchCodeIJCV/' 
%
% saves new image in 'meta_path/classification/images/'
% saves labels in 'meta_path/classification/labels.txt' 
%   one line per image:  image_name label_index

%TODO - clean removal of skinny boxes
%     - output map from new_id to category name
%     - auto balancing between classes
%     - make perturb nums based on gt_box size and/or input


%CLEANED - no
%TESTED  - no

init;

%set up paths to data
scene_path = fullfile(ROHIT_BASE_PATH, scene_name);
meta_path = fullfile(ROHIT_META_BASE_PATH, scene_name);
proposal_path = fullfile(meta_path, 'region_proposals');
gt_boxes_path = fullfile(meta_path, LABELING_DIR, VERIFIED_LABELS, BBOXES_BY_IMAGE_INSTANCE);
image_load_path = fullfile(scene_path, JPG_RGB);
image_save_path = fullfile(meta_path, 'classification', 'images');
label_file_path = fullfile(meta_path, 'classification', 'labels.txt'); 

add_gt_boxes = 1;


if(~exist(image_save_path, 'dir'))
  mkdir(image_save_path);
end

%create_label_file
label_fid = fopen(label_file_path,'wt');
assert(label_fid > 0);

%category id for background
bg_cat_id = 0;

%get list of image names
image_names = dir(fullfile(image_load_path, '*.jpg'));
image_names = {image_names.name};


all_selected_props = cell(1,length(image_names));

nums = [0 0];
%for each image, select some region proposals
for il=1:length(image_names)

  cur_image_name = image_names{il};
  
  if(strcmp(cur_image_name,'0000560101.jpg'))
    breakp=1;
  end
  cur_mat_name = strcat(cur_image_name(1:10), '.mat');
  cur_rgb_img = imread(fullfile(scene_path,'jpg_rgb', strcat(cur_image_name(1:10),'.jpg')));
  %load all the proposals, and the labels
  full_props = load(fullfile(proposal_path,cur_mat_name));
  full_props = full_props.boxes;
  gt_boxes = load(fullfile(gt_boxes_path,cur_mat_name));
  gt_boxes = gt_boxes.boxes; 
  
  
  %get rid of really skinny boxes
  min_dims = min(full_props(:,3)-full_props(:,1) , full_props(:,4)-full_props(:,2));
  aspect_ratios = (full_props(:,3)-full_props(:,1)) ./ (full_props(:,4)-full_props(:,2));
  skinny_boxes = find(min_dims < 25);
  full_props(skinny_boxes,:) = [];
  aspect_ratios = (full_props(:,3)-full_props(:,1)) ./ (full_props(:,4)-full_props(:,2));
  skinny_boxes = find(aspect_ratios < .12);
  full_props(skinny_boxes,:) = [];
  aspect_ratios = (full_props(:,3)-full_props(:,1)) ./ (full_props(:,4)-full_props(:,2));
  skinny_boxes = find(aspect_ratios > 8);
  full_props(skinny_boxes,:) = [];
  
  
  
  %will hold indicies of possilbe background boxes
  possible_bgs = [1:size(full_props,1)];
  
  %to store proposals for objects
  object_selects = zeros(size(gt_boxes,1)*2, 2);
  os_counter = 1;
  
  made_boxes = [];
  bg_boxes = [];
  
  %for each gt_box, pick up to 2 proposals to use with this image.
  % also perturb the gt box
  for jl=1:size(gt_boxes,1)
    cur_box = gt_boxes(jl,:);

    %only use chosen categories
    if(~any(cat_ids_to_use==cur_box(5)))
      continue;
    end

    %get the new id for this category from the reduced set of categories
    new_id = find(cat_ids_to_use == cur_box(5));

    %get the intersection over union of the proposals and the gt box
    ious = get_bboxes_iou(cur_box(1:4), full_props);

    %make the gt box bigger (+ grow) and them randomly perturb it(+ perturbs)
    if(add_gt_boxes)
      num_add = 7;
      grow = repmat([-30 -30 30 30], num_add,1);
      perturbs = randi([-30 30], num_add, 4);
      p_boxes = repmat(cur_box([1:4]),num_add,1) + grow + perturbs;       
      %keep boxes in image boundary
      p_boxes(p_boxes<1) = 1;
      p_boxes(p_boxes(:,3) > 1920,3) = 1920;
      p_boxes(p_boxes(:,4) > 1080,4) = 1080;
      
      p_boxes = [p_boxes repmat(cur_box([5 6]),num_add,1)];
      made_boxes(end+1:end+num_add,:) = p_boxes;
    end
    %object_selects(os_counter:os_counter+num_add-1,:) = p_boxes; 


  
    %valid poposals for this instance have large iou
    valid_props = find(ious > .3);
    
    %background proposals do not have the object at all
    cur_pos_bgs = find(ious == 0);
    %update list of possible backgrounds to remove this object
   
    possible_bgs = intersect(possible_bgs, cur_pos_bgs);
   
    bg_boxes = full_props(possible_bgs,:);

    
    
    %choose at most 2 of them
    num_val = length(valid_props);
    if(num_val==0)
      continue;
    end
    inds = randperm(num_val);
    inds = valid_props(inds(1:min(3,num_val)));
    object_selects(os_counter,:) = [inds(1) new_id];
    os_counter = os_counter+1;
    if(length(inds)>1)
      object_selects(os_counter,:) = [inds(2) new_id];
      os_counter = os_counter+1;
    end
    if(length(inds)>2)
      object_selects(os_counter,:) = [inds(3) new_id];
      os_counter = os_counter+1;
    end
    



  end%for jl, each gt_box  

  %lose empty slots
  object_selects(os_counter:end,:) = [];
  
  %make sure a box is not used twice for two different objects
  [a,b,c] = unique(object_selects(:,1));
  object_selects = object_selects(b,:);
  
  %choose two random backgrounds
  if(isempty(bg_boxes))
    bg_boxes = full_props;
  end
  bgs = randperm(size(bg_boxes,1));
  bgs = bgs(1:2);
  bgs = [bg_boxes(bgs,:) [bg_cat_id; bg_cat_id]];
  %bgs = bgs(1);
  %bgs = [full_props(bgs,:) bg_cat_id];
  
  %get the selected object boxes
  objs = [full_props(object_selects(:,1),:) object_selects(:,2)];
  if(~isempty(made_boxes))
    made_boxes = [made_boxes(:,1:4) repmat(new_id,num_add,1)];
    objs = [objs; made_boxes];
  end
  
  %put the backgrounds and objects together
  boxes = [bgs; objs];

  nums = nums + [size(bgs,1) size(objs,1)];

  %save images and write to label file
  for jl=1:size(boxes,1)
    box = boxes(jl,:);
    crop_img = cur_rgb_img(box(2):box(4),box(1):box(3),:);
    save_name = fullfile(image_save_path, ...
                      strcat(cur_image_name(1:10), '_', num2str(jl), ...
                       '_', num2str(box(end)) , '.jpg'));
    imwrite(crop_img, save_name);
    fprintf(label_fid, '%s %d\n', save_name, box(end));
  end 
  %add image index to each box
  img_index = str2double(cur_mat_name(1:6));
  boxes = [boxes repmat(img_index, size(boxes,1), 1)];
  
  %keep track of globla data structure
  all_selected_props{il} = boxes;
end%for il, each image name  


fclose(label_fid);


%boxes = cell2mat(all_selected_props');
%inds = randperm(size(boxes,1));
%boxes = boxes(inds,:);
%save(fullfile(proposal_path, 'all_selected_proposals.mat'), 'boxes');


















