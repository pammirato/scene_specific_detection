%TODO - pick better boxes(higher ious first)



%set up paths to data
scene_path = '/playpen/ammirato/Data/RohitData/Bedroom_01_1';
proposal_path = fullfile(scene_path, 'region_proposals');
full_proposal_path = fullfile(scene_path, 'region_proposals', 'full_region_proposals');
selected_proposal_path = fullfile(scene_path, 'region_proposals', 'selected_region_proposals');
gt_boxes_path = fullfile(scene_path, 'labels', 'bounding_boxes_by_image_instance');
image_path = fullfile(scene_path, 'jpg_rgb');

%category id for background
bg_cat_id = 33;


%get list of image names
image_names = dir(fullfile(image_path, '*.jpg'));
image_names = {image_names.name};


all_selected_props = cell(1,length(image_names));


%for each image, select some region proposals
for il=1:length(image_names)

  cur_image_name = image_names{il};
  cur_mat_name = strcat(cur_image_name(1:10), '.mat');

  %load all the proposals, and the labels
  full_props = load(fullfile(full_proposal_path,cur_mat_name));
  full_props = full_props.boxes;
  gt_boxes = load(fullfile(gt_boxes_path,cur_mat_name));
  gt_boxes = gt_boxes.boxes; 
  

  %will hold indicies of possilbe background boxes
  possible_bgs = [1:size(full_props,1)];
  
  %to store proposals for objects, max 2 per object(gt_box)
  object_selects = zeros(size(gt_boxes,1)*2, 2);
  os_counter = 1;
  
  %for each gt_box, pick up to 2 proposals to use with this image.
  %  if there is no good proposal, perturb the gt box
  for jl=1:size(gt_boxes,1)
    cur_box = gt_boxes(jl,:);

    %get the intersection over union of the proposals and the gt box
    ious = get_bboxes_iou(cur_box(1:4), full_props);
  
    %valid poposals for this instance have large iou
    valid_props = find(ious > .3);
    
    %choose at most 2 of them
    num_val = length(valid_props);
    if(num_val==0)
      continue;
    end
    inds = randperm(num_val);
    inds = valid_props(inds(1:min(2,num_val)));
    object_selects(os_counter,:) = [inds(1) cur_box(5)];
    os_counter = os_counter+1;
    if(length(inds)>1)
      object_selects(os_counter,:) = [inds(2) cur_box(5)];
      os_counter = os_counter+1;
    end
    
    
    %background proposals do not have the object at all
    cur_pos_bgs = find(ious == 0);
    %update list of possible backgrounds to remove this object
    possible_bgs = intersect(possible_bgs, cur_pos_bgs);
    

  end%for jl, each gt_box  

  %lose empty slots
  object_selects(os_counter:end,:) = [];
  
  %make sure a box is not used twice for two different objects
  [a,b,c] = unique(object_selects(:,1));
  object_selects = object_selects(b,:);
  
  %choose two random backgrounds
  bgs = randperm(length(possible_bgs));
  bgs = bgs(1:2);
  bgs = [full_props(bgs,:) [bg_cat_id; bg_cat_id]];
  
  %get the selected object boxes
  objs = [full_props(object_selects(:,1),:) object_selects(:,2)];
  
  %put the backgrounds and objects together
  boxes = [bgs; objs];
  %save to file
  save(fullfile(selected_proposal_path, cur_mat_name), 'boxes');
 
  %add image index to each box
  img_index = str2double(cur_mat_name(1:6));
  boxes = [boxes repmat(img_index, size(boxes,1), 1)];
  
  %keep track of globla data structure
  all_selected_props{il} = boxes;
end%for il, each image name  



boxes = cell2mat(all_selected_props');

save(fullfile(proposal_path, 'all_selected_proposals.mat'), 'boxes');


















