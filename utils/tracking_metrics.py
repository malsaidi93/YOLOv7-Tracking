import math
import os
from os.path import isfile, join



def normalize_to_pixels(annotation, width, height):
    """
    Normalized annotation to pixel value 
    args:
    - annotation line (not file path)
    - image width
    - image height
    
    returns (obj_id, x1, y1, x2, y2)
    """
    
    object_id = annotation[0]
    x, y, w, h = map(float, annotation[1:])
    
    # object_id, x_norm, y_norm, w_norm, h_norm = annotation
    x1 = int((x - w / 2) * width)
    y1 = int((y - h / 2) * height)
    x2 = int((x + w / 2) * width)
    y2 = int((y + h / 2) * height)
    return object_id, x1, y1, x2, y2

def normalize_file_to_pixels(annotation, width, height):
    """
    Convert normalized coordinates to pixel values
     
    args:
    - annotation line (not file path)
    - image width
    - image height
    
    returns (obj_id, x1, y1, x2, y2)
    """
    data = annotation.split()
    object_id = int(annotation[0])
    x, y, w, h = map(float, data[1:])
    
    # object_id, x_norm, y_norm, w_norm, h_norm = annotation
    x1 = int((x - w / 2) * width)
    y1 = int((y - h / 2) * height)
    x2 = int((x + w / 2) * width)
    y2 = int((y + h / 2) * height)
    return object_id, x1, y1, x2, y2


def reorder_system_files(gt_dir, sys_dir, output_dir):
    
    """
    Get the list of files in both directories
    
    To fix a problem with annotation order 
    Now boxes will match based on line order
    
    Args: 
        - GT label dir
        - SYS label dir
        - New output dir
    
    """
    

    gt_files = [f for f in os.listdir(gt_dir) if isfile(join(gt_dir, f))]
    sys_files = [f for f in os.listdir(sys_dir) if isfile(join(sys_dir, f))]
    
    

    for filename in gt_files:
        gt_path = os.path.join(gt_dir, filename)
        sys_path = os.path.join(sys_dir, filename)
        

        with open(gt_path, 'r', encoding='ascii') as gt_file, open(sys_path, 'r', encoding='ascii') as sys_file:
            gt_annotations = [tuple(map(float, line.strip().split())) for line in gt_file.readlines()]
            sys_annotations = [tuple(map(float, line.strip().split())) for line in sys_file.readlines()]
            gts=[]
            for a in gt_annotations:
                gt = normalize_to_pixels(a, 3840,2160)
                gts.append(gt)
            
            gt_annotations = gts
            reordered_sys_annotations = []

            for gt_annotation in gt_annotations:
                highest_iou = 0
                best_sys_annotation = None

                for sys_annotation in sys_annotations:
                    iou = calculate_iou(gt_annotation[1:], sys_annotation[1:])
                    if iou > highest_iou:
                        highest_iou = iou
                        best_sys_annotation = sys_annotation

                if best_sys_annotation is not None:
                    reordered_sys_annotations.append(best_sys_annotation)
                    sys_annotations.remove(best_sys_annotation)

            output_path = os.path.join(output_dir, filename)
            with open(output_path, 'w', encoding='ascii') as output_file:
                for sys_annotation in reordered_sys_annotations:
                    output_file.write(" ".join(map(str, sys_annotation)) + "\n")

    return gt_annotations, sys_annotations
    
    
def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.
    
    Args:
    - point1: A tuple representing the (x, y) coordinates of the first point.
    - point2: A tuple representing the (x, y) coordinates of the second point.
    
    Returns:
    - The Euclidean distance between the two points.
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
    - box1: A tuple representing the coordinates (x1, y1, x2, y2) of the first bounding box.
    - box2: A tuple representing the coordinates (x1, y1, x2, y2) of the second bounding box.

    Returns:
    - The IoU (Intersection over Union) value between the two bounding boxes.
    """
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    intersection_area = max(0, min(x2, x4) - max(x1, x3)) * max(0, min(y2, y4) - max(y1, y3))
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    union_area = area1 + area2 - intersection_area

    iou = intersection_area / union_area
    return iou

def calculate_distance(box1, box2):
    """
    Calculate the distance between the centers of two bounding boxes.

    Args:
    - box1: A tuple representing the coordinates (x1, y1, x2, y2) of the first bounding box.
    - box2: A tuple representing the coordinates (x1, y1, x2, y2) of the second bounding box.

    Returns:
    - The distance between the centers of the two bounding boxes.
    """
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    center_x1 = (x1 + x2) / 2
    center_y1 = (y1 + y2) / 2

    center_x2 = (x3 + x4) / 2
    center_y2 = (y3 + y4) / 2

    c1 = (center_x1, center_y1)
    c2 = (center_x2, center_y2)

    dc1c2 = euclidean_distance(c1,c2)


    return dc1c2


def find_bbox_center(x1, y1, x2, y2):
    """
    Calculate the center of a bounding box.

    Args:
    - x1, y1: The top-left coordinates of the bounding box.
    - x2, y2: The bottom-right coordinates of the bounding box.

    Returns:
    - A tuple representing the (x, y) coordinates of the center of the bounding box.
    """
    
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y
    

def calculate_mota(ground_truth, system, iou_threshold=0.5):
    """
    Calculate the Multiple Object Tracking Accuracy (MOTA).

    Args:
    - ground_truth: List of ground truth annotations.
    - system: List of system annotations.
    - iou_threshold: IoU threshold for matching objects.

    Returns:
    - MOTA value and a list of MOTP values.
    """
    
    TP, FP, FN = 0, 0, 0
    matched_gt = set()
    matched_sys = set()
    ious=[]
    MOTP=[]

    if len(ground_truth) != len(system):
            r = min(len(ground_truth), len(system))
    else:
        r = len(ground_truth)

    for i in range(0,r):
       
        sys_id = int(system[i][0])
        sys_box = system[i][1:]
        # best_iou = 0
        gt_id = int(ground_truth[i][0])
        gt_box = ground_truth[i][1:]
        iou = calculate_iou(sys_box, gt_box)
        IDS = sys_id - gt_id 
        
        if iou >= iou_threshold:
            TP += 1
            matched_gt.add(1)
            matched_sys.add(1)
            
        else:
            FP += 1
            
        ious.append(iou)
        if TP==0:
            motp=0
        else:
            motp = 1-iou/TP
        MOTP.append(motp)

    FN = len(ground_truth) - len(matched_gt)

    MOTA = abs(1 - (FN + FP + IDS) / (TP + FN))

    return MOTA,MOTP

    
def calculate_dir_mota(gt_dir, sys_dir):
    """
    Calculate the directory-level MOTA and per-file MOTA values.

    Args:
    - gt_path: Path to the directory containing ground truth annotation files.
    - annot_path: Path to the directory containing system annotation files.

    Returns:
    - List of per-file MOTA values and the average MOTA across all files.
    """
    gt_files = [f for f in os.listdir(gt_dir) if isfile(join(gt_dir, f))]
    sys_files = [f for f in os.listdir(sys_dir) if isfile(join(sys_dir, f))]
    
    # print(gt_files[:5])
    
    MOTA = []
    
    for filename in gt_files:
        gt_path = os.path.join(gt_dir, filename)
        sys_path = os.path.join(sys_dir, filename)
        # print(gt_path)

        with open(gt_path, 'r') as gt_file, open(sys_path, 'r') as sys_file:
            gt_annotations = [tuple(map(float, line.strip().split())) for line in gt_file.readlines()]
            sys_annotations = [tuple(map(float, line.strip().split())) for line in sys_file.readlines()]
            gts=[]
            for a in gt_annotations:
                gt = normalize_to_pixels(a, 3840,2160)
                gts.append(gt)
            
            gt_annotations = gts

            
            mota,_ = calculate_mota(gt_annotations, sys_annotations, iou_threshold=0.5)
            MOTA.append(mota)

    avg_mota = sum(MOTA)/len(MOTA)
    
    return MOTA, avg_mota