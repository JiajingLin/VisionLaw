import torch

def compute_eval_loss(renders_dict, gt_dict, pixel_loss):
    loss = 0.0
    for (renders_view, renders_frame_dict), (gt_view, gt_frame_dict) in zip(renders_dict.items(), gt_dict.items()):
        sorted_renders_frames = sorted(renders_frame_dict.keys())
        sorted_gt_frames = sorted(gt_frame_dict.keys())
        renders = [
            renders_frame_dict[frame]for frame in sorted_renders_frames
        ]
        gts = [
            gt_frame_dict[frame]for frame in sorted_gt_frames
        ]
        renders = torch.stack(renders, dim=0)
        gts = torch.stack(gts, dim=0)
        
        loss_rgb = pixel_loss(renders, gts)
        loss += loss_rgb
        
    return loss