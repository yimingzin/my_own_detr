python main.py --coco_path ../datasets/minicoco --output_dir output --pre_norm

image:
torch.Size([2, 3, 1167, 1039])
backbone out: 
torch.Size([2, 2048, 37, 33])
pos out:
torch.Size([2, 256, 37, 33])
torch.Size([6, 2, 100, 256])    # (num_decoder_layers, batch_size, num_queries, hidden_dim)
torch.Size([2, 256, 37, 33])