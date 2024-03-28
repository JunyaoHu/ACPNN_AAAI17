def count_parameters(model):
    res = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"count_training_parameters: {res}")
    res = sum(p.numel() for p in model.parameters())
    print(f"count_all_parameters:      {res}")
    