def get_stack_out_channels(cellstack):
    cells=cellstack.cells
    assert len(cells) >= 1
    out_channels = cells[-1].out_channels
    if len(cells) == 1:
        out_prev_channels = cells[-1].in_channels
    else:
        out_prev_channels = cells[-2].out_channels

    return out_channels,out_prev_channels
