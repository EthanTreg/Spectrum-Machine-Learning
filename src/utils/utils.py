import torch


def progress_bar(i: int, total: int, text: str = ''):
    """
    Terminal progress bar

    Parameters
    ----------
    i : integer
        Current progress
    total : integer
        Completion number
    text : string, default = '
        Optional text to place at the end of the progress bar
    """
    length = 50
    i += 1

    filled = int(i * length / total)
    percent = i * 100 / total
    bar_fill = '█' * filled + '-' * (length - filled)
    print(f'\rProgress: |{bar_fill}| {int(percent)}%\t{text}\t', end='')

    if i == total:
        print()


def even_length(x: torch.Tensor) -> torch.Tensor:
    """
    Returns a tensor of even length in the last
    dimension by merging the last two values

    Parameters
    ----------
    x : Tensor
        Input data

    Returns
    -------
    Tensor
        Output data with even length
    """
    if x.size(-1) % 2 != 0:
        x = torch.cat((
            x[..., :-2],
            torch.mean(x[..., -2:], dim=-1, keepdim=True)
        ), dim=-1)

    return x
