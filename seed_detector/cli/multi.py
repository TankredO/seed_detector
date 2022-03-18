import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

from cloup import command, option, help_option, Path as PathType, Choice, option_group
from .single import run_single


def single_wrapped(args):
    run_single(*args)


@command('multi', help='Detect seeds in a multiple images.')
@option_group(
    'Required options',
    option(
        '-i',
        '--image_dir',
        type=PathType(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            path_type=Path,
        ),
        help='Directory containing images.',
        required=True,
    ),
    option(
        '-o',
        '--out_dir',
        type=PathType(
            file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
        ),
        help=f'''
            Output directory. Will be created if it does not exist. By default
            a directory with the name "<IMAGE_DIR>_detections" will be created in
            the current working directory ({Path(".").resolve()}).
        ''',
        default=None,
        required=True,
    ),
)
@option(
    '-c',
    '--background_color',
    type=str,
    default=None,
    help='''
        Background color. Can be passed as comma-separated list of RGB values
        (e.g., "12, 52, 128") or alternatively as hex color code (e.g., "#04245f").
        By default the most common color range will be assumed to be the background
        color.
    ''',
)
@option_group(
    'Output options',
    option(
        '-m',
        '--masks_output',
        is_flag=True,
        help='Generate mask output for every detected object.',
    ),
    option(
        '-b',
        '--bbox_output',
        is_flag=True,
        help='Extract bounding boxes for every detected object.',
    ),
    option(
        '-u',
        '--contour_output',
        is_flag=True,
        help='Generate additional output image with contours.',
    ),
    option(
        '-p',
        '--padding',
        type=int,
        default=0,
        help='Padding around objects for mask and bounding box outputs.',
        show_default=True,
    ),
    option(
        '--rm_bg',
        is_flag=True,
        help='Set background pixel in bbox output to 0 (black).',
    ),
)
@option(
    '-t',
    '--n_proc',
    default=1,
    help='''
        Number of parallel processes to run. Should not be set higher than
        the number of CPU cores.
    ''',
)
@option(
    '--adaptive',
    is_flag=True,
    help='EXPERIMENTAL: use adaptive segmentation algorithm',
)
@help_option('-h', '--help')
def multi(
    image_dir: Path,
    out_dir: Optional[Path],
    background_color: Optional[str],
    masks_output: bool,
    bbox_output: bool,
    contour_output: bool,
    padding: int,
    rm_bg: bool,
    n_proc: int,
    adaptive: bool,
):
    import sys
    import multiprocessing
    from tqdm import tqdm
    from ..defaults import IMAGE_EXTENSIONS

    image_dir_name = image_dir.name
    if out_dir is None:
        out_dir = Path(f'{image_dir_name}_detections')

    image_extensions = IMAGE_EXTENSIONS
    image_files = [
        f for f in image_dir.glob('*') if f.suffix.lower() in image_extensions
    ]
    if len(image_files) < 1:
        sys.stderr.write(f'ERROR: Could not find any image in {image_dir}.')
        sys.exit(1)

    args_list = [
        (
            f,
            out_dir.joinpath(f.with_suffix('').name),
            background_color,
            masks_output,
            bbox_output,
            contour_output,
            padding,
            rm_bg,
            adaptive,
        )
        for f in image_files
    ]
    out_dir.mkdir(parents=True, exist_ok=True)

    with multiprocessing.Pool(processes=n_proc) as pool:
        list(
            tqdm(
                pool.imap_unordered(single_wrapped, args_list),
                total=len(image_files),
            )
        )
